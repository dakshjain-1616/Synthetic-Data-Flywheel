"""Judge module - Ollama client for Gemma4 E2B, quality scoring rubric, filtering logic."""

import json
import re
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

import httpx

from synthetic_data_flywheel.config import get_settings
from synthetic_data_flywheel.models import JudgmentResult, QualityScores, SyntheticPair


QUALITY_RUBRIC_TEMPLATE = """Evaluate this instruction-response pair:

INSTRUCTION:
{instruction}

RESPONSE:
{response}

Rate 0-10 on coherence, accuracy, helpfulness. Return JSON:
{{"coherence": X, "accuracy": X, "helpfulness": X, "overall": X, "passed": true/false, "reasoning": "brief explanation"}}"""


class OllamaClient:
    """Client for Ollama local LLM API."""
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        timeout: Optional[int] = None,
    ):
        """Initialize Ollama client."""
        settings = get_settings()
        
        self.base_url = base_url or settings.ollama_base_url
        self.model = model or settings.ollama_model
        self.max_tokens = max_tokens or settings.judge_max_tokens
        self.temperature = temperature or settings.judge_temperature
        self.timeout = timeout or settings.judge_timeout
        
        self._client: Optional[httpx.Client] = None
    
    def _get_client(self) -> httpx.Client:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.Client(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout),
            )
        return self._client
    
    def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            self._client.close()
            self._client = None
    
    def __enter__(self) -> "OllamaClient":
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text using Ollama API."""
        client = self._get_client()
        
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        response = client.post("/api/generate", json=payload)
        response.raise_for_status()
        
        data = response.json()
        return data.get("response", "").strip()
    
    def check_health(self) -> bool:
        """Check if Ollama server is available."""
        try:
            client = self._get_client()
            response = client.get("/api/tags")
            response.raise_for_status()
            return True
        except Exception:
            return False


class QualityJudge:
    """Judge for assessing quality of synthetic data pairs."""
    
    def __init__(
        self,
        client: Optional[OllamaClient] = None,
        min_overall_score: Optional[float] = None,
        min_coherence: Optional[float] = None,
        min_accuracy: Optional[float] = None,
        min_helpfulness: Optional[float] = None,
    ):
        """Initialize quality judge."""
        settings = get_settings()
        
        self.client = client or OllamaClient()
        self.min_overall_score = min_overall_score or settings.quality_min_score
        self.min_coherence = min_coherence or settings.quality_min_coherence
        self.min_accuracy = min_accuracy or settings.quality_min_accuracy
        self.min_helpfulness = min_helpfulness or settings.quality_min_helpfulness
    
    def __enter__(self) -> "QualityJudge":
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.client.close()
    
    def _parse_judgment(self, response_text: str) -> Dict[str, Any]:
        """Parse judgment response into structured scores."""
        try:
            json_str = response_text
            
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                json_str = response_text.split("```")[1].split("```")[0]
            
            json_match = re.search(r'\{[^}]+\}', json_str)
            if json_match:
                json_str = json_match.group(0)
            
            data = json.loads(json_str.strip())
            
            coherence = float(data.get("coherence", 5.0))
            accuracy = float(data.get("accuracy", 5.0))
            helpfulness = float(data.get("helpfulness", 5.0))
            overall = float(data.get("overall", (coherence + accuracy + helpfulness) / 3))
            
            passed = data.get("passed")
            if passed is None:
                passed = (
                    overall >= self.min_overall_score
                    and coherence >= self.min_coherence
                    and accuracy >= self.min_accuracy
                    and helpfulness >= self.min_helpfulness
                )
            else:
                passed = bool(passed)
            
            reasoning = data.get("reasoning", "No reasoning provided")
            
            return {
                "coherence": coherence,
                "accuracy": accuracy,
                "helpfulness": helpfulness,
                "overall": overall,
                "passed": passed,
                "reasoning": reasoning,
            }
            
        except (json.JSONDecodeError, ValueError, KeyError):
            return {
                "coherence": 5.0,
                "accuracy": 5.0,
                "helpfulness": 5.0,
                "overall": 5.0,
                "passed": False,
                "reasoning": "Parse error",
            }
    
    def judge(self, pair: SyntheticPair) -> JudgmentResult:
        """Judge a single synthetic pair."""
        prompt = QUALITY_RUBRIC_TEMPLATE.format(
            instruction=pair.instruction,
            response=pair.output[:2000],
        )
        
        response_text = self.client.generate(prompt)
        result = self._parse_judgment(response_text)
        
        scores = QualityScores(
            coherence=result["coherence"],
            accuracy=result["accuracy"],
            helpfulness=result["helpfulness"],
            overall=result["overall"],
        )
        
        return JudgmentResult(
            pair_id=pair.id,
            scores=scores,
            passed=result["passed"],
            judge_model=self.client.model,
            judgment_reasoning=result["reasoning"],
        )
    
    def judge_batch(self, pairs: List[SyntheticPair]) -> List[JudgmentResult]:
        """Judge multiple pairs sequentially."""
        results = []
        for pair in pairs:
            try:
                result = self.judge(pair)
                results.append(result)
            except Exception:
                results.append(JudgmentResult(
                    pair_id=pair.id,
                    scores=QualityScores(
                        coherence=0.0,
                        accuracy=0.0,
                        helpfulness=0.0,
                        overall=0.0,
                    ),
                    passed=False,
                    judge_model=self.client.model,
                    judgment_reasoning="Judgment failed",
                ))
        
        return results
    
    def filter_pairs(
        self,
        pairs: List[SyntheticPair],
        judgments: List[JudgmentResult],
    ) -> Tuple[List[SyntheticPair], List[JudgmentResult]]:
        """Filter pairs based on judgments."""
        passed_pairs = []
        passed_judgments = []
        
        # Create judgment map using string comparison for IDs
        judgment_map = {}
        for j in judgments:
            # Handle both UUID and string IDs
            key = str(j.pair_id)
            judgment_map[key] = j
        
        for pair in pairs:
            # Use string comparison for pair ID
            pair_id_str = str(pair.id)
            judgment = judgment_map.get(pair_id_str)
            if judgment and judgment.passed:
                passed_pairs.append(pair)
                passed_judgments.append(judgment)
        
        return passed_pairs, passed_judgments


def create_judge(
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    **kwargs
) -> QualityJudge:
    """Factory function to create a quality judge."""
    settings = get_settings()
    client = OllamaClient(
        model=model or settings.ollama_model,
        base_url=base_url or settings.ollama_base_url,
    )
    return QualityJudge(client=client, **kwargs)
