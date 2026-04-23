"""Judge module — quality scoring with pluggable rubric + backend.

Keeps the original sync `OllamaClient` + `QualityJudge` surface for backwards
compatibility (used by `engine.py` and existing tests). Adds `AsyncQualityJudge`
for use with any `JudgeBackend` — this is what the `flywheel judge` CLI uses.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Dict, List, Optional, Tuple

import httpx

from synthetic_data_flywheel.config import get_settings
from synthetic_data_flywheel.judge_backends.base import JudgeBackend
from synthetic_data_flywheel.models import JudgmentResult, QualityScores, RubricRef, SyntheticPair
from synthetic_data_flywheel.rubrics import Rubric, default_rubric, render_prompt


QUALITY_RUBRIC_TEMPLATE = """Evaluate this instruction-response pair:

INSTRUCTION:
{instruction}

RESPONSE:
{response}

Rate 0-10 on coherence, accuracy, helpfulness. Return JSON:
{{"coherence": X, "accuracy": X, "helpfulness": X, "overall": X, "passed": true/false, "reasoning": "brief explanation"}}"""


class OllamaClient:
    """Sync client for Ollama — retained for backwards compatibility."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        timeout: Optional[int] = None,
    ):
        s = get_settings()
        self.base_url = base_url or s.ollama_base_url
        self.model = model or s.ollama_model
        self.max_tokens = max_tokens or s.judge_max_tokens
        self.temperature = temperature if temperature is not None else s.judge_temperature
        self.timeout = timeout or s.judge_timeout
        self._client: Optional[httpx.Client] = None

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                base_url=self.base_url, timeout=httpx.Timeout(self.timeout)
            )
        return self._client

    def close(self) -> None:
        if self._client:
            self._client.close()
            self._client = None

    def __enter__(self) -> "OllamaClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
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
        r = client.post("/api/generate", json=payload)
        r.raise_for_status()
        return r.json().get("response", "").strip()

    def check_health(self) -> bool:
        try:
            r = self._get_client().get("/api/tags")
            r.raise_for_status()
            return True
        except Exception:
            return False


def parse_judgment(
    response_text: str,
    *,
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Parse a JSON judgment response. Robust to ```json fences and prose."""
    thresholds = thresholds or {}
    try:
        json_str = response_text
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            json_str = response_text.split("```")[1].split("```")[0]

        # Match innermost JSON object if prose surrounds it
        m = re.search(r"\{[^}]+\}", json_str)
        if m:
            json_str = m.group(0)

        data = json.loads(json_str.strip())

        coherence = float(data.get("coherence", 5.0))
        accuracy = float(data.get("accuracy", 5.0))
        helpfulness = float(data.get("helpfulness", 5.0))
        overall = float(data.get("overall", (coherence + accuracy + helpfulness) / 3))

        passed = data.get("passed")
        if passed is None:
            passed = (
                overall >= thresholds.get("overall", 7.0)
                and coherence >= thresholds.get("coherence", 6.0)
                and accuracy >= thresholds.get("accuracy", 6.0)
                and helpfulness >= thresholds.get("helpfulness", 6.0)
            )
        else:
            passed = bool(passed)

        return {
            "coherence": coherence,
            "accuracy": accuracy,
            "helpfulness": helpfulness,
            "overall": overall,
            "passed": passed,
            "reasoning": data.get("reasoning", "No reasoning provided"),
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


class QualityJudge:
    """Sync judge used by the flywheel engine.

    Back-compat: `client` defaults to a local `OllamaClient`; the default rubric
    renders the same prompt the tool used historically.
    """

    def __init__(
        self,
        client: Optional[OllamaClient] = None,
        min_overall_score: Optional[float] = None,
        min_coherence: Optional[float] = None,
        min_accuracy: Optional[float] = None,
        min_helpfulness: Optional[float] = None,
        rubric: Optional[Rubric] = None,
    ):
        s = get_settings()
        self.client = client or OllamaClient()
        self.min_overall_score = min_overall_score or s.quality_min_score
        self.min_coherence = min_coherence or s.quality_min_coherence
        self.min_accuracy = min_accuracy or s.quality_min_accuracy
        self.min_helpfulness = min_helpfulness or s.quality_min_helpfulness
        self.rubric = rubric or default_rubric()

    def __enter__(self) -> "QualityJudge":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.client.close()

    def _thresholds(self) -> Dict[str, float]:
        return {
            "overall": self.min_overall_score,
            "coherence": self.min_coherence,
            "accuracy": self.min_accuracy,
            "helpfulness": self.min_helpfulness,
        }

    def _parse_judgment(self, response_text: str) -> Dict[str, Any]:
        return parse_judgment(response_text, thresholds=self._thresholds())

    def _build_prompt(self, pair: SyntheticPair) -> str:
        # Default rubric mirrors legacy prompt; custom rubrics render via Jinja2.
        if self.rubric.name == "default":
            return QUALITY_RUBRIC_TEMPLATE.format(
                instruction=pair.instruction,
                response=pair.output[:2000],
            )
        return render_prompt(self.rubric, pair)

    def judge(self, pair: SyntheticPair) -> JudgmentResult:
        prompt = self._build_prompt(pair)
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
            rubric=RubricRef(name=self.rubric.name, version=self.rubric.version),
        )

    def judge_batch(self, pairs: List[SyntheticPair]) -> List[JudgmentResult]:
        results: List[JudgmentResult] = []
        for pair in pairs:
            try:
                results.append(self.judge(pair))
            except Exception:
                results.append(
                    JudgmentResult(
                        pair_id=pair.id,
                        scores=QualityScores(coherence=0.0, accuracy=0.0, helpfulness=0.0, overall=0.0),
                        passed=False,
                        judge_model=self.client.model,
                        judgment_reasoning="Judgment failed",
                    )
                )
        return results

    def filter_pairs(
        self,
        pairs: List[SyntheticPair],
        judgments: List[JudgmentResult],
    ) -> Tuple[List[SyntheticPair], List[SyntheticPair]]:
        judgment_map = {str(j.pair_id): j for j in judgments}
        passed_pairs: List[SyntheticPair] = []
        failed_pairs: List[SyntheticPair] = []
        for pair in pairs:
            j = judgment_map.get(str(pair.id))
            if j and j.passed:
                passed_pairs.append(pair)
            else:
                failed_pairs.append(pair)
        return passed_pairs, failed_pairs


class AsyncQualityJudge:
    """Async judge over an arbitrary `JudgeBackend` with concurrency + custom rubric."""

    def __init__(
        self,
        backend: JudgeBackend,
        rubric: Optional[Rubric] = None,
        thresholds: Optional[Dict[str, float]] = None,
        tag: Optional[str] = None,
        cache=None,  # JudgmentCache | None — optional to avoid circular import
        backend_name: str = "",
    ):
        s = get_settings()
        self.backend = backend
        self.rubric = rubric or default_rubric()
        self.thresholds = thresholds or {
            "overall": s.quality_min_score,
            "coherence": s.quality_min_coherence,
            "accuracy": s.quality_min_accuracy,
            "helpfulness": s.quality_min_helpfulness,
        }
        self.tag = tag
        self.cache = cache
        self.backend_name = backend_name

    def _cache_key(self, pair: SyntheticPair) -> str:
        from synthetic_data_flywheel.judge_cache import JudgmentCache
        return JudgmentCache.key(
            pair_id=str(pair.id),
            rubric_name=self.rubric.name,
            rubric_version=self.rubric.version,
            backend=self.backend_name or type(self.backend).__name__,
            model=getattr(self.backend, "model", ""),
        )

    async def judge(self, pair: SyntheticPair) -> JudgmentResult:
        if self.cache is not None:
            key = self._cache_key(pair)
            cached = self.cache.get(key)
            if cached is not None:
                # Re-stamp tag so downstream filtering by tag still works
                cached.tag = self.tag
                return cached

        prompt = render_prompt(self.rubric, pair)
        temp = self.rubric.temperature_override
        try:
            text = await self.backend.generate(prompt, temperature=temp)
            result = parse_judgment(text, thresholds=self.thresholds)
        except Exception as e:
            reason = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
            return JudgmentResult(
                pair_id=pair.id,
                scores=QualityScores(coherence=0.0, accuracy=0.0, helpfulness=0.0, overall=0.0),
                passed=False,
                judge_model=getattr(self.backend, "model", ""),
                judgment_reasoning=f"Judgment failed: {reason}",
                tag=self.tag,
                rubric=RubricRef(name=self.rubric.name, version=self.rubric.version),
            )
        judgment = JudgmentResult(
            pair_id=pair.id,
            scores=QualityScores(
                coherence=result["coherence"],
                accuracy=result["accuracy"],
                helpfulness=result["helpfulness"],
                overall=result["overall"],
            ),
            passed=result["passed"],
            judge_model=getattr(self.backend, "model", ""),
            judgment_reasoning=result["reasoning"],
            tag=self.tag,
            rubric=RubricRef(name=self.rubric.name, version=self.rubric.version),
        )
        if self.cache is not None:
            self.cache.put(self._cache_key(pair), judgment)
        return judgment

    async def judge_batch(
        self,
        pairs: List[SyntheticPair],
        concurrency: int = 4,
    ) -> List[JudgmentResult]:
        sem = asyncio.Semaphore(max(1, concurrency))

        async def _one(p: SyntheticPair) -> JudgmentResult:
            async with sem:
                return await self.judge(p)

        return await asyncio.gather(*(_one(p) for p in pairs))

    async def close(self) -> None:
        await self.backend.close()


def create_judge(
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    **kwargs,
) -> QualityJudge:
    """Legacy factory — constructs a sync QualityJudge over Ollama."""
    settings = get_settings()
    client = OllamaClient(
        model=model or settings.ollama_model,
        base_url=base_url or settings.ollama_base_url,
    )
    return QualityJudge(client=client, **kwargs)
