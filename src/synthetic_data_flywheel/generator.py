"""Generator module - OpenRouter client for Qwen3 8B, prompt templates for QA/instruction generation."""

import asyncio
import json
from typing import Any, Dict, List, Optional
from uuid import UUID

import httpx

from synthetic_data_flywheel.config import get_settings
from synthetic_data_flywheel.models import SyntheticPair


# Prompt Templates
QA_GENERATION_TEMPLATE = """Generate a question-answer pair based on this topic:

{seed}

Respond in JSON format:
{{
    "instruction": "The question",
    "output": "The detailed answer",
    "category": "qa"
}}"""

INSTRUCTION_GENERATION_TEMPLATE = """Generate an instruction-following training example based on this topic:

{seed}

Respond in JSON format:
{{
    "instruction": "The instruction",
    "output": "The ideal response",
    "category": "instruction"
}}"""

REASONING_GENERATION_TEMPLATE = """Generate a reasoning problem with step-by-step solution based on this topic:

{seed}

Respond in JSON format:
{{
    "instruction": "The problem",
    "output": "The step-by-step solution",
    "category": "reasoning"
}}"""

CREATIVE_GENERATION_TEMPLATE = """Generate a creative writing task based on this topic:

{seed}

Respond in JSON format:
{{
    "instruction": "The creative task",
    "output": "An example response",
    "category": "creative"
}}"""


class PromptTemplate:
    """Available prompt templates for generation."""
    
    QA = "QA"
    INSTRUCTION = "INSTRUCTION"
    REASONING = "REASONING"
    CREATIVE = "CREATIVE"
    
    TEMPLATES = {
        QA: QA_GENERATION_TEMPLATE,
        INSTRUCTION: INSTRUCTION_GENERATION_TEMPLATE,
        REASONING: REASONING_GENERATION_TEMPLATE,
        CREATIVE: CREATIVE_GENERATION_TEMPLATE,
    }
    
    @classmethod
    def get(cls, template_type: str) -> str:
        """Get template by type."""
        # Try exact match first, then case-insensitive
        if template_type in cls.TEMPLATES:
            return cls.TEMPLATES[template_type]
        # Try lowercase match
        lower_type = template_type.lower()
        for key, template in cls.TEMPLATES.items():
            if key.lower() == lower_type:
                return template
        # Default to instruction if not found
        return cls.TEMPLATES[cls.INSTRUCTION]
    
    @classmethod
    def list_templates(cls) -> List[str]:
        """List available template types."""
        return list(cls.TEMPLATES.keys())


class OpenRouterClient:
    """Client for OpenRouter API to generate synthetic data."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        timeout: Optional[int] = None,
    ):
        """Initialize OpenRouter client."""
        settings = get_settings()
        
        self.api_key = api_key or settings.openrouter_api_key
        self.base_url = base_url or settings.openrouter_base_url
        self.model = model or settings.openrouter_model
        self.max_tokens = max_tokens or settings.generator_max_tokens
        self.temperature = temperature or settings.generator_temperature
        self.top_p = top_p or settings.generator_top_p
        self.timeout = timeout or settings.generator_timeout
        
        self._client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://synthetic-data-flywheel.local",
                    "X-Title": "Synthetic Data Flywheel",
                },
                timeout=httpx.Timeout(self.timeout),
            )
        return self._client
    
    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
    
    async def __aenter__(self) -> "OpenRouterClient":
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Generate text using OpenRouter API."""
        client = await self._get_client()
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature or self.temperature,
            "top_p": self.top_p,
        }
        
        response = await client.post("/chat/completions", json=payload)
        response.raise_for_status()
        
        data = response.json()
        return data["choices"][0]["message"]["content"]
    
    async def generate_synthetic_pair(
        self,
        seed: str,
        template_type: str = PromptTemplate.INSTRUCTION,
        cycle_id: Optional[int] = None,
        system_prompt: Optional[str] = None,
    ) -> SyntheticPair:
        """Generate a synthetic training pair."""
        template = PromptTemplate.get(template_type)
        prompt = template.format(seed=seed)
        
        response_text = await self.generate(prompt, system_prompt)
        
        # Parse JSON response
        try:
            json_str = response_text
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                json_str = response_text.split("```")[1].split("```")[0]
            
            data = json.loads(json_str.strip())
            
            return SyntheticPair(
                instruction=data.get("instruction", ""),
                output=data.get("output", ""),
                context=data.get("context"),
                category=data.get("category"),
                difficulty=data.get("difficulty"),
                source_seed=seed,
                cycle_id=cycle_id,
            )
            
        except json.JSONDecodeError:
            # Create a pair with raw response for debugging
            return SyntheticPair(
                instruction=seed,
                output=response_text,
                source_seed=seed,
                cycle_id=cycle_id,
            )
    
    async def generate_batch(
        self,
        seeds: List[str],
        template_type: str = PromptTemplate.INSTRUCTION,
        cycle_id: Optional[int] = None,
        max_concurrent: int = 5,
    ) -> List[SyntheticPair]:
        """Generate multiple synthetic pairs concurrently."""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def generate_with_limit(seed: str) -> SyntheticPair:
            async with semaphore:
                try:
                    return await self.generate_synthetic_pair(
                        seed=seed,
                        template_type=template_type,
                        cycle_id=cycle_id,
                    )
                except Exception:
                    return SyntheticPair(
                        instruction=seed,
                        output="[Generation failed]",
                        source_seed=seed,
                        cycle_id=cycle_id,
                    )
        
        tasks = [generate_with_limit(seed) for seed in seeds]
        return await asyncio.gather(*tasks)


def create_generator(
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs
) -> OpenRouterClient:
    """Factory function to create a generator client."""
    settings = get_settings()
    return OpenRouterClient(
        api_key=api_key or settings.openrouter_api_key,
        model=model or settings.openrouter_model,
        **kwargs
    )
