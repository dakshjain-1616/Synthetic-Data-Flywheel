"""OpenRouter judge backend."""

from __future__ import annotations

from typing import Any, Dict, Optional

import httpx

from synthetic_data_flywheel.config import get_settings


class OpenRouterBackend:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        timeout: Optional[int] = None,
    ):
        s = get_settings()
        self.api_key = api_key or s.openrouter_api_key
        self.base_url = base_url or s.openrouter_base_url
        self.model = model or s.openrouter_model
        self.max_tokens = max_tokens or s.judge_max_tokens
        self.temperature = temperature if temperature is not None else s.judge_temperature
        self.timeout = timeout or s.judge_timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def _aclient(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key or ''}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://synthetic-data-flywheel.local",
                    "X-Title": "Synthetic Data Flywheel (Judge)",
                },
                timeout=httpx.Timeout(self.timeout),
            )
        return self._client

    async def generate(
        self,
        prompt: str,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        client = await self._aclient()
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
        }
        r = await client.post("/chat/completions", json=payload)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    def health(self) -> bool:
        return bool(self.api_key)

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
