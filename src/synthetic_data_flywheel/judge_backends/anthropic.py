"""Anthropic judge backend (Claude via /v1/messages)."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import httpx

from synthetic_data_flywheel.config import get_settings


class AnthropicBackend:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.anthropic.com",
        model: str = "claude-haiku-4-5-20251001",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        timeout: Optional[int] = None,
    ):
        s = get_settings()
        self.api_key = api_key or getattr(s, "anthropic_api_key", None) or os.getenv("ANTHROPIC_API_KEY")
        self.base_url = base_url
        self.model = model
        self.max_tokens = max_tokens or s.judge_max_tokens
        self.temperature = temperature if temperature is not None else s.judge_temperature
        self.timeout = timeout or s.judge_timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def _aclient(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={
                    "x-api-key": self.api_key or "",
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
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
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        r = await client.post("/v1/messages", json=payload)
        r.raise_for_status()
        data = r.json()
        parts = data.get("content", [])
        return "".join(p.get("text", "") for p in parts if p.get("type") == "text").strip()

    def health(self) -> bool:
        return bool(self.api_key)

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
