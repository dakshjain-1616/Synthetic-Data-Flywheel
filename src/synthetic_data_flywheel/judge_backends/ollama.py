"""Ollama judge backend — async wrapper over the local Ollama server."""

from __future__ import annotations

from typing import Any, Dict, Optional

import httpx

from synthetic_data_flywheel.config import get_settings


class OllamaBackend:
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
        self._client: Optional[httpx.AsyncClient] = None

    def _client_sync(self) -> httpx.Client:
        return httpx.Client(base_url=self.base_url, timeout=httpx.Timeout(self.timeout))

    async def _aclient(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url, timeout=httpx.Timeout(self.timeout)
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
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature if temperature is not None else self.temperature,
                "num_predict": max_tokens or self.max_tokens,
            },
        }
        r = await client.post("/api/generate", json=payload)
        r.raise_for_status()
        return r.json().get("response", "").strip()

    def health(self) -> bool:
        try:
            with self._client_sync() as c:
                r = c.get("/api/tags")
                r.raise_for_status()
            return True
        except Exception:
            return False

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
