"""Common interface for judge backends."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class JudgeBackend(Protocol):
    """Any LLM client usable for judging implements this protocol."""

    model: str

    async def generate(
        self,
        prompt: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        ...

    def health(self) -> bool:
        ...

    async def close(self) -> None:
        ...
