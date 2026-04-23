"""Factory for judge backends selected by name."""

from __future__ import annotations

from typing import Any

from synthetic_data_flywheel.judge_backends.base import JudgeBackend


def get_backend(name: str, **kwargs: Any) -> JudgeBackend:
    name = (name or "").lower()
    if name == "ollama":
        from synthetic_data_flywheel.judge_backends.ollama import OllamaBackend
        return OllamaBackend(**kwargs)
    if name == "openrouter":
        from synthetic_data_flywheel.judge_backends.openrouter import OpenRouterBackend
        return OpenRouterBackend(**kwargs)
    if name == "anthropic":
        from synthetic_data_flywheel.judge_backends.anthropic import AnthropicBackend
        return AnthropicBackend(**kwargs)
    raise ValueError(f"Unknown judge backend: {name!r}. Expected ollama|openrouter|anthropic.")
