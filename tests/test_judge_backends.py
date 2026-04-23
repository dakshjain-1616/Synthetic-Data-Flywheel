"""Tests for judge_backends: registry + Ollama backend."""

from unittest.mock import AsyncMock, patch

import pytest

from synthetic_data_flywheel.judge_backends import get_backend
from synthetic_data_flywheel.judge_backends.base import JudgeBackend
from synthetic_data_flywheel.judge_backends.ollama import OllamaBackend


def test_registry_returns_correct_types():
    assert isinstance(get_backend("ollama"), JudgeBackend)
    assert isinstance(get_backend("openrouter"), JudgeBackend)
    assert isinstance(get_backend("anthropic"), JudgeBackend)


def test_registry_unknown_raises():
    with pytest.raises(ValueError):
        get_backend("does-not-exist")


@pytest.mark.asyncio
async def test_ollama_backend_generate_posts_correct_payload():
    be = OllamaBackend(base_url="http://x", model="m1")

    class FakeResp:
        def raise_for_status(self): pass
        def json(self): return {"response": "  hello  "}

    async def fake_post(path, json=None):
        assert path == "/api/generate"
        assert json["model"] == "m1"
        assert json["prompt"] == "P"
        return FakeResp()

    client = AsyncMock()
    client.post = fake_post
    client.is_closed = False
    with patch.object(be, "_aclient", AsyncMock(return_value=client)):
        out = await be.generate("P")
    assert out == "hello"
