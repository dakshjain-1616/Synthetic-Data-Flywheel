"""Pluggable LLM backends for judging."""

from synthetic_data_flywheel.judge_backends.base import JudgeBackend
from synthetic_data_flywheel.judge_backends.registry import get_backend

__all__ = ["JudgeBackend", "get_backend"]
