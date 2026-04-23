"""Rubric loading & prompt rendering for LLM-as-judge.

Rubrics are YAML/JSON files that describe how to judge a pair: the scoring
criteria, pass condition, prompt template (Jinja2), and the expected output
schema. Shipping rubrics live under `./rubrics/`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from jinja2 import Environment, StrictUndefined
from pydantic import BaseModel, Field

from synthetic_data_flywheel.models import SyntheticPair


class Criterion(BaseModel):
    id: str
    prompt: Optional[str] = None
    scale: List[float] = Field(default_factory=lambda: [0.0, 10.0])


class Rubric(BaseModel):
    name: str
    version: int = 1
    description: str = ""
    criteria: List[Criterion] = Field(default_factory=list)
    pass_condition: Optional[str] = None
    prompt_template: str
    output_schema: Dict[str, Any] = Field(default_factory=dict)
    temperature_override: Optional[float] = None

    def score_keys(self) -> List[str]:
        return [c.id for c in self.criteria]


_JINJA = Environment(undefined=StrictUndefined, autoescape=False)


def load_rubric(path: str | Path) -> Rubric:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Rubric not found: {p}")
    text = p.read_text(encoding="utf-8")
    data = yaml.safe_load(text) if p.suffix.lower() in (".yaml", ".yml") else json.loads(text)
    return Rubric(**data)


def render_prompt(rubric: Rubric, pair: SyntheticPair) -> str:
    tpl = _JINJA.from_string(rubric.prompt_template)
    return tpl.render(pair=pair, rubric=rubric)


def default_rubric() -> Rubric:
    """Fallback rubric matching the previously hardcoded prompt."""
    return Rubric(
        name="default",
        version=1,
        description="General instruction/response quality.",
        criteria=[
            Criterion(id="coherence"),
            Criterion(id="accuracy"),
            Criterion(id="helpfulness"),
        ],
        pass_condition=(
            "overall >= 7 and coherence >= 6 and accuracy >= 6 and helpfulness >= 6"
        ),
        temperature_override=0.3,
        prompt_template=(
            "Evaluate this instruction-response pair:\n\n"
            "INSTRUCTION:\n{{ pair.instruction }}\n\n"
            "RESPONSE:\n{{ pair.output[:2000] }}\n\n"
            "Rate 0-10 on coherence, accuracy, helpfulness. Return JSON:\n"
            "{\"coherence\": X, \"accuracy\": X, \"helpfulness\": X, "
            "\"overall\": X, \"passed\": true/false, "
            "\"reasoning\": \"brief explanation\"}"
        ),
        output_schema={
            "type": "object",
            "required": ["coherence", "accuracy", "helpfulness", "overall"],
        },
    )
