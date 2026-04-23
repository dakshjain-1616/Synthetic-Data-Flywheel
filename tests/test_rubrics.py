"""Tests for rubric loading and rendering."""

from pathlib import Path

import pytest

from synthetic_data_flywheel.models import SyntheticPair
from synthetic_data_flywheel.rubrics import Rubric, default_rubric, load_rubric, render_prompt


def test_default_rubric_has_criteria():
    r = default_rubric()
    assert r.name == "default"
    assert {c.id for c in r.criteria} == {"coherence", "accuracy", "helpfulness"}


def test_render_default_prompt_has_instruction_and_output():
    r = default_rubric()
    p = SyntheticPair(instruction="What is 2+2?", output="4")
    prompt = render_prompt(r, p)
    assert "What is 2+2?" in prompt
    assert "4" in prompt


def test_load_rubric_yaml(tmp_path: Path):
    y = tmp_path / "r.yaml"
    y.write_text(
        "name: t\nversion: 2\ncriteria: [{id: a}, {id: b}]\n"
        "prompt_template: 'INST: {{ pair.instruction }}'\n"
    )
    r = load_rubric(y)
    assert r.name == "t" and r.version == 2
    assert r.score_keys() == ["a", "b"]
    assert "INST:" in r.prompt_template


def test_load_rubric_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_rubric(tmp_path / "missing.yaml")
