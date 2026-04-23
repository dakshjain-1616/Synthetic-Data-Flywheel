"""Tests for the visualization module."""

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from synthetic_data_flywheel.cli import main
from synthetic_data_flywheel.models import (
    JudgmentResult,
    QualityScores,
    SyntheticPair,
    ValidationReport,
)
from synthetic_data_flywheel.viz import (
    VizInputs,
    chart_category_breakdown,
    chart_criteria_means,
    chart_judge_agreement,
    chart_pass_fail,
    chart_score_histogram,
    render_all,
)


def _mk_pair(**kw):
    return SyntheticPair(**kw)


def _mk_judgment(pair_id, overall, passed=True):
    return JudgmentResult(
        pair_id=pair_id, passed=passed,
        scores=QualityScores(coherence=overall, accuracy=overall,
                              helpfulness=overall, overall=overall),
        judge_model="t",
    )


def test_pass_fail_chart_is_written(tmp_path):
    js = [_mk_judgment("a", 9, True), _mk_judgment("b", 2, False)]
    out = chart_pass_fail(js, tmp_path / "pf.png", "t")
    assert out.exists() and out.stat().st_size > 0


def test_score_histogram_handles_empty(tmp_path):
    out = chart_score_histogram([], tmp_path / "h.png", "t")
    assert out.exists()


def test_criteria_means_with_data(tmp_path):
    js = [_mk_judgment("a", 8), _mk_judgment("b", 6)]
    out = chart_criteria_means(js, tmp_path / "c.png", "t")
    assert out.exists() and out.stat().st_size > 0


def test_category_breakdown(tmp_path):
    pairs = [_mk_pair(instruction="x", output="y", category="qa"),
             _mk_pair(instruction="x2", output="y2", category="qa"),
             _mk_pair(instruction="x3", output="y3", category="creative")]
    out = chart_category_breakdown(pairs, tmp_path / "cat.png", "t")
    assert out.exists()


def test_judge_agreement_requires_two_tags(tmp_path):
    one = {"v1": [_mk_judgment("a", 8, True)]}
    assert chart_judge_agreement(one, tmp_path / "agr.png", "t") is None

    two = {
        "v1": [_mk_judgment("a", 8, True), _mk_judgment("b", 2, False)],
        "v2": [_mk_judgment("a", 7, True), _mk_judgment("b", 5, True)],
    }
    out = chart_judge_agreement(two, tmp_path / "agr.png", "t")
    assert out is not None and out.exists()


def test_render_all_produces_index_and_images(tmp_path):
    pairs = [_mk_pair(instruction=f"q{i}", output=f"a{i}", category="qa")
             for i in range(3)]
    judgments = {"v1": [_mk_judgment(str(p.id), 8, True) for p in pairs]}
    inputs = VizInputs(name="toy", pairs=pairs, judgments_by_tag=judgments,
                        labels={}, validation=None)
    images, index = render_all(inputs, tmp_path)
    assert index.exists()
    html = index.read_text()
    assert "toy" in html and "img src" in html
    for img in images:
        assert img.exists()


def test_visualize_cli_smoke(tmp_path, monkeypatch):
    for key, sub in [("USER_DATA_DIR", "data/user"),
                     ("VALIDATION_DIR", "data/validation"),
                     ("LABELS_DIR", "data/labels"),
                     ("JUDGMENTS_DIR", "data/judgments"),
                     ("REPORT_OUTPUT_DIR", "reports")]:
        monkeypatch.setenv(key, str(tmp_path / sub))
    from synthetic_data_flywheel import config as _cfg
    _cfg.get_settings.cache_clear()

    user = tmp_path / "data/user"
    user.mkdir(parents=True)
    pairs_path = user / "toy.jsonl"
    pairs_path.write_text("\n".join(
        json.dumps({"instruction": f"q{i}", "output": f"a{i}",
                    "category": "qa", "id": f"00000000-0000-0000-0000-00000000000{i}"})
        for i in range(3)
    ) + "\n")

    runner = CliRunner()
    r = runner.invoke(main, ["visualize", "-d", "toy"])
    assert r.exit_code == 0, r.output
    assert (tmp_path / "reports/toy/index.html").exists()
    assert (tmp_path / "reports/toy/categories.png").exists()
