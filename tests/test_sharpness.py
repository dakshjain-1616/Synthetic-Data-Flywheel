"""Tests for judge cache, Cohen's kappa, and calibration CLI commands."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from click.testing import CliRunner

from synthetic_data_flywheel.cli import main
from synthetic_data_flywheel.judge_cache import JudgmentCache
from synthetic_data_flywheel.models import JudgmentResult, QualityScores, RubricRef
from synthetic_data_flywheel.stats import cohens_kappa, pearson, prf


def test_cohens_kappa_perfect_agreement():
    assert cohens_kappa([True, True, False, False], [True, True, False, False]) == 1.0


def test_cohens_kappa_complete_disagreement():
    k = cohens_kappa([True, False, True, False], [False, True, False, True])
    assert k < 0


def test_cohens_kappa_chance():
    # Everyone always says True → κ undefined; implementation returns 0
    assert cohens_kappa([True] * 5, [True] * 5) == 1.0
    assert cohens_kappa([True, True, True, False], [True, False, True, True]) < 1.0


def test_pearson_monotonic():
    assert pearson([1, 2, 3, 4], [2, 4, 6, 8]) == pytest.approx(1.0)
    assert pearson([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert pearson([1, 1, 1], [1, 2, 3]) == 0.0  # zero variance branch


def test_prf_basic():
    m = prf([True, True, False, False], [True, False, True, False])
    assert m["tp"] == 1 and m["fp"] == 1 and m["tn"] == 1 and m["fn"] == 1
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["f1"] == 0.5


def test_judgment_cache_hits_and_writes(tmp_path):
    cache = JudgmentCache(tmp_path / "c", enabled=True)
    key = JudgmentCache.key("pid", "default", 1, "ollama", "gemma")
    assert cache.get(key) is None
    j = JudgmentResult(pair_id="pid", passed=True,
                       scores=QualityScores(coherence=8, accuracy=8, helpfulness=8, overall=8),
                       judge_model="gemma", rubric=RubricRef(name="default", version=1))
    cache.put(key, j)
    got = cache.get(key)
    assert got is not None and got.passed is True
    assert cache.hits == 1 and cache.writes == 1


def test_judgment_cache_disabled_bypass(tmp_path):
    cache = JudgmentCache(tmp_path / "c", enabled=False)
    key = JudgmentCache.key("pid", "default", 1, "ollama", "gemma")
    j = JudgmentResult(pair_id="pid", passed=True, judge_model="gemma")
    cache.put(key, j)
    assert cache.get(key) is None
    assert cache.writes == 0 and cache.misses == 1


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    for key, sub in [("USER_DATA_DIR", "data/user"),
                     ("VALIDATION_DIR", "data/validation"),
                     ("LABELS_DIR", "data/labels"),
                     ("JUDGMENTS_DIR", "data/judgments"),
                     ("REPORT_OUTPUT_DIR", "reports")]:
        monkeypatch.setenv(key, str(tmp_path / sub))
    from synthetic_data_flywheel import config as _cfg
    _cfg.get_settings.cache_clear()
    return tmp_path


def _write_judgments(dir_: Path, dataset: str, tag: str, rows):
    dir_.mkdir(parents=True, exist_ok=True)
    path = dir_ / f"{dataset}.{tag}.jsonl"
    with path.open("w") as f:
        for pid, passed, overall in rows:
            j = JudgmentResult(
                pair_id=pid, passed=passed,
                scores=QualityScores(coherence=overall, accuracy=overall,
                                      helpfulness=overall, overall=overall),
                judge_model="m", tag=tag)
            f.write(json.dumps(j.to_dict()) + "\n")
    return path


def test_compare_cli(workspace):
    j_dir = workspace / "data/judgments"
    _write_judgments(j_dir, "toy", "v1",
                     [("a", True, 9), ("b", True, 8), ("c", False, 3)])
    _write_judgments(j_dir, "toy", "v2",
                     [("a", True, 8), ("b", False, 5), ("c", False, 4)])

    runner = CliRunner()
    r = runner.invoke(main, ["compare", "-d", "toy", "--tags", "v1,v2"])
    assert r.exit_code == 0, r.output
    out = json.loads((workspace / "reports/toy/compare.json").read_text())
    assert out["n_common"] == 3
    assert 0.0 <= out["pass_agreement"] <= 1.0


def test_calibrate_cli(workspace):
    j_dir = workspace / "data/judgments"
    _write_judgments(j_dir, "toy", "v1",
                     [("a", True, 9), ("b", True, 8), ("c", False, 3), ("d", False, 4)])

    # Labels: truth has (a, b) approved, (c, d) rejected  →  judge got all 4 right → F1=1
    labels_path = workspace / "data/labels"
    labels_path.mkdir(parents=True, exist_ok=True)
    with (labels_path / "toy.jsonl").open("w") as f:
        for pid, status in [("a", "approved"), ("b", "approved"),
                             ("c", "rejected"), ("d", "rejected")]:
            f.write(json.dumps({"pair_id": pid, "status": status}) + "\n")

    runner = CliRunner()
    r = runner.invoke(main, ["calibrate", "-d", "toy", "--tag", "v1"])
    assert r.exit_code == 0, r.output
    out = json.loads((workspace / "reports/toy/calibrate.v1.json").read_text())
    assert out["precision"] == 1.0 and out["recall"] == 1.0 and out["f1"] == 1.0
