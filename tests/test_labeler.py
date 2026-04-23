"""Tests for labeler module."""

import pytest

from synthetic_data_flywheel.labeler import (
    LabelStore, SafeEval, auto_from_judge, bulk_apply,
)
from synthetic_data_flywheel.models import JudgmentResult, QualityScores, SyntheticPair


def test_safe_eval_basic():
    assert SafeEval("x > 3 and y == 'a'")({"x": 5, "y": "a"}) is True


def test_safe_eval_rejects_dunder():
    with pytest.raises(ValueError):
        SafeEval("__import__('os')")


def test_safe_eval_rejects_attr_access():
    with pytest.raises(ValueError):
        SafeEval("x.upper()")


def test_bulk_apply_filters_on_scores():
    pairs = [SyntheticPair(instruction="a", output="b"),
             SyntheticPair(instruction="c", output="d")]
    judgments = {
        str(pairs[0].id): JudgmentResult(pair_id=pairs[0].id,
                                          scores=QualityScores(overall=9, coherence=9,
                                                                accuracy=9, helpfulness=9),
                                          passed=True, judge_model="t"),
        str(pairs[1].id): JudgmentResult(pair_id=pairs[1].id,
                                          scores=QualityScores(overall=3, coherence=3,
                                                                accuracy=3, helpfulness=3),
                                          passed=False, judge_model="t"),
    }
    labels = bulk_apply(pairs, judgments, expr="scores['overall'] >= 7",
                        status="approved", tag="hq")
    assert len(labels) == 1
    assert str(labels[0].pair_id) == str(pairs[0].id)


def test_auto_from_judge_maps_statuses():
    pair_id = "abc"
    judgments = [
        JudgmentResult(pair_id=pair_id, passed=True,
                        scores=QualityScores(overall=8, coherence=8, accuracy=8, helpfulness=8),
                        judge_model="t"),
        JudgmentResult(pair_id="def", passed=False,
                        scores=QualityScores(overall=2, coherence=2, accuracy=2, helpfulness=2),
                        judge_model="t"),
        JudgmentResult(pair_id="ghi", passed=False,
                        scores=QualityScores(overall=5, coherence=5, accuracy=5, helpfulness=5),
                        judge_model="t"),
    ]
    labels = auto_from_judge(judgments, reject_below=3.5)
    by_id = {str(l.pair_id): l.status for l in labels}
    assert by_id[pair_id] == "approved"
    assert by_id["def"] == "rejected"
    assert by_id["ghi"] == "needs_edit"


def test_label_store_latest_wins(tmp_path):
    store = LabelStore(tmp_path / "lbl.jsonl")
    from synthetic_data_flywheel.models import Label
    store.append(Label(pair_id="x", status="approved"))
    store.append(Label(pair_id="x", status="rejected"))
    loaded = store.load()
    assert loaded["x"].status == "rejected"
