"""Tests for validator module."""

from synthetic_data_flywheel.models import SyntheticPair
from synthetic_data_flywheel.validator import Validator


def test_schema_flags_empty_fields():
    pairs = [SyntheticPair(instruction="", output="x"),
             SyntheticPair(instruction="y", output="")]
    report = Validator().validate(pairs, checks=["schema"])
    assert report.counts.get("schema") == 2
    assert report.counts.get("severity:error") == 2


def test_dedup_flags_duplicates():
    pairs = [SyntheticPair(instruction="a", output="b"),
             SyntheticPair(instruction="a", output="b"),
             SyntheticPair(instruction="c", output="d")]
    report = Validator().validate(pairs, checks=["dedup"])
    assert report.counts.get("dedup") == 1


def test_pii_detects_email():
    pairs = [SyntheticPair(instruction="contact me at foo@bar.com", output="ok")]
    report = Validator().validate(pairs, checks=["pii"])
    assert report.counts.get("pii") == 1


def test_length_check_bounds():
    pairs = [SyntheticPair(instruction="hi", output="ok")]  # instruction < 3
    report = Validator(options={"min_instruction_len": 3, "max_output_len": 10}).validate(
        pairs, checks=["length"])
    assert report.counts.get("length") == 1


def test_filter_clean_drops_errors_and_duplicates():
    pairs = [SyntheticPair(instruction="", output="x"),        # schema error
             SyntheticPair(instruction="a", output="b"),        # ok
             SyntheticPair(instruction="a", output="b")]        # dup
    v = Validator()
    report = v.validate(pairs, checks=["schema", "dedup"])
    cleaned = v.filter_clean(pairs, report)
    assert len(cleaned) == 1
    assert cleaned[0].instruction == "a"
