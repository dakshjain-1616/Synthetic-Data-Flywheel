"""Tests for ingest module."""

import csv
import json
from pathlib import Path

from synthetic_data_flywheel.ingest import DatasetIngestor, load_dataset_jsonl, normalize_row


def test_normalize_row_aliases_prompt_and_completion():
    p = normalize_row({"prompt": "hi", "completion": "hello"}, mapping={})
    assert p.instruction == "hi" and p.output == "hello"


def test_normalize_row_mapping_wins_over_aliases():
    p = normalize_row({"question": "q", "answer": "a"},
                      mapping={"instruction": "question", "output": "answer"})
    assert p.instruction == "q" and p.output == "a"


def test_deterministic_id():
    a = normalize_row({"prompt": "x", "completion": "y"}, mapping={})
    b = normalize_row({"prompt": "x", "completion": "y"}, mapping={})
    assert a.id == b.id


def test_ingest_jsonl_round_trip(tmp_path: Path):
    src = tmp_path / "in.jsonl"
    src.write_text(
        '{"prompt": "a", "completion": "b"}\n'
        '{"prompt": "c", "completion": "d"}\n'
    )
    ing = DatasetIngestor(user_data_dir=str(tmp_path / "user"))
    out, meta = ing.ingest(str(src), name="toy", fmt="jsonl")
    assert meta.row_count == 2
    pairs = load_dataset_jsonl(out)
    assert [p.instruction for p in pairs] == ["a", "c"]
    assert (tmp_path / "user" / "toy.meta.json").exists()


def test_ingest_csv(tmp_path: Path):
    src = tmp_path / "in.csv"
    with src.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["prompt", "completion"])
        w.writerow(["hello", "world"])
    ing = DatasetIngestor(user_data_dir=str(tmp_path / "user"))
    out, meta = ing.ingest(str(src), name="t", fmt="csv")
    assert meta.row_count == 1
    pairs = load_dataset_jsonl(out)
    assert pairs[0].output == "world"


def test_ingest_json_list(tmp_path: Path):
    src = tmp_path / "in.json"
    src.write_text(json.dumps([{"prompt": "a", "completion": "b"}]))
    ing = DatasetIngestor(user_data_dir=str(tmp_path / "user"))
    _, meta = ing.ingest(str(src), name="t", fmt="json")
    assert meta.row_count == 1
