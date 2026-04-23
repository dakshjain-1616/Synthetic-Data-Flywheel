"""End-to-end smoke tests for new CLI commands using Click's runner."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from click.testing import CliRunner

from synthetic_data_flywheel.cli import main


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Point all data dirs at tmp_path/<subdir> via environment variables."""
    for key, sub in [
        ("USER_DATA_DIR", "data/user"),
        ("VALIDATION_DIR", "data/validation"),
        ("LABELS_DIR", "data/labels"),
        ("JUDGMENTS_DIR", "data/judgments"),
        ("RUBRICS_DIR", "rubrics"),
        ("DATA_DIR", "data"),
    ]:
        monkeypatch.setenv(key, str(tmp_path / sub))

    # Invalidate the cached settings so env vars are picked up
    from synthetic_data_flywheel import config as _cfg
    _cfg.get_settings.cache_clear()
    return tmp_path


def _write_sample(path: Path):
    path.write_text(
        '{"prompt": "What is 2+2?", "completion": "4"}\n'
        '{"prompt": "What is 3+3?", "completion": "6"}\n'
        '{"prompt": "Bad example", "completion": ""}\n'
    )


def test_ingest_and_validate(workspace):
    runner = CliRunner()
    src = workspace / "toy.jsonl"
    _write_sample(src)

    r = runner.invoke(main, ["ingest", "-i", str(src), "-n", "toy", "-f", "jsonl",
                              "--map", "instruction=prompt,output=completion"])
    assert r.exit_code == 0, r.output
    assert (workspace / "data/user/toy.jsonl").exists()

    r = runner.invoke(main, ["validate", "-d", "toy", "--checks", "schema,dedup"])
    assert r.exit_code == 0, r.output
    report = json.loads((workspace / "data/validation/toy.report.json").read_text())
    # One pair has empty output => schema error
    assert report["counts"].get("schema", 0) >= 1


def test_dataset_ls_and_info(workspace):
    runner = CliRunner()
    src = workspace / "toy.jsonl"
    _write_sample(src)
    runner.invoke(main, ["ingest", "-i", str(src), "-n", "toy", "-f", "jsonl",
                          "--map", "instruction=prompt,output=completion"])

    r = runner.invoke(main, ["dataset", "ls"])
    assert r.exit_code == 0
    assert "toy" in r.output

    r = runner.invoke(main, ["dataset", "info", "toy"])
    assert r.exit_code == 0
    assert "Dataset: toy" in r.output
    assert "present" in r.output


def test_label_bulk(workspace):
    runner = CliRunner()
    src = workspace / "toy.jsonl"
    _write_sample(src)
    runner.invoke(main, ["ingest", "-i", str(src), "-n", "toy", "-f", "jsonl",
                          "--map", "instruction=prompt,output=completion"])

    r = runner.invoke(main, ["label", "-d", "toy", "--mode", "bulk",
                              "--where", "instruction == 'What is 2+2?'",
                              "--set-status", "approved", "--tag", "math"])
    assert r.exit_code == 0, r.output
    labels_path = workspace / "data/labels/toy.jsonl"
    lines = [json.loads(l) for l in labels_path.read_text().splitlines() if l]
    assert len(lines) == 1
    assert lines[0]["status"] == "approved"


def test_judge_command_end_to_end_with_mocked_backend(workspace):
    runner = CliRunner()
    src = workspace / "toy.jsonl"
    _write_sample(src)
    runner.invoke(main, ["ingest", "-i", str(src), "-n", "toy", "-f", "jsonl",
                          "--map", "instruction=prompt,output=completion"])

    fake_payload = (
        '{"coherence": 8, "accuracy": 8, "helpfulness": 8, '
        '"overall": 8, "passed": true, "reasoning": "ok"}'
    )

    async def fake_generate(self, prompt, *, temperature=None, max_tokens=None):
        return fake_payload

    with patch(
        "synthetic_data_flywheel.judge_backends.ollama.OllamaBackend.generate",
        new=fake_generate,
    ):
        r = runner.invoke(main, ["judge", "-d", "toy", "--backend", "ollama",
                                  "--tag", "t1", "--concurrency", "2"])
    assert r.exit_code == 0, r.output
    out = workspace / "data/judgments/toy.t1.jsonl"
    assert out.exists()
    lines = [json.loads(l) for l in out.read_text().splitlines() if l]
    assert len(lines) == 3
    assert all(j["passed"] for j in lines)


def test_dataset_export_with_filter_and_split(workspace):
    runner = CliRunner()
    src = workspace / "toy.jsonl"
    src.write_text("\n".join(
        json.dumps({"prompt": f"q{i}", "completion": f"a{i}"}) for i in range(10)
    ) + "\n")
    runner.invoke(main, ["ingest", "-i", str(src), "-n", "toy", "-f", "jsonl",
                          "--map", "instruction=prompt,output=completion"])

    export_dir = workspace / "exp"
    export_dir.mkdir()
    r = runner.invoke(main, ["dataset", "export", "toy", "--to", str(export_dir / "t.jsonl"),
                              "--split", "train=0.8,val=0.2"])
    assert r.exit_code == 0, r.output
    train = export_dir / "t.train.jsonl"
    val = export_dir / "t.val.jsonl"
    assert train.exists() and val.exists()
    assert len(train.read_text().strip().splitlines()) == 8
    assert len(val.read_text().strip().splitlines()) == 2
