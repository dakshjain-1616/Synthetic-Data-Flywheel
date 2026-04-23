"""End-to-end test for the YAML pipeline runner."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from synthetic_data_flywheel.cli import main


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    for key, sub in [("USER_DATA_DIR", "data/user"),
                     ("VALIDATION_DIR", "data/validation"),
                     ("LABELS_DIR", "data/labels"),
                     ("JUDGMENTS_DIR", "data/judgments"),
                     ("REPORT_OUTPUT_DIR", "reports"),
                     ("RUBRICS_DIR", str(tmp_path / "rubrics"))]:
        monkeypatch.setenv(key, str(tmp_path / sub) if key != "RUBRICS_DIR" else sub)
    from synthetic_data_flywheel import config as _cfg
    _cfg.get_settings.cache_clear()
    return tmp_path


def test_pipeline_end_to_end_with_mocked_backend(workspace):
    src = workspace / "src.jsonl"
    src.write_text(
        '{"prompt": "What is 2+2?", "completion": "4"}\n'
        '{"prompt": "Say hi.", "completion": "hi"}\n'
    )

    pipeline_yaml = workspace / "p.yaml"
    pipeline_yaml.write_text(f"""
dataset: toy
steps:
  - ingest:
      input: {src}
      format: jsonl
      map: {{instruction: prompt, output: completion}}
  - validate:
      checks: [schema, dedup]
      fail_on: never
      write_clean: true
  - judge:
      backend: ollama
      tag: v1
      max_pairs: 2
      concurrency: 1
  - label:
      mode: auto-from-judge
      judgments: auto
  - export:
      to: {workspace}/data/exports/toy.jsonl
      filter: "label['status'] == 'approved'"
  - visualize: {{}}
""")

    fake_payload = (
        '{"coherence": 8, "accuracy": 8, "helpfulness": 8, "overall": 8, '
        '"passed": true, "reasoning": "ok"}'
    )

    async def fake_generate(self, prompt, *, temperature=None, max_tokens=None):
        return fake_payload

    with patch(
        "synthetic_data_flywheel.judge_backends.ollama.OllamaBackend.generate",
        new=fake_generate,
    ):
        r = CliRunner().invoke(main, ["pipeline", "run", str(pipeline_yaml)])

    assert r.exit_code == 0, r.output
    assert (workspace / "data/user/toy.jsonl").exists()
    assert (workspace / "data/validation/toy.report.json").exists()
    assert (workspace / "data/judgments/toy.v1.jsonl").exists()
    assert (workspace / "data/labels/toy.jsonl").exists()
    assert (workspace / "data/exports/toy.jsonl").exists()
    assert (workspace / "reports/toy/index.html").exists()


def test_pipeline_fail_on_error_stops(workspace):
    # Reference a missing input file; ingest should fail and pipeline should bail.
    (workspace / "p.yaml").write_text("""
dataset: toy
steps:
  - ingest: {input: /no/such/file.jsonl, format: jsonl}
  - visualize: {}
""")
    r = CliRunner().invoke(main, ["pipeline", "run", str(workspace / "p.yaml")])
    assert r.exit_code != 0
    # visualize should not run because ingest failed
    assert not (workspace / "reports/toy/index.html").exists()
