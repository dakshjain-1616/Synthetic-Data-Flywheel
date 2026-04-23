"""Pipeline runner — executes a declarative YAML spec end-to-end.

Example `pipeline.yaml`:

    dataset: toy
    steps:
      - ingest:
          input: toy.jsonl
          format: jsonl
          map: {instruction: prompt, output: completion}
      - validate:
          checks: [schema, dedup, pii, length]
          write_clean: true
      - judge:
          backend: ollama
          model: gemma4:latest
          tag: v1
          max_pairs: 10
          concurrency: 2
      - label:
          mode: auto-from-judge
          judgments: auto   # resolves to data/judgments/<dataset>.<tag>.jsonl
      - export:
          to: data/exports/toy.jsonl
          filter: "label['status'] == 'approved'"
          split: {train: 0.8, val: 0.2}
      - visualize: {}

Reuses CLI-layer logic by invoking the same registered Click commands via
`CliRunner`, so behaviour is identical to manual runs and we don't duplicate
argument-handling.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from click.testing import CliRunner


@dataclass
class StepResult:
    name: str
    ok: bool
    args: List[str]
    output: str
    exit_code: int


@dataclass
class PipelineResult:
    dataset: str
    steps: List[StepResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(s.ok for s in self.steps)


def load_pipeline(path: str | Path) -> Dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Step → CLI argv translators
# ---------------------------------------------------------------------------

def _as_map_str(m: Dict[str, str]) -> str:
    return ",".join(f"{k}={v}" for k, v in m.items())


def _resolve_auto_judgments(spec_val: Any, dataset: str, tag: Optional[str],
                            settings) -> Optional[str]:
    if spec_val in (None, False):
        return None
    if spec_val == "auto" or spec_val is True:
        if tag is None:
            return None
        return str(Path(settings.judgments_dir) / f"{dataset}.{tag}.jsonl")
    return str(spec_val)


def _build_argv(step_name: str, spec: Dict[str, Any], dataset: str,
                state: Dict[str, Any], settings) -> List[str]:
    spec = spec or {}
    argv: List[str] = [step_name]

    if step_name == "ingest":
        argv += ["-i", str(spec["input"]), "-n", dataset,
                 "-f", spec.get("format", "auto")]
        mp = spec.get("map") or {}
        if mp:
            argv += ["--map", _as_map_str(mp)]
        if spec.get("tag"): argv += ["--tag", str(spec["tag"])]
        if spec.get("limit") is not None: argv += ["--limit", str(spec["limit"])]
        if spec.get("hf_split"): argv += ["--hf-split", str(spec["hf_split"])]
        if spec.get("dry_run"): argv += ["--dry-run"]

    elif step_name == "validate":
        argv += ["-d", dataset]
        if spec.get("checks"):
            argv += ["--checks", ",".join(spec["checks"])]
        for k, flag in [("min_instruction_len", "--min-instruction-len"),
                         ("max_output_len", "--max-output-len"),
                         ("lang", "--lang"),
                         ("pii", "--pii"),
                         ("fail_on", "--fail-on")]:
            if spec.get(k) is not None:
                argv += [flag, str(spec[k])]
        if spec.get("write_clean"):
            wc = spec["write_clean"]
            target = wc if isinstance(wc, str) else \
                str(Path(settings.user_data_dir) / f"{dataset}.clean.jsonl")
            argv += ["--write-clean", target]

    elif step_name == "judge":
        argv += ["-d", dataset]
        tag = spec.get("tag")
        if tag: argv += ["--tag", str(tag)]
        state["judge_tag"] = tag
        for k, flag in [("rubric", "--rubric"), ("backend", "--backend"),
                         ("model", "--model"), ("concurrency", "--concurrency"),
                         ("max_pairs", "--max-pairs"), ("sample", "--sample")]:
            if spec.get(k) is not None:
                argv += [flag, str(spec[k])]
        if spec.get("no_cache"):
            argv += ["--no-cache"]

    elif step_name == "label":
        argv += ["-d", dataset, "--mode", str(spec["mode"])]
        for k, flag in [("where", "--where"), ("set_status", "--set-status"),
                         ("tag", "--tag"), ("note", "--note"),
                         ("reject_below", "--reject-below")]:
            if spec.get(k) is not None:
                argv += [flag, str(spec[k])]
        judgments = _resolve_auto_judgments(
            spec.get("judgments"), dataset, state.get("judge_tag"), settings)
        if judgments:
            argv += ["--judgments", judgments]
        if spec.get("resume"):
            argv += ["--resume"]

    elif step_name == "export":
        argv = ["dataset", "export", dataset]
        argv += ["--to", str(spec["to"])]
        if spec.get("format"):
            argv += ["--format", str(spec["format"])]
        if spec.get("filter"):
            argv += ["--filter", str(spec["filter"])]
        if spec.get("judgments"):
            j = _resolve_auto_judgments(spec["judgments"], dataset,
                                         state.get("judge_tag"), settings)
            if j: argv += ["--judgments", j]
        sp = spec.get("split")
        if sp:
            argv += ["--split", ",".join(f"{k}={v}" for k, v in sp.items())]
        if spec.get("seed") is not None:
            argv += ["--seed", str(spec["seed"])]

    elif step_name == "visualize":
        argv += ["-d", dataset]
        if spec.get("output"):
            argv += ["-o", str(spec["output"])]

    elif step_name == "compare":
        argv += ["-d", dataset, "--tags", ",".join(spec["tags"])]
        if spec.get("output"):
            argv += ["-o", str(spec["output"])]

    elif step_name == "calibrate":
        argv += ["-d", dataset, "--tag", str(spec["tag"])]
        if spec.get("approved_is"):
            argv += ["--approved-is", str(spec["approved_is"])]
        if spec.get("output"):
            argv += ["-o", str(spec["output"])]

    else:
        raise ValueError(f"Unknown pipeline step: {step_name}")

    return argv


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_pipeline(config: Dict[str, Any], runner: Optional[CliRunner] = None,
                 stop_on_error: bool = True, echo=None) -> PipelineResult:
    from synthetic_data_flywheel.cli import main as cli_main
    from synthetic_data_flywheel.config import get_settings

    dataset = config.get("dataset")
    if not dataset:
        raise ValueError("pipeline config must have a top-level 'dataset' field")
    steps = config.get("steps") or []
    result = PipelineResult(dataset=dataset)
    runner = runner or CliRunner()
    state: Dict[str, Any] = {}
    settings = get_settings()

    for i, step in enumerate(steps, start=1):
        if not isinstance(step, dict) or len(step) != 1:
            raise ValueError(f"step #{i} must be a single-key mapping like '- ingest: {{...}}'")
        (name, spec), = step.items()
        argv = _build_argv(name, spec or {}, dataset, state, settings)
        if echo:
            echo(f"[{i}/{len(steps)}] flywheel {' '.join(shlex.quote(a) for a in argv)}")
        r = runner.invoke(cli_main, argv, catch_exceptions=False)
        step_result = StepResult(
            name=name, ok=r.exit_code == 0, args=argv,
            output=r.output, exit_code=r.exit_code,
        )
        result.steps.append(step_result)
        if echo:
            echo(r.output)
        if not step_result.ok and stop_on_error:
            break

    return result
