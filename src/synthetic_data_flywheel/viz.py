"""Dataset visualizations — PNGs + an index.html.

Produces a set of charts from a dataset's on-disk artifacts (pairs, judgments,
labels, validation report). Uses matplotlib (non-interactive Agg backend) so
the CLI works headless.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt

from synthetic_data_flywheel.ingest import load_dataset_jsonl
from synthetic_data_flywheel.models import (
    JudgmentResult,
    Label,
    SyntheticPair,
    ValidationReport,
)


# Keep each chart small and consistent
_FIGSIZE = (7.5, 4.5)
_DPI = 140
plt.rcParams.update({"figure.autolayout": True, "axes.spines.top": False,
                     "axes.spines.right": False})


@dataclass
class VizInputs:
    name: str
    pairs: List[SyntheticPair]
    judgments_by_tag: Dict[str, List[JudgmentResult]]
    labels: Dict[str, Label]
    validation: Optional[ValidationReport]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_judgments(path: Path) -> List[JudgmentResult]:
    out: List[JudgmentResult] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(JudgmentResult.from_dict(json.loads(line)))
    return out


def _load_labels(path: Path) -> Dict[str, Label]:
    out: Dict[str, Label] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                lbl = Label.from_dict(json.loads(line))
            except Exception:
                continue
            out[str(lbl.pair_id)] = lbl
    return out


def load_inputs(
    dataset: str,
    *,
    user_data_dir: str,
    judgments_dir: str,
    labels_dir: str,
    validation_dir: str,
) -> VizInputs:
    pairs_path = Path(user_data_dir) / f"{dataset}.jsonl"
    pairs = load_dataset_jsonl(pairs_path) if pairs_path.exists() else []

    judgments_by_tag: Dict[str, List[JudgmentResult]] = {}
    j_dir = Path(judgments_dir)
    if j_dir.exists():
        for jp in sorted(j_dir.glob(f"{dataset}.*.jsonl")):
            tag = jp.name[len(f"{dataset}."):-len(".jsonl")]
            judgments_by_tag[tag] = _load_judgments(jp)

    labels = _load_labels(Path(labels_dir) / f"{dataset}.jsonl")

    validation: Optional[ValidationReport] = None
    vp = Path(validation_dir) / f"{dataset}.report.json"
    if vp.exists():
        raw = json.loads(vp.read_text())
        issues = [i for i in raw.get("issues", [])]
        validation = ValidationReport(
            dataset=raw.get("dataset", dataset),
            total_pairs=raw.get("total_pairs", len(pairs)),
            counts=raw.get("counts", {}),
            issues=[],  # details not needed for aggregate plots
        )
        # Keep the raw issues on the side as a list of dicts for chart use
        validation.issues = issues  # type: ignore[assignment]

    return VizInputs(
        name=dataset, pairs=pairs, judgments_by_tag=judgments_by_tag,
        labels=labels, validation=validation,
    )


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def chart_pass_fail(judgments: List[JudgmentResult], out: Path, title: str) -> Path:
    passed = sum(1 for j in judgments if j.passed)
    failed = len(judgments) - passed
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    bars = ax.bar(["passed", "failed"], [passed, failed], color=["#4caf50", "#e57373"])
    for b, n in zip(bars, [passed, failed]):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), str(n),
                ha="center", va="bottom", fontsize=11)
    ax.set_title(title)
    ax.set_ylabel("count")
    return _save(fig, out)


def chart_score_histogram(judgments: List[JudgmentResult], out: Path, title: str) -> Path:
    scores = [j.scores.overall for j in judgments]
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    ax.hist(scores, bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], color="#5c6bc0",
            edgecolor="white")
    ax.set_xlabel("overall score (0-10)")
    ax.set_ylabel("pairs")
    ax.set_title(title)
    ax.set_xticks(range(0, 11))
    return _save(fig, out)


def chart_criteria_means(judgments: List[JudgmentResult], out: Path, title: str) -> Path:
    if not judgments:
        return _empty(out, title)
    keys = ["coherence", "accuracy", "helpfulness", "overall"]
    means = [
        sum(getattr(j.scores, k) for j in judgments) / len(judgments) for k in keys
    ]
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    ax.bar(keys, means, color=["#26a69a", "#42a5f5", "#ab47bc", "#ffa726"])
    for i, m in enumerate(means):
        ax.text(i, m, f"{m:.2f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, 10.5)
    ax.set_ylabel("mean score")
    ax.set_title(title)
    return _save(fig, out)


def chart_category_breakdown(pairs: List[SyntheticPair], out: Path, title: str) -> Path:
    counts = Counter((p.category or "(uncategorized)") for p in pairs)
    items = counts.most_common(20)
    if not items:
        return _empty(out, title)
    labels, values = zip(*items)
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    ax.barh(labels, values, color="#5c6bc0")
    ax.invert_yaxis()
    ax.set_xlabel("pairs")
    ax.set_title(title)
    return _save(fig, out)


def chart_length_hist(pairs: List[SyntheticPair], out: Path, title: str) -> Path:
    if not pairs:
        return _empty(out, title)
    inst = [len(p.instruction or "") for p in pairs]
    outp = [len(p.output or "") for p in pairs]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(_FIGSIZE[0] * 1.4, _FIGSIZE[1]))
    a1.hist(inst, bins=20, color="#29b6f6", edgecolor="white")
    a1.set_title("instruction length (chars)")
    a1.set_ylabel("pairs")
    a2.hist(outp, bins=20, color="#ef6c00", edgecolor="white")
    a2.set_title("output length (chars)")
    fig.suptitle(title)
    return _save(fig, out)


def chart_label_distribution(labels: Dict[str, Label], out: Path, title: str) -> Path:
    counts = Counter(l.status for l in labels.values())
    if not counts:
        return _empty(out, title)
    order = ["approved", "needs_edit", "rejected", "skipped"]
    keys = [k for k in order if k in counts] + [k for k in counts if k not in order]
    vals = [counts[k] for k in keys]
    palette = {"approved": "#43a047", "needs_edit": "#fbc02d",
               "rejected": "#e53935", "skipped": "#90a4ae"}
    colors = [palette.get(k, "#5c6bc0") for k in keys]
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    bars = ax.bar(keys, vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), str(v),
                ha="center", va="bottom", fontsize=11)
    ax.set_ylabel("labels (latest-wins per pair)")
    ax.set_title(title)
    return _save(fig, out)


def chart_validation_issues(validation, out: Path, title: str) -> Path:
    if validation is None:
        return _empty(out, title)
    checks = {k: v for k, v in validation.counts.items() if not k.startswith("severity:")}
    if not checks:
        return _empty(out, title)
    keys = sorted(checks, key=lambda k: -checks[k])
    vals = [checks[k] for k in keys]
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    ax.barh(keys, vals, color="#ec407a")
    ax.invert_yaxis()
    ax.set_xlabel("issues")
    ax.set_title(title)
    return _save(fig, out)


def chart_judge_agreement(
    judgments_by_tag: Dict[str, List[JudgmentResult]],
    out: Path,
    title: str,
) -> Optional[Path]:
    tags = list(judgments_by_tag.keys())
    if len(tags) < 2:
        return None
    a_map = {str(j.pair_id): j.passed for j in judgments_by_tag[tags[0]]}
    b_map = {str(j.pair_id): j.passed for j in judgments_by_tag[tags[1]]}
    common = set(a_map) & set(b_map)
    if not common:
        return None
    # 2x2 contingency
    matrix = [[0, 0], [0, 0]]  # a: pass/fail rows; b: pass/fail cols
    for pid in common:
        i = 0 if a_map[pid] else 1
        j = 0 if b_map[pid] else 1
        matrix[i][j] += 1
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], [f"{tags[1]}: pass", f"{tags[1]}: fail"])
    ax.set_yticks([0, 1], [f"{tags[0]}: pass", f"{tags[0]}: fail"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i][j]), ha="center", va="center",
                    color="black", fontsize=14)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    return _save(fig, out)


def _empty(out: Path, title: str) -> Path:
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    ax.text(0.5, 0.5, "no data", ha="center", va="center", fontsize=14,
            color="#999")
    ax.set_axis_off()
    ax.set_title(title)
    return _save(fig, out)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def render_all(inputs: VizInputs, out_dir: Path) -> Tuple[List[Path], Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    produced: List[Path] = []

    # Aggregate judgments across all tags for overall views
    all_judgments: List[JudgmentResult] = []
    for tag, js in inputs.judgments_by_tag.items():
        all_judgments.extend(js)

    produced.append(chart_category_breakdown(
        inputs.pairs, out_dir / "categories.png",
        f"{inputs.name} — category breakdown ({len(inputs.pairs)} pairs)",
    ))
    produced.append(chart_length_hist(
        inputs.pairs, out_dir / "lengths.png", f"{inputs.name} — text lengths",
    ))
    produced.append(chart_validation_issues(
        inputs.validation, out_dir / "validation.png",
        f"{inputs.name} — validation issues",
    ))
    produced.append(chart_pass_fail(
        all_judgments, out_dir / "pass_fail.png",
        f"{inputs.name} — judge pass/fail ({len(all_judgments)} judgments)",
    ))
    produced.append(chart_score_histogram(
        all_judgments, out_dir / "scores.png",
        f"{inputs.name} — overall score distribution",
    ))
    produced.append(chart_criteria_means(
        all_judgments, out_dir / "criteria.png",
        f"{inputs.name} — mean per-criterion score",
    ))
    produced.append(chart_label_distribution(
        inputs.labels, out_dir / "labels.png",
        f"{inputs.name} — label distribution",
    ))

    agree = chart_judge_agreement(
        inputs.judgments_by_tag, out_dir / "judge_agreement.png",
        f"{inputs.name} — judge agreement",
    )
    if agree is not None:
        produced.append(agree)

    index = _write_index(inputs, produced, out_dir / "index.html")
    return produced, index


def _write_index(inputs: VizInputs, images: List[Path], out: Path) -> Path:
    cards = "\n".join(
        f'<figure><img src="{img.name}" alt="{img.stem}"/><figcaption>{img.stem}</figcaption></figure>'
        for img in images
    )
    summary = (
        f"pairs={len(inputs.pairs)} · "
        f"judgment_sets={len(inputs.judgments_by_tag)} "
        f"({sum(len(v) for v in inputs.judgments_by_tag.values())} judgments) · "
        f"labels={len(inputs.labels)}"
    )
    html = f"""<!doctype html><meta charset="utf-8">
<title>{inputs.name} — visualizations</title>
<style>
 body {{ font: 14px system-ui, sans-serif; margin: 2rem; color: #222; }}
 h1 {{ margin-bottom: 0; }}
 .meta {{ color: #666; margin-bottom: 1.5rem; }}
 .grid {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(460px,1fr));
          gap: 1.25rem; }}
 figure {{ margin: 0; border: 1px solid #eee; border-radius: 10px; padding: 10px;
           background: #fff; }}
 figure img {{ width: 100%; display: block; }}
 figcaption {{ color: #555; padding-top: 6px; font-size: 13px; }}
</style>
<h1>{inputs.name}</h1>
<div class="meta">{summary}</div>
<div class="grid">{cards}</div>
"""
    out.write_text(html, encoding="utf-8")
    return out
