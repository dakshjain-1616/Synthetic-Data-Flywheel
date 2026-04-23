"""Labeling — bulk expression filters, auto-from-judge, and interactive TUI.

Labels are persisted append-only as JSONL at `<labels_dir>/<dataset>.jsonl`.
Reads use latest-wins semantics by `pair_id`.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from synthetic_data_flywheel.models import JudgmentResult, Label, SyntheticPair


VALID_STATUSES = {"approved", "rejected", "needs_edit", "skipped"}


# ---------------------------------------------------------------------------
# Safe expression evaluator
# ---------------------------------------------------------------------------

_ALLOWED_NODES = {
    ast.Expression, ast.BoolOp, ast.And, ast.Or, ast.Not, ast.UnaryOp,
    ast.USub, ast.UAdd, ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.Div,
    ast.Mod, ast.FloorDiv, ast.Pow, ast.Compare, ast.Eq, ast.NotEq,
    ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn, ast.Is, ast.IsNot,
    ast.Constant, ast.Name, ast.Load, ast.Subscript, ast.Slice,
    ast.List, ast.Tuple, ast.Set, ast.Dict, ast.IfExp,
}


def _validate_ast(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if type(node) not in _ALLOWED_NODES:
            raise ValueError(f"Disallowed expression: {type(node).__name__}")
        if isinstance(node, ast.Name):
            if node.id.startswith("__"):
                raise ValueError("Dunder names are not allowed")


class SafeEval:
    """Evaluate a boolean expression against a context dict. No attr access."""

    def __init__(self, expr: str):
        self.expr = expr
        try:
            self._tree = ast.parse(expr, mode="eval")
        except SyntaxError as e:
            raise ValueError(f"Invalid expression: {e}") from e
        _validate_ast(self._tree)
        self._code = compile(self._tree, "<safe-eval>", "eval")

    def __call__(self, ctx: Dict[str, Any]) -> bool:
        # globals empty + no builtins => no name resolution outside ctx
        return bool(eval(self._code, {"__builtins__": {}}, ctx))


def _pair_context(pair: SyntheticPair, judgment: Optional[JudgmentResult],
                  label: Optional[Label]) -> Dict[str, Any]:
    ctx: Dict[str, Any] = {
        "id": str(pair.id),
        "instruction": pair.instruction,
        "input": pair.input,
        "output": pair.output,
        "category": pair.category,
        "difficulty": pair.difficulty,
        "metadata": pair.metadata,
    }
    if judgment is not None:
        ctx["scores"] = {
            "coherence": judgment.scores.coherence,
            "accuracy": judgment.scores.accuracy,
            "helpfulness": judgment.scores.helpfulness,
            "overall": judgment.scores.overall,
        }
        ctx["passed"] = judgment.passed
        ctx["judge_model"] = judgment.judge_model
    else:
        ctx["scores"] = {"coherence": 0, "accuracy": 0, "helpfulness": 0, "overall": 0}
        ctx["passed"] = False
        ctx["judge_model"] = None
    if label is not None:
        ctx["label"] = {"status": label.status, "tag": label.tag, "note": label.note}
    else:
        ctx["label"] = {"status": None, "tag": None, "note": None}
    return ctx


# ---------------------------------------------------------------------------
# LabelStore — JSONL, latest-wins
# ---------------------------------------------------------------------------

class LabelStore:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> Dict[str, Label]:
        out: Dict[str, Label] = {}
        if not self.path.exists():
            return out
        with self.path.open("r", encoding="utf-8") as f:
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

    def append(self, label: Label) -> None:
        if label.status not in VALID_STATUSES:
            raise ValueError(f"Invalid status: {label.status}")
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(label.to_dict(), ensure_ascii=False) + "\n")

    def extend(self, labels: Iterable[Label]) -> int:
        count = 0
        for lbl in labels:
            self.append(lbl)
            count += 1
        return count


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def bulk_apply(
    pairs: List[SyntheticPair],
    judgments: Optional[Dict[str, JudgmentResult]],
    expr: str,
    status: str,
    tag: Optional[str] = None,
    note: Optional[str] = None,
    labeler: str = "bulk",
    existing: Optional[Dict[str, Label]] = None,
) -> List[Label]:
    if status not in VALID_STATUSES:
        raise ValueError(f"Invalid status: {status}")
    judgments = judgments or {}
    existing = existing or {}
    predicate = SafeEval(expr)
    out: List[Label] = []
    seen: set = set()
    for p in pairs:
        pid = str(p.id)
        if pid in seen:
            continue
        seen.add(pid)
        j = judgments.get(pid)
        lbl = existing.get(pid)
        ctx = _pair_context(p, j, lbl)
        try:
            match = predicate(ctx)
        except Exception:
            match = False
        if match:
            out.append(Label(pair_id=p.id, status=status, tag=tag, note=note, labeler=labeler))
    return out


def auto_from_judge(
    judgments: Iterable[JudgmentResult],
    reject_below: float = 3.5,
    approve_if_passed: bool = True,
) -> List[Label]:
    out: List[Label] = []
    for j in judgments:
        if approve_if_passed and j.passed:
            status = "approved"
        elif j.scores.overall < reject_below:
            status = "rejected"
        else:
            status = "needs_edit"
        out.append(Label(pair_id=j.pair_id, status=status, labeler="auto-from-judge",
                         tag=f"overall={j.scores.overall:.1f}"))
    return out


def interactive_loop(
    pairs: List[SyntheticPair],
    judgments: Optional[Dict[str, JudgmentResult]] = None,
    already_labeled: Optional[Dict[str, Label]] = None,
    prompt_fn=None,
    echo_fn=print,
) -> List[Label]:
    """CLI-driven review loop. `prompt_fn(msg, default)` returns the raw input.

    Keys: a=approve, r=reject, e=needs_edit, s=skip, q=quit.
    """
    from rich.prompt import Prompt  # local import to avoid mandatory dep at module load

    pf = prompt_fn or (lambda m, default="": Prompt.ask(m, default=default))
    already_labeled = already_labeled or {}
    judgments = judgments or {}
    out: List[Label] = []
    for idx, pair in enumerate(pairs, start=1):
        if str(pair.id) in already_labeled:
            continue
        echo_fn(f"\n[{idx}/{len(pairs)}] id={pair.id}")
        echo_fn(f"INSTRUCTION: {pair.instruction}")
        echo_fn(f"OUTPUT: {pair.output[:600]}")
        j = judgments.get(str(pair.id))
        if j is not None:
            echo_fn(f"JUDGE: overall={j.scores.overall:.1f} passed={j.passed} — {j.judgment_reasoning[:120]}")
        key = (pf("Action [a]pprove/[r]eject/[e]dit/[s]kip/[q]uit", "s") or "s").strip().lower()
        if key in ("q", "quit"):
            break
        status = {"a": "approved", "r": "rejected", "e": "needs_edit", "s": "skipped"}.get(key, "skipped")
        if status == "skipped":
            continue
        note = pf("Note (optional)", "") or None
        tag = pf("Tag (optional)", "") or None
        out.append(Label(pair_id=pair.id, status=status, tag=tag, note=note, labeler="interactive"))
    return out
