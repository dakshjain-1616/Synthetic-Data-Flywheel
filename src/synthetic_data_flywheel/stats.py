"""Statistics helpers for judge sharpness — pure Python, no scipy/numpy dep."""

from __future__ import annotations

from math import sqrt
from typing import Dict, List, Optional, Sequence, Tuple


def cohens_kappa(a: Sequence[bool], b: Sequence[bool]) -> float:
    """Cohen's κ for two binary raters. Returns 0 if degenerate."""
    if not a or len(a) != len(b):
        return 0.0
    n = len(a)
    # Observed agreement
    obs = sum(1 for x, y in zip(a, b) if x == y) / n
    # Expected agreement (marginal probabilities)
    pa_true = sum(1 for x in a if x) / n
    pb_true = sum(1 for y in b if y) / n
    pe = pa_true * pb_true + (1 - pa_true) * (1 - pb_true)
    if pe == 1.0:
        return 1.0 if obs == 1.0 else 0.0
    return (obs - pe) / (1 - pe)


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Pearson correlation. Returns 0 if degenerate."""
    n = len(xs)
    if n == 0 or n != len(ys):
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sqrt(sum((x - mx) ** 2 for x in xs))
    dy = sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


def confusion(a: Sequence[bool], b: Sequence[bool]) -> Dict[str, int]:
    tp = fp = tn = fn = 0
    for pred, truth in zip(a, b):
        if truth and pred:
            tp += 1
        elif truth and not pred:
            fn += 1
        elif not truth and pred:
            fp += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def prf(a: Sequence[bool], b: Sequence[bool]) -> Dict[str, float]:
    """Precision/recall/F1 treating `a` as predictions, `b` as ground truth."""
    c = confusion(a, b)
    p = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) else 0.0
    r = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    acc = (c["tp"] + c["tn"]) / sum(c.values()) if sum(c.values()) else 0.0
    return {"precision": p, "recall": r, "f1": f1, "accuracy": acc, **c}
