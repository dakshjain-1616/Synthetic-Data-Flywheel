"""Dataset validation — schema, length, dedup, PII, language, profanity."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from typing import Callable, Dict, List, Optional, Tuple

from synthetic_data_flywheel.models import SyntheticPair, ValidationIssue, ValidationReport


CheckFn = Callable[[List[SyntheticPair], Dict], List[ValidationIssue]]


_PII_PATTERNS = [
    ("email", re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b")),
    ("phone", re.compile(r"\b(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{3}\)?[\s-]?)?\d{3}[\s-]?\d{4}\b")),
    ("api_key", re.compile(r"\b(?:sk-[A-Za-z0-9]{16,}|ghp_[A-Za-z0-9]{16,}|AKIA[0-9A-Z]{16})\b")),
]


_PROFANITY_STUB = {"badword1", "badword2"}  # intentionally minimal; users can override


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _hash_pair(pair: SyntheticPair) -> str:
    h = hashlib.sha1()
    h.update(_norm(pair.instruction).encode())
    h.update(b"\x1f")
    h.update(_norm(pair.output).encode())
    return h.hexdigest()


def check_schema(pairs, opts) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    for p in pairs:
        if not p.instruction:
            issues.append(ValidationIssue(pair_id=p.id, check="schema", severity="error",
                                          field="instruction", message="instruction is empty"))
        if not p.output:
            issues.append(ValidationIssue(pair_id=p.id, check="schema", severity="error",
                                          field="output", message="output is empty"))
    return issues


def check_length(pairs, opts) -> List[ValidationIssue]:
    min_i = opts.get("min_instruction_len", 3)
    max_o = opts.get("max_output_len", 8000)
    issues: List[ValidationIssue] = []
    for p in pairs:
        if len(p.instruction) < min_i:
            issues.append(ValidationIssue(pair_id=p.id, check="length", severity="warning",
                                          field="instruction",
                                          message=f"instruction shorter than {min_i} chars"))
        if len(p.output) > max_o:
            issues.append(ValidationIssue(pair_id=p.id, check="length", severity="warning",
                                          field="output",
                                          message=f"output longer than {max_o} chars"))
    return issues


def check_dedup(pairs, opts) -> List[ValidationIssue]:
    seen: Dict[str, str] = {}
    issues: List[ValidationIssue] = []
    for p in pairs:
        h = _hash_pair(p)
        if h in seen:
            issues.append(ValidationIssue(pair_id=p.id, check="dedup", severity="warning",
                                          message=f"duplicate of {seen[h]}"))
        else:
            seen[h] = str(p.id)
    return issues


def check_pii(pairs, opts) -> List[ValidationIssue]:
    policy = opts.get("pii_policy", "warn")
    if policy == "off":
        return []
    severity = "error" if policy == "strict" else "warning"
    issues: List[ValidationIssue] = []
    for p in pairs:
        blob = (p.instruction or "") + "\n" + (p.output or "")
        for kind, pat in _PII_PATTERNS:
            if pat.search(blob):
                issues.append(ValidationIssue(pair_id=p.id, check="pii", severity=severity,
                                              message=f"potential {kind} detected"))
                break
    return issues


def check_lang(pairs, opts) -> List[ValidationIssue]:
    want = opts.get("lang")
    if not want:
        return []
    try:
        from langdetect import detect  # type: ignore
    except Exception:
        return []  # soft-disable
    issues: List[ValidationIssue] = []
    for p in pairs:
        sample = (p.instruction or "")[:500]
        if not sample:
            continue
        try:
            if detect(sample) != want:
                issues.append(ValidationIssue(pair_id=p.id, check="lang", severity="warning",
                                              message=f"language != {want}"))
        except Exception:
            continue
    return issues


def check_profanity(pairs, opts) -> List[ValidationIssue]:
    words = set(opts.get("profanity_words") or _PROFANITY_STUB)
    issues: List[ValidationIssue] = []
    for p in pairs:
        tokens = set(re.findall(r"[a-z]+", (p.instruction + " " + p.output).lower()))
        if tokens & words:
            issues.append(ValidationIssue(pair_id=p.id, check="profanity", severity="warning",
                                          message="flagged term detected"))
    return issues


CHECKS: Dict[str, CheckFn] = {
    "schema": check_schema,
    "length": check_length,
    "dedup": check_dedup,
    "pii": check_pii,
    "lang": check_lang,
    "profanity": check_profanity,
}


class Validator:
    def __init__(self, options: Optional[Dict] = None):
        self.options = options or {}

    def validate(
        self,
        pairs: List[SyntheticPair],
        checks: Optional[List[str]] = None,
        dataset: str = "dataset",
    ) -> ValidationReport:
        checks = checks or list(CHECKS.keys())
        all_issues: List[ValidationIssue] = []
        for name in checks:
            fn = CHECKS.get(name)
            if fn is None:
                continue
            all_issues.extend(fn(pairs, self.options))

        counts: Counter = Counter()
        for i in all_issues:
            counts[i.check] += 1
            counts[f"severity:{i.severity}"] += 1
        return ValidationReport(
            dataset=dataset, total_pairs=len(pairs),
            counts=dict(counts), issues=all_issues,
        )

    def filter_clean(
        self,
        pairs: List[SyntheticPair],
        report: ValidationReport,
        drop_severities: Tuple[str, ...] = ("error",),
        drop_duplicates: bool = True,
    ) -> List[SyntheticPair]:
        bad = {str(i.pair_id) for i in report.issues if i.severity in drop_severities}
        out: List[SyntheticPair] = []
        seen_hash: Dict[str, bool] = {}
        for p in pairs:
            if str(p.id) in bad:
                continue
            if drop_duplicates:
                h = _hash_pair(p)
                if h in seen_hash:
                    continue
                seen_hash[h] = True
            out.append(p)
        return out
