"""Content-addressed cache for LLM judgments.

Key = sha256(pair_id | rubric_name | rubric_version | backend | model).
Value = the JudgmentResult dict on disk as a small JSON file.

Skips the API call when a prior result exists for the exact (pair, rubric,
backend, model) tuple. Safe to delete — it only accelerates re-runs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

from synthetic_data_flywheel.models import JudgmentResult


class JudgmentCache:
    def __init__(self, root: str | Path, enabled: bool = True):
        self.root = Path(root)
        self.enabled = enabled
        self.hits = 0
        self.misses = 0
        self.writes = 0
        if enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(pair_id: str, rubric_name: str, rubric_version: int,
            backend: str, model: str) -> str:
        h = hashlib.sha256()
        h.update(f"{pair_id}\x1f{rubric_name}\x1f{rubric_version}\x1f{backend}\x1f{model}".encode())
        return h.hexdigest()

    def _path(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def get(self, key: str) -> Optional[JudgmentResult]:
        if not self.enabled:
            self.misses += 1
            return None
        p = self._path(key)
        if not p.exists():
            self.misses += 1
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            self.hits += 1
            return JudgmentResult.from_dict(data)
        except Exception:
            self.misses += 1
            return None

    def put(self, key: str, result: JudgmentResult) -> None:
        if not self.enabled:
            return
        p = self._path(key)
        p.write_text(json.dumps(result.to_dict(), ensure_ascii=False), encoding="utf-8")
        self.writes += 1

    def stats(self) -> dict:
        return {"hits": self.hits, "misses": self.misses, "writes": self.writes}
