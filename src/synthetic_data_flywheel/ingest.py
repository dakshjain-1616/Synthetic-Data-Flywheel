"""Ingest user-supplied datasets into the flywheel's JSONL format."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional
from uuid import NAMESPACE_URL, uuid5

from synthetic_data_flywheel.config import get_settings
from synthetic_data_flywheel.models import DatasetMeta, SyntheticPair


_FIELD_ALIASES = {
    "instruction": ("instruction", "prompt", "question", "query"),
    "output": ("output", "completion", "response", "answer"),
    "input": ("input", "context", "passage"),
    "category": ("category", "label", "topic"),
    "difficulty": ("difficulty", "level"),
}


def _atomic_write(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(data, encoding="utf-8")
    tmp.replace(path)


def _deterministic_id(instruction: str, output: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"sdf::{instruction}::{output}"))


def _extract(row: Mapping[str, Any], mapping: Mapping[str, str], key: str) -> Any:
    """Look up `key` using mapping, then aliases."""
    src = mapping.get(key)
    if src and src in row:
        return row[src]
    for alias in _FIELD_ALIASES.get(key, ()):  # type: ignore[arg-type]
        if alias in row:
            return row[alias]
    return None


def normalize_row(
    row: Mapping[str, Any],
    mapping: Mapping[str, str],
    tag: Optional[str] = None,
) -> SyntheticPair:
    instruction = str(_extract(row, mapping, "instruction") or "").strip()
    output = str(_extract(row, mapping, "output") or "").strip()
    input_ = _extract(row, mapping, "input")
    category = _extract(row, mapping, "category")
    difficulty = _extract(row, mapping, "difficulty")
    md: Dict[str, Any] = {"source": "user"}
    if tag:
        md["tag"] = tag
    return SyntheticPair(
        id=_deterministic_id(instruction, output),
        instruction=instruction,
        output=output,
        input=str(input_) if input_ is not None else None,
        category=str(category) if category is not None else None,
        difficulty=str(difficulty) if difficulty is not None else None,
        metadata=md,
    )


class DatasetIngestor:
    def __init__(self, user_data_dir: Optional[str] = None):
        s = get_settings()
        self.root = Path(user_data_dir or s.user_data_dir)
        self.root.mkdir(parents=True, exist_ok=True)

    # ----- readers -----
    @staticmethod
    def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)

    @staticmethod
    def _read_json(path: Path) -> Iterable[Dict[str, Any]]:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            yield from data
        elif isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
            yield from data["data"]
        else:
            raise ValueError("Unsupported JSON shape: expected a list of rows")

    @staticmethod
    def _read_csv(path: Path) -> Iterable[Dict[str, Any]]:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            yield from reader

    @staticmethod
    def _read_hf(ref: str, split: str = "train") -> Iterable[Dict[str, Any]]:
        from datasets import load_dataset  # local import — heavy dep

        ds = load_dataset(ref, split=split)
        for row in ds:
            yield dict(row)

    # ----- public API -----
    def ingest(
        self,
        source: str,
        name: str,
        fmt: str = "auto",
        mapping: Optional[Dict[str, str]] = None,
        tag: Optional[str] = None,
        limit: Optional[int] = None,
        hf_split: str = "train",
    ) -> tuple[Path, DatasetMeta]:
        mapping = mapping or {}
        rows = self._iter_rows(source, fmt, hf_split)

        out_path = self.root / f"{name}.jsonl"
        pairs: List[SyntheticPair] = []
        seen_ids: set[str] = set()
        hasher = hashlib.sha256()
        for i, row in enumerate(rows):
            if limit is not None and i >= limit:
                break
            pair = normalize_row(row, mapping, tag=tag)
            if not pair.instruction and not pair.output:
                continue
            pid = str(pair.id)
            if pid in seen_ids:
                # Exact duplicates collapse by deterministic ID — keeps the
                # stored dataset 1:1 with the labeling/judgment key space.
                continue
            seen_ids.add(pid)
            pairs.append(pair)
            hasher.update((pair.instruction + "\x1f" + pair.output).encode("utf-8"))

        body = "\n".join(json.dumps(p.to_dict(), ensure_ascii=False) for p in pairs) + "\n"
        _atomic_write(out_path, body)

        meta = DatasetMeta(
            name=name,
            source=self._source_kind(source, fmt),
            row_count=len(pairs),
            tags=[tag] if tag else [],
            mapping=dict(mapping),
            checksum=hasher.hexdigest(),
        )
        _atomic_write(self.root / f"{name}.meta.json", json.dumps(meta.to_dict(), indent=2))
        return out_path, meta

    # ----- helpers -----
    def _iter_rows(self, source: str, fmt: str, hf_split: str) -> Iterable[Dict[str, Any]]:
        if source.startswith("hf://") or fmt == "hf":
            ref = source[5:] if source.startswith("hf://") else source
            return self._read_hf(ref, split=hf_split)
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(source)
        kind = fmt if fmt != "auto" else self._autodetect(path)
        if kind == "jsonl":
            return self._read_jsonl(path)
        if kind == "json":
            return self._read_json(path)
        if kind == "csv":
            return self._read_csv(path)
        raise ValueError(f"Unsupported format: {kind}")

    @staticmethod
    def _autodetect(path: Path) -> str:
        s = path.suffix.lower()
        if s == ".jsonl":
            return "jsonl"
        if s == ".json":
            return "json"
        if s == ".csv":
            return "csv"
        raise ValueError(f"Cannot autodetect format for {path.name}")

    def _source_kind(self, source: str, fmt: str) -> str:
        if source.startswith("hf://") or fmt == "hf":
            return "hf"
        if fmt != "auto":
            return fmt
        return self._autodetect(Path(source))


def load_dataset_jsonl(path: Path | str) -> List[SyntheticPair]:
    """Helper to load a dataset jsonl into SyntheticPair objects."""
    p = Path(path)
    out: List[SyntheticPair] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(SyntheticPair.from_dict(json.loads(line)))
    return out
