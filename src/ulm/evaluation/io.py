"""ULM-Bench JSONL 입출력."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from .schema import BenchmarkItem


def iter_items(path: str | Path) -> Iterator[BenchmarkItem]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield BenchmarkItem.from_dict(json.loads(line))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(f"{source}:{line_number}을 읽을 수 없습니다: {exc}") from exc


def read_items(path: str | Path) -> list[BenchmarkItem]:
    return list(iter_items(path))
