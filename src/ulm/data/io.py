"""canonical JSONL 입출력."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from .schema import DatasetRecord, validate_record


def iter_jsonl(path: str | Path, *, require_split: bool = False) -> Iterator[DatasetRecord]:
    """JSONL을 한 줄씩 읽어 검증한다."""

    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw: Any = json.loads(line)
                yield validate_record(raw, require_split=require_split)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(f"{source}:{line_number}을 읽을 수 없습니다: {exc}") from exc


def read_jsonl(path: str | Path, *, require_split: bool = False) -> list[DatasetRecord]:
    return list(iter_jsonl(path, require_split=require_split))


def _atomic_write(path: Path, content: str, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"기존 파일을 덮어쓰지 않습니다: {path}")
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_name = handle.name
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists() and not overwrite:
            raise FileExistsError(f"동시 변경을 보호하기 위해 덮어쓰지 않습니다: {path}")
        os.replace(temp_name, path)
        temp_name = None
    finally:
        if temp_name:
            Path(temp_name).unlink(missing_ok=True)


def write_jsonl(
    path: str | Path,
    records: Iterable[DatasetRecord],
    *,
    overwrite: bool = False,
    require_split: bool = False,
) -> None:
    """검증이 끝난 record들을 안전하게 JSONL로 기록한다.

    기본값은 기존 파일을 덮어쓰지 않는다. 모든 record를 먼저 검증한 뒤 파일을 교체한다.
    """

    validated = [validate_record(record, require_split=require_split) for record in records]
    content = "".join(
        json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
        for record in validated
    )
    _atomic_write(Path(path), content, overwrite=overwrite)
