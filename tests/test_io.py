from __future__ import annotations

import pytest

from ulm.data.io import read_jsonl, write_jsonl
from ulm.data.schema import DatasetRecord


def record(identifier: str, split: str = "train") -> DatasetRecord:
    return DatasetRecord(
        id=identifier,
        source="developer_fixture",
        task="standard_to_dialect",
        speaker_id=f"speaker-{identifier}",
        dialect_text="나는 간다.",
        standard_text="저는 갑니다.",
        split=split,
    )


def test_jsonl_round_trip_and_overwrite_guard(tmp_path) -> None:
    destination = tmp_path / "records.jsonl"
    write_jsonl(destination, [record("one")], require_split=True)
    assert read_jsonl(destination, require_split=True) == [record("one")]
    with pytest.raises(FileExistsError):
        write_jsonl(destination, [record("two")], require_split=True)


def test_jsonl_writer_validates_before_writing(tmp_path) -> None:
    destination = tmp_path / "records.jsonl"
    invalid = DatasetRecord(
        id="bad",
        source="fixture",
        task="standard_to_dialect",
        speaker_id="speaker-bad",
        dialect_text=None,
        standard_text="문장",
    )
    with pytest.raises(ValueError):
        write_jsonl(destination, [invalid])
    assert not destination.exists()
