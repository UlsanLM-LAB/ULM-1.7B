from __future__ import annotations

import pytest

from ulm.data.schema import DatasetRecord, ValidationError
from ulm.data.split import assert_no_speaker_leakage, split_by_speaker


def records(count: int = 10) -> list[DatasetRecord]:
    return [
        DatasetRecord(
            id=f"sample-{index}",
            source="developer_fixture",
            task="dialect_to_standard",
            speaker_id=f"speaker-{index}",
            dialect_text=f"방언 문장 {index}",
            standard_text=f"표준 문장 {index}",
        )
        for index in range(count)
    ]


def test_split_is_deterministic_and_disjoint() -> None:
    first = split_by_speaker(records(), seed=7)
    second = split_by_speaker(records(), seed=7)
    assert first == second
    assert sum(len(items) for items in first.values()) == 10
    assert {record.split for items in first.values() for record in items} == {
        "train",
        "validation",
        "test",
    }
    assert_no_speaker_leakage(first)


def test_seed_changes_assignment_without_changing_records() -> None:
    first = split_by_speaker(records(), seed=1)
    second = split_by_speaker(records(), seed=2)
    first_assignment = {
        record.speaker_id: record.split for items in first.values() for record in items
    }
    second_assignment = {
        record.speaker_id: record.split for items in second.values() for record in items
    }
    assert first_assignment != second_assignment
    assert set(first_assignment) == set(second_assignment)


def test_leakage_is_rejected() -> None:
    source = records(1)[0]
    with pytest.raises(ValidationError, match="speaker leakage"):
        assert_no_speaker_leakage({"train": [source], "validation": [source], "test": []})


def test_existing_split_is_not_silently_overwritten() -> None:
    source = records(1)[0].with_split("train")
    with pytest.raises(ValidationError, match="이미 split"):
        split_by_speaker([source])


def test_empty_input_has_all_keys() -> None:
    result = split_by_speaker([])
    assert set(result) == {"train", "validation", "test"}
    assert all(not values for values in result.values())
