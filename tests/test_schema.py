from __future__ import annotations

import pytest

from ulm.data.schema import DatasetRecord, ValidationError, validate_record


def make_record(**overrides: object) -> DatasetRecord:
    values: dict[str, object] = {
        "id": "sample-1",
        "source": "developer_fixture",
        "task": "standard_to_dialect",
        "speaker_id": "spk-1",
        "dialect_text": "나는 지금 집에 간다.",
        "standard_text": "저는 지금 집에 갑니다.",
        "quality_grade": "C",
    }
    values.update(overrides)
    return DatasetRecord(**values)  # type: ignore[arg-type]


def test_record_round_trip() -> None:
    record = make_record(dialect_strength=2, split="validation", metadata={"fixture": True})
    rendered = record.to_dict()
    restored = DatasetRecord.from_dict(rendered, require_split=True)
    assert restored == record


def test_mapping_is_validated() -> None:
    record = validate_record(make_record().to_dict())
    assert record.id == "sample-1"


@pytest.mark.parametrize(
    "overrides",
    [
        {"task": "not-a-task"},
        {"dialect_strength": 4},
        {"quality_grade": "D"},
        {"split": "dev"},
        {"metadata": {"real_name": "금지"}},
        {"dialect_text": None, "standard_text": None},
    ],
)
def test_invalid_record_is_rejected(overrides: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        make_record(**overrides).validate()


def test_task_requires_parallel_text() -> None:
    with pytest.raises(ValidationError, match="standard_text"):
        make_record(task="dialect_to_standard", standard_text=None).validate()


def test_missing_required_field_is_rejected() -> None:
    raw = make_record().to_dict()
    del raw["speaker_id"]
    with pytest.raises(ValidationError, match="speaker_id"):
        DatasetRecord.from_dict(raw)


def test_split_is_required_when_requested() -> None:
    with pytest.raises(ValidationError, match="split"):
        make_record().validate(require_split=True)


def test_synthetic_and_human_review_can_coexist() -> None:
    record = make_record(synthetic=True, human_verified=True)
    assert record.validate().human_verified is True
