"""ULM 내부 canonical dataset schema와 경계 검증."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from typing import Any

TASKS = frozenset(
    {
        "dialect_to_standard",
        "standard_to_dialect",
        "dialect_understanding",
        "dialect_chat",
        "region_classification",
        "dialect_strength_control",
    }
)
SPLITS = frozenset({"train", "validation", "test"})
QUALITY_GRADES = frozenset({"A", "B", "C"})
_PRIVATE_METADATA_KEYS = frozenset(
    {
        "name",
        "full_name",
        "real_name",
        "speaker_name",
        "phone",
        "phone_number",
        "email",
        "address",
        "resident_registration_number",
        "rrn",
    }
)
_FIELDS = frozenset(
    {
        "id",
        "source",
        "task",
        "speaker_id",
        "birthplace",
        "raised_region",
        "current_region",
        "age_group",
        "gender",
        "dialect_text",
        "standard_text",
        "dialect_strength",
        "synthetic",
        "human_verified",
        "quality_grade",
        "split",
        "metadata",
    }
)


class ValidationError(ValueError):
    """Record 또는 benchmark 입력이 contract를 위반했을 때 발생하는 오류."""


def _is_nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _contains_private_key(value: Any) -> str | None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).strip().lower()
            if normalized in _PRIVATE_METADATA_KEYS:
                return str(key)
            found = _contains_private_key(child)
            if found:
                return found
    elif isinstance(value, (list, tuple)):
        for child in value:
            found = _contains_private_key(child)
            if found:
                return found
    return None


@dataclass(frozen=True, slots=True)
class DatasetRecord:
    """학습·평가용으로 정규화된 한 발화 record.

    `speaker_id`는 원본 실명이나 원본 식별자가 아닌 pseudonymous ID를 사용한다.
    split은 split 이전 record를 표현하기 위해 `None`을 허용한다.
    """

    id: str
    source: str
    task: str
    speaker_id: str
    dialect_text: str | None = None
    standard_text: str | None = None
    birthplace: str | None = None
    raised_region: str | None = None
    current_region: str | None = None
    age_group: str | None = None
    gender: str | None = None
    dialect_strength: int | None = None
    synthetic: bool = False
    human_verified: bool = False
    quality_grade: str = "C"
    split: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self, *, require_split: bool = False) -> DatasetRecord:
        errors: list[str] = []
        for name in ("id", "source", "task", "speaker_id"):
            if not _is_nonempty_text(getattr(self, name)):
                errors.append(f"{name}은(는) 비어 있을 수 없습니다")
        if self.task not in TASKS:
            errors.append(f"지원하지 않는 task: {self.task!r}")
        if self.dialect_text is not None and not _is_nonempty_text(self.dialect_text):
            errors.append("dialect_text는 비어 있는 문자열 대신 None을 사용해야 합니다")
        if self.standard_text is not None and not _is_nonempty_text(self.standard_text):
            errors.append("standard_text는 비어 있는 문자열 대신 None을 사용해야 합니다")
        if self.dialect_text is None and self.standard_text is None:
            errors.append("dialect_text 또는 standard_text 중 하나는 필요합니다")
        if self.task in {"dialect_to_standard", "standard_to_dialect", "dialect_strength_control"}:
            if self.dialect_text is None:
                errors.append(f"{self.task}에는 dialect_text가 필요합니다")
            if self.standard_text is None:
                errors.append(f"{self.task}에는 standard_text가 필요합니다")
        if self.task in {"dialect_understanding", "dialect_chat", "region_classification"}:
            if self.dialect_text is None:
                errors.append(f"{self.task}에는 dialect_text가 필요합니다")
        if not isinstance(self.synthetic, bool):
            errors.append("synthetic은 bool이어야 합니다")
        if not isinstance(self.human_verified, bool):
            errors.append("human_verified는 bool이어야 합니다")
        if self.dialect_strength is not None and (
            not isinstance(self.dialect_strength, int) or self.dialect_strength not in range(4)
        ):
            errors.append("dialect_strength는 None 또는 0~3 정수여야 합니다")
        if self.quality_grade not in QUALITY_GRADES:
            errors.append(f"quality_grade는 A/B/C 중 하나여야 합니다: {self.quality_grade!r}")
        if self.split is not None and self.split not in SPLITS:
            errors.append(f"split은 train/validation/test 중 하나여야 합니다: {self.split!r}")
        if require_split and self.split is None:
            errors.append("분할이 완료된 record에는 split이 필요합니다")
        if not isinstance(self.metadata, dict):
            errors.append("metadata는 dict이어야 합니다")
        else:
            private_key = _contains_private_key(self.metadata)
            if private_key:
                errors.append(
                    f"metadata에 개인정보로 해석될 수 있는 key가 있습니다: {private_key!r}"
                )
        if errors:
            raise ValidationError("; ".join(errors))
        return self

    def with_split(self, split: str) -> DatasetRecord:
        if split not in SPLITS:
            raise ValidationError(f"지원하지 않는 split: {split!r}")
        return replace(self, split=split)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], *, require_split: bool = False) -> DatasetRecord:
        if not isinstance(raw, Mapping):
            raise ValidationError("record는 mapping이어야 합니다")
        unknown = set(raw) - _FIELDS
        if unknown:
            raise ValidationError(f"알 수 없는 record field: {sorted(unknown)!r}")
        required = {"id", "source", "task", "speaker_id"}
        missing = required - set(raw)
        if missing:
            raise ValidationError(f"필수 record field가 없습니다: {sorted(missing)!r}")
        try:
            record = cls(**dict(raw))
        except TypeError as exc:
            raise ValidationError(f"record field type 또는 값이 올바르지 않습니다: {exc}") from exc
        return record.validate(require_split=require_split)


def validate_record(
    record: DatasetRecord | Mapping[str, Any], *, require_split: bool = False
) -> DatasetRecord:
    """record를 검증하고 동일 객체를 반환한다."""

    if isinstance(record, DatasetRecord):
        return record.validate(require_split=require_split)
    return DatasetRecord.from_dict(record, require_split=require_split)
