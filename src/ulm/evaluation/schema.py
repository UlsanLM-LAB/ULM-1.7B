"""ULM-Bench item schema."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from ulm.data.schema import TASKS, ValidationError, _contains_private_key

_FIELDS = frozenset(
    {
        "id",
        "task",
        "input",
        "choices",
        "gold",
        "reference",
        "variants",
        "phenomena",
        "source",
        "annotators",
        "human_verified",
        "review_status",
        "metadata",
    }
)
_CLASSIFICATION_TASKS = frozenset({"dialect_understanding", "region_classification"})
_REFERENCE_TASKS = frozenset({"dialect_to_standard", "standard_to_dialect", "dialect_chat"})


@dataclass(frozen=True, slots=True)
class BenchmarkItem:
    """한 benchmark prompt와 검수 provenance."""

    id: str
    task: str
    input: str
    choices: tuple[str, ...] = ()
    gold: str | None = None
    reference: str | None = None
    variants: dict[str, str] = field(default_factory=dict)
    phenomena: tuple[str, ...] = ()
    source: str = "unknown"
    annotators: int = 0
    human_verified: bool = False
    review_status: str = "unreviewed"
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> BenchmarkItem:
        errors: list[str] = []
        if not isinstance(self.id, str) or not self.id.strip():
            errors.append("id는 비어 있을 수 없습니다")
        if self.task not in TASKS:
            errors.append(f"지원하지 않는 benchmark task: {self.task!r}")
        if not isinstance(self.input, str) or not self.input.strip():
            errors.append("input은 비어 있을 수 없습니다")
        if not isinstance(self.choices, tuple) or any(
            not isinstance(choice, str) or not choice.strip() for choice in self.choices
        ):
            errors.append("choices는 비어 있지 않은 문자열 tuple이어야 합니다")
        if isinstance(self.choices, tuple) and len(set(self.choices)) != len(self.choices):
            errors.append("choices에는 중복 항목이 있을 수 없습니다")
        if self.gold is not None and not isinstance(self.gold, str):
            errors.append("gold는 문자열이어야 합니다")
        if self.reference is not None and (
            not isinstance(self.reference, str) or not self.reference.strip()
        ):
            errors.append("reference는 비어 있지 않은 문자열이어야 합니다")
        if self.gold is not None and self.choices and self.gold not in self.choices:
            errors.append("gold는 choices 중 하나여야 합니다")
        if self.task in _CLASSIFICATION_TASKS and (self.gold is None or not self.choices):
            errors.append(f"{self.task}에는 choices와 gold가 필요합니다")
        if self.task in _REFERENCE_TASKS and self.reference is None:
            errors.append(f"{self.task}에는 reference가 필요합니다")
        if self.task == "dialect_strength_control" and not self.variants:
            errors.append("dialect_strength_control에는 variants가 필요합니다")
        if not isinstance(self.variants, dict):
            errors.append("variants는 dict이어야 합니다")
        elif any(
            not isinstance(key, str) or not isinstance(value, str) or not value.strip()
            for key, value in self.variants.items()
        ):
            errors.append("variants는 0~3 key와 비어 있지 않은 문자열 value를 가져야 합니다")
        elif not set(self.variants).issubset({"0", "1", "2", "3"}):
            errors.append("variants key는 0~3이어야 합니다")
        if not isinstance(self.phenomena, tuple) or any(
            not isinstance(value, str) or not value.strip() for value in self.phenomena
        ):
            errors.append("phenomena는 문자열 tuple이어야 합니다")
        if not isinstance(self.source, str) or not self.source.strip():
            errors.append("source는 비어 있을 수 없습니다")
        if not isinstance(self.annotators, int) or self.annotators < 0:
            errors.append("annotators는 0 이상의 정수여야 합니다")
        if not isinstance(self.human_verified, bool):
            errors.append("human_verified는 bool이어야 합니다")
        if self.human_verified and self.annotators < 1:
            errors.append("human_verified item에는 annotators가 필요합니다")
        if self.review_status not in {"unreviewed", "reviewed", "fixture_only"}:
            errors.append(f"지원하지 않는 review_status: {self.review_status!r}")
        if not isinstance(self.metadata, dict):
            errors.append("metadata는 dict이어야 합니다")
        elif (private_key := _contains_private_key(self.metadata)) is not None:
            errors.append(f"metadata에 개인정보로 해석될 수 있는 key가 있습니다: {private_key!r}")
        if errors:
            raise ValidationError("; ".join(errors))
        return self

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        result = asdict(self)
        result["choices"] = list(self.choices)
        result["phenomena"] = list(self.phenomena)
        return result

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> BenchmarkItem:
        if not isinstance(raw, Mapping):
            raise ValidationError("benchmark item은 mapping이어야 합니다")
        unknown = set(raw) - _FIELDS
        if unknown:
            raise ValidationError(f"알 수 없는 benchmark field: {sorted(unknown)!r}")
        missing = {"id", "task", "input"} - set(raw)
        if missing:
            raise ValidationError(f"필수 benchmark field가 없습니다: {sorted(missing)!r}")
        values = dict(raw)
        if isinstance(values.get("choices"), list):
            values["choices"] = tuple(values["choices"] or ())
        if isinstance(values.get("phenomena"), list):
            values["phenomena"] = tuple(values["phenomena"] or ())
        try:
            item = cls(**values)
        except TypeError as exc:
            raise ValidationError(
                f"benchmark field type 또는 값이 올바르지 않습니다: {exc}"
            ) from exc
        return item.validate()
