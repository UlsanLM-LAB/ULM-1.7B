"""speaker-level deterministic split."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import replace

from .schema import SPLITS, DatasetRecord, ValidationError, validate_record

DEFAULT_RATIOS = (0.8, 0.1, 0.1)


def _validate_ratios(ratios: Sequence[float]) -> tuple[float, float, float]:
    if len(ratios) != 3:
        raise ValueError("ratios는 train, validation, test 세 값이어야 합니다")
    values = tuple(float(value) for value in ratios)
    if any(value <= 0 for value in values) or abs(sum(values) - 1.0) > 1e-8:
        raise ValueError("ratios는 양수이고 합이 1이어야 합니다")
    return values  # type: ignore[return-value]


def _counts(total: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    raw = [total * ratio for ratio in ratios]
    result = [int(value) for value in raw]
    remainder = total - sum(result)
    order = sorted(range(3), key=lambda index: raw[index] - result[index], reverse=True)
    for index in order[:remainder]:
        result[index] += 1
    # 표본이 충분하면 양수 비율의 split이 비어 있지 않도록 한다.
    if total >= 3:
        for index in range(3):
            if result[index] == 0:
                donor = max(range(3), key=lambda candidate: result[candidate])
                if result[donor] <= 1:
                    break
                result[donor] -= 1
                result[index] = 1
    return tuple(result)  # type: ignore[return-value]


def _speaker_order(speaker_ids: Iterable[str], seed: int) -> list[str]:
    def key(speaker_id: str) -> bytes:
        payload = f"{seed}\0{speaker_id}".encode()
        return hashlib.sha256(payload).digest()

    return sorted(set(speaker_ids), key=key)


def split_by_speaker(
    records: Iterable[DatasetRecord],
    *,
    seed: int = 42,
    ratios: Sequence[float] = DEFAULT_RATIOS,
) -> dict[str, list[DatasetRecord]]:
    """record를 speaker 단위로 분할한다.

    입력 순서를 보존한 채 각 speaker 전체를 하나의 split에 배정한다. seed가 같으면
    Python random 구현에 의존하지 않고 같은 결과를 만든다.
    """

    ratio_values = _validate_ratios(ratios)
    validated = [validate_record(record) for record in records]
    if any(record.split is not None for record in validated):
        raise ValidationError("이미 split된 record는 다시 분할하기 전에 split을 제거해야 합니다")
    if not validated:
        return {split: [] for split in SPLITS}

    speaker_ids = _speaker_order((record.speaker_id for record in validated), seed)
    train_count, validation_count, _ = _counts(len(speaker_ids), ratio_values)
    assignments: dict[str, str] = {}
    for index, speaker_id in enumerate(speaker_ids):
        if index < train_count:
            split = "train"
        elif index < train_count + validation_count:
            split = "validation"
        else:
            split = "test"
        assignments[speaker_id] = split

    result = {split: [] for split in SPLITS}
    for record in validated:
        split = assignments[record.speaker_id]
        result[split].append(replace(record, split=split))
    assert_no_speaker_leakage(result)
    return result


def assert_no_speaker_leakage(splits: dict[str, Iterable[DatasetRecord]]) -> None:
    """split 간 speaker overlap이 있으면 실패시킨다."""

    seen: dict[str, str] = {}
    for split, records in splits.items():
        if split not in SPLITS:
            raise ValueError(f"지원하지 않는 split key: {split!r}")
        for record in records:
            speaker_id = validate_record(record).speaker_id
            previous = seen.get(speaker_id)
            if previous is not None and previous != split:
                raise ValidationError(
                    f"speaker leakage: {speaker_id!r}가 {previous!r}와 {split!r}에 함께 존재합니다"
                )
            seen[speaker_id] = split
