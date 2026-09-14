from __future__ import annotations

import pytest

from ulm.data.schema import ValidationError
from ulm.evaluation.metrics import character_f1, evaluate_predictions
from ulm.evaluation.schema import BenchmarkItem


def test_benchmark_item_round_trip() -> None:
    item = BenchmarkItem(
        id="fixture-understand-1",
        task="dialect_understanding",
        input="문장",
        choices=("A", "B"),
        gold="B",
        source="developer_fixture",
        review_status="fixture_only",
    )
    assert BenchmarkItem.from_dict(item.to_dict()) == item


def test_invalid_benchmark_item_is_rejected() -> None:
    with pytest.raises(ValidationError, match="gold"):
        BenchmarkItem(
            id="bad",
            task="region_classification",
            input="문장",
            choices=("울산", "부산"),
            gold="대구",
        ).validate()


def test_character_f1_and_task_metrics() -> None:
    assert character_f1("가 나", "가나") == 1.0
    items = [
        BenchmarkItem(
            id="understand-1",
            task="dialect_understanding",
            input="입력",
            choices=("A", "B"),
            gold="B",
            source="fixture",
            review_status="fixture_only",
        ),
        BenchmarkItem(
            id="normalize-1",
            task="dialect_to_standard",
            input="입력",
            reference="표준 문장",
            source="fixture",
            review_status="fixture_only",
        ),
    ]
    result = evaluate_predictions(items, {"understand-1": "B", "normalize-1": "표준 문장"})
    assert result["evaluated"] == 2
    assert result["tasks"]["dialect_understanding"]["exact_accuracy"] == 1.0
    assert result["tasks"]["dialect_understanding"]["macro_f1"] == 1.0
    assert result["tasks"]["dialect_to_standard"]["character_f1_mean"] == 1.0


def test_missing_prediction_is_reported() -> None:
    item = BenchmarkItem(
        id="understand-1",
        task="dialect_understanding",
        input="입력",
        choices=("A", "B"),
        gold="B",
        source="fixture",
        review_status="fixture_only",
    )
    assert evaluate_predictions([item], {})["missing_predictions"] == 1
