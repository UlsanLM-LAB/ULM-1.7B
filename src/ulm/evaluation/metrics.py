"""외부 ML dependency가 없는 초기 benchmark metric."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from .schema import BenchmarkItem


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).strip()
    return re.sub(r"\s+", " ", text)


def _characters(text: str) -> list[str]:
    return [char for char in normalize_text(text) if not char.isspace()]


def character_f1(prediction: str, reference: str) -> float:
    predicted = Counter(_characters(prediction))
    expected = Counter(_characters(reference))
    overlap = sum((predicted & expected).values())
    if not predicted or not expected:
        return 1.0 if not predicted and not expected else 0.0
    precision = overlap / sum(predicted.values())
    recall = overlap / sum(expected.values())
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def exact_match(prediction: str, gold: str | Iterable[str]) -> bool:
    expected = (
        {normalize_text(gold)} if isinstance(gold, str) else {normalize_text(item) for item in gold}
    )
    return normalize_text(prediction) in expected


def _classification_f1(rows: list[tuple[str, str]]) -> dict[str, float]:
    labels = sorted({label for pair in rows for label in pair})
    f1_values: list[float] = []
    for label in labels:
        true_positive = sum(gold == label and pred == label for gold, pred in rows)
        false_positive = sum(gold != label and pred == label for gold, pred in rows)
        false_negative = sum(gold == label and pred != label for gold, pred in rows)
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1_values.append(
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
    return {"macro_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0}


def evaluate_predictions(
    items: Iterable[BenchmarkItem], predictions: Mapping[str, str]
) -> dict[str, Any]:
    """prediction mapping을 task별로 집계한다.

    자동 결과는 baseline 지표일 뿐 authenticity 또는 의미 보존의 human score를 대체하지 않는다.
    """

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    classification_grouped: dict[str, list[tuple[str, str]]] = defaultdict(list)
    missing = 0
    for item in items:
        if item.id not in predictions:
            missing += 1
            continue
        prediction = predictions[item.id]
        result: dict[str, Any] = {"id": item.id, "task": item.task}
        if item.gold is not None:
            result["exact_match"] = exact_match(prediction, item.gold)
        if item.reference is not None:
            result["character_f1"] = character_f1(prediction, item.reference)
        grouped[item.task].append(result)
        if item.choices and item.gold is not None:
            classification_grouped[item.task].append((item.gold, prediction))

    task_metrics: dict[str, Any] = {}
    for task, rows in sorted(grouped.items()):
        metrics: dict[str, Any] = {"count": len(rows)}
        exact_rows = [row for row in rows if "exact_match" in row]
        char_rows = [row for row in rows if "character_f1" in row]
        if exact_rows:
            metrics["exact_accuracy"] = sum(row["exact_match"] for row in exact_rows) / len(
                exact_rows
            )
        if char_rows:
            metrics["character_f1_mean"] = sum(row["character_f1"] for row in char_rows) / len(
                char_rows
            )
        if classification_grouped.get(task):
            metrics.update(_classification_f1(classification_grouped[task]))
        task_metrics[task] = metrics

    all_evaluated = [row for rows in grouped.values() for row in rows]
    summary: dict[str, Any] = {
        "items": len(all_evaluated) + missing,
        "evaluated": len(all_evaluated),
        "missing_predictions": missing,
        "tasks": task_metrics,
    }
    if all_evaluated:
        exact_rows = [row for row in all_evaluated if "exact_match" in row]
        char_rows = [row for row in all_evaluated if "character_f1" in row]
        if exact_rows:
            summary["exact_accuracy"] = sum(row["exact_match"] for row in exact_rows) / len(
                exact_rows
            )
        if char_rows:
            summary["character_f1_mean"] = sum(row["character_f1"] for row in char_rows) / len(
                char_rows
            )
    return summary
