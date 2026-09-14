"""benchmark 평가 CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .io import read_items
from .metrics import evaluate_predictions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ULM-Bench prediction JSONL을 평가합니다.")
    parser.add_argument("--benchmark", required=True, type=Path, help="benchmark item JSONL")
    parser.add_argument("--predictions", required=True, type=Path, help="id,prediction JSONL")
    parser.add_argument("--output", type=Path, help="결과 JSON 경로; 생략하면 stdout")
    parser.add_argument("--overwrite", action="store_true", help="기존 결과 JSON을 덮어씀")
    return parser


def _read_predictions(path: Path) -> dict[str, str]:
    predictions: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                identifier = str(row["id"])
                prediction = row["prediction"]
                if not isinstance(prediction, str):
                    raise TypeError("prediction은 문자열이어야 합니다")
                if identifier in predictions:
                    raise ValueError(f"중복 prediction id: {identifier}")
                predictions[identifier] = prediction
            except (KeyError, json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}을 읽을 수 없습니다: {exc}") from exc
    return predictions


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = evaluate_predictions(read_items(args.benchmark), _read_predictions(args.predictions))
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        if args.output.exists() and not args.overwrite:
            raise FileExistsError(f"기존 결과를 덮어쓰지 않습니다: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
