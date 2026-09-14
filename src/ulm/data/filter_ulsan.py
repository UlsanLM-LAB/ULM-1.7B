"""AI Hub JSON에서 울산 tier record를 추출하고 canonical split을 만든다."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .aihub import (
    DEFAULT_ULSAN_ALIASES,
    _first_text,
    _flag,
    _nested_list,
    _speaker_id,
    _utterance_speaker_id,
    clean_location,
    speaker_tier,
)
from .io import write_jsonl
from .schema import DatasetRecord
from .split import split_by_speaker

DEFAULT_TIERS = ("U0", "U1")


def _age_group(speaker: Mapping[str, Any]) -> str | None:
    value = speaker.get("age_group", speaker.get("age"))
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)) and 10 <= value < 100:
        return f"{int(value) // 10 * 10}s"
    return str(value).strip()


def _stable_id(namespace: str, raw_value: str, *, prefix: str) -> str:
    digest = hashlib.sha256(f"{namespace}\0{raw_value}".encode()).hexdigest()[:20]
    return f"{prefix}_{digest}"


def _density(utterance: Mapping[str, Any]) -> float | None:
    eojeols = utterance.get("eojeolList", utterance.get("eojeol_list", []))
    if not isinstance(eojeols, list):
        return None
    flags = [
        _flag(item.get("isDialect", item.get("is_dialect", False)))
        for item in eojeols
        if isinstance(item, Mapping)
    ]
    return sum(flags) / len(flags) if flags else None


def extract_ulsan_records(
    root: str | Path,
    *,
    source: str = "aihub_gyeongsang",
    tiers: Sequence[str] = DEFAULT_TIERS,
    aliases: Iterable[str] = DEFAULT_ULSAN_ALIASES,
) -> list[DatasetRecord]:
    """합법적으로 확보한 AI Hub JSON에서 지정 tier의 parallel record를 추출한다.

    `speaker_id`와 `id`는 raw 값을 직접 공개하지 않도록 stable hash로 만든다. 실제 원본 sample
    ID는 비공개 처리용 `metadata`에만 남기므로, 이 출력물을 공개하기 전에는 provenance policy를
    다시 확인해야 한다.
    """

    root_path = Path(root)
    if not root_path.is_dir():
        raise NotADirectoryError(f"JSON root directory가 아닙니다: {root_path}")
    selected_tiers = set(tiers)
    invalid_tiers = selected_tiers - {"U0", "U1", "U2", "GX"}
    if invalid_tiers:
        raise ValueError(f"지원하지 않는 tier: {sorted(invalid_tiers)!r}")
    alias_values = tuple(aliases)
    output: list[DatasetRecord] = []
    for path in sorted(root_path.rglob("*.json")):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"AI Hub JSON을 읽을 수 없습니다: {path}: {exc}") from exc
        if not isinstance(document, Mapping):
            raise ValueError(f"AI Hub JSON 최상위 값이 object가 아닙니다: {path}")
        speakers = _nested_list(document, "speaker")
        speaker_by_id = {
            _speaker_id(speaker): speaker for speaker in speakers if _speaker_id(speaker)
        }
        sole_speaker = _speaker_id(speakers[0]) if len(speakers) == 1 else ""
        for index, utterance in enumerate(_nested_list(document, "utterance")):
            raw_speaker_id = _utterance_speaker_id(utterance) or sole_speaker
            if not raw_speaker_id:
                continue
            speaker = speaker_by_id.get(raw_speaker_id, {})
            tier = speaker_tier(speaker, alias_values)
            if tier not in selected_tiers:
                continue
            dialect_text = _first_text(utterance, ("dialect_form", "dialect_text", "form"))
            standard_text = _first_text(utterance, ("standard_form", "standard_text", "standard"))
            if not dialect_text or not standard_text:
                continue
            raw_utterance_id = str(utterance.get("id", f"{path.stem}:{index}"))
            speaker_id = _stable_id(source, raw_speaker_id, prefix="spk")
            record_id = _stable_id(
                source,
                f"{raw_speaker_id}\0{raw_utterance_id}",
                prefix="sample",
            )
            metadata: dict[str, Any] = {
                "source_sample_id": raw_utterance_id,
                "ulsan_tier": tier,
                "fixture": False,
            }
            density = _density(utterance)
            if density is not None:
                metadata["dialect_density"] = density
            output.append(
                DatasetRecord(
                    id=record_id,
                    source=source,
                    task="dialect_to_standard",
                    speaker_id=speaker_id,
                    birthplace=clean_location(speaker.get("birthplace")) or None,
                    raised_region=clean_location(
                        speaker.get("raised_region", speaker.get("principal_residence"))
                    )
                    or None,
                    current_region=clean_location(speaker.get("current_residence")) or None,
                    age_group=_age_group(speaker),
                    gender=str(speaker.get("sex", speaker.get("gender", ""))).strip() or None,
                    dialect_text=dialect_text,
                    standard_text=standard_text,
                    dialect_strength=None,
                    synthetic=False,
                    human_verified=False,
                    quality_grade={"U0": "A", "U1": "B", "U2": "C", "GX": "C"}[tier],
                    metadata=metadata,
                ).validate()
            )
    return output


def expand_parallel_tasks(records: Iterable[DatasetRecord]) -> list[DatasetRecord]:
    """parallel record를 양방향 변환 task로 확장한다."""

    expanded: list[DatasetRecord] = []
    for record in records:
        record.validate()
        if record.dialect_text is None or record.standard_text is None:
            continue
        for task in ("dialect_to_standard", "standard_to_dialect"):
            expanded.append(
                DatasetRecord(
                    id=f"{record.id}:{task}",
                    source=record.source,
                    task=task,
                    speaker_id=record.speaker_id,
                    birthplace=record.birthplace,
                    raised_region=record.raised_region,
                    current_region=record.current_region,
                    age_group=record.age_group,
                    gender=record.gender,
                    dialect_text=record.dialect_text,
                    standard_text=record.standard_text,
                    dialect_strength=record.dialect_strength,
                    synthetic=record.synthetic,
                    human_verified=record.human_verified,
                    quality_grade=record.quality_grade,
                    metadata={**record.metadata, "derived_from": record.id, "derived_task": task},
                ).validate()
            )
    return expanded


def build_ulsan_splits(
    root: str | Path,
    *,
    source: str = "aihub_gyeongsang",
    tiers: Sequence[str] = DEFAULT_TIERS,
    seed: int = 42,
) -> dict[str, list[DatasetRecord]]:
    records = expand_parallel_tasks(extract_ulsan_records(root, source=source, tiers=tiers))
    return split_by_speaker(records, seed=seed)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "AI Hub JSON에서 지정 울산 tier를 추출해 canonical speaker split JSONL을 생성합니다."
        )
    )
    parser.add_argument(
        "--input-root", required=True, type=Path, help="합법적으로 확보한 JSON directory"
    )
    parser.add_argument(
        "--output-dir", required=True, type=Path, help="train/validation/test 출력 directory"
    )
    parser.add_argument(
        "--tiers", nargs="+", default=list(DEFAULT_TIERS), help="포함할 tier (기본: U0 U1)"
    )
    parser.add_argument("--seed", type=int, default=42, help="speaker split seed")
    parser.add_argument(
        "--overwrite", action="store_true", help="기존 출력 파일을 명시적으로 덮어씀"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"기존 dataset output을 덮어쓰지 않습니다: {args.output_dir}")
    splits = build_ulsan_splits(args.input_root, tiers=args.tiers, seed=args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, records in splits.items():
        write_jsonl(
            args.output_dir / f"{split}.jsonl",
            records,
            overwrite=args.overwrite,
            require_split=True,
        )
    manifest = {
        "source": "aihub_gyeongsang",
        "tiers": args.tiers,
        "seed": args.seed,
        "record_counts": {split: len(records) for split, records in splits.items()},
        "speaker_counts": {
            split: len({record.speaker_id for record in records})
            for split, records in splits.items()
        },
        "contains_raw_audio": False,
        "contains_raw_source_ids": True,
        "public_release": False,
    }
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(f"기존 manifest를 덮어쓰지 않습니다: {manifest_path}")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
