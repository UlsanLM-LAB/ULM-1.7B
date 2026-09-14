"""AI Hub 방언 JSON audit와 울산 화자 tier 분류."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_ULSAN_ALIASES = frozenset({"울산", "울산광역시"})
LOCATION_FIELDS = ("birthplace", "principal_residence", "current_residence")


def clean_location(value: Any) -> str:
    """위치 문자열의 공백만 정규화한다. 의미를 추측해 보정하지 않는다."""

    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).strip())


def is_ulsan(value: Any, aliases: Iterable[str] = DEFAULT_ULSAN_ALIASES) -> bool:
    normalized = clean_location(value)
    if not normalized:
        return False
    alias_values = {clean_location(alias) for alias in aliases if clean_location(alias)}
    return normalized in alias_values or normalized.startswith("울산광역시 ")


def _principal_value(speaker: Mapping[str, Any]) -> Any:
    return speaker.get("raised_region") or speaker.get("principal_residence")


def speaker_tier(speaker: Mapping[str, Any], aliases: Iterable[str] = DEFAULT_ULSAN_ALIASES) -> str:
    """보고서의 연구용 U0/U1/U2/GX 규칙을 적용한다."""

    birthplace = is_ulsan(speaker.get("birthplace"), aliases)
    principal = is_ulsan(_principal_value(speaker), aliases)
    current = is_ulsan(speaker.get("current_residence"), aliases)
    if birthplace and principal:
        return "U0"
    if principal and current:
        return "U1"
    if birthplace or principal or current:
        return "U2"
    return "GX"


def _as_list(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, Mapping)]
    return []


def _nested_list(obj: Mapping[str, Any], key: str) -> list[Mapping[str, Any]]:
    direct = _as_list(obj.get(key))
    if direct:
        return direct
    metadata = obj.get("metadata")
    if isinstance(metadata, Mapping):
        return _as_list(metadata.get(key))
    return []


def _first_text(obj: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _speaker_id(speaker: Mapping[str, Any]) -> str:
    value = speaker.get("id", speaker.get("speaker_id"))
    return str(value).strip() if value is not None else ""


def _utterance_speaker_id(utterance: Mapping[str, Any]) -> str:
    value = utterance.get("speaker_id", utterance.get("speakerId"))
    return str(value).strip() if value is not None else ""


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result


def _flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def _audio_present(utterance: Mapping[str, Any], document: Mapping[str, Any]) -> bool:
    keys = ("audio", "audio_path", "audioPath", "wav", "file", "file_name", "fileName")
    return any(utterance.get(key) not in (None, "") for key in keys) or any(
        document.get(key) not in (None, "") for key in keys
    )


@dataclass
class _TierAccumulator:
    speaker_ids: set[str] = field(default_factory=set)
    utterances: int = 0
    duration_sec: float = 0.0
    duration_count: int = 0
    parallel_pairs: int = 0
    dialect_tokens: int = 0
    eojeol_tokens: int = 0
    male: int = 0
    female: int = 0
    unknown_gender: int = 0
    ages: Counter[str] = field(default_factory=Counter)
    missing_audio: int = 0
    missing_standard_form: int = 0
    missing_dialect_form: int = 0

    def register_speaker(self, identifier: str, speaker: Mapping[str, Any]) -> None:
        if not identifier or identifier in self.speaker_ids:
            return
        self.speaker_ids.add(identifier)
        sex = str(speaker.get("sex", speaker.get("gender", ""))).strip().lower()
        if sex in {"m", "male", "남", "남성"}:
            self.male += 1
        elif sex in {"f", "female", "여", "여성"}:
            self.female += 1
        else:
            self.unknown_gender += 1
        age = speaker.get("age_group", speaker.get("age"))
        self.ages[str(age).strip() if age not in (None, "") else "unknown"] += 1

    def row(self, tier: str) -> dict[str, Any]:
        duration_average = self.duration_sec / self.duration_count if self.duration_count else 0.0
        dialect_ratio = self.dialect_tokens / self.eojeol_tokens if self.eojeol_tokens else 0.0
        gender = {"male": self.male, "female": self.female, "unknown": self.unknown_gender}
        return {
            "tier": tier,
            "unique_speakers": len(self.speaker_ids),
            "utterances": self.utterances,
            "audio_hours": round(self.duration_sec / 3600, 6),
            "parallel_pairs": self.parallel_pairs,
            "avg_duration_sec": round(duration_average, 6),
            "dialect_token_ratio": round(dialect_ratio, 6),
            "gender_distribution": json.dumps(gender, ensure_ascii=False, sort_keys=True),
            "age_distribution": json.dumps(dict(sorted(self.ages.items())), ensure_ascii=False),
            "missing_audio": self.missing_audio,
            "missing_standard_form": self.missing_standard_form,
            "missing_dialect_form": self.missing_dialect_form,
        }


@dataclass
class AuditSummary:
    """AI Hub raw directory audit 결과."""

    rows: list[dict[str, Any]]
    location_values: dict[str, dict[str, int]]
    files_scanned: int
    parse_errors: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "files_scanned": self.files_scanned,
            "parse_errors": self.parse_errors,
            "rows": self.rows,
            "location_values": self.location_values,
        }


def _location_counts(counts: dict[str, Counter[str]], speaker: Mapping[str, Any]) -> None:
    for field_name in LOCATION_FIELDS:
        value = speaker.get(field_name)
        if field_name == "principal_residence" and value in (None, ""):
            value = speaker.get("raised_region")
        counts[field_name][clean_location(value)] += 1


def audit_json_root(root: str | Path) -> AuditSummary:
    """root 아래 JSON을 읽고 tier별 inventory를 계산한다.

    이 함수는 다운로드·로그인·약관 동의를 수행하지 않는다. 이미 합법적으로 확보한
    local archive만 입력으로 받는다.
    """

    root_path = Path(root)
    if not root_path.is_dir():
        raise NotADirectoryError(f"JSON root directory가 아닙니다: {root_path}")

    accumulators = {tier: _TierAccumulator() for tier in ("U0", "U1", "U2", "GX")}
    location_counts = {field_name: Counter() for field_name in LOCATION_FIELDS}
    speaker_info: dict[str, Mapping[str, Any]] = {}
    files_scanned = 0
    parse_errors: list[str] = []

    for path in sorted(root_path.rglob("*.json")):
        files_scanned += 1
        try:
            with path.open("r", encoding="utf-8") as handle:
                document = json.load(handle)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            parse_errors.append(f"{path}: {exc}")
            continue
        if not isinstance(document, Mapping):
            parse_errors.append(f"{path}: 최상위 JSON이 object가 아닙니다")
            continue

        speakers = _nested_list(document, "speaker")
        for speaker in speakers:
            identifier = _speaker_id(speaker)
            if identifier:
                speaker_info[identifier] = speaker
                _location_counts(location_counts, speaker)
                accumulators[speaker_tier(speaker)].register_speaker(identifier, speaker)

        utterances = _nested_list(document, "utterance")
        sole_speaker = _speaker_id(speakers[0]) if len(speakers) == 1 else ""
        for utterance in utterances:
            identifier = _utterance_speaker_id(utterance) or sole_speaker
            speaker = speaker_info.get(identifier, {})
            tier = speaker_tier(speaker)
            accumulator = accumulators[tier]
            if identifier and identifier not in speaker_info:
                accumulator.register_speaker(identifier, speaker)
            accumulator.utterances += 1

            standard_text = _first_text(utterance, ("standard_form", "standard_text", "standard"))
            dialect_text = _first_text(utterance, ("dialect_form", "dialect_text", "form"))
            if not standard_text:
                accumulator.missing_standard_form += 1
            if not dialect_text:
                accumulator.missing_dialect_form += 1
            if standard_text and dialect_text:
                accumulator.parallel_pairs += 1

            start = _number(utterance.get("start", utterance.get("start_sec")))
            end = _number(utterance.get("end", utterance.get("end_sec")))
            if start is not None and end is not None and end >= start:
                accumulator.duration_sec += end - start
                accumulator.duration_count += 1
            if not _audio_present(utterance, document):
                accumulator.missing_audio += 1

            eojeols = utterance.get("eojeolList", utterance.get("eojeol_list", []))
            if isinstance(eojeols, list):
                for eojeol in eojeols:
                    if not isinstance(eojeol, Mapping):
                        continue
                    accumulator.eojeol_tokens += 1
                    if _flag(eojeol.get("isDialect", eojeol.get("is_dialect", False))):
                        accumulator.dialect_tokens += 1

    rows = [accumulators[tier].row(tier) for tier in ("U0", "U1", "U2", "GX")]
    return AuditSummary(
        rows=rows,
        location_values={
            field_name: dict(sorted(counter.items()))
            for field_name, counter in location_counts.items()
        },
        files_scanned=files_scanned,
        parse_errors=parse_errors,
    )


def write_inventory_csv(
    summary: AuditSummary, path: str | Path, *, overwrite: bool = False
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"기존 audit 파일을 덮어쓰지 않습니다: {destination}")
    fieldnames = list(summary.rows[0]) if summary.rows else []
    mode = "w" if overwrite else "x"
    with destination.open(mode, encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary.rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="합법적으로 확보한 AI Hub JSON archive의 위치·화자·발화 inventory를 생성합니다."
    )
    parser.add_argument(
        "--input-root", required=True, type=Path, help="JSON 파일이 있는 local directory"
    )
    parser.add_argument("--output", required=True, type=Path, help="tier inventory CSV 출력 경로")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="기존 CSV를 명시적으로 덮어씀 (기본값: 거부)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = audit_json_root(args.input_root)
    write_inventory_csv(summary, args.output, overwrite=args.overwrite)
    print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
