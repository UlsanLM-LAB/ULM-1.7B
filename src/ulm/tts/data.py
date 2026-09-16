"""ULM-TTS data contracts and validation."""

from __future__ import annotations

import json
import wave
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TTSRecord:
    utterance_id: str
    speaker_id: str
    audio_path: str
    transcript: str
    region_tier: str
    age_group: str
    dialect_strength: int
    sample_rate: int
    duration: float
    consent_scope: str
    quality_grade: str
    split: str
    metadata: dict

    @classmethod
    def from_dict(cls, row: dict) -> TTSRecord:
        required = {f.name for f in cls.__dataclass_fields__.values()}
        missing = sorted(required - row.keys())
        if missing:
            raise ValueError(f"missing TTS fields: {', '.join(missing)}")
        rec = cls(**{k: row[k] for k in required})
        rec.validate()
        return rec

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if not self.utterance_id.strip() or not self.speaker_id.strip():
            raise ValueError("utterance_id and speaker_id must be non-empty")
        if not self.transcript.strip():
            raise ValueError(f"{self.utterance_id}: transcript is empty")
        if self.region_tier not in {"U0", "U1", "U2", "GX"}:
            raise ValueError(f"{self.utterance_id}: invalid region_tier")
        if not 0 <= int(self.dialect_strength) <= 3:
            raise ValueError(f"{self.utterance_id}: dialect_strength must be 0..3")
        if self.sample_rate <= 0 or self.duration <= 0:
            raise ValueError(f"{self.utterance_id}: invalid audio metadata")
        if self.split not in {"train", "validation", "test"}:
            raise ValueError(f"{self.utterance_id}: invalid split")
        if not isinstance(self.metadata, dict):
            raise ValueError(f"{self.utterance_id}: metadata must be an object")


def load_manifest(path: str | Path) -> list[TTSRecord]:
    records: list[TTSRecord] = []
    with Path(path).open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                records.append(TTSRecord.from_dict(json.loads(line)))
            except Exception as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
    validate_no_speaker_leakage(records)
    return records


def validate_no_speaker_leakage(records: Iterable[TTSRecord]) -> None:
    owner: dict[str, str] = {}
    for rec in records:
        previous = owner.setdefault(rec.speaker_id, rec.split)
        if previous != rec.split:
            raise ValueError(
                f"speaker leakage: {rec.speaker_id!r} appears in {previous!r} and {rec.split!r}"
            )


def inspect_wav(path: str | Path) -> tuple[int, float]:
    with wave.open(str(path), "rb") as wav:
        sr = wav.getframerate()
        duration = wav.getnframes() / float(sr)
    return sr, duration


def validate_audio_files(
    records: Iterable[TTSRecord],
    audio_root: str | Path,
    min_duration: float = 0.4,
    max_duration: float = 15.0,
) -> list[str]:
    root = Path(audio_root)
    errors: list[str] = []
    for rec in records:
        path = root / rec.audio_path
        if not path.is_file():
            errors.append(f"{rec.utterance_id}: missing {path}")
            continue
        try:
            sr, duration = inspect_wav(path)
        except (wave.Error, EOFError) as exc:
            errors.append(f"{rec.utterance_id}: unreadable wav: {exc}")
            continue
        if sr != rec.sample_rate:
            errors.append(f"{rec.utterance_id}: sample_rate manifest={rec.sample_rate} wav={sr}")
        if abs(duration - rec.duration) > 0.1:
            errors.append(
                f"{rec.utterance_id}: duration manifest={rec.duration:.3f} wav={duration:.3f}"
            )
        if not min_duration <= duration <= max_duration:
            errors.append(f"{rec.utterance_id}: duration {duration:.3f}s outside quality gate")
    return errors
