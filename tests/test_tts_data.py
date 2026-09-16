import json
import wave

import pytest

from ulm.tts.data import TTSRecord, load_manifest, validate_audio_files


def row(**overrides):
    value = {
        "utterance_id": "u1",
        "speaker_id": "s1",
        "audio_path": "u1.wav",
        "transcript": "밥 묵었나",
        "region_tier": "U0",
        "age_group": "10s",
        "dialect_strength": 2,
        "sample_rate": 16000,
        "duration": 1.0,
        "consent_scope": "research-training",
        "quality_grade": "A",
        "split": "train",
        "metadata": {},
    }
    value.update(overrides)
    return value


def test_record_rejects_invalid_strength():
    with pytest.raises(ValueError, match="0..3"):
        TTSRecord.from_dict(row(dialect_strength=4))


def test_manifest_rejects_speaker_leakage(tmp_path):
    path = tmp_path / "manifest.jsonl"
    path.write_text(
        "\n".join([json.dumps(row()), json.dumps(row(utterance_id="u2", split="test"))]),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="speaker leakage"):
        load_manifest(path)


def test_audio_validation(tmp_path):
    audio = tmp_path / "u1.wav"
    with wave.open(str(audio), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(16000)
        f.writeframes(b"\0\0" * 16000)
    rec = TTSRecord.from_dict(row())
    assert validate_audio_files([rec], tmp_path) == []
