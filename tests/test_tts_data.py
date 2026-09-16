import json
import wave
from pathlib import Path

import pytest
import yaml

from ulm.tts.data import TTSRecord, inspect_wav, load_manifest, validate_audio_files
from ulm.tts.train import main as train_main


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


def write_test_wav(path: Path, sr: int = 16000, duration: float = 1.0):
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        num_frames = int(sr * duration)
        f.writeframes(b"\0\0" * num_frames)


def test_record_to_dict_roundtrip():
    data = row()
    rec = TTSRecord.from_dict(data)
    assert rec.to_dict() == data


def test_record_rejects_invalid_strength():
    with pytest.raises(ValueError, match="0..3"):
        TTSRecord.from_dict(row(dialect_strength=4))


def test_record_rejects_missing_fields():
    data = row()
    del data["transcript"]
    with pytest.raises(ValueError, match="missing TTS fields"):
        TTSRecord.from_dict(data)


def test_record_rejects_empty_transcript():
    with pytest.raises(ValueError, match="transcript is empty"):
        TTSRecord.from_dict(row(transcript="  "))


def test_record_rejects_invalid_region_or_split():
    with pytest.raises(ValueError, match="invalid region_tier"):
        TTSRecord.from_dict(row(region_tier="INVALID"))
    with pytest.raises(ValueError, match="invalid split"):
        TTSRecord.from_dict(row(split="train_test"))


def test_manifest_rejects_speaker_leakage(tmp_path):
    path = tmp_path / "manifest.jsonl"
    path.write_text(
        "\n".join([json.dumps(row()), json.dumps(row(utterance_id="u2", split="test"))]),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="speaker leakage"):
        load_manifest(path)


def test_inspect_wav(tmp_path):
    audio = tmp_path / "u1.wav"
    write_test_wav(audio, sr=16000, duration=1.0)
    sr, duration = inspect_wav(audio)
    assert sr == 16000
    assert abs(duration - 1.0) < 0.01


def test_audio_validation(tmp_path):
    audio = tmp_path / "u1.wav"
    write_test_wav(audio, sr=16000, duration=1.0)
    rec = TTSRecord.from_dict(row())
    assert validate_audio_files([rec], tmp_path) == []


def test_audio_validation_missing_and_corrupt(tmp_path):
    rec = TTSRecord.from_dict(row(audio_path="nonexistent.wav"))
    errors = validate_audio_files([rec], tmp_path)
    assert any("missing" in err for err in errors)

    corrupt = tmp_path / "corrupt.wav"
    corrupt.write_text("not a real wav header", encoding="utf-8")
    rec_corrupt = TTSRecord.from_dict(row(audio_path="corrupt.wav"))
    errors2 = validate_audio_files([rec_corrupt], tmp_path)
    assert any("unreadable wav" in err for err in errors2)


def test_tts_train_validate_only(tmp_path):
    audio_path = tmp_path / "u1.wav"
    write_test_wav(audio_path, sr=16000, duration=1.0)
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(json.dumps(row(audio_path="u1.wav")) + "\n", encoding="utf-8")

    out_dir = tmp_path / "tts_out"
    cfg = {
        "output_dir": str(out_dir),
        "data": {
            "manifest": str(manifest_path),
            "audio_root": str(tmp_path),
            "min_duration": 0.4,
            "max_duration": 15.0,
        },
    }
    cfg_path = tmp_path / "test_cfg.yaml"
    cfg_path.write_text(yaml.dump(cfg), encoding="utf-8")

    ret = train_main(["--config", str(cfg_path), "--validate-only"])
    assert ret == 0
    assert (out_dir / "resolved_config.json").exists()


def test_tts_train_gated_acoustic(tmp_path):
    audio_path = tmp_path / "u1.wav"
    write_test_wav(audio_path, sr=16000, duration=1.0)
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(json.dumps(row(audio_path="u1.wav")) + "\n", encoding="utf-8")

    out_dir = tmp_path / "tts_out"
    cfg = {
        "output_dir": str(out_dir),
        "data": {
            "manifest": str(manifest_path),
            "audio_root": str(tmp_path),
        },
    }
    cfg_path = tmp_path / "test_cfg.yaml"
    cfg_path.write_text(yaml.dump(cfg), encoding="utf-8")

    with pytest.raises(SystemExit, match="Acoustic fine-tuning is gated"):
        train_main(["--config", str(cfg_path)])
