import json
import wave
from pathlib import Path

from ulm.tts.preprocess import (
    normalize_transcript,
    preprocess_audio,
    preprocess_manifest,
)


def write_test_wav(path: Path, sr: int = 22050, duration: float = 1.0):
    with wave.open(str(path), "wb") as f:
        f.setnchannels(2)  # stereo
        f.setsampwidth(2)
        f.setframerate(sr)
        num_frames = int(sr * duration)
        f.writeframes(b"\x10\x00\x10\x00" * num_frames)


def test_normalize_transcript_strips_and_collapses():
    assert normalize_transcript("  밥   묵었나  ") == "밥 묵었나"


def test_normalize_transcript_removes_control_chars():
    assert normalize_transcript("밥\x00묵었나\t잘") == "밥묵었나잘"


def test_normalize_transcript_nfc():
    # Decomposed hangul should normalize to NFC
    import unicodedata

    decomposed = unicodedata.normalize("NFD", "밥")
    result = normalize_transcript(decomposed)
    assert result == unicodedata.normalize("NFC", decomposed)


def test_preprocess_audio(tmp_path):
    in_wav = tmp_path / "in.wav"
    out_wav = tmp_path / "out.wav"
    write_test_wav(in_wav, sr=22050, duration=1.0)

    sr, dur = preprocess_audio(in_wav, out_wav, target_sr=16000)
    assert sr == 16000
    assert out_wav.exists()
    assert abs(dur - 1.0) < 0.1

    # Check output is mono
    with wave.open(str(out_wav), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getframerate() == 16000


def test_preprocess_manifest(tmp_path):
    in_wav = tmp_path / "sample.wav"
    write_test_wav(in_wav, sr=22050, duration=1.0)
    manifest = tmp_path / "manifest.jsonl"
    record = {
        "utterance_id": "u1",
        "audio_path": "sample.wav",
        "transcript": "  밥   묵었나  ",
    }
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")

    out_dir = tmp_path / "preprocessed"
    errors = preprocess_manifest(manifest, tmp_path, out_dir, target_sr=16000)
    assert errors == []
    out_manifest = out_dir / "manifest.jsonl"
    assert out_manifest.exists()
    out_rec = json.loads(out_manifest.read_text(encoding="utf-8").strip())
    assert out_rec["transcript"] == "밥 묵었나"
    assert out_rec["sample_rate"] == 16000
    assert (out_dir / out_rec["audio_path"]).exists()
