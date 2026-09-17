import json
import wave

from ulm.tts.evaluate import (
    compare_checkpoints,
    compute_duration_stats,
    compute_rtf,
    generate_eval_report,
)


def test_compute_rtf():
    assert compute_rtf(2.0, 1.0) == 0.5
    assert compute_rtf(1.0, 2.0) == 2.0
    assert compute_rtf(0.0, 1.0) == 0.0


def test_compute_duration_stats(tmp_path):
    wav_path = tmp_path / "test.wav"
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 16000)

    stats = compute_duration_stats(wav_path)
    assert stats["duration"] == 1.0
    assert stats["sample_rate"] == 16000
    assert "peak_amplitude" in stats
    assert "rms_energy" in stats


def test_compare_checkpoints():
    results_a = [
        {"rtf": 0.5, "f0_mean": 150.0, "voiced_ratio": 0.6, "duration": 2.0, "rms_energy": 0.1},
        {"rtf": 0.7, "f0_mean": 170.0, "voiced_ratio": 0.4, "duration": 3.0, "rms_energy": 0.2},
    ]
    results_b = [
        {"rtf": 0.4, "f0_mean": 140.0, "voiced_ratio": 0.6, "duration": 2.0, "rms_energy": 0.1},
        {"rtf": 0.6, "f0_mean": 160.0, "voiced_ratio": 0.5, "duration": 3.0, "rms_energy": 0.2},
    ]
    summary = compare_checkpoints(results_a, results_b)
    assert summary["rtf_mean_a"] == 0.6
    assert summary["rtf_mean_b"] == 0.5
    assert summary["rtf_delta"] < 0
    assert summary["f0_mean_delta"] == -10.0


def test_compare_checkpoints_empty():
    assert compare_checkpoints([], []) == {}


def test_generate_eval_report(tmp_path):
    results = [
        {"id": 0, "rtf": 0.5, "f0_mean": 150.0, "duration": 2.0},
        {"id": 1, "rtf": 0.7, "f0_mean": 170.0, "duration": 3.0},
    ]
    out_path = tmp_path / "report.json"
    generate_eval_report(results, out_path)

    assert out_path.exists()
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert "aggregate" in data
    assert "per_utterance" in data
    assert data["aggregate"]["rtf_mean"] == 0.6
    assert data["aggregate"]["f0_mean_mean"] == 160.0
    assert len(data["per_utterance"]) == 2
