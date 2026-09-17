"""ULM-TTS evaluation metrics and batch synthesis."""

from __future__ import annotations

import argparse
import json
import logging
import time
import wave
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def compute_rtf(audio_duration: float, synthesis_time: float) -> float:
    """Real-Time Factor: synthesis_time / audio_duration. Lower is better."""
    if audio_duration <= 0.0:
        return 0.0
    return synthesis_time / audio_duration


def compute_duration_stats(wav_path: str | Path) -> dict[str, Any]:
    """Extract duration, sample rate, peak amplitude, and RMS energy from WAV."""
    import numpy as np

    with wave.open(str(wav_path), "rb") as wf:
        n_frames = wf.getnframes()
        sr = wf.getframerate()
        duration = n_frames / float(sr)
        raw = wf.readframes(n_frames)
        width = wf.getsampwidth()

    if width == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif width == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        data = np.zeros(n_frames, dtype=np.float32)

    peak = float(np.max(np.abs(data))) if len(data) > 0 else 0.0
    rms = float(np.sqrt(np.mean(data**2))) if len(data) > 0 else 0.0
    return {"duration": duration, "sample_rate": sr, "peak_amplitude": peak, "rms_energy": rms}


def compute_pitch_stats(waveform: Any, sample_rate: int) -> dict[str, Any]:
    """Compute F0 statistics using torchaudio pitch detection."""
    try:
        import torch
        import torchaudio.functional as F
    except ImportError:
        return _empty_pitch()

    if not isinstance(waveform, torch.Tensor):
        waveform = torch.tensor(waveform, dtype=torch.float32)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    try:
        f0 = F.detect_pitch_frequency(waveform, sample_rate)
        voiced = f0[f0 > 0.0]
        if len(voiced) == 0:
            return _empty_pitch()
        return {
            "f0_mean": float(voiced.mean()),
            "f0_std": float(voiced.std()),
            "f0_min": float(voiced.min()),
            "f0_max": float(voiced.max()),
            "voiced_ratio": float(len(voiced) / f0.numel()),
        }
    except Exception as exc:
        log.warning("pitch detection failed: %s", exc)
        return _empty_pitch()


def _empty_pitch() -> dict[str, float]:
    return {"f0_mean": 0.0, "f0_std": 0.0, "f0_min": 0.0, "f0_max": 0.0, "voiced_ratio": 0.0}


def batch_synthesize(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    output_dir: str | Path,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Synthesize texts, measure latency, and compute per-utterance stats."""
    import scipy.io.wavfile
    import torch

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    sr = config.get("sample_rate", 22050)

    model.eval()
    device = next(model.parameters()).device
    results: list[dict[str, Any]] = []

    for idx, text in enumerate(texts):
        t0 = time.time()
        try:
            inputs = tokenizer(text, return_tensors="pt").to(device)
            with torch.inference_mode():
                waveform = model(**inputs).waveform[0].cpu().numpy()
            synth_time = time.time() - t0
            audio_dur = len(waveform) / sr
            rtf = compute_rtf(audio_dur, synth_time)

            wav_path = out / f"eval_{idx:04d}.wav"
            scipy.io.wavfile.write(str(wav_path), sr, waveform)

            dur_stats = compute_duration_stats(wav_path)
            pitch_stats = compute_pitch_stats(torch.from_numpy(waveform).unsqueeze(0), sr)

            entry: dict[str, Any] = {
                "id": idx,
                "text": text,
                "synthesis_time": synth_time,
                "rtf": rtf,
                "out_path": str(wav_path),
            }
            entry.update(dur_stats)
            entry.update(pitch_stats)
            results.append(entry)
        except Exception as exc:
            log.error("synthesis failed for text %d: %s", idx, exc)

    return results


def compare_checkpoints(
    results_a: list[dict[str, Any]],
    results_b: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare aggregate metrics between two evaluation runs."""
    if not results_a or not results_b:
        return {}

    def _mean(results: list[dict[str, Any]], key: str) -> float:
        vals = [r[key] for r in results if key in r]
        return sum(vals) / len(vals) if vals else 0.0

    summary: dict[str, Any] = {}
    for k in ("rtf", "f0_mean", "voiced_ratio", "duration", "rms_energy"):
        a, b = _mean(results_a, k), _mean(results_b, k)
        summary[f"{k}_mean_a"] = a
        summary[f"{k}_mean_b"] = b
        summary[f"{k}_delta"] = b - a
    return summary


def generate_eval_report(results: list[dict[str, Any]], output_path: str | Path) -> None:
    """Write per-utterance and aggregate evaluation report as JSON."""
    if not results:
        return
    numeric_keys = [k for k, v in results[0].items() if isinstance(v, int | float)]
    agg: dict[str, float] = {}
    for key in numeric_keys:
        vals = [r[key] for r in results]
        agg[f"{key}_mean"] = sum(vals) / len(vals)
        agg[f"{key}_min"] = min(vals)
        agg[f"{key}_max"] = max(vals)
    report = {"aggregate": agg, "per_utterance": results}
    Path(output_path).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for TTS evaluation."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Evaluate ULM-TTS checkpoints")
    parser.add_argument("--config", required=True, help="TTS config YAML path")
    parser.add_argument("--checkpoint", default=None, help="model checkpoint path")
    parser.add_argument("--texts-file", required=True, help="one text per line")
    parser.add_argument("--output-dir", required=True, help="output directory")
    args = parser.parse_args(argv)

    try:
        import yaml
        from transformers import AutoTokenizer, VitsModel
    except ImportError as exc:
        raise SystemExit("Install TTS deps: pip install -e '.[tts]'") from exc

    with Path(args.config).open(encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_id = args.checkpoint or config.get("model_id", "facebook/mms-tts-kor")
    log.info("loading model: %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = VitsModel.from_pretrained(model_id)

    texts = Path(args.texts_file).read_text(encoding="utf-8").strip().splitlines()
    texts = [t.strip() for t in texts if t.strip()]

    log.info("evaluating %d utterances", len(texts))
    results = batch_synthesize(model, tokenizer, texts, args.output_dir, config)

    report_path = Path(args.output_dir) / "eval_report.json"
    generate_eval_report(results, report_path)
    log.info("report saved to %s", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
