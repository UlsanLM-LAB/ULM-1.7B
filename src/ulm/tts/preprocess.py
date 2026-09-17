"""ULM-TTS audio preprocessing pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def load_audio(path: str | Path) -> tuple[Any, int]:
    """Load audio file as float32 torch.Tensor [channels, samples] and sample rate."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch required. Install with: pip install -e '.[tts]'") from exc

    try:
        import torchaudio

        return torchaudio.load(str(path))
    except (ImportError, RuntimeError):
        import numpy as np
        import scipy.io.wavfile

        sr, data = scipy.io.wavfile.read(str(path))
        if data.ndim == 1:
            data = data[None, :]
        else:
            data = data.T
        if data.dtype == np.int16:
            tensor = torch.from_numpy(data.astype(np.float32) / 32768.0)
        elif data.dtype == np.int32:
            tensor = torch.from_numpy(data.astype(np.float32) / 2147483648.0)
        else:
            tensor = torch.from_numpy(data.astype(np.float32))
        return tensor, sr


def save_audio(path: str | Path, waveform: Any, sr: int) -> None:
    """Save audio tensor [channels, samples] as 16-bit PCM WAV."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        import torchaudio

        torchaudio.save(str(out), waveform, sr, bits_per_sample=16, encoding="PCM_S")
    except (ImportError, RuntimeError):
        import numpy as np
        import scipy.io.wavfile

        data = waveform.detach().cpu().numpy()
        if data.ndim == 2:
            if data.shape[0] == 1:
                data = data[0]
            else:
                data = data.T
        data = (np.clip(data, -1.0, 1.0) * 32767.0).astype(np.int16)
        scipy.io.wavfile.write(str(out), rate=sr, data=data)


def preprocess_audio(
    input_path: str | Path,
    output_path: str | Path,
    target_sr: int = 16000,
) -> tuple[int, float]:
    """Load, normalize, and save audio as 16-bit mono PCM WAV."""
    try:
        import torch
        import torchaudio.functional as F
    except ImportError as exc:
        raise RuntimeError(
            "torch/torchaudio required. Install with: pip install -e '.[tts]'"
        ) from exc

    waveform, sr = load_audio(input_path)

    # Stereo to mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample
    if sr != target_sr:
        waveform = F.resample(waveform, sr, target_sr)
        sr = target_sr

    # Silence trimming (leading/trailing below -40dB ~ amplitude < 0.01)
    if waveform.shape[1] > 0:
        abs_wave = torch.abs(waveform[0])
        mask = abs_wave > 0.01
        if mask.any():
            start = mask.nonzero()[0].item()
            end = mask.nonzero()[-1].item()
            waveform = waveform[:, start : end + 1]

    # Clipping detection
    peak = torch.max(torch.abs(waveform)).item()
    if peak > 0.99:
        log.warning("clipping detected in %s (peak=%.3f)", input_path, peak)

    # Peak normalization to -1.0 dB
    target_peak = 10 ** (-1.0 / 20)
    if peak > 0:
        waveform = waveform * (target_peak / peak)

    # Save
    out = Path(output_path)
    save_audio(out, waveform, sr)

    duration = waveform.shape[1] / sr
    return sr, duration


def normalize_transcript(text: str) -> str:
    """Strip, NFC-normalize, collapse whitespace, remove control chars."""
    text = text.strip()
    text = unicodedata.normalize("NFC", text)
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C")
    text = re.sub(r"\s+", " ", text)
    return text


def compute_mel_spectrogram(
    waveform: Any,
    sample_rate: int,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mels: int = 80,
) -> Any:
    """Compute log-scale mel spectrogram. Returns tensor [n_mels, time]."""
    try:
        import torch
        import torchaudio.transforms as T
    except ImportError as exc:
        raise RuntimeError("torch/torchaudio required") from exc

    mel_transform = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    mel = mel_transform(waveform).squeeze(0)
    return torch.log(mel + 1e-5)


def preprocess_manifest(
    manifest_path: str | Path,
    audio_root: str | Path,
    output_dir: str | Path,
    target_sr: int = 16000,
) -> list[str]:
    """Preprocess all audio in a manifest and write updated manifest."""
    manifest_path = Path(manifest_path)
    audio_root = Path(audio_root)
    output_dir = Path(output_dir)
    out_audio = output_dir / "audio"
    out_audio.mkdir(parents=True, exist_ok=True)
    out_manifest = output_dir / "manifest.jsonl"

    errors: list[str] = []
    with (
        manifest_path.open(encoding="utf-8") as fin,
        out_manifest.open("w", encoding="utf-8") as fout,
    ):
        for line_no, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                rel = record.get("audio_path", "")
                if not rel:
                    raise ValueError("missing audio_path")

                src = audio_root / rel
                dst = out_audio / Path(rel).with_suffix(".wav").name

                sr, dur = preprocess_audio(src, dst, target_sr)

                if "transcript" in record:
                    record["transcript"] = normalize_transcript(record["transcript"])
                record["audio_path"] = f"audio/{dst.name}"
                record["sample_rate"] = sr
                record["duration"] = round(dur, 4)

                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            except Exception as exc:
                msg = f"line {line_no}: {exc}"
                log.error(msg)
                errors.append(msg)

    return errors


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for TTS audio preprocessing."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Preprocess audio for ULM-TTS training")
    parser.add_argument("--manifest", required=True, help="input manifest JSONL path")
    parser.add_argument("--audio-root", required=True, help="root directory of input audio")
    parser.add_argument("--output-dir", required=True, help="output directory")
    parser.add_argument("--target-sr", type=int, default=16000, help="target sample rate")
    args = parser.parse_args(argv)

    log.info("preprocessing: manifest=%s audio_root=%s", args.manifest, args.audio_root)
    errors = preprocess_manifest(args.manifest, args.audio_root, args.output_dir, args.target_sr)

    if errors:
        log.warning("completed with %d error(s)", len(errors))
        return 1

    log.info("preprocessing complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
