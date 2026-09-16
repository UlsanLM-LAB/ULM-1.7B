"""Local ULM-TTS inference using the initial Korean MMS/VITS baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize Korean/Ulsan-dialect text")
    parser.add_argument("text")
    parser.add_argument("--config", default="configs/tts/mms_vits.yaml")
    parser.add_argument("--output", default="outputs/tts/sample.wav")
    parser.add_argument("--model", default=None, help="checkpoint/model id override")
    args = parser.parse_args()

    try:
        import scipy.io.wavfile
        import torch
        from transformers import AutoTokenizer, VitsModel
    except ImportError as exc:
        raise SystemExit("Install TTS dependencies with: pip install -e '.[tts]'") from exc

    with Path(args.config).open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    model_id = args.model or cfg["model_id"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = VitsModel.from_pretrained(model_id).to(device).eval()
    inputs = tokenizer(args.text, return_tensors="pt").to(device)

    inf = cfg.get("inference", {})
    with torch.inference_mode():
        waveform = model(
            **inputs,
            noise_scale=float(inf.get("noise_scale", 0.667)),
            noise_scale_duration=float(inf.get("noise_scale_duration", 0.8)),
            speaking_rate=float(inf.get("speaking_rate", 1.0)),
        ).waveform[0].detach().cpu().float().numpy()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    scipy.io.wavfile.write(output, rate=model.config.sampling_rate, data=waveform)
    print(output)


if __name__ == "__main__":
    main()
