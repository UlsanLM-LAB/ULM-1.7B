"""ULM-TTS training preparation entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from ulm.tts.data import load_manifest, validate_audio_files


def _read_config(path: str) -> dict:
    with Path(path).open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and validate ULM-TTS training data")
    parser.add_argument("--config", default="configs/tts/mms_vits.yaml")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    cfg = _read_config(args.config)
    records = load_manifest(cfg["data"]["manifest"])
    errors = validate_audio_files(
        records,
        cfg["data"]["audio_root"],
        cfg["data"].get("min_duration", 0.4),
        cfg["data"].get("max_duration", 15.0),
    )
    if errors:
        raise SystemExit("TTS data validation failed:\n" + "\n".join(errors[:50]))

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if args.validate_only:
        print(f"validated {len(records)} TTS records")
        return

    # Hugging Face VitsModel is used as the first Korean inference baseline, but its
    # public interface does not expose a stable generic waveform-supervised training
    # objective. Failing here is intentional: do not claim training happened when it did not.
    raise SystemExit(
        f"Prepared {len(records)} samples. Acoustic fine-tuning is gated until code review "
        "pins a trainable Korean TTS backend/checkpoint. Run with --validate-only for now."
    )


if __name__ == "__main__":
    main()
