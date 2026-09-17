"""ULM-TTS VITS fine-tuning trainer.

Implements mel-spectrogram reconstruction fine-tuning for the HuggingFace
VitsModel on Ulsan-dialect audio.  The approach freezes the discriminator and
posterior encoder, then optimizes the text encoder + decoder (generator) with
an L1 mel-spectrogram loss.  This is a practical approximation for dialect
adaptation when the original VITS adversarial training setup is not available
through the Transformers public API.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import yaml

from ulm.tts.data import load_manifest, validate_audio_files, validate_consent

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _read_config(path: str) -> dict:
    with Path(path).open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _save_resolved_config(cfg: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class TTSDataset:
    """Torch Dataset that loads preprocessed audio and returns mel + text."""

    def __init__(
        self,
        records: list,
        audio_root: str | Path,
        tokenizer: Any,
        target_sr: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 80,
    ) -> None:
        import torch
        import torchaudio

        self.records = records
        self.audio_root = Path(audio_root)
        self.tokenizer = tokenizer
        self.target_sr = target_sr
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        self._torch = torch

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        from ulm.tts.preprocess import load_audio

        rec = self.records[idx]
        wav_path = self.audio_root / rec.audio_path
        waveform, sr = load_audio(wav_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.target_sr:
            import torchaudio.functional as F

            waveform = F.resample(waveform, sr, self.target_sr)

        mel = self.mel_transform(waveform).squeeze(0)
        mel = self._torch.log(mel + 1e-5)

        tokens = self.tokenizer(rec.transcript, return_tensors="pt")
        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "mel_target": mel,
            "utterance_id": rec.utterance_id,
        }


def _collate_fn(batch: list[dict]) -> dict[str, Any]:
    """Pad variable-length sequences in a batch."""
    import torch
    from torch.nn.utils.rnn import pad_sequence

    input_ids = pad_sequence([b["input_ids"] for b in batch], batch_first=True)
    attention_mask = pad_sequence([b["attention_mask"] for b in batch], batch_first=True)
    max_mel_len = max(b["mel_target"].shape[-1] for b in batch)
    n_mels = batch[0]["mel_target"].shape[0]
    mel_targets = torch.zeros(len(batch), n_mels, max_mel_len)
    mel_lengths = []
    for i, b in enumerate(batch):
        length = b["mel_target"].shape[-1]
        mel_targets[i, :, :length] = b["mel_target"]
        mel_lengths.append(length)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "mel_targets": mel_targets,
        "mel_lengths": torch.tensor(mel_lengths, dtype=torch.long),
    }


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


def _freeze_module(module: Any) -> None:
    for param in module.parameters():
        param.requires_grad = False


def _count_params(model: Any) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def run_training(cfg: dict, *, enable_dashboard: bool = False) -> Path:
    """Execute the VITS fine-tuning loop."""
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, VitsModel

    training_cfg = cfg["training"]
    data_cfg = cfg["data"]
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = training_cfg.get("mixed_precision", "no") != "no" and device.type == "cuda"
    amp_dtype = torch.float16 if training_cfg.get("mixed_precision") == "fp16" else torch.bfloat16

    model_id = cfg.get("model_id", "facebook/mms-tts-kor")
    log.info("loading model: %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = VitsModel.from_pretrained(model_id).to(device)

    # Freeze discriminator and flow (keep text_encoder + decoder trainable)
    if hasattr(model, "discriminator"):
        _freeze_module(model.discriminator)
    if hasattr(model, "flow"):
        _freeze_module(model.flow)
    if hasattr(model, "posterior_encoder"):
        _freeze_module(model.posterior_encoder)
    use_grad_ckpt = training_cfg.get("gradient_checkpointing", False)
    if use_grad_ckpt and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    total_params, trainable_params = _count_params(model)
    pct = 100 * trainable_params / max(total_params, 1)
    log.info("params total=%d trainable=%d (%.1f%%)", total_params, trainable_params, pct)

    # Data
    all_records = load_manifest(data_cfg["manifest"])
    train_split = data_cfg.get("train_split", "train")
    val_split = data_cfg.get("val_split", "validation")
    train_records = [r for r in all_records if r.split == train_split]
    val_records = [r for r in all_records if r.split == val_split]
    if not train_records:
        raise SystemExit("no training records found in manifest")

    target_sr = cfg.get("sample_rate", 16000)
    train_ds = TTSDataset(train_records, data_cfg["audio_root"], tokenizer, target_sr)
    train_loader = DataLoader(
        train_ds,
        batch_size=training_cfg.get("batch_size", 4),
        shuffle=True,
        num_workers=training_cfg.get("num_workers", 0),
        collate_fn=_collate_fn,
        drop_last=True,
    )

    val_ds = (
        TTSDataset(val_records, data_cfg["audio_root"], tokenizer, target_sr)
        if val_records
        else None
    )
    val_loader = (
        DataLoader(val_ds, batch_size=training_cfg.get("batch_size", 4), collate_fn=_collate_fn)
        if val_ds
        else None
    )

    # Optimizer
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(training_cfg.get("learning_rate", 2e-5)),
        weight_decay=float(training_cfg.get("weight_decay", 0.01)),
    )

    epochs = int(training_cfg.get("epochs", 20))
    grad_accum = int(training_cfg.get("gradient_accumulation_steps", 1))
    max_grad_norm = float(training_cfg.get("max_grad_norm", 1.0))
    save_every = int(training_cfg.get("save_every_steps", 500))
    eval_every = int(training_cfg.get("eval_every_steps", 500))
    log_every = int(training_cfg.get("log_every_steps", 20))
    warmup_ratio = float(training_cfg.get("warmup_ratio", 0.05))

    total_steps = (len(train_loader) // grad_accum) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scaler = torch.amp.GradScaler(enabled=use_amp)

    # Warmup + cosine decay scheduler
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return max(step / max(warmup_steps, 1), 1e-6)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        import math

        return max(0.5 * (1 + math.cos(math.pi * progress)), 1e-6)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Resume
    global_step = 0
    start_epoch = 0
    resume_path = training_cfg.get("resume_from_checkpoint")
    if resume_path and Path(resume_path).is_dir():
        ckpt_dir = Path(resume_path)
        ckpt = torch.load(ckpt_dir / "trainer_state.pt", map_location=device, weights_only=False)
        model.load_state_dict(
            torch.load(ckpt_dir / "model.pt", map_location=device, weights_only=False)
        )
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        global_step = ckpt["global_step"]
        start_epoch = ckpt["epoch"]
        log.info("resumed from checkpoint: step=%d epoch=%d", global_step, start_epoch)

    # Mel loss
    mel_loss_fn = torch.nn.L1Loss()

    # Training log
    train_log: list[dict] = []

    def _save_checkpoint(epoch: int, step: int, tag: str = "") -> Path:
        ckpt_name = f"checkpoint-{step}" if not tag else f"checkpoint-{tag}"
        ckpt_dir = output_dir / ckpt_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt_dir / "model.pt")
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "global_step": step,
                "epoch": epoch,
            },
            ckpt_dir / "trainer_state.pt",
        )
        log.info("saved checkpoint: %s", ckpt_dir)
        return ckpt_dir

    def _compute_mel_from_model(batch: dict) -> torch.Tensor:
        """Run the model and extract output waveform, then compute mel."""
        from ulm.tts.preprocess import compute_mel_spectrogram

        output = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        )
        # VitsModel output has .waveform [batch, 1, time]
        waveform = output.waveform  # [batch, 1, time]
        # Compute mel from predicted waveform
        mels = []
        for i in range(waveform.shape[0]):
            mel = compute_mel_spectrogram(
                waveform[i], target_sr, n_fft=1024, hop_length=256, n_mels=80
            )
            mels.append(mel)
        return torch.stack(mels).to(device)

    log.info(
        "starting training: epochs=%d steps=%d batch=%d accum=%d",
        epochs,
        total_steps,
        training_cfg.get("batch_size", 4),
        grad_accum,
    )

    live = None
    events: list[str] = []
    loss_history: list[float] = []
    t_start = time.time()
    t_last_step = time.time()
    speed_sec_per_step: float | None = None
    last_train_loss: float | None = None
    last_eval_loss: float | None = None
    last_lr: float | None = float(training_cfg.get("learning_rate", 2e-5))

    if enable_dashboard:
        try:
            from rich.console import Console
            from rich.live import Live

            from ulm.utils.dashboard import build_dashboard_layout

            console = Console()
            ts = time.strftime("%H:%M:%S")
            events.append(f"[{ts}] TTS training started. Target steps: {total_steps}")
            layout = build_dashboard_layout(
                step=0,
                max_steps=total_steps,
                epoch=0.0,
                max_epochs=float(epochs),
                loss=None,
                eval_loss=None,
                learning_rate=last_lr,
                loss_history=[],
                speed_sec_per_step=None,
                elapsed_sec=0.0,
                events=events,
                model_name=cfg.get("model_id", "facebook/mms-tts-kor"),
            )
            live = Live(layout, console=console, refresh_per_second=2)
            live.start()
        except Exception:
            live = None

    model.train()
    for epoch in range(start_epoch, epochs):
        epoch_loss = 0.0
        epoch_steps = 0
        for batch_idx, batch in enumerate(train_loader):
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                pred_mel = _compute_mel_from_model(batch)
                target_mel = batch["mel_targets"].to(device)
                # Align lengths (truncate to shorter)
                min_len = min(pred_mel.shape[-1], target_mel.shape[-1])
                loss = mel_loss_fn(pred_mel[:, :, :min_len], target_mel[:, :, :min_len])
                loss = loss / grad_accum

            scaler.scale(loss).backward()

            if (batch_idx + 1) % grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_parameters, max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                epoch_loss += loss.item() * grad_accum
                epoch_steps += 1

                if global_step % log_every == 0:
                    avg = epoch_loss / max(epoch_steps, 1)
                    lr = scheduler.get_last_lr()[0]
                    last_train_loss = avg
                    last_lr = lr
                    now = time.time()
                    speed_sec_per_step = (now - t_last_step) / max(log_every, 1)
                    t_last_step = now
                    loss_history.append(round(avg, 4))
                    if len(loss_history) > 30:
                        loss_history.pop(0)

                    log.info(
                        "step=%d epoch=%d/%d loss=%.4f lr=%.2e",
                        global_step,
                        epoch + 1,
                        epochs,
                        avg,
                        lr,
                    )
                    train_log.append(
                        {
                            "step": global_step,
                            "epoch": epoch + 1,
                            "loss": round(avg, 6),
                            "lr": lr,
                        }
                    )
                    if live:
                        cur_epoch = epoch + (batch_idx + 1) / max(len(train_loader), 1)
                        layout = build_dashboard_layout(
                            step=global_step,
                            max_steps=total_steps,
                            epoch=cur_epoch,
                            max_epochs=float(epochs),
                            loss=last_train_loss,
                            eval_loss=last_eval_loss,
                            learning_rate=last_lr,
                            loss_history=loss_history,
                            speed_sec_per_step=speed_sec_per_step,
                            elapsed_sec=now - t_start,
                            events=events,
                            model_name=cfg.get("model_id", "facebook/mms-tts-kor"),
                        )
                        live.update(layout)

                if global_step % save_every == 0:
                    _save_checkpoint(epoch, global_step)
                    if live:
                        ts = time.strftime("%H:%M:%S")
                        l_str = f"{last_train_loss:.4f}" if last_train_loss else "N/A"
                        events.append(f"[{ts}] Checkpoint step {global_step} saved (loss: {l_str})")

                if global_step % eval_every == 0 and val_loader:
                    model.eval()
                    val_loss_sum = 0.0
                    val_count = 0
                    with torch.inference_mode():
                        for vbatch in val_loader:
                            pred = _compute_mel_from_model(vbatch)
                            tgt = vbatch["mel_targets"].to(device)
                            ml = min(pred.shape[-1], tgt.shape[-1])
                            val_loss_sum += mel_loss_fn(pred[:, :, :ml], tgt[:, :, :ml]).item()
                            val_count += 1
                    val_avg = val_loss_sum / max(val_count, 1)
                    last_eval_loss = val_avg
                    log.info("eval step=%d val_loss=%.4f", global_step, val_avg)
                    train_log.append(
                        {
                            "step": global_step,
                            "eval_loss": round(val_avg, 6),
                        }
                    )
                    if live:
                        ts = time.strftime("%H:%M:%S")
                        events.append(f"[{ts}] Eval step {global_step}: val_loss={val_avg:.4f}")
                    model.train()

    # Final save
    final_dir = _save_checkpoint(epochs, global_step, tag="final")

    if live:
        ts = time.strftime("%H:%M:%S")
        events.append(f"[{ts}] Training completed. Total steps: {global_step}")
        layout = build_dashboard_layout(
            step=global_step,
            max_steps=total_steps,
            epoch=float(epochs),
            max_epochs=float(epochs),
            loss=last_train_loss,
            eval_loss=last_eval_loss,
            learning_rate=last_lr,
            loss_history=loss_history,
            speed_sec_per_step=speed_sec_per_step,
            elapsed_sec=time.time() - t_start,
            events=events,
            model_name=cfg.get("model_id", "facebook/mms-tts-kor"),
        )
        live.update(layout)
        live.stop()

    # Save training log
    (output_dir / "train_log.json").write_text(json.dumps(train_log, indent=2), encoding="utf-8")
    log.info("training complete: %d steps, output=%s", global_step, output_dir)
    return final_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="ULM-TTS VITS fine-tuning trainer")
    parser.add_argument("--config", default="configs/tts/mms_vits.yaml")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--dashboard", action="store_true", help="Enable terminal Rich live dashboard"
    )
    args = parser.parse_args(argv)

    cfg = _read_config(args.config)
    data_cfg = cfg["data"]

    # Validate data
    records = load_manifest(data_cfg["manifest"])
    errors = validate_audio_files(
        records,
        data_cfg["audio_root"],
        data_cfg.get("min_duration", 0.4),
        data_cfg.get("max_duration", 15.0),
    )
    if errors:
        for err in errors[:20]:
            log.error(err)
        raise SystemExit(f"TTS data validation failed: {len(errors)} error(s)")

    if data_cfg.get("require_consent", False):
        validate_consent(records)

    output_dir = Path(cfg["output_dir"])
    _save_resolved_config(cfg, output_dir)

    if args.validate_only:
        train_count = sum(1 for r in records if r.split == data_cfg.get("train_split", "train"))
        val_count = sum(1 for r in records if r.split == data_cfg.get("val_split", "validation"))
        print(f"validated {len(records)} TTS records (train={train_count}, val={val_count})")
        return 0

    # Actual training
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise SystemExit("Install TTS dependencies: pip install -e '.[tts]'") from exc

    run_training(cfg, enable_dashboard=args.dashboard)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
