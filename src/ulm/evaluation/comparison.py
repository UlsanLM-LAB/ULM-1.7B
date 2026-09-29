"""Input contracts and reproducibility records for the UlsanBench comparison scripts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

TASKS = ("comprehension", "generation", "identification", "grammar", "context")
ENCODER = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DECODING_MODES = ("neutral", "context_guard")
DTYPES = ("bfloat16", "float16", "float32")


def utc_timestamp() -> str:
    return datetime.now(UTC).isoformat()


def validate_name(name: str) -> str:
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", name):
        raise ValueError(
            "Model id/name must be 1–128 letters, digits, dots, underscores or hyphens"
        )
    return name


def load_models_config(path: str) -> list[dict]:
    """Accept a model list or the id-keyed mapping used by historical models.json."""
    raw = json.loads(sys.stdin.read() if path == "-" else Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        models = []
        for model_id, cfg in raw.items():
            if not isinstance(cfg, dict) or cfg.get("id", model_id) != model_id:
                raise ValueError("Model mapping keys must match each model's id")
            models.append({**cfg, "id": model_id})
    else:
        models = raw
    if not isinstance(models, list) or not models:
        raise ValueError("Models config must contain a nonempty model list or id-keyed mapping")
    result = []
    names = set()
    for cfg in models:
        if not isinstance(cfg, dict):
            raise ValueError("Every model configuration must be an object")
        name = validate_name(cfg.get("id"))
        if name in ("summary.json", "config.json", "models.json", "raw"):
            raise ValueError(f"Model id conflicts with a reserved output path: {name}")
        if name in names:
            raise ValueError(f"Duplicate model id: {name}")
        names.add(name)
        if not isinstance(cfg.get("model"), str) or not cfg["model"].strip():
            raise ValueError(f"Model {name} requires a nonempty model path or Hugging Face id")
        for key in ("adapter", "revision", "adapter_revision"):
            if cfg.get(key) is not None and (not isinstance(cfg[key], str) or not cfg[key].strip()):
                raise ValueError(f"Model {name}: {key} must be a nonempty string or null")
        display_name = cfg.get("display_name", name)
        if not isinstance(display_name, str) or not display_name.strip():
            raise ValueError(f"Model {name}: display_name must be a nonempty string")
        mode = cfg.get("decoding_mode", "neutral")
        if mode not in DECODING_MODES:
            raise ValueError(f"Model {name}: unsupported decoding_mode {mode!r}")
        result.append(
            {
                **cfg,
                "adapter": cfg.get("adapter"),
                "display_name": display_name,
                "decoding_mode": mode,
            }
        )
    return result


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_benchmark(path: str | Path, subset: int = 0) -> tuple[list[dict], dict]:
    if subset < 0:
        raise ValueError("subset must be nonnegative")
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Benchmark dataset does not exist: {path}; supply --dataset explicitly")
    rows = []
    seen = set()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for number, raw_line in enumerate(stream, 1):
            digest.update(raw_line)
            line = raw_line.decode("utf-8")
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid benchmark JSON on line {number}: {exc.msg}") from exc
            if not isinstance(row, dict) or row.get("task") not in TASKS:
                raise ValueError(f"Line {number}: expected UlsanBench task in {TASKS}")
            for key in ("prompt", "reference", "standard_reference"):
                if not isinstance(row.get(key), str) or not row[key].strip():
                    raise ValueError(f"Line {number}: {key} must be a nonempty string")
            if "id" in row:
                if not isinstance(row["id"], str) or not row["id"].strip():
                    raise ValueError(f"Line {number}: id must be a nonempty string")
                if row["id"] in seen:
                    raise ValueError(f"Line {number}: duplicate benchmark id {row['id']!r}")
                seen.add(row["id"])
            if row.get("ending") is not None and not isinstance(row["ending"], str):
                raise ValueError(f"Line {number}: ending must be a string or null")
            if row["task"] == "identification" and row["reference"] not in (
                "ULSAN",
                "OTHER_GYEONGSANG",
                "STANDARD",
            ):
                raise ValueError(f"Line {number}: invalid identification reference")
            rows.append(row)
    if not rows:
        raise ValueError("Benchmark dataset is empty")
    dataset = {
        "path": str(path),
        "sha256": digest.hexdigest(),
        "total_items": len(rows),
        "task_counts": dict(Counter(row["task"] for row in rows)),
    }
    if subset and subset < len(rows):
        by_task = defaultdict(list)
        for row in rows:
            by_task[row["task"]].append(row)
        # Preserve the runner's deterministic stratified sampling policy.
        rows = [
            row
            for task_rows in by_task.values()
            for row in task_rows[: max(1, round(subset * len(task_rows) / len(rows)))]
        ]
    dataset["evaluated_items"] = len(rows)
    dataset["evaluated_task_counts"] = dict(Counter(row["task"] for row in rows))
    return rows, dataset


def asset_identity(source: str, revision: str | None = None) -> dict:
    """Hash local assets or resolve a Hub revision without downloading model weights."""
    path = Path(source).expanduser()
    if path.exists():
        path = path.resolve()
        if not path.is_dir():
            raise ValueError(f"Model/adapter/encoder must be a directory: {path}")
        files = [
            p
            for p in sorted(path.rglob("*"))
            if p.is_file()
            and not any(part in (".git", ".cache") for part in p.relative_to(path).parts)
        ]
        if not files:
            raise ValueError(f"Model/adapter/encoder directory is empty: {path}")
        manifest = [(str(p.relative_to(path)), file_sha256(p)) for p in files]
        digest = hashlib.sha256(json.dumps(manifest, ensure_ascii=False).encode()).hexdigest()
        return {
            "source": source,
            "kind": "local",
            "path": str(path),
            "sha256": digest,
            "file_count": len(files),
        }
    if source.startswith(("/", "./", "../", "~")):
        raise ValueError(f"Local model/adapter/encoder directory does not exist: {source}")
    from huggingface_hub import model_info

    info = model_info(source, revision=revision)
    if not info.sha:
        raise ValueError(f"Could not resolve model revision for {source}")
    return {
        "source": source,
        "kind": "hub",
        "requested_revision": revision,
        "resolved_commit": info.sha,
    }


def evaluator_identity(root: Path) -> dict:
    files = [
        root / "scripts/evaluate_ulsanbench_v2.py",
        root / "scripts/run_model_comparison.py",
        root / "scripts/run_all_comparisons.py",
        Path(__file__).resolve(),
    ]
    return {str(path.relative_to(root)): file_sha256(path) for path in files}


def software_versions() -> dict:
    versions = {"python": platform.python_version()}
    for package in ("torch", "transformers", "peft", "huggingface-hub"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def hardware_info(device: str) -> dict:
    import torch

    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "device": device,
        "cuda_version": torch.version.cuda,
        "gpu": None,
        "driver_version": None,
    }
    if device.startswith("cuda") and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(torch.device(device))
        info["gpu"] = {
            "name": props.name,
            "total_memory_bytes": props.total_memory,
            "compute_capability": [props.major, props.minor],
        }
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
            info["driver_version"] = sorted(set(result.stdout.strip().splitlines()))
        except (OSError, subprocess.SubprocessError):
            pass
    return info


def build_request(
    root: Path,
    cfg: dict,
    dataset: dict,
    *,
    device: str,
    dtype: str,
    batch_size: int,
    subset: int = 0,
    identities: dict | None = None,
) -> dict:
    if batch_size < 1:
        raise ValueError("batch-size must be positive")
    if dtype not in DTYPES or cfg["decoding_mode"] not in DECODING_MODES:
        raise ValueError("Unsupported dtype or decoding mode")
    if identities is None:
        identities = {
            "model": asset_identity(cfg["model"], cfg.get("revision")),
            "adapter": asset_identity(cfg["adapter"], cfg.get("adapter_revision"))
            if cfg.get("adapter")
            else None,
            "encoder": asset_identity(ENCODER),
        }
    identities = {**identities, "tokenizer": identities["model"]}
    execution_config = {
        "id": cfg["id"],
        "model": cfg["model"],
        "adapter": cfg.get("adapter"),
        "revision": cfg.get("revision"),
        "adapter_revision": cfg.get("adapter_revision"),
        "decoding_mode": cfg["decoding_mode"],
    }
    return {
        "model_config": execution_config,
        "dataset": dataset,
        "assets": identities,
        "runtime": {
            "device": device,
            "dtype": dtype,
            "batch_size": batch_size,
            "decoding_mode": cfg["decoding_mode"],
            "subset": subset,
        },
        "evaluator_sha256": evaluator_identity(root),
        "software": software_versions(),
    }


def reusable_summary(summary: dict, request: dict, model_config: dict | None = None) -> bool:
    """Legacy or changed runs must be recomputed, never relabelled as current results."""
    if not isinstance(summary, dict):
        return False
    provenance = summary.get("provenance", {})
    return (
        isinstance(provenance, dict)
        and provenance.get("version") == 1
        and provenance.get("request") == request
        and isinstance(provenance.get("started_at"), str)
        and isinstance(provenance.get("completed_at"), str)
        and summary.get("n") == request["dataset"]["evaluated_items"]
        and (model_config is None or summary.get("meta") == model_config)
    )
