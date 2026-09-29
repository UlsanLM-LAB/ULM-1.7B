"""Run sequential UlsanBench comparisons from an explicit, portable model config."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ulm.evaluation.comparison import (  # noqa: E402
    DTYPES,
    ENCODER,
    asset_identity,
    build_request,
    hardware_info,
    load_benchmark,
    load_models_config,
    reusable_summary,
    software_versions,
    utc_timestamp,
)

SCRIPT_RUNNER = ROOT / "scripts/run_model_comparison.py"
OUTPUT_DIR = ROOT / "outputs/model-comparison"


def save(path: Path, value: dict) -> None:
    """Replace each result atomically so interruption cannot leave a partial JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
            stream.flush()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def run_benchmark_for_model(
    cfg: dict,
    *,
    output_dir: Path,
    dataset: Path,
    device: str,
    dtype: str,
    batch_size: int,
    request: dict,
) -> dict:
    model_out = output_dir / cfg["id"]
    cmd = [
        sys.executable,
        str(SCRIPT_RUNNER),
        "--model",
        cfg["model"],
        "--name",
        cfg["id"],
        "--output-dir",
        str(model_out),
        "--raw-dir",
        str(output_dir / "raw"),
        "--dataset",
        str(dataset),
        "--batch-size",
        str(batch_size),
        "--decoding-mode",
        cfg["decoding_mode"],
        "--dtype",
        dtype,
        "--device",
        device,
    ]
    if cfg.get("adapter"):
        cmd.extend(["--adapter", cfg["adapter"]])
    if cfg.get("revision"):
        cmd.extend(["--revision", cfg["revision"]])
    if cfg.get("adapter_revision"):
        cmd.extend(["--adapter-revision", cfg["adapter_revision"]])
    print(f"[START] {cfg['display_name']} ({cfg['id']})", flush=True)
    start = time.perf_counter()
    subprocess.run(cmd, text=True, check=True)
    summary_file = model_out / "summary.json"
    if not summary_file.is_file():
        raise RuntimeError(f"Missing summary file: {summary_file}")
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    if not reusable_summary(summary, request):
        raise ValueError(f"New result provenance does not match requested inputs: {summary_file}")
    summary["meta"] = cfg
    summary["display_name"] = cfg["display_name"]
    save(summary_file, summary)
    print(f"[DONE] {cfg['display_name']} in {time.perf_counter() - start:.1f}s", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Sequential UlsanBench v2 comparison")
    parser.add_argument("--dataset", required=True, help="Explicit UlsanBench JSONL input")
    parser.add_argument(
        "--models-config",
        required=True,
        help="JSON list or historical models.json mapping; '-' reads stdin",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=DTYPES)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse only matching provenance; stale/legacy results fail clearly",
    )
    args = parser.parse_args()
    try:
        if args.batch_size < 1:
            raise ValueError("batch-size must be positive")
        models = load_models_config(args.models_config)
        _, dataset = load_benchmark(args.dataset)
        # Hash each unique local asset / resolve each Hub revision once per invocation.
        identities = {}

        def identity(source, revision=None):
            if source is None:
                return None
            key = (source, revision)
            if key not in identities:
                identities[key] = asset_identity(source, revision)
            return identities[key]

        requests = {}
        existing = {}
        for cfg in models:
            assets = {
                "model": identity(cfg["model"], cfg.get("revision")),
                "adapter": identity(cfg.get("adapter"), cfg.get("adapter_revision")),
                "encoder": identity(ENCODER),
            }
            request = build_request(
                ROOT,
                cfg,
                dataset,
                device=args.device,
                dtype=args.dtype,
                batch_size=args.batch_size,
                identities=assets,
            )
            requests[cfg["id"]] = request
            path = args.output_dir / cfg["id"] / "summary.json"
            if args.skip_existing and path.exists():
                summary = json.loads(path.read_text(encoding="utf-8"))
                if not reusable_summary(summary, request, cfg):
                    raise ValueError(
                        f"Cannot reuse {path}: missing/stale provenance or changed model, "
                        "dataset, evaluator or runtime settings. Use a new --output-dir or "
                        "remove --skip-existing to explicitly rerun."
                    )
                existing[cfg["id"]] = summary
    except (ValueError, OSError) as exc:
        parser.error(str(exc))

    output_dir = args.output_dir
    invocation = {
        "benchmark": "UlsanBench v2",
        "dataset": dataset,
        "invocation_started_at": utc_timestamp(),
        "invocation_hardware": hardware_info(args.device),
        "software": software_versions(),
        "runtime": {"device": args.device, "dtype": args.dtype, "batch_size": args.batch_size},
        "note": "Each model summary retains its original execution provenance",
    }
    save(output_dir / "models.json", {cfg["id"]: cfg for cfg in models})
    save(output_dir / "config.json", invocation)
    all_summaries = {}
    for cfg in models:
        if cfg["id"] in existing:
            print(f"Skipping matching result: {cfg['id']}", flush=True)
            summary = existing[cfg["id"]]
        else:
            summary = run_benchmark_for_model(
                cfg,
                output_dir=output_dir,
                dataset=Path(dataset["path"]),
                device=args.device,
                dtype=args.dtype,
                batch_size=args.batch_size,
                request=requests[cfg["id"]],
            )
        all_summaries[cfg["id"]] = summary
        # Preserve completed work even if a later subprocess fails, including skipped runs.
        save(output_dir / "summary.json", all_summaries)
    save(output_dir / "summary.json", all_summaries)
    print(f"All comparisons completed: {output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
