from __future__ import annotations

import io
import json
import sys
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from scripts import run_all_comparisons as all_runner
from scripts import run_model_comparison as runner

from ulm.evaluation import comparison


@pytest.mark.parametrize(
    "tokens,eos,pad,expected",
    [
        ([8, 2, 0], 2, 0, 2),
        ([8, 9, 2], 2, 0, 3),
        ([8, 2, 2], 2, 2, 2),
        ([8, 9, 9], 2, 2, 3),
        ([8, 9, 0], 2, 0, 3),
        ([8, 3, 2], [2, 3], 0, 2),
        ([8, 9, 0], None, 0, 3),
    ],
)
def test_token_count_excludes_post_eos_padding(tokens, eos, pad, expected):
    assert runner.generated_token_count(torch.tensor(tokens), eos) == expected


@pytest.mark.parametrize("pad", [0, 2])
def test_actual_cpu_batch_inference_counts_unequal_lengths(monkeypatch, pad):
    class Batch(dict):
        input_ids = torch.tensor([[1, 1], [1, 1]])

        def __init__(self):
            super().__init__(input_ids=self.input_ids)

        def to(self, device):
            assert device == "cpu"
            return self

    tokenizer = SimpleNamespace(
        pad_token_id=pad,
        eos_token_id=2,
        padding_side=None,
        apply_chat_template=lambda *a, **kw: "prompt",
        decode=lambda ids, **kw: "reply",
    )

    class Tokenizer:
        def __call__(self, *args, **kwargs):
            return Batch()

        def __getattr__(self, name):
            return getattr(tokenizer, name)

    model = SimpleNamespace(
        eval=lambda: model,
        generate=lambda **kw: torch.tensor([[1, 1, 8, 2, pad], [1, 1, 8, 9, 2]]),
    )
    monkeypatch.setattr(runner.AutoTokenizer, "from_pretrained", lambda *a, **kw: Tokenizer())
    monkeypatch.setattr(runner.AutoModelForCausalLM, "from_pretrained", lambda *a, **kw: model)

    def forbidden_cuda(*a, **kw):
        raise AssertionError("CPU execution touched CUDA")

    monkeypatch.setattr(torch.cuda, "empty_cache", forbidden_cuda)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", forbidden_cuda)
    outputs, stats = runner.run_model_inference(
        "stub", None, [{"task": "comprehension", "prompt": "a"}] * 2, device="cpu"
    )
    assert outputs == ["reply", "reply"]
    assert stats["total_tokens_generated"] == 5
    assert stats["peak_vram_gib"] is None
    assert "first EOS inclusive" in stats["token_count_definition"]


def row(task="comprehension", item_id="one"):
    return {
        "id": item_id,
        "task": task,
        "prompt": "질문",
        "reference": "응답",
        "standard_reference": "응답",
    }


def write_dataset(path, rows=None):
    rows = [row()] if rows is None else rows
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "invalid", [[], [row(task="dialect_understanding")], [row(), row()], [{**row(), "prompt": 123}]]
)
def test_benchmark_boundary_rejects_invalid_or_cross_schema_input(tmp_path, invalid):
    with pytest.raises(ValueError):
        comparison.load_benchmark(write_dataset(tmp_path / "data.jsonl", invalid))


def test_dataset_identity_records_real_checksum_and_counts(tmp_path):
    path = write_dataset(tmp_path / "data.jsonl", [row(), row("grammar", "two")])
    rows, identity = comparison.load_benchmark(path)
    assert len(rows) == identity["total_items"] == 2
    assert identity["task_counts"] == {"comprehension": 1, "grammar": 1}
    assert identity["sha256"] == comparison.file_sha256(path)
    before = identity["sha256"]
    path.write_text(path.read_text() + "\n", encoding="utf-8")
    assert comparison.load_benchmark(path)[1]["sha256"] != before


@pytest.mark.parametrize(
    "cfg",
    [
        [],
        [{"id": "../escape", "model": "base"}],
        [{"id": "summary.json", "model": "base"}],
        [{"id": "raw", "model": "base"}],
        [{"id": "ok", "model": "base"}] * 2,
        [{"id": "ok", "model": "base", "adapter": 3}],
        [{"id": "ok", "model": "base", "decoding_mode": "other"}],
    ],
)
def test_models_config_boundary_validation(monkeypatch, cfg):
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(cfg)))
    with pytest.raises(ValueError):
        comparison.load_models_config("-")


def test_historical_model_mapping_preserves_metadata(tmp_path):
    path = tmp_path / "models.json"
    path.write_text(json.dumps({"ok": {"model": "base", "params": "4B", "type": "specialized"}}))
    cfg = comparison.load_models_config(str(path))[0]
    assert cfg["id"] == "ok"
    assert cfg["params"] == "4B"
    assert cfg["type"] == "specialized"
    assert cfg["adapter"] is None
    assert cfg["decoding_mode"] == "neutral"


def test_local_asset_identity_detects_content_changes(tmp_path):
    path = tmp_path / "model"
    path.mkdir()
    weight = path / "weights.bin"
    weight.write_bytes(b"first")
    before = comparison.asset_identity(str(path))
    weight.write_bytes(b"other")
    after = comparison.asset_identity(str(path))
    assert before["sha256"] != after["sha256"]
    assert after["file_count"] == 1


def test_hub_asset_identity_resolves_commit_without_downloading(monkeypatch):
    import huggingface_hub

    calls = []

    def info(source, revision=None):
        calls.append((source, revision))
        return SimpleNamespace(sha="immutable-sha")

    monkeypatch.setattr(huggingface_hub, "model_info", info)
    result = comparison.asset_identity("org/model", "release")
    assert result["resolved_commit"] == "immutable-sha"
    assert calls == [("org/model", "release")]


@pytest.fixture
def configured_run(tmp_path, monkeypatch):
    dataset = write_dataset(tmp_path / "benchmark.jsonl")
    models = [
        {
            "id": "first",
            "display_name": "First",
            "model": "org/model",
            "adapter": None,
            "decoding_mode": "neutral",
        },
        {
            "id": "second",
            "display_name": "Second",
            "model": "org/model",
            "adapter": None,
            "decoding_mode": "context_guard",
        },
    ]
    config = tmp_path / "models.json"
    config.write_text(json.dumps(models), encoding="utf-8")
    output = tmp_path / "results"
    calls = []

    def asset(source, revision=None):
        calls.append((source, revision))
        return {
            "source": source,
            "kind": "hub",
            "requested_revision": revision,
            "resolved_commit": "immutable-sha",
        }

    monkeypatch.setattr(all_runner, "asset_identity", asset)
    monkeypatch.setattr(all_runner, "hardware_info", lambda device: {"device": device, "gpu": None})
    _, identity = comparison.load_benchmark(dataset)
    assets = {"model": asset("org/model"), "adapter": None, "encoder": asset(comparison.ENCODER)}
    requests = {
        cfg["id"]: comparison.build_request(
            all_runner.ROOT,
            cfg,
            identity,
            device="cpu",
            dtype="float32",
            batch_size=2,
            identities=assets,
        )
        for cfg in models
    }
    calls.clear()

    def summary(cfg):
        return {
            "n": 1,
            "metrics": {"comprehension": {"n": 1}},
            "meta": cfg,
            "provenance": {
                "version": 1,
                "request": requests[cfg["id"]],
                "started_at": "original-start",
                "completed_at": "original-end",
                "hardware": {"gpu": "Original GPU"},
            },
        }

    argv = [
        "run_all",
        "--dataset",
        str(dataset),
        "--models-config",
        str(config),
        "--output-dir",
        str(output),
        "--device",
        "cpu",
        "--dtype",
        "float32",
        "--batch-size",
        "2",
    ]
    return SimpleNamespace(
        dataset=dataset,
        models=models,
        config=config,
        output=output,
        summary=summary,
        argv=argv,
        requests=requests,
        asset_calls=calls,
    )


def test_all_skipped_always_creates_combined_summary_and_preserves_provenance(
    configured_run, monkeypatch
):
    cfg = configured_run
    originals = {}
    for model in cfg.models:
        value = cfg.summary(model)
        originals[model["id"]] = value
        all_runner.save(cfg.output / model["id"] / "summary.json", value)
    monkeypatch.setattr(sys, "argv", cfg.argv + ["--skip-existing"])
    monkeypatch.setattr(all_runner.subprocess, "run", lambda *a, **kw: pytest.fail("GPU rerun"))
    all_runner.main()
    assert json.loads((cfg.output / "summary.json").read_text()) == originals
    assert len(cfg.asset_calls) == 2  # shared model + encoder, each resolved once
    invocation = json.loads((cfg.output / "config.json").read_text())
    assert invocation["dataset"]["total_items"] == 1
    assert "ec2_instance" not in invocation["invocation_hardware"]


@pytest.mark.parametrize(
    "change", ["legacy", "model", "dataset", "runtime", "encoder", "evaluator", "metadata", "null"]
)
def test_skip_rejects_stale_or_unverifiable_results_before_mutation(
    configured_run, monkeypatch, change
):
    cfg = configured_run
    model = cfg.models[0]
    summary = deepcopy(cfg.summary(model))
    request = summary["provenance"]["request"]
    if change == "legacy":
        summary.pop("provenance")
    elif change == "model":
        request["model_config"]["model"] = "wrong"
    elif change == "dataset":
        request["dataset"]["sha256"] = "old"
    elif change == "runtime":
        request["runtime"]["batch_size"] = 99
    elif change == "encoder":
        request["assets"]["encoder"]["resolved_commit"] = "old"
    elif change == "evaluator":
        request["evaluator_sha256"] = {}
    elif change == "metadata":
        summary["meta"]["display_name"] = "Other name"
    elif change == "null":
        summary = None
    all_runner.save(cfg.output / model["id"] / "summary.json", summary)
    monkeypatch.setattr(sys, "argv", cfg.argv + ["--skip-existing"])
    monkeypatch.setattr(all_runner.subprocess, "run", lambda *a, **kw: pytest.fail("GPU rerun"))
    with pytest.raises(SystemExit) as exc:
        all_runner.main()
    assert exc.value.code == 2
    assert not (cfg.output / "models.json").exists()
    assert not (cfg.output / "config.json").exists()


def test_partial_skip_preserves_progress_when_later_model_fails(configured_run, monkeypatch):
    cfg = configured_run
    first = cfg.models[0]
    all_runner.save(cfg.output / first["id"] / "summary.json", cfg.summary(first))
    monkeypatch.setattr(sys, "argv", cfg.argv + ["--skip-existing"])

    def fail(*a, **kw):
        raise RuntimeError("second model failed")

    monkeypatch.setattr(all_runner, "run_benchmark_for_model", fail)
    with pytest.raises(RuntimeError, match="second model failed"):
        all_runner.main()
    result = json.loads((cfg.output / "summary.json").read_text())
    assert result == {first["id"]: cfg.summary(first)}


def test_partial_skip_combines_existing_and_new_results(configured_run, monkeypatch):
    cfg = configured_run
    first = cfg.models[0]
    all_runner.save(cfg.output / first["id"] / "summary.json", cfg.summary(first))
    calls = []

    def run(model, **kwargs):
        calls.append(model["id"])
        return cfg.summary(model)

    monkeypatch.setattr(all_runner, "run_benchmark_for_model", run)
    monkeypatch.setattr(sys, "argv", cfg.argv + ["--skip-existing"])
    all_runner.main()
    result = json.loads((cfg.output / "summary.json").read_text())
    assert result == {model["id"]: cfg.summary(model) for model in cfg.models}
    assert calls == [cfg.models[1]["id"]]


def test_fresh_subprocess_uses_current_python_and_explicit_inputs(configured_run, monkeypatch):
    cfg = configured_run
    cmds = []

    def run(cmd, **kw):
        cmds.append(cmd)
        name = cmd[cmd.index("--name") + 1]
        model = next(m for m in cfg.models if m["id"] == name)
        all_runner.save(cfg.output / name / "summary.json", cfg.summary(model))

    monkeypatch.setattr(all_runner.subprocess, "run", run)
    monkeypatch.setattr(sys, "argv", cfg.argv)
    all_runner.main()
    result = json.loads((cfg.output / "summary.json").read_text())
    assert set(result) == {"first", "second"}
    assert len(cmds) == 2
    for cmd in cmds:
        assert cmd[0] == sys.executable
        assert cmd[cmd.index("--dataset") + 1] == str(cfg.dataset)
        assert cmd[cmd.index("--device") + 1] == "cpu"
        assert cmd[cmd.index("--dtype") + 1] == "float32"
        assert cmd[cmd.index("--raw-dir") + 1] == str(cfg.output / "raw")


def test_single_runner_requires_dataset_before_loading_models(monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["runner", "--model", "m", "--name", "safe", "--output-dir", "unused"]
    )
    with pytest.raises(SystemExit) as exc:
        runner.main()
    assert exc.value.code == 2


@pytest.mark.parametrize("extra", [[], ["--model", "unused"]])
def test_standalone_evaluator_requires_explicit_inputs_before_generation(tmp_path, extra):
    import evaluate_ulsanbench_v2 as evaluator

    with pytest.raises(SystemExit) as exc:
        evaluator.main(["--output", str(tmp_path / "result"), *extra])
    assert exc.value.code == 2
    assert not (tmp_path / "result").exists()


def test_standalone_evaluator_reads_explicit_dataset(tmp_path, monkeypatch):
    import evaluate_ulsanbench_v2 as evaluator

    dataset = write_dataset(tmp_path / "data.jsonl")
    calls = []

    def generate(model, adapter, rows):
        calls.extend(rows)
        return ["응답"]

    monkeypatch.setattr(evaluator, "generate_rows", generate)
    monkeypatch.setattr(evaluator, "score_rows", lambda rows, outputs, *a: outputs)
    assert evaluator.evaluate("unused", None, tmp_path, dataset_path=dataset) == ["응답"]
    assert calls == [row()]


def test_single_runner_pins_revisions_records_provenance_and_keeps_raw_local(tmp_path, monkeypatch):
    dataset = write_dataset(tmp_path / "data.jsonl")
    output = tmp_path / "output"
    identities = {
        "model": {"kind": "hub", "source": "org/model", "resolved_commit": "model-sha"},
        "adapter": None,
        "encoder": {"kind": "hub", "resolved_commit": "encoder-sha"},
    }
    monkeypatch.setattr(
        comparison,
        "asset_identity",
        lambda source, revision=None: (
            identities["encoder"] if source == comparison.ENCODER else identities["model"]
        ),
    )
    monkeypatch.setattr(runner, "hardware_info", lambda device: {"device": device, "gpu": None})
    calls = {}

    def inference(*a, **kw):
        calls["inference"] = kw
        return ["응답"], {"total_tokens_generated": 2}

    def metric(**kw):
        calls["metric"] = kw
        return object()

    def score(rows, outputs, output_dir, *a, **kw):
        output_dir.mkdir(parents=True)
        (output_dir / "predictions.jsonl").write_text("prediction\n", encoding="utf-8")
        return {"n": 1, "config": {}, "metrics": {}}

    monkeypatch.setattr(runner, "run_model_inference", inference)
    monkeypatch.setattr(runner, "Embedder", metric)
    monkeypatch.setattr(runner, "score_rows", score)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runner",
            "--model",
            "org/model",
            "--name",
            "safe",
            "--dataset",
            str(dataset),
            "--output-dir",
            str(output),
            "--device",
            "cpu",
            "--dtype",
            "float32",
        ],
    )
    runner.main()
    result = json.loads((output / "summary.json").read_text())
    assert result["provenance"]["request"]["assets"]["tokenizer"] == identities["model"]
    assert calls["inference"]["assets"]["model"]["resolved_commit"] == "model-sha"
    assert calls["metric"]["revision"] == "encoder-sha"
    assert result["provenance"]["started_at"] <= result["provenance"]["completed_at"]
    assert (output / "raw/safe.jsonl").read_text() == "prediction\n"
