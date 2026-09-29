from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ulm.inference import policy
from ulm.inference.server import ChatCompletionRequest

ROOT = Path(__file__).resolve().parents[1]
RELEASE = json.loads((ROOT / "configs/release/ulm4b-v1.0.json").read_text())


def test_v1_preserves_original_arm_b_training_inference_and_evaluation():
    reference = RELEASE["legacy_release_config"]
    original = ROOT / reference["path"]
    assert hashlib.sha256(original.read_bytes()).hexdigest() == reference["sha256"]
    candidate = json.loads(original.read_text())
    for field in ("base_model", "lineage", "training", "inference", "evaluation"):
        assert RELEASE[field] == candidate[field]
    assert RELEASE["weights_frozen"] is True
    assert RELEASE["further_training_planned"] is False
    assert RELEASE["checkpoint_lineage_name"] == "Arm B"
    assert RELEASE["adapter"]["sha256"] in (ROOT / "MODEL_CARD.md").read_text()


def test_v1_preserves_historical_benchmark_snapshot():
    snapshot = RELEASE["benchmark_snapshot"]
    original = ROOT / snapshot["path"]
    assert hashlib.sha256(original.read_bytes()).hexdigest() == snapshot["sha256"]
    models = json.loads(original.read_text())
    assert models["ulm-4b-arm-b-neutral"]["n"] == snapshot["n"] == 500
    assert snapshot["historical_padding_inclusive_throughput_valid"] is False


def test_v1_serving_defaults_match_frozen_release_policy():
    request = ChatCompletionRequest(messages=[{"role": "user", "content": "안녕"}])
    serving = RELEASE["serving"]
    assert request.temperature == serving["default_temperature"] == policy.DEFAULT_TEMPERATURE
    assert request.top_p == serving["default_top_p"] == policy.DEFAULT_TOP_P
    assert request.max_tokens == serving["default_max_new_tokens"] == policy.DEFAULT_MAX_NEW_TOKENS
    assert request.dialect_strength == serving["default_dialect_strength"] == 2
    assert serving["default_top_k"] == policy.DEFAULT_TOP_K
    assert RELEASE["inference"]["repetition_penalty"] == policy.REPETITION_PENALTY
    assert RELEASE["inference"]["no_repeat_ngram_size"] == policy.NO_REPEAT_NGRAM_SIZE
