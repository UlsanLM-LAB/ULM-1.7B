from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from ulm.inference import chat, cli
from ulm.inference.prompt import PHASE4_SYSTEM_PROMPT, build_dialect_instruction


@pytest.mark.parametrize("path_kind", ["missing", "file", "empty"])
@pytest.mark.parametrize("entrypoint", ["chat", "infer"])
def test_invalid_adapter_is_rejected_before_model_loading(
    monkeypatch, tmp_path, path_kind, entrypoint
):
    path = tmp_path / "adapter"
    if path_kind == "file":
        path.write_text("not a directory", encoding="utf-8")
    if path_kind == "empty":
        path = ""
    calls = []
    monkeypatch.setattr(chat, "run_chat", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(cli, "generate_text", lambda *a, **kwargs: calls.append(kwargs))
    module = chat if entrypoint == "chat" else cli
    args = [] if entrypoint == "chat" else ["--model-name", "unused", "--text", "안녕"]
    with pytest.raises(SystemExit) as exc:
        module.main([*args, "--adapter", str(path)])
    assert exc.value.code == 2
    assert calls == []


@pytest.mark.parametrize("path_kind", ["missing", "file", "empty"])
def test_direct_chat_rejects_invalid_adapter(tmp_path, path_kind):
    path = tmp_path / "adapter"
    if path_kind == "file":
        path.write_text("not a directory", encoding="utf-8")
    if path_kind == "empty":
        path = ""
    with pytest.raises(ValueError, match="Adapter directory"):
        chat.run_chat(adapter_path=path)


@pytest.mark.parametrize("use_adapter", [False, True])
@pytest.mark.parametrize("entrypoint", ["chat", "infer"])
def test_chat_cli_preserves_omitted_or_valid_adapter(
    monkeypatch, tmp_path, use_adapter, entrypoint
):
    calls = []
    monkeypatch.setattr(chat, "run_chat", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(cli, "generate_text", lambda *a, **kwargs: calls.append(kwargs))
    args = ["--adapter", str(tmp_path)] if use_adapter else []
    module = chat if entrypoint == "chat" else cli
    if entrypoint == "infer":
        args.extend(["--model-name", "unused", "--text", "안녕"])
    assert module.main(args) == 0
    assert calls[0]["adapter_path"] == (str(tmp_path) if use_adapter else None)


@pytest.mark.parametrize("strength", [None, 0, 1, 2, 3])
def test_cli_uses_shared_prompt_without_accumulating_style(monkeypatch, strength):
    import transformers

    recorded_messages = []

    class Tokenizer:
        eos_token_id = 9

        def apply_chat_template(self, messages, **kwargs):
            recorded_messages.append(deepcopy(messages))
            return "formatted"

        def __call__(self, prompt, return_tensors):
            return {"input_ids": torch.tensor([[1]])}

        def decode(self, tokens, **kwargs):
            return "답변"

    class Model:
        device = "cpu"

        def generate(self, **kwargs):
            return torch.tensor([[1, 9]])

    monkeypatch.setattr(
        transformers, "AutoTokenizer", SimpleNamespace(from_pretrained=lambda *a, **k: Tokenizer())
    )
    monkeypatch.setattr(
        transformers, "AutoModelForCausalLM",
        SimpleNamespace(from_pretrained=lambda *a, **k: Model()),
    )
    turns = iter([f"질문 {index}" for index in range(12)] + ["q"])
    monkeypatch.setattr("builtins.input", lambda _: next(turns))
    chat.run_chat(model_name="cpu-stub", dialect_strength=strength)
    expected_strength = 2 if strength is None else strength
    instruction = build_dialect_instruction(expected_strength)
    expected_system = PHASE4_SYSTEM_PROMPT
    if instruction:
        expected_system += "\n\n[기본 응답 말투 설정]\n" + instruction
    assert len(recorded_messages) == 12
    for messages in recorded_messages:
        assert messages[0] == {"role": "system", "content": expected_system}
        assert sum(message["role"] == "system" for message in messages) == 1
        expected_marker_count = 0 if expected_strength == 2 else 1
        assert messages[0]["content"].count("[기본 응답 말투 설정]") == expected_marker_count
    assert recorded_messages[1][1:] == [
        {"role": "user", "content": "질문 0"},
        {"role": "assistant", "content": "답변"},
        {"role": "user", "content": "질문 1"},
    ]
    assert len(recorded_messages[-1]) == 22  # one system, ten previous turns, current user
    assert recorded_messages[-1][1] == {"role": "user", "content": "질문 1"}
    if strength == 0:
        assert "표준 한국어로 답하고" in expected_system
