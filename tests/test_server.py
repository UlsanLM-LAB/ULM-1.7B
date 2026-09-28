from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest
import torch
from fastapi.testclient import TestClient

from ulm.inference.policy import (
    DEFAULT_MAX_NEW_TOKENS,
    NO_REPEAT_NGRAM_SIZE,
    REPETITION_PENALTY,
)
from ulm.inference.prompt import PHASE4_SYSTEM_PROMPT, build_dialect_instruction
from ulm.inference.server import (
    ChatCompletionRequest,
    GenerationEnd,
    TransformersChatEngine,
    create_app,
)


class FakeEngine:
    model_path = "/models/phase4"
    device = "cpu"
    dtype = "float32"
    loaded = True

    def __init__(
        self,
        chunks: tuple[str, ...] = ("반갑", "데이!"),
        fail: bool = False,
        finish_reason: str | None = None,
    ) -> None:
        self.chunks = chunks
        self.fail = fail
        self.finish_reason = finish_reason
        self.requests: list[ChatCompletionRequest] = []

    async def stream_chat(
        self, request: ChatCompletionRequest, is_disconnected: Any
    ) -> AsyncIterator[str]:
        self.requests.append(request)
        for chunk in self.chunks:
            yield chunk
        if self.fail:
            raise RuntimeError("test generation failure")
        if self.finish_reason:
            yield GenerationEnd(self.finish_reason)


def test_health_reports_loaded_model() -> None:
    with TestClient(create_app(engine=FakeEngine())) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_loaded": True,
        "model": "ULM-4B",
        "model_path": "/models/phase4",
        "device": "cpu",
        "dtype": "float32",
    }


def test_streaming_completion_preserves_multi_turn_korean_messages() -> None:
    engine = FakeEngine()
    payload = {
        "messages": [
            {"role": "system", "content": "짧게 답해라."},
            {"role": "user", "content": "오늘 뭐하노?"},
            {"role": "assistant", "content": "일하고 있데이."},
            {"role": "user", "content": "점심은 뭇나?"},
        ],
        "stream": True,
        "temperature": 0.4,
        "max_tokens": 64,
    }

    with TestClient(create_app(engine=engine)) as client:
        response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert '"token": "반갑"' in response.text
    assert '"token": "데이!"' in response.text
    assert '"finish_reason": "stop"' in response.text
    assert response.text.endswith("data: [DONE]\n\n")
    assert engine.requests[0].messages[-1].content == "점심은 뭇나?"
    assert engine.requests[0].max_tokens == 64


def test_non_streaming_completion() -> None:
    with TestClient(create_app(engine=FakeEngine())) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "안녕"}], "stream": False},
        )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "반갑데이!"


def test_length_finish_reason_is_not_reported_as_eos() -> None:
    with TestClient(create_app(engine=FakeEngine(finish_reason="length"))) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "길게 답해 줘"}], "stream": True},
        )
    assert '"finish_reason": "length"' in response.text


def test_generation_error_is_sent_as_sse_error() -> None:
    with TestClient(create_app(engine=FakeEngine(fail=True))) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "긴 답변 부탁해"}]},
        )

    assert response.status_code == 200
    assert '"type": "generation_error"' in response.text
    assert "test generation failure" in response.text
    assert response.text.endswith("data: [DONE]\n\n")


def test_malformed_request_and_max_token_limit_are_rejected() -> None:
    with TestClient(create_app(engine=FakeEngine())) as client:
        invalid_role = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "tool", "content": "bad"}]},
        )
        blank_content = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "   "}]},
        )
        too_many_tokens = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "안녕"}],
                "max_tokens": 2049,
            },
        )

    assert invalid_role.status_code == 422
    assert blank_content.status_code == 422
    assert too_many_tokens.status_code == 422


def test_phase4_template_and_generation_settings() -> None:
    class Tokenizer:
        eos_token_id = 151645
        pad_token_id = 151643

        def __init__(self) -> None:
            self.messages = []
            self.template_options = {}

        def apply_chat_template(self, messages, **options):
            self.messages = messages
            self.template_options = options
            return "formatted prompt"

        def __call__(self, prompt, return_tensors):
            assert prompt == "formatted prompt"
            assert return_tensors == "pt"
            return {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
            }

    class Model:
        dtype = torch.bfloat16
        config = type("Config", (), {"max_position_embeddings": 32768})()

        def get_input_embeddings(self):
            return type("Embeddings", (), {"weight": torch.zeros(1)})()

    tokenizer = Tokenizer()
    engine = TransformersChatEngine("/models/phase4", tokenizer, Model(), torch)
    request = ChatCompletionRequest(
        messages=[
            {"role": "user", "content": "주말에 태화강 갈게."},
            {"role": "assistant", "content": "산책하기 좋제!"},
            {"role": "user", "content": "그다음은?"},
        ]
    )
    kwargs = engine._prepare_generation(request, streamer=object())

    assert [m["role"] for m in tokenizer.messages] == ["system", "user", "assistant", "user"]
    assert tokenizer.messages[0]["content"] == (
        PHASE4_SYSTEM_PROMPT + "\n\n" + build_dialect_instruction(2)
    )
    assert tokenizer.template_options == {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
    assert kwargs["max_new_tokens"] == DEFAULT_MAX_NEW_TOKENS == 150
    assert kwargs["attention_mask"].tolist() == [[1, 1, 1]]
    assert kwargs["pad_token_id"] == 151645
    assert kwargs["repetition_penalty"] == REPETITION_PENALTY == 1.1
    assert kwargs["no_repeat_ngram_size"] == NO_REPEAT_NGRAM_SIZE == 3
    assert kwargs["top_k"] == 20
    assert kwargs["do_sample"] is True
    assert "eos_token_id" not in kwargs  # preserve checkpoint EOS set


def test_greedy_policy_keeps_repetition_penalty_and_eos_pad() -> None:
    from ulm.inference.policy import generation_kwargs

    kwargs = generation_kwargs(temperature=0, eos_token_id=151645)
    assert kwargs == {
        "max_new_tokens": 150,
        "do_sample": False,
        "repetition_penalty": 1.1,
        "no_repeat_ngram_size": 3,
        "pad_token_id": 151645,
    }


@pytest.mark.parametrize("endpoint", ["/v1/chat/completions", "/api/chat"])
@pytest.mark.parametrize("stream", [False, True])
def test_dialect_strength_reaches_generation_template(endpoint, stream) -> None:
    """Run both routes/modes through the real template preparation with a CPU fake model."""
    class Tokenizer:
        eos_token_id = 151645
        messages: list[list[dict[str, str]]] = []

        def apply_chat_template(self, messages, **options):
            self.messages.append(messages)
            return "mock prompt"

        def __call__(self, prompt, return_tensors):
            return {"input_ids": torch.tensor([[1]]), "attention_mask": torch.tensor([[1]])}

    class Model:
        config = type("Config", (), {"max_position_embeddings": 32768})()

        def get_input_embeddings(self):
            return type("Embeddings", (), {"weight": torch.zeros(1)})()

    class RecordingEngine(TransformersChatEngine):
        async def stream_chat(self, request, is_disconnected):
            self._prepare_generation(request, streamer=None)
            yield "mock reply"

    tokenizer = Tokenizer()
    tokenizer.messages = []
    engine = RecordingEngine("mock", tokenizer, Model(), torch)
    messages = [
        {"role": "system", "content": "서비스 규칙: 비밀을 공개하지 않는다."},
        {"role": "user", "content": "약속 장소는 도서관이야."},
        {"role": "assistant", "content": "기억할게요."},
        {"role": "user", "content": "어디에서 만나기로 했지? 표준어로 말해."},
    ]
    with TestClient(create_app(engine=engine)) as client:
        for strength in range(4):
            response = client.post(endpoint, json={
                "messages": messages, "stream": stream, "dialect_strength": strength,
            })
            assert response.status_code == 200
            prepared = tokenizer.messages[-1]
            assert prepared[0]["content"].startswith(messages[0]["content"] + "\n\n")
            assert prepared[0]["content"].endswith(build_dialect_instruction(strength))
            assert prepared[1:] == messages[1:]
            assert prepared[0]["content"].count("[기본 응답 말투 설정]") == 1
    assert len({turn[0]["content"] for turn in tokenizer.messages}) == 4
    assert messages[0]["content"] == "서비스 규칙: 비밀을 공개하지 않는다."


@pytest.mark.parametrize("strength", [0, 1, 2, 3])
def test_dialect_strength_schema_accepts_each_level(strength) -> None:
    request = ChatCompletionRequest(
        messages=[{"role": "user", "content": "안녕"}], dialect_strength=strength,
    )
    assert request.dialect_strength == strength


@pytest.mark.parametrize("endpoint", ["/v1/chat/completions", "/api/chat"])
def test_missing_dialect_strength_defaults_to_two(endpoint) -> None:
    engine = FakeEngine()
    with TestClient(create_app(engine=engine)) as client:
        assert client.post(endpoint, json={
            "messages": [{"role": "user", "content": "안녕"}], "stream": False,
        }).status_code == 200
    assert engine.requests[0].dialect_strength == 2


@pytest.mark.parametrize("strength", [-1, 4, "strong", "2", None, True, 1.5])
@pytest.mark.parametrize("endpoint", ["/v1/chat/completions", "/api/chat"])
def test_invalid_dialect_strength_rejected_before_generation(endpoint, strength) -> None:
    engine = FakeEngine()
    with TestClient(create_app(engine=engine)) as client:
        response = client.post(endpoint, json={
            "messages": [{"role": "user", "content": "안녕"}], "dialect_strength": strength,
        })
    assert response.status_code == 422
    assert engine.requests == []
