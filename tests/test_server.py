from __future__ import annotations

import asyncio
import queue
import threading
from collections.abc import AsyncIterator
from types import SimpleNamespace
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
    InvalidChatRequest,
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

    def validate_request(self, request: ChatCompletionRequest) -> None:
        pass

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


class CpuTokenizer:
    eos_token_id = 9

    def __init__(self, input_tokens=5):
        self.input_tokens = input_tokens

    def apply_chat_template(self, messages, **kwargs):
        return "formatted prompt"

    def __call__(self, prompt, return_tensors):
        return {"input_ids": torch.ones((1, self.input_tokens), dtype=torch.long)}


class CpuModel:
    config = SimpleNamespace(max_position_embeddings=8)
    generation_config = SimpleNamespace(eos_token_id=9)

    def get_input_embeddings(self):
        return SimpleNamespace(weight=torch.zeros(1))


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("endpoint", ["/api/chat", "/v1/chat/completions"])
@pytest.mark.parametrize("input_tokens, max_tokens", [(5, 4), (1, 9)])
def test_context_overflow_is_rejected_before_headers(endpoint, stream, input_tokens, max_tokens):
    model = CpuModel()
    model.generate = lambda **kwargs: pytest.fail("invalid request must not generate")
    engine = TransformersChatEngine("cpu-stub", CpuTokenizer(input_tokens), model, torch)
    with TestClient(create_app(engine=engine)) as client:
        response = client.post(endpoint, json={
            "messages": [{"role": "user", "content": "안녕"}],
            "max_tokens": max_tokens,
            "stream": stream,
        })
    assert response.status_code == 400
    assert response.headers["content-type"] == "application/json"
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert "context window" in response.json()["error"]["message"]


def test_context_budget_accepts_exact_limit_and_rejects_extra_token():
    engine = TransformersChatEngine("cpu-stub", CpuTokenizer(), CpuModel(), torch)
    request = ChatCompletionRequest(messages=[{"role": "user", "content": "안녕"}], max_tokens=3)
    engine.validate_request(request)
    assert engine._prepare_generation(request, None)["max_new_tokens"] == 3
    request.max_tokens = 4
    with pytest.raises(InvalidChatRequest):
        engine._prepare_generation(request, None)


@pytest.mark.parametrize("cancel_request, stream_done_before_exit", [
    (False, False), (True, False), (True, True),
])
def test_generation_lock_is_held_until_worker_exits(
    monkeypatch, cancel_request, stream_done_before_exit
):
    """A release barrier represents an uninterruptible model forward, without sleep timing."""
    import transformers

    class Streamer:
        def __init__(self, *args, **kwargs):
            self.queue = queue.Queue()

        def __iter__(self):
            return self

        def __next__(self):
            item = self.queue.get(timeout=0.01)
            if item is None:
                raise StopIteration
            return item

        def on_finalized_text(self, text, stream_end=False):
            if stream_end:
                self.queue.put(None)

    monkeypatch.setattr(transformers, "TextIteratorStreamer", Streamer)

    async def scenario():
        loop = asyncio.get_running_loop()
        first_entered = asyncio.Event()
        disconnect_checked = asyncio.Event()
        second_started = asyncio.Event()
        join_waiting = asyncio.Event()
        release_first = threading.Event()
        active = 0
        calls = 0
        maximum_active = 0
        stop_checks = []
        original_shield = asyncio.shield

        def record_join_wait(future):
            join_waiting.set()
            return original_shield(future)

        monkeypatch.setattr(asyncio, "shield", record_join_wait)

        class Model(CpuModel):
            def generate(self, **kwargs):
                nonlocal active, calls, maximum_active
                calls += 1
                active += 1
                maximum_active = max(maximum_active, active)
                stop_checks.append(kwargs["stopping_criteria"][0])
                try:
                    if calls == 1:
                        if stream_done_before_exit:
                            kwargs["streamer"].on_finalized_text("", stream_end=True)
                        loop.call_soon_threadsafe(first_entered.set)
                        if not release_first.wait(timeout=5):
                            raise RuntimeError("test did not release generation barrier")
                    kwargs["streamer"].on_finalized_text("", stream_end=True)
                    return torch.tensor([[1, 9]])
                finally:
                    active -= 1

        engine = TransformersChatEngine("cpu-stub", CpuTokenizer(1), Model(), torch)
        request = ChatCompletionRequest(
            messages=[{"role": "user", "content": "안녕"}], max_tokens=1
        )

        async def disconnected():
            disconnect_checked.set()
            return not cancel_request

        async def connected():
            return False

        async def consume(check, started=None):
            if started is not None:
                started.set()
            return [part async for part in engine.stream_chat(request, check)]

        first = asyncio.create_task(consume(disconnected))
        second = None
        try:
            await asyncio.wait_for(first_entered.wait(), timeout=3)
            await asyncio.wait_for(disconnect_checked.wait(), timeout=3)
            if stream_done_before_exit:
                await asyncio.wait_for(join_waiting.wait(), timeout=3)
            if cancel_request:
                first.cancel()
                await asyncio.sleep(0)
                first.cancel()  # cancel again while its cleanup is awaiting the worker
                await asyncio.sleep(0)
            second = asyncio.create_task(consume(connected, second_started))
            await second_started.wait()
            assert engine._generation_lock.locked()
            assert active == 1
            assert calls == 1
            assert not first.done()
            assert not second.done()
            assert stop_checks[0](None, None)  # cancellation/disconnect requests a stop immediately
            release_first.set()
            if cancel_request:
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(first, timeout=3)
            else:
                assert await asyncio.wait_for(first, timeout=3) == []
            assert await asyncio.wait_for(second, timeout=3) == [GenerationEnd("stop")]
            assert maximum_active == 1
            assert active == 0
            assert not engine._generation_lock.locked()
        finally:
            release_first.set()
            await asyncio.gather(first, *([second] if second else []), return_exceptions=True)

    asyncio.run(scenario())
