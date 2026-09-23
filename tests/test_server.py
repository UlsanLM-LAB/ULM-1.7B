from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from fastapi.testclient import TestClient

from ulm.inference.server import ChatCompletionRequest, create_app


class FakeEngine:
    model_path = "/models/phase4"
    device = "cpu"
    dtype = "float32"
    loaded = True

    def __init__(self, chunks: tuple[str, ...] = ("반갑", "데이!"), fail: bool = False) -> None:
        self.chunks = chunks
        self.fail = fail
        self.requests: list[ChatCompletionRequest] = []

    async def stream_chat(
        self, request: ChatCompletionRequest, is_disconnected: Any
    ) -> AsyncIterator[str]:
        self.requests.append(request)
        for chunk in self.chunks:
            yield chunk
        if self.fail:
            raise RuntimeError("test generation failure")


def test_health_reports_loaded_model() -> None:
    with TestClient(create_app(engine=FakeEngine())) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_loaded": True,
        "model": "ULM-1.7B",
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
