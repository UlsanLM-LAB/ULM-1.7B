"""FastAPI server for streaming chat with a merged ULM model."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import queue
import threading
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from typing import Any, Literal, Protocol

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator

DEFAULT_MODEL_PATH = "outputs/ulm-1.7b-phase4-best-merged"
MODEL_NAME = "ULM-1.7B"
DEFAULT_SYSTEM_PROMPT = (
    "울산 지역어 대화 assistant로서 자연스럽고 일상적인 울산 사투리로 상대방과 친근하게 대화한다. "
    "자연스럽고 편안한 일상 울산 말투를 기본으로 구사한다."
)


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)

    @field_validator("content")
    @classmethod
    def content_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message content must not be blank")
        return value


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    model: str = "ulm-1.7b"
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = True
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    max_tokens: int = Field(default=512, ge=1, le=2048)
    system_prompt: str | None = Field(
        default=None,
        validation_alias=AliasChoices("system_prompt", "systemPrompt"),
    )


DisconnectCheck = Callable[[], Awaitable[bool]]


class ChatEngine(Protocol):
    model_path: str
    device: str
    dtype: str
    loaded: bool

    async def stream_chat(
        self,
        request: ChatCompletionRequest,
        is_disconnected: DisconnectCheck,
    ) -> AsyncIterator[str]: ...


class TransformersChatEngine:
    """One loaded Transformers model shared by all requests."""

    def __init__(self, model_path: str, tokenizer: Any, model: Any, torch_module: Any) -> None:
        self.model_path = model_path
        self.tokenizer = tokenizer
        self.model = model
        self._torch = torch_module
        self._generation_lock = asyncio.Lock()
        self.loaded = True

        embedding_device = model.get_input_embeddings().weight.device
        self._input_device = embedding_device
        self.device = str(embedding_device)
        self.dtype = str(getattr(model, "dtype", "unknown")).removeprefix("torch.")

    @classmethod
    def load(cls, model_path: str) -> TransformersChatEngine:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on installation extras
            raise RuntimeError(
                "모델 서버 의존성이 없습니다. `uv sync --extra ml`을 먼저 실행하세요."
            ) from exc

        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

        if torch.cuda.is_available():
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            model_kwargs: dict[str, Any] = {"dtype": dtype, "device_map": "auto"}
        else:
            # float16 CPU kernels are incomplete and commonly fail at generation time.
            model_kwargs = {"dtype": torch.float32}

        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        model.eval()
        return cls(model_path, tokenizer, model, torch)

    def _messages_for_template(self, request: ChatCompletionRequest) -> list[dict[str, str]]:
        messages = [message.model_dump() for message in request.messages]
        if not any(message["role"] == "system" for message in messages):
            messages.insert(
                0,
                {
                    "role": "system",
                    "content": request.system_prompt or DEFAULT_SYSTEM_PROMPT,
                },
            )
        return messages

    def _prepare_generation(self, request: ChatCompletionRequest, streamer: Any) -> dict[str, Any]:
        messages = self._messages_for_template(request)
        template_kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        try:
            prompt = self.tokenizer.apply_chat_template(messages, **template_kwargs)
        except TypeError:
            # Older chat templates do not expose Qwen's enable_thinking option.
            template_kwargs.pop("enable_thinking")
            prompt = self.tokenizer.apply_chat_template(messages, **template_kwargs)

        encoded = self.tokenizer(prompt, return_tensors="pt")
        context_limit = int(getattr(self.model.config, "max_position_embeddings", 32768))
        max_input_tokens = max(1, context_limit - request.max_tokens)
        if encoded["input_ids"].shape[-1] > max_input_tokens:
            encoded = {key: value[:, -max_input_tokens:] for key, value in encoded.items()}
        encoded = {key: value.to(self._input_device) for key, value in encoded.items()}

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id

        generation_kwargs: dict[str, Any] = {
            **encoded,
            "streamer": streamer,
            "max_new_tokens": request.max_tokens,
            "do_sample": request.temperature > 0,
            "pad_token_id": pad_token_id,
        }
        if request.temperature > 0:
            generation_kwargs.update(temperature=request.temperature, top_p=request.top_p)
        return generation_kwargs

    async def stream_chat(
        self,
        request: ChatCompletionRequest,
        is_disconnected: DisconnectCheck,
    ) -> AsyncIterator[str]:
        from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

        stop_event = threading.Event()

        class StopWhenRequested(StoppingCriteria):
            def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
                return stop_event.is_set()

        async with self._generation_lock:
            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
                timeout=0.25,
            )
            generation_kwargs = self._prepare_generation(request, streamer)
            generation_kwargs["stopping_criteria"] = StoppingCriteriaList([StopWhenRequested()])
            generation_errors: list[BaseException] = []

            def run_generation() -> None:
                try:
                    with self._torch.inference_mode():
                        self.model.generate(**generation_kwargs)
                except BaseException as exc:  # propagate failures to the response stream
                    generation_errors.append(exc)
                    streamer.on_finalized_text("", stream_end=True)

            thread = threading.Thread(target=run_generation, name="ulm-generation", daemon=True)
            thread.start()

            try:
                iterator = iter(streamer)
                while True:
                    if await is_disconnected():
                        stop_event.set()
                        break

                    state, chunk = await asyncio.to_thread(_next_stream_item, iterator)
                    if state == "chunk":
                        if chunk:
                            yield chunk
                        continue
                    if state == "waiting":
                        if generation_errors:
                            raise RuntimeError("모델 응답 생성에 실패했습니다.") from (
                                generation_errors[0]
                            )
                        continue
                    break

                if generation_errors:
                    raise RuntimeError("모델 응답 생성에 실패했습니다.") from generation_errors[0]
            finally:
                stop_event.set()
                await asyncio.to_thread(thread.join, 2.0)


def _next_stream_item(iterator: Any) -> tuple[str, str]:
    try:
        return "chunk", next(iterator)
    except queue.Empty:
        return "waiting", ""
    except StopIteration:
        return "done", ""


def _completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex}"


def _stream_chunk(completion_id: str, content: str, finish_reason: str | None = None) -> str:
    payload = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "delta": {"content": content} if content else {},
                "finish_reason": finish_reason,
            }
        ],
    }
    if content:
        # Kept for the existing ULM-chat provider while retaining OpenAI-compatible choices.
        payload["token"] = content
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def create_app(
    model_path: str | None = None,
    engine: ChatEngine | None = None,
) -> FastAPI:
    resolved_model_path = model_path or os.environ.get("ULM_MODEL_PATH", DEFAULT_MODEL_PATH)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if app.state.engine is None:
            app.state.engine = await asyncio.to_thread(
                TransformersChatEngine.load, resolved_model_path
            )
        yield

    app = FastAPI(title="ULM-1.7B Inference API", version="1.0.0", lifespan=lifespan)
    app.state.engine = engine

    @app.get("/health")
    async def health() -> dict[str, Any]:
        active_engine: ChatEngine = app.state.engine
        return {
            "status": "ok",
            "model_loaded": active_engine.loaded,
            "model": MODEL_NAME,
            "model_path": active_engine.model_path,
            "device": active_engine.device,
            "dtype": active_engine.dtype,
        }

    @app.post("/api/chat", include_in_schema=False)
    @app.post("/v1/chat/completions")
    async def chat_completion(
        chat_request: ChatCompletionRequest,
        http_request: Request,
    ) -> Response:
        active_engine: ChatEngine = app.state.engine
        completion_id = _completion_id()

        if not chat_request.stream:
            try:
                chunks = [
                    chunk
                    async for chunk in active_engine.stream_chat(
                        chat_request, http_request.is_disconnected
                    )
                ]
            except Exception as exc:
                return JSONResponse(
                    status_code=500,
                    content={"error": {"message": str(exc), "type": "generation_error"}},
                )
            return JSONResponse(
                {
                    "id": completion_id,
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": MODEL_NAME,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "".join(chunks)},
                            "finish_reason": "stop",
                        }
                    ],
                }
            )

        async def event_stream() -> AsyncIterator[str]:
            try:
                async for chunk in active_engine.stream_chat(
                    chat_request, http_request.is_disconnected
                ):
                    yield _stream_chunk(completion_id, chunk)
                if not await http_request.is_disconnected():
                    yield _stream_chunk(completion_id, "", "stop")
            except Exception as exc:
                error = {"error": {"message": str(exc), "type": "generation_error"}}
                yield f"data: {json.dumps(error, ensure_ascii=False)}\n\n"
            if not await http_request.is_disconnected():
                yield "data: [DONE]\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
            },
        )

    return app


app = create_app()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="ULM-1.7B streaming inference server")
    parser.add_argument(
        "--model",
        default=os.environ.get("ULM_MODEL_PATH", DEFAULT_MODEL_PATH),
        help="Merged Hugging Face model path (or ULM_MODEL_PATH)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    args = parser.parse_args(argv)

    import uvicorn

    uvicorn.run(create_app(args.model), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
