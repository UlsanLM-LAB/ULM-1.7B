"""FastAPI server for streaming chat with a merged ULM model."""

from __future__ import annotations

import argparse
import asyncio
import hmac
import json
import os
import queue
import threading
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator

from .policy import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
)
from .policy import (
    generation_kwargs as phase4_generation_kwargs,
)
from .prompt import PHASE4_SYSTEM_PROMPT, build_chat_messages

DEFAULT_MODEL_PATH = "outputs/ulm-4b-arm-b-merged"
MODEL_NAME = "ULM-4B"
DEFAULT_SYSTEM_PROMPT = PHASE4_SYSTEM_PROMPT


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=20000)

    @field_validator("content")
    @classmethod
    def content_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message content must not be blank")
        return value


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    model: str = "ulm-4b"
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = True
    temperature: float = Field(default=DEFAULT_TEMPERATURE, ge=0.0, le=2.0)
    top_p: float = Field(default=DEFAULT_TOP_P, gt=0.0, le=1.0)
    max_tokens: int = Field(default=DEFAULT_MAX_NEW_TOKENS, ge=1, le=2048)
    dialect_strength: int = Field(default=2, ge=0, le=3, strict=True)
    system_prompt: str | None = Field(
        default=None,
        validation_alias=AliasChoices("system_prompt", "systemPrompt"),
    )


DisconnectCheck = Callable[[], Awaitable[bool]]


@dataclass(frozen=True)
class GenerationEnd:
    reason: Literal["stop", "length"]


class InvalidChatRequest(ValueError):
    """An input error that must be reported before response headers are sent."""


class ChatEngine(Protocol):
    model_path: str
    device: str
    dtype: str
    loaded: bool

    def validate_request(self, request: ChatCompletionRequest) -> None: ...

    async def stream_chat(
        self,
        request: ChatCompletionRequest,
        is_disconnected: DisconnectCheck,
    ) -> AsyncIterator[str | GenerationEnd]: ...


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

        adapter_path = os.environ.get("ULM_ADAPTER_PATH")
        tokenizer_path = adapter_path or model_path
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)

        if torch.cuda.is_available():
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            model_kwargs: dict[str, Any] = {"dtype": dtype, "device_map": "auto"}
        else:
            # float16 CPU kernels are incomplete and commonly fail at generation time.
            model_kwargs = {"dtype": torch.float32}

        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        resolved_path = model_path
        if adapter_path:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError("ULM_ADAPTER_PATH requires peft") from exc
            model = PeftModel.from_pretrained(model, adapter_path)
            resolved_path = f"{model_path} + {adapter_path}"
        model.eval()
        return cls(resolved_path, tokenizer, model, torch)

    def _messages_for_template(self, request: ChatCompletionRequest) -> list[dict[str, str]]:
        return build_chat_messages(
            [message.model_dump() for message in request.messages],
            dialect_strength=request.dialect_strength,
            system_prompt=request.system_prompt or DEFAULT_SYSTEM_PROMPT,
        )

    def _encode_request(self, request: ChatCompletionRequest) -> dict[str, Any]:
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
        input_tokens = encoded["input_ids"].shape[-1]
        if input_tokens + request.max_tokens > context_limit:
            raise InvalidChatRequest(
                f"Chat history ({input_tokens} tokens) plus requested output "
                f"({request.max_tokens} tokens) exceeds the model context window "
                f"({context_limit} tokens)"
            )
        return encoded

    def validate_request(self, request: ChatCompletionRequest) -> None:
        self._encode_request(request)

    def _prepare_generation(self, request: ChatCompletionRequest, streamer: Any) -> dict[str, Any]:
        encoded = self._encode_request(request)
        encoded = {key: value.to(self._input_device) for key, value in encoded.items()}

        generation_kwargs: dict[str, Any] = {
            **encoded,
            "streamer": streamer,
            **phase4_generation_kwargs(
                temperature=request.temperature,
                top_p=request.top_p,
                max_new_tokens=request.max_tokens,
                eos_token_id=self.tokenizer.eos_token_id,
                prompt_length=encoded["input_ids"].shape[-1],
            ),
        }
        return generation_kwargs

    async def stream_chat(
        self,
        request: ChatCompletionRequest,
        is_disconnected: DisconnectCheck,
    ) -> AsyncIterator[str | GenerationEnd]:
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
            generation_outputs: list[Any] = []

            def run_generation() -> None:
                try:
                    with self._torch.inference_mode():
                        generation_outputs.append(self.model.generate(**generation_kwargs))
                except BaseException as exc:  # propagate failures to the response stream
                    generation_errors.append(exc)
                    streamer.on_finalized_text("", stream_end=True)

            thread = threading.Thread(target=run_generation, name="ulm-generation", daemon=True)
            thread.start()
            generation_finished = asyncio.create_task(asyncio.to_thread(thread.join))

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

                await asyncio.shield(generation_finished)
                if generation_errors:
                    raise RuntimeError("모델 응답 생성에 실패했습니다.") from generation_errors[0]
                if not generation_outputs and not stop_event.is_set():
                    raise RuntimeError("Model generation ended without a result")
                if generation_outputs and not stop_event.is_set():
                    eos = self.model.generation_config.eos_token_id
                    eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
                    last_id = int(generation_outputs[0][0, -1])
                    yield GenerationEnd("stop" if last_id in eos_ids else "length")
            finally:
                stop_event.set()
                # Stopping criteria cannot interrupt an in-flight model forward. Retain
                # ownership until the worker exits, including repeated request cancellation.
                await _wait_for_generation(generation_finished)


async def _wait_for_generation(generation_finished: asyncio.Task[None]) -> None:
    cancelled = False
    while not generation_finished.done():
        try:
            await asyncio.shield(generation_finished)
        except asyncio.CancelledError:
            cancelled = True
    generation_finished.result()
    if cancelled:
        raise asyncio.CancelledError


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
    api_key = os.environ.get("ULM_API_KEY")

    def authorize(
        credentials: HTTPAuthorizationCredentials | None = Depends(HTTPBearer(auto_error=False)),
    ) -> None:
        if api_key and (
            credentials is None
            or not hmac.compare_digest(credentials.credentials.encode(), api_key.encode())
        ):
            raise HTTPException(401, "Unauthorized", headers={"WWW-Authenticate": "Bearer"})

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if app.state.engine is None:
            app.state.engine = await asyncio.to_thread(
                TransformersChatEngine.load, resolved_model_path
            )
        yield

    app = FastAPI(
        title="ULM-4B Inference API",
        version="1.0.0",
        lifespan=lifespan,
        dependencies=[Depends(authorize)],
    )
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

        try:
            await asyncio.to_thread(active_engine.validate_request, chat_request)
        except InvalidChatRequest as exc:
            return JSONResponse(
                status_code=400,
                content={"error": {"message": str(exc), "type": "invalid_request_error"}},
            )
        except Exception as exc:
            return JSONResponse(
                status_code=500,
                content={"error": {"message": str(exc), "type": "generation_error"}},
            )

        if not chat_request.stream:
            try:
                chunks = []
                finish_reason = "stop"
                async for part in active_engine.stream_chat(
                    chat_request, http_request.is_disconnected
                ):
                    if isinstance(part, GenerationEnd):
                        finish_reason = part.reason
                    else:
                        chunks.append(part)
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
                            "finish_reason": finish_reason,
                        }
                    ],
                }
            )

        async def event_stream() -> AsyncIterator[str]:
            try:
                finish_reason = "stop"
                async for part in active_engine.stream_chat(
                    chat_request, http_request.is_disconnected
                ):
                    if isinstance(part, GenerationEnd):
                        finish_reason = part.reason
                    else:
                        yield _stream_chunk(completion_id, part)
                if not await http_request.is_disconnected():
                    yield _stream_chunk(completion_id, "", finish_reason)
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
    parser = argparse.ArgumentParser(description="ULM-4B streaming inference server")
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
