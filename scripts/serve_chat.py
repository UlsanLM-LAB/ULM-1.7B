"""ULM-1.7B Local Streaming Inference Server for ULM-chat.

Serves /api/chat with Server-Sent Events (SSE) streaming.
"""

from __future__ import annotations

import json
import threading
from typing import Any, AsyncGenerator

import torch
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

MODEL_PATH = os.environ.get("ULM_MODEL_PATH", "outputs/ulm-1.7b-phase4-best-merged")
DEFAULT_SYSTEM_PROMPT = (
    "울산 지역어 대화 assistant로서 자연스럽고 일상적인 울산 사투리로 상대방과 친근하게 대화한다. "
    "자연스럽고 편안한 일상 울산 말투를 기본으로 구사한다."
)

app = FastAPI(title="ULM-1.7B Inference API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print(f"[ULM Server] Loading tokenizer from {MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)

print(f"[ULM Server] Loading model from {MODEL_PATH} on GPU...")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=dtype,
    device_map=device,
)
model.eval()
print(f"[ULM Server] Model successfully loaded on {device} ({dtype})!")


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "ulm-1.7b"
    messages: list[ChatMessage]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    max_tokens: int = Field(default=512, ge=1, le=2048)
    systemPrompt: str | None = None
    stream: bool = True


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_PATH, "device": device}


@app.post("/api/chat")
@app.post("/v1/chat/completions")
async def chat_endpoint(req: ChatCompletionRequest):
    # Prepare messages
    msgs: list[dict[str, str]] = []
    has_system = False

    for m in req.messages:
        if m.role == "system":
            has_system = True
            msgs.append({"role": "system", "content": m.content})
        else:
            msgs.append({"role": m.role, "content": m.content})

    if not has_system:
        system_text = req.systemPrompt or DEFAULT_SYSTEM_PROMPT
        msgs.insert(0, {"role": "system", "content": system_text})

    template_kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
    try:
        prompt = tokenizer.apply_chat_template(msgs, **template_kwargs)
    except TypeError:
        template_kwargs.pop("enable_thinking", None)
        prompt = tokenizer.apply_chat_template(msgs, **template_kwargs)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    gen_kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": req.max_tokens,
        "temperature": max(req.temperature, 0.01),
        "top_p": req.top_p,
        "do_sample": req.temperature > 0.05,
    }

    # Run generation in a background thread
    thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    async def sse_event_generator() -> AsyncGenerator[str, None]:
        in_thinking_block = False
        try:
            for text_chunk in streamer:
                # Filter out <think>...</think> reasoning blocks if present
                if "<think>" in text_chunk:
                    in_thinking_block = True
                    text_chunk = text_chunk.split("<think>", 1)[0]
                if in_thinking_block:
                    if "</think>" in text_chunk:
                        in_thinking_block = False
                        text_chunk = text_chunk.split("</think>", 1)[1]
                    else:
                        continue

                if text_chunk:
                    # Send SSE data line expected by ULM-chat: data: {"token": "..."}\n\n
                    data = json.dumps({"token": text_chunk}, ensure_ascii=False)
                    yield f"data: {data}\n\n"
        finally:
            thread.join()
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        sse_event_generator(),
        media_type="text/event-stream; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="ULM-1.7B Streaming Inference Server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=8008, help="Bind port")
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

