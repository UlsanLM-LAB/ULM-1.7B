"""Real HTTP/SSE smoke against the loaded Phase4 backend or ULM-chat proxy."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import httpx

from audit_phase4_inference import issues

PROMPTS = [
    "대한민국의 수도는 어디야?",
    "태양계에서 가장 큰 행성은 뭐야?",
    "요즘 어떻게 지내?",
    "밥 뭇나 하고 자연스럽게 물어봐 줘.",
    "오늘 날씨가 좋으면 친구한테 뭐 하자고 할까?",
    "태화강 산책을 한 문장으로 추천해 줘.",
    "친구야를 반복하지 말고 비 오는 날 인사해 줘.",
    "울산에 처음 온 사람에게 짧게 인사해 줘.",
]


def request_sse(client: httpx.Client, url: str, messages: list[dict]) -> dict:
    started = time.monotonic()
    chunks = []
    done = False
    finish_reason = None
    first_token_ms = None
    with client.stream(
        "POST",
        url,
        json={"model": "ulm-1.7b", "messages": messages, "stream": True},
        headers={"Accept": "text/event-stream"},
    ) as response:
        response.raise_for_status()
        assert "text/event-stream" in response.headers.get("content-type", "")
        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                done = True
                break
            event = json.loads(data)
            if "error" in event:
                raise RuntimeError(f"Backend SSE error: {event['error']}")
            token = event.get(
                "token", event.get("choices", [{}])[0].get("delta", {}).get("content", "")
            )
            if token:
                if first_token_ms is None:
                    first_token_ms = round((time.monotonic() - started) * 1000)
                chunks.append(token)
            reason = event.get("choices", [{}])[0].get("finish_reason")
            if reason:
                finish_reason = reason
    text = "".join(chunks)
    prompt = messages[-1]["content"]
    problems = issues(
        prompt,
        text,
        ended_on_eos=finish_reason == "stop",
        token_count=150 if finish_reason == "length" else 0,
        limit=150,
    )
    return {
        "prompt": prompt,
        "messages": messages,
        "response": text,
        "token_events": len(chunks),
        "first_token_ms": first_token_ms,
        "total_ms": round((time.monotonic() - started) * 1000),
        "finish_reason": finish_reason,
        "done": done,
        "issues": problems,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--health-url", required=True)
    parser.add_argument("--proxy", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with httpx.Client(timeout=180) as client:
        health_response = client.get(args.health_url)
        health_response.raise_for_status()
        health = health_response.json()
        if args.proxy:
            assert health["mockMode"] is False, health
            assert health["backendTarget"], health
        else:
            assert health["model_loaded"] is True, health
            assert "ulm-1.7b-phase4-best-merged" in health["model_path"], health
        print("health", json.dumps(health, ensure_ascii=False), flush=True)

        rows = []
        for prompt in PROMPTS:
            rows.append(request_sse(client, args.url, [{"role": "user", "content": prompt}]))

        first_messages = [{"role": "user", "content": "내 이름은 민수야. 기억해 줘."}]
        first = request_sse(client, args.url, first_messages)
        rows.append(first)
        history = first_messages + [
            {"role": "assistant", "content": first["response"]},
            {"role": "user", "content": "내 이름이 뭐라고 했지?"},
        ]
        rows.append(request_sse(client, args.url, history))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out:
        for index, row in enumerate(rows, 1):
            row["index"] = index
            row["path"] = "ulm_chat_proxy" if args.proxy else "backend_direct"
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(index, row["finish_reason"], row["issues"], row["response"][:95], flush=True)
    assert len(rows) == 10
    assert all(row["done"] and row["first_token_ms"] is not None for row in rows)


if __name__ == "__main__":
    main()
