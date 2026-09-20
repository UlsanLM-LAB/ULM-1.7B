"""Web E2E Verification for ULM-chat -> FastAPI -> Phase 3 Best Merged Model."""

import json
import urllib.request

TEST_PROMPTS = [
    "안녕",
    "너 뭐하노?",
    "오늘 뭐 먹을까?",
    "1+1은 뭐야?",
    "왜 하늘은 파래?",
    "파이썬 리스트 정렬 알려줘",
    "오늘 기분이 안 좋다",
]

API_URL = "http://localhost:3001/api/chat"

print(f"=== Testing ULM-chat Web API E2E: {API_URL} ===")
results = []

for prompt in TEST_PROMPTS:
    payload = {
        "model": "ulm-1.7b",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
    }
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200, f"Expected 200, got {resp.status}"
        tokens = []
        for line in resp:
            line_str = line.decode("utf-8").strip()
            if line_str.startswith("data: ") and line_str != "data: [DONE]":
                data_json = json.loads(line_str[6:])
                tokens.append(data_json.get("token", ""))
        full_text = "".join(tokens).strip()

    # Check for Mock contamination
    is_mock = "게이야 에 대해 물어보셨네예!" in full_text or "ULM은 일상적인 질문도 정겹고" in full_text
    assert not is_mock, f"MOCK DETECTED in response: {full_text}"

    results.append({"prompt": prompt, "response": full_text})
    print(f"\n[Prompt]   {prompt}")
    print(f"[Response] {full_text}")

print("\n=== ALL 7 WEB E2E TESTS PASSED SUCCESSFULLY! ===")
with open("reports/phase3_web_e2e_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
