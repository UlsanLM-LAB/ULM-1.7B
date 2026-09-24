"""Raw Phase4 checkpoint audit; run only on the GPU host.

Uses the original Phase4 prompt suite and records token-level termination.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from ulm.inference.prompt import PHASE4_SYSTEM_PROMPT as SYSTEM

MODEL = "/home/ubuntu/ULM-1.7B/outputs/ulm-1.7b-phase4-best-merged"
REPEATED_PHRASE = re.compile(r"(.{3,40}?)\1{2,}")


def issues(
    prompt: str, response: str, *, ended_on_eos: bool, token_count: int, limit: int
) -> list[str]:
    problems = []
    if not response.strip():
        problems.append("empty")
    if response.strip().startswith(prompt.strip()) and prompt.strip():
        problems.append("prompt_echo")
    if REPEATED_PHRASE.search(re.sub(r"\s+", " ", response)):
        problems.append("repeated_phrase")
    if "\ufffd" in response:
        problems.append("replacement_character")
    if token_count >= limit and not ended_on_eos:
        problems.append("token_limit")
    if len(response) > 800:
        problems.append("abnormally_long")
    return problems


def baseline_prompts() -> list[dict]:
    return [
        {"category": "short_general", "prompt": "안녕하세요. 오늘 하루는 어땠어요?"},
        {"category": "factual", "prompt": "대한민국의 수도는 어디야?"},
        {"category": "casual", "prompt": "친구가 피곤하다고 하면 어떻게 답할래?"},
        {"category": "dialect", "prompt": "친구에게 밥 뭇나 하고 자연스럽게 물어봐 줘."},
        {"category": "long_answer", "prompt": "태화강을 산책하는 좋은 점을 세 문장으로 설명해 줘."},
        {"category": "short_answer", "prompt": "1+1은? 한 단어로 답해 줘."},
        {
            "category": "multi_turn",
            "prompt": "그곳에서 무엇을 하면 좋을까?",
            "history": [["주말에 태화강에 갈 거야.", "태화강 산책 좋제!"]],
        },
        {
            "category": "stress",
            "prompt": "친구야를 여러 번 쓰지 말고 비 오는 날 친구에게 한 문장으로 인사해 줘.",
        },
    ]


def regression_prompts() -> list[dict]:
    from run_phase4_final_regression import load_200_prompts

    source = load_200_prompts()
    selected = []
    for category, count, group in [
        ("factual QA", 10, "general"),
        ("casual chat", 5, "general"),
        ("recommendation", 5, "general"),
        ("Ulsan QA", 10, "dialect"),
        ("slang", 5, "dialect"),
        ("multi-turn", 10, "multi_turn"),
    ]:
        matches = [x for x in source if x["category"] == category]
        if len(matches) < count:
            raise ValueError(f"Missing Phase4 prompts: {category}")
        selected.extend({**x, "group": group} for x in matches[:count])
    selected.extend(
        {"group": "stress", "category": "stress", "prompt": prompt}
        for prompt in [
            "친구야를 반복하지 말고 비 오는 날 친구에게 한 문장으로 인사해 줘.",
            "울산에 왔다는 말을 같은 표현 없이 두 문장으로 해 줘.",
            "아주 짧게 답해 줘. 밥 뭇나?",
            "긴 이야기를 해 줘. 태화강에서 보낸 하루를 네 문장으로 설명해 줘.",
            "이전 답을 그대로 따라 하지 말고 오늘 날씨가 좋다고 한 문장으로 말해 줘.",
        ]
    )
    assert len(selected) == 50
    return selected


def make_messages(item: dict) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": SYSTEM}]
    for user, assistant in item.get("history", []):
        messages.extend(
            [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
        )
    messages.append({"role": "user", "content": item["prompt"]})
    return messages


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=("baseline", "regression"), required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    model_path = Path(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda:0"
    ).eval()
    eos_ids = model.generation_config.eos_token_id
    eos_ids = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids)
    config_sha = hashlib.sha256((model_path / "config.json").read_bytes()).hexdigest()
    tokenizer_sha = hashlib.sha256((model_path / "tokenizer_config.json").read_bytes()).hexdigest()
    prompts = baseline_prompts() if args.suite == "baseline" else regression_prompts()
    variants = ("greedy", "sampling", "phase4") if args.suite == "baseline" else ("phase4",)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out:
        for index, item in enumerate(prompts, 1):
            prompt_text = tokenizer.apply_chat_template(
                make_messages(item),
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda:0")
            for variant in variants:
                set_seed(42)
                kwargs = {
                    "max_new_tokens": 150,
                    "do_sample": variant != "greedy",
                    "pad_token_id": tokenizer.eos_token_id,
                }
                if variant != "greedy":
                    kwargs.update(temperature=0.7, top_p=0.9)
                if variant == "phase4":
                    kwargs["repetition_penalty"] = 1.1
                with torch.inference_mode():
                    generated = model.generate(**inputs, **kwargs)[
                        0, inputs["input_ids"].shape[-1] :
                    ]
                response = tokenizer.decode(generated, skip_special_tokens=True).strip()
                ended_on_eos = bool(len(generated) and generated[-1].item() in eos_ids)
                row = {
                    "index": index,
                    "group": item.get("group", item["category"]),
                    "category": item["category"],
                    "prompt": item["prompt"],
                    "history": item.get("history", []),
                    "variant": variant,
                    "response": response,
                    "generated_tokens": len(generated),
                    "last_token_id": generated[-1].item() if len(generated) else None,
                    "ended_on_eos": ended_on_eos,
                    "issues": issues(
                        item["prompt"],
                        response,
                        ended_on_eos=ended_on_eos,
                        token_count=len(generated),
                        limit=150,
                    ),
                    "model_path": args.model,
                    "config_sha256": config_sha,
                    "tokenizer_sha256": tokenizer_sha,
                    "tokenizer_class": type(tokenizer).__name__,
                    "eos_token_ids": sorted(eos_ids),
                    "pad_token_id": tokenizer.pad_token_id,
                    "prompt_token_count": inputs["input_ids"].shape[-1],
                }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                print(
                    f"{index}/{len(prompts)} {variant}: {row['issues']} {response[:100]}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
