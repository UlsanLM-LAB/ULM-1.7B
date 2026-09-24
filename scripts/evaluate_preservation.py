"""Evaluation runner for 100-Prompt Preservation Benchmark.

Used for Base Baseline measurement, training gate monitoring, and checkpoint selection.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

CJK_REGEX = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff]")
REPETITION_REGEX = re.compile(r"(.{4,})\1{2,}")
REPEATED_PHRASE_REGEX = re.compile(r"(.{3,40}?)\1{2,}")


def format_chat_prompt(tokenizer: Any, item: dict) -> str:
    messages = []
    if "history" in item:
        for u, a in item["history"]:
            messages.append({"role": "user", "content": u})
            messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": item["prompt"]})

    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def separate_thinking(raw_text: str) -> tuple[str, str, bool]:
    if "<think>" in raw_text:
        parts = raw_text.split("</think>", 1)
        if len(parts) == 2:
            return parts[0].replace("<think>", "").strip(), parts[1].strip(), True
        else:
            return raw_text.replace("<think>", "").strip(), "", True
    if raw_text.strip().startswith("Thinking Process:"):
        return raw_text.strip(), "", True
    return "", raw_text.strip(), False


def evaluate_response(item: dict, final_text: str, raw_text: str, token_count: int, max_tokens: int) -> dict[str, Any]:
    cat = item["category"]
    eval_text = final_text if final_text else raw_text
    norm = eval_text.strip()

    empty_final = len(norm) == 0
    replacement_char = "\ufffd" in raw_text
    repetition = bool(REPETITION_REGEX.search(raw_text)) or bool(REPEATED_PHRASE_REGEX.search(raw_text))
    cjk_leakage = bool(CJK_REGEX.search(final_text))
    token_limit = token_count >= max_tokens

    factual_hit = False
    if cat == "factual_qa":
        kw = item.get("keywords", [])
        factual_hit = any(k.lower() in norm.lower() for k in kw)

    memory_hit = False
    if cat == "multi_turn":
        m_kw = item.get("memory_keywords", [])
        memory_hit = any(k.lower() in norm.lower() for k in m_kw)

    instruction_hit = False
    if cat == "instruction_trap":
        itype = item.get("instruction_type", "")
        if itype == "exact_apple":
            clean = re.sub(r"[^\w\s]", "", norm).strip()
            instruction_hit = clean == "사과"
        elif itype == "exact_banana":
            clean = re.sub(r"[^\w\s]", "", norm).strip()
            instruction_hit = clean == "바나나"
        elif itype == "numbered_list_1_5":
            lines = [line.strip() for line in norm.splitlines() if line.strip()]
            has_1 = any(re.match(r"^1[\.\)]", line) for line in lines)
            has_5 = any(re.match(r"^5[\.\)]", line) for line in lines)
            instruction_hit = has_1 and has_5 and len(lines) <= 7
        elif itype == "numbered_list_1_4":
            lines = [line.strip() for line in norm.splitlines() if line.strip()]
            has_1 = any(re.match(r"^1[\.\)]", line) for line in lines)
            has_4 = any(re.match(r"^4[\.\)]", line) for line in lines)
            instruction_hit = has_1 and has_4 and len(lines) <= 6
        elif itype == "repeat_hello_3":
            instruction_hit = (len(re.findall(r"안녕하세요", norm)) == 3)
        elif itype == "repeat_thanks_2":
            instruction_hit = (len(re.findall(r"감사합니다", norm)) == 2)
        elif itype == "extract_animals":
            instruction_hit = ("고양이" in norm and "참새" in norm and "철수" not in norm)
        elif itype == "extract_dog_rabbit":
            instruction_hit = ("강아지" in norm and "토끼" in norm and "민지" not in norm)
        elif itype == "yes_no":
            clean = re.sub(r"[^\w\s]", "", norm).strip()
            instruction_hit = clean in ("네", "예")
        elif itype.startswith("hallucination_trap_"):
            refusal_kw = [
                "없", "허구", "사실이 아니", "존재하지 않", "지어지지 않", "건설된 적",
                "건설되지 않", "불가능", "증거가 없", "낭설", "가공", "신화", "개발하지 않",
                "선출된 적", "발명하지 않", "아닙니다", "역사적 사실과 다릅니다"
            ]
            instruction_hit = any(kw in norm for kw in refusal_kw)
        elif itype == "exact_three_fruits":
            instruction_hit = "apple" in norm.lower() and "banana" in norm.lower() and "orange" in norm.lower()
        elif itype == "exact_three_animals":
            instruction_hit = "cat" in norm.lower() and "dog" in norm.lower() and "bird" in norm.lower()
        elif itype == "exact_json_status":
            instruction_hit = '{"status": "ok"}' in norm or '{"status":"ok"}' in norm
        elif itype == "exact_json_pass":
            instruction_hit = '{"result": "pass"}' in norm or '{"result":"pass"}' in norm

    dialect_hit = False
    if cat == "dialect_eval":
        if item.get("dialect_gen"):
            markers = item.get("markers", [])
            dialect_hit = any(m in norm for m in markers)
        else:
            kw = item.get("keywords", [])
            dialect_hit = any(k in norm for k in kw)

    return {
        "empty_final": empty_final,
        "replacement_char": replacement_char,
        "repetition": repetition,
        "cjk_leakage": cjk_leakage,
        "token_limit": token_limit,
        "factual_hit": factual_hit,
        "memory_hit": memory_hit,
        "instruction_hit": instruction_hit,
        "dialect_hit": dialect_hit,
    }


def run_evaluation(
    model_path: str,
    adapter_path: str | None = None,
    benchmark_path: str = "data/preservation_benchmark_100.jsonl",
    device: str = "cuda:0",
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 20,
    repetition_penalty: float = 1.1,
    seed: int = 42,
) -> tuple[list[dict], dict]:
    print(f"\n=======================================================")
    print(f"Loading Base Model: {model_path}")
    if adapter_path:
        print(f"Loading Adapter: {adapter_path}")
    print(f"Benchmark: {benchmark_path}")
    print(f"=======================================================")

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map=device,
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
        print(f"Attached PeftModel adapter from {adapter_path}")

    load_time = round(time.time() - t0, 2)
    vram_gb = round(torch.cuda.memory_allocated() / (1024**3), 2)
    print(f"Model ready in {load_time}s | VRAM: {vram_gb} GB")

    with open(benchmark_path, "r", encoding="utf-8") as f:
        bench_items = [json.loads(line) for line in f]

    rows = []
    for idx, item in enumerate(bench_items, start=1):
        prompt_text = format_chat_prompt(tokenizer, item)
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
        input_len = inputs.input_ids.shape[1]

        set_seed(seed)
        gen_t0 = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=True,
            )
        latency_ms = round((time.perf_counter() - gen_t0) * 1000, 2)
        gen_tokens = outputs[0][input_len:].tolist()
        gen_token_count = len(gen_tokens)
        raw_text = tokenizer.decode(gen_tokens, skip_special_tokens=False)

        thinking_text, final_text, has_thinking = separate_thinking(raw_text)
        clean_final = final_text.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()

        eval_res = evaluate_response(item, clean_final, raw_text, gen_token_count, max_new_tokens)

        row = {
            "id": item["id"],
            "category": item["category"],
            "prompt": item["prompt"],
            "raw_output": raw_text,
            "thinking_content": thinking_text,
            "final_answer": clean_final,
            "has_thinking_block": has_thinking,
            "generated_token_count": gen_token_count,
            "latency_ms": latency_ms,
            **eval_res,
        }
        rows.append(row)

        short_ans = clean_final[:50].replace("\n", " ") if clean_final else "(EMPTY)"
        if idx % 10 == 0 or idx == len(bench_items):
            print(f"  [{idx:03d}/{len(bench_items)}] {item['category']:16s} | Latency: {latency_ms:6.1f}ms | Tokens: {gen_token_count:3d} | Ans: {short_ans}")

    # Summary metrics
    factual_rows = [r for r in rows if r["category"] == "factual_qa"]
    memory_rows = [r for r in rows if r["category"] == "multi_turn"]
    trap_rows = [r for r in rows if r["category"] == "instruction_trap"]
    dialect_rows = [r for r in rows if r["category"] == "dialect_eval"]
    general_rows = [r for r in rows if r["category"] == "general_korean"]

    factual_hits = sum(1 for r in factual_rows if r["factual_hit"])
    memory_hits = sum(1 for r in memory_rows if r["memory_hit"])
    trap_hits = sum(1 for r in trap_rows if r["instruction_hit"])
    dialect_hits = sum(1 for r in dialect_rows if r["dialect_hit"])

    summary = {
        "model_path": model_path,
        "adapter_path": adapter_path,
        "total_prompts": len(rows),
        "load_time_s": load_time,
        "vram_gb": vram_gb,
        "avg_latency_ms": round(sum(r["latency_ms"] for r in rows) / len(rows), 2),
        "avg_tokens": round(sum(r["generated_token_count"] for r in rows) / len(rows), 1),
        "empty_final_count": sum(1 for r in rows if r["empty_final"]),
        "token_limit_count": sum(1 for r in rows if r["token_limit"]),
        "cjk_leakage_count": sum(1 for r in rows if r["cjk_leakage"]),
        "repetition_count": sum(1 for r in rows if r["repetition"]),
        "malformed_unicode_count": sum(1 for r in rows if r["replacement_char"]),
        "factual_qa": {
            "total": len(factual_rows),
            "hits": factual_hits,
            "accuracy_pct": round(factual_hits / len(factual_rows) * 100, 1),
        },
        "multi_turn": {
            "total": len(memory_rows),
            "hits": memory_hits,
            "accuracy_pct": round(memory_hits / len(memory_rows) * 100, 1),
        },
        "instruction_trap": {
            "total": len(trap_rows),
            "hits": trap_hits,
            "accuracy_pct": round(trap_hits / len(trap_rows) * 100, 1),
        },
        "dialect_eval": {
            "total": len(dialect_rows),
            "hits": dialect_hits,
            "accuracy_pct": round(dialect_hits / len(dialect_rows) * 100, 1),
        },
        "general_korean": {
            "total": len(general_rows),
            "avg_latency_ms": round(sum(r["latency_ms"] for r in general_rows) / len(general_rows), 2),
            "avg_tokens": round(sum(r["generated_token_count"] for r in general_rows) / len(general_rows), 1),
        },
    }

    del model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    return rows, summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate 100-Prompt Preservation Benchmark")
    parser.add_argument("--model-path", default="/home/ubuntu/models/Qwen3.8-4B-Distill")
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--benchmark-path", default="data/preservation_benchmark_100.jsonl")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_jsonl = Path(args.output_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_summary = Path(args.output_summary)
    out_summary.parent.mkdir(parents=True, exist_ok=True)

    rows, summary = run_evaluation(
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        benchmark_path=args.benchmark_path,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
    )

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n=======================================================")
    print("Evaluation Complete!")
    print(f"  Factual QA:       {summary['factual_qa']['hits']}/{summary['factual_qa']['total']} ({summary['factual_qa']['accuracy_pct']}%)")
    print(f"  Multi-turn Memory:{summary['multi_turn']['hits']}/{summary['multi_turn']['total']} ({summary['multi_turn']['accuracy_pct']}%)")
    print(f"  Instruction/Trap: {summary['instruction_trap']['hits']}/{summary['instruction_trap']['total']} ({summary['instruction_trap']['accuracy_pct']}%)")
    print(f"  Dialect Eval:     {summary['dialect_eval']['hits']}/{summary['dialect_eval']['total']} ({summary['dialect_eval']['accuracy_pct']}%)")
    print(f"  Avg Latency:      {summary['avg_latency_ms']} ms | Avg Tokens: {summary['avg_tokens']}")
    print(f"  Empty Final:      {summary['empty_final_count']} | Repetition: {summary['repetition_count']}")
    print(f"Saved: {out_jsonl} & {out_summary}")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
