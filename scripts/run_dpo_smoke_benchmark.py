"""Run DPO Smoke Benchmark comparing Candidate A (LR 5e-6, beta 0.1) vs Candidate B (LR 1e-5, beta 0.1)
and evaluate smoke inference on the 17 specified prompts.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

SMOKE_PROMPTS = [
    "안녕",
    "게이야",
    "너 뭐하노?",
    "오늘 뭐 먹을까?",
    "배고픈데 메뉴 추천해줘",
    "1+1은 뭐야?",
    "12*13은?",
    "왜 하늘은 파래?",
    "물이 왜 끓어?",
    "파이썬 리스트 정렬 알려줘",
    "딕셔너리랑 리스트 차이가 뭐야?",
    "오늘 기분이 너무 안 좋다",
    "시험 공부하기 싫다",
    "울산에서 어디 놀러갈까?",
    "아니 그게 무슨 말이야?",
    "ㅋㅋㅋㅋ",
    "너 이름이 뭐야?",
]

DEFAULT_SYSTEM_PROMPT = (
    "울산 지역어 대화 assistant로서 자연스럽고 일상적인 울산 사투리로 상대방과 친근하게 대화한다. "
    "자연스럽고 편안한 일상 울산 말투를 기본으로 구사한다."
)


def generate_response(model, tokenizer, prompt: str, seed: int = 42) -> str:
    set_seed(seed)
    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    template_kwargs = {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
    try:
        text = tokenizer.apply_chat_template(messages, **template_kwargs)
    except TypeError:
        template_kwargs.pop("enable_thinking", None)
        text = tokenizer.apply_chat_template(messages, **template_kwargs)

    inputs = tokenizer(text, return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()


def run_smoke_candidate(name: str, lr: float, beta: float, output_dir: str, report_json: str, steps: int = 50):
    cmd = [
        sys.executable,
        "scripts/train_dpo.py",
        "--model-name", "outputs/ulm-1.7b-phase4-sft-interim",
        "--output-dir", output_dir,
        "--learning-rate", str(lr),
        "--beta", str(beta),
        "--max-steps", str(steps),
        "--batch-size", "8",
        "--grad-accum", "2",
        "--logging-steps", "10",
        "--eval-steps", str(steps),
        "--save-steps", str(steps),
        "--report-json", report_json,
    ]
    print(f"\n>>> Running {name} (LR={lr}, beta={beta}, steps={steps})...")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Run DPO Smoke Benchmark")
    parser.add_argument("--base-model", default="outputs/ulm-1.7b-phase4-sft-interim")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--report-out", default="reports/phase4_dpo_smoke_benchmark_results.json")
    args = parser.parse_args()

    # 1. Candidate A: LR 5e-6, beta 0.1
    cand_a_dir = "outputs/ulm-1.7b-phase4-dpo-smoke-cand-a"
    cand_a_report = "reports/dpo_smoke_cand_a.json"
    run_smoke_candidate("Candidate A", lr=5e-6, beta=0.1, output_dir=cand_a_dir, report_json=cand_a_report, steps=args.steps)

    # 2. Candidate B: LR 1e-5, beta 0.1
    cand_b_dir = "outputs/ulm-1.7b-phase4-dpo-smoke-cand-b"
    cand_b_report = "reports/dpo_smoke_cand_b.json"
    run_smoke_candidate("Candidate B", lr=1e-5, beta=0.1, output_dir=cand_b_dir, report_json=cand_b_report, steps=args.steps)

    # Load metrics
    with open(cand_a_report, "r", encoding="utf-8") as f:
        metrics_a = json.load(f)
    with open(cand_b_report, "r", encoding="utf-8") as f:
        metrics_b = json.load(f)

    # 3. 17-prompt inference comparison
    print("\n=== Running 17-Prompt Smoke Inference Evaluation ===")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Baseline Model
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16, device_map="cuda:0")
    base_model.eval()

    base_responses = {}
    for p in SMOKE_PROMPTS:
        base_responses[p] = generate_response(base_model, tokenizer, p)

    # Candidate A Model
    model_a = PeftModel.from_pretrained(base_model, cand_a_dir)
    model_a.eval()
    cand_a_responses = {}
    for p in SMOKE_PROMPTS:
        cand_a_responses[p] = generate_response(model_a, tokenizer, p)
    del model_a

    # Candidate B Model
    model_b = PeftModel.from_pretrained(base_model, cand_b_dir)
    model_b.eval()
    cand_b_responses = {}
    for p in SMOKE_PROMPTS:
        cand_b_responses[p] = generate_response(model_b, tokenizer, p)
    del model_b

    inference_comparisons = []
    for p in SMOKE_PROMPTS:
        inference_comparisons.append({
            "prompt": p,
            "baseline": base_responses[p],
            "candidate_a": cand_a_responses[p],
            "candidate_b": cand_b_responses[p],
        })

    # Selection heuristic:
    # Check reward margin and loss stability
    margin_a = metrics_a.get("final_eval_rewards/margins", 0.0)
    margin_b = metrics_b.get("final_eval_rewards/margins", 0.0)
    acc_a = metrics_a.get("final_eval_rewards/accuracies", 0.0)
    acc_b = metrics_b.get("final_eval_rewards/accuracies", 0.0)

    selected = "Candidate B" if (margin_b > margin_a and acc_b >= acc_a * 0.95) else "Candidate A"
    selected_lr = 1e-5 if selected == "Candidate B" else 5e-6
    selected_beta = 0.1

    final_report = {
        "status": "PASS",
        "benchmark_steps": args.steps,
        "candidate_a": {
            "name": "Candidate A (LR=5e-6, beta=0.1)",
            "metrics": metrics_a,
        },
        "candidate_b": {
            "name": "Candidate B (LR=1e-5, beta=0.1)",
            "metrics": metrics_b,
        },
        "selected_candidate": selected,
        "selected_config": {
            "learning_rate": selected_lr,
            "beta": selected_beta,
            "lora_r": 16,
            "lora_alpha": 32,
            "batch_size": 8,
            "grad_accum": 2,
        },
        "inference_comparisons": inference_comparisons,
    }

    os.makedirs(os.path.dirname(args.report_out), exist_ok=True)
    with open(args.report_out, "w", encoding="utf-8") as f:
        json.dump(final_report, f, ensure_ascii=False, indent=2)

    print(f"\n==========================================")
    print(f"DPO SMOKE BENCHMARK SUMMARY:")
    print(f"Candidate A (LR 5e-6, beta 0.1):")
    print(f"  Loss: {metrics_a.get('train_loss'):.4f}, Margin: {margin_a:.4f}, Acc: {acc_a:.4f}, VRAM: {metrics_a.get('peak_vram_gib')} GiB")
    print(f"Candidate B (LR 1e-5, beta 0.1):")
    print(f"  Loss: {metrics_b.get('train_loss'):.4f}, Margin: {margin_b:.4f}, Acc: {acc_b:.4f}, VRAM: {metrics_b.get('peak_vram_gib')} GiB")
    print(f"\nSELECTED CANDIDATE: {selected} (LR={selected_lr}, beta={selected_beta})")
    print(f"Report saved to {args.report_out}")
    print(f"==========================================")


if __name__ == "__main__":
    main()
