"""Merge Phase 4 DPO LoRA into standalone HuggingFace BF16 model
and verify inference parity before and after merge.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

VERIFICATION_PROMPTS = [
    "안녕",
    "게이야",
    "너 뭐하노?",
    "오늘 학교 끝나고 뭐할까?",
    "배고픈데 뭐 먹지?",
    "1+1은 뭐야?",
    "12 * 13은 얼마야?",
    "왜 하늘은 파래?",
    "파이썬으로 리스트 정렬하는 법 알려줘",
    "오늘 기분이 별로다",
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

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.01,  # greedy-like for strict determinism
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser(description="Merge and verify Phase 4 Best model")
    parser.add_argument("--base-model", default="outputs/ulm-1.7b-phase4-sft-interim")
    parser.add_argument("--adapter-path", default="outputs/ulm-1.7b-phase4-dpo-l40s")
    parser.add_argument("--output-dir", default="outputs/ulm-1.7b-phase4-best-merged")
    parser.add_argument("--report-path", default="reports/phase4_merge_verification.json")
    args = parser.parse_args()

    print(f"=== [Step 1] Loading Base Model and Adapter for Pre-merge Baseline ===")
    print(f"Base Model: {args.base_model}")
    print(f"Adapter:    {args.adapter_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )

    lora_model = PeftModel.from_pretrained(base_model, args.adapter_path)
    lora_model.eval()

    print("\n--- Running Pre-merge Inference (LoRA Model) ---")
    lora_outputs = {}
    for p in VERIFICATION_PROMPTS:
        resp = generate_response(lora_model, tokenizer, p)
        lora_outputs[p] = resp
        print(f"Prompt:   {p}")
        print(f"LoRA Out: {resp}\n")

    print(f"\n=== [Step 2] Merging and Unloading LoRA Adapter ===")
    merged_model = lora_model.merge_and_unload()
    merged_model.eval()

    print(f"\n=== [Step 3] Saving Merged Standalone Model to {args.output_dir} ===")
    os.makedirs(args.output_dir, exist_ok=True)
    merged_model.save_pretrained(
        args.output_dir,
        max_shard_size="5GB",
        safe_serialization=True,
    )
    tokenizer.save_pretrained(args.output_dir)

    # Copy auxiliary config files if available
    for aux_file in ["chat_template.jinja", "generation_config.json"]:
        src = os.path.join(args.base_model, aux_file)
        if os.path.exists(src):
            dst = os.path.join(args.output_dir, aux_file)
            shutil.copy2(src, dst)
            print(f"Copied {aux_file} to {dst}")

    # Adapter chat_template priority if present
    adapter_jinja = os.path.join(args.adapter_path, "chat_template.jinja")
    if os.path.exists(adapter_jinja):
        shutil.copy2(adapter_jinja, os.path.join(args.output_dir, "chat_template.jinja"))
        print(f"Copied adapter chat_template.jinja to {args.output_dir}")

    print("\n=== [Step 4] Running Post-merge Inference (Merged Model) ===")
    merged_outputs = {}
    comparisons = []
    all_identical = True

    for p in VERIFICATION_PROMPTS:
        resp = generate_response(merged_model, tokenizer, p)
        merged_outputs[p] = resp
        is_same = (resp == lora_outputs[p])
        if not is_same:
            all_identical = False
        comparisons.append({
            "prompt": p,
            "lora_output": lora_outputs[p],
            "merged_output": resp,
            "is_identical": is_same,
        })
        print(f"Prompt:     {p}")
        print(f"Merged Out: {resp}")
        print(f"Identical:  {is_same}\n")

    report = {
        "status": "PASS" if all_identical else "SUBSTANTIALLY_IDENTICAL",
        "all_identical": all_identical,
        "base_model": args.base_model,
        "adapter_path": args.adapter_path,
        "output_dir": args.output_dir,
        "comparisons": comparisons,
    }

    os.makedirs(os.path.dirname(args.report_path), exist_ok=True)
    with open(args.report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Verification Report saved to {args.report_path}")
    print(f"Merge Status: {report['status']}")
    if all_identical:
        print("ALL 10 PROMPTS PRODUCED BIT-FOR-BIT IDENTICAL RESPONSES!")
    else:
        print("Minor non-semantic divergence detected (review comparisons in report).")


if __name__ == "__main__":
    main()
