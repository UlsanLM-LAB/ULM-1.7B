"""Diagnose adapter/merged first-token parity for the selected v3 model."""

import json

import torch
from peft import PeftModel
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

from merge_and_verify_phase3_v2 import PARITY_PROMPTS, format_prompt

BASE = "/home/ubuntu/models/Qwen3.8-4B-Distill"
ADAPTER = "outputs/ulm-4b-phase3-v3-lora"
MERGED = "outputs/ulm-4b-phase3-v3-best-merged"

with safe_open(ADAPTER + "/adapter_model.safetensors", framework="pt") as file:
    key = next(iter(file.keys()))
    print("adapter weight dtype", file.get_tensor(key).dtype, key, flush=True)

tokenizer = AutoTokenizer.from_pretrained(BASE)
base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda:0")
adapter = PeftModel.from_pretrained(base, ADAPTER).eval()
merged = AutoModelForCausalLM.from_pretrained(MERGED, dtype=torch.bfloat16, device_map="cuda:0").eval()

for item in PARITY_PROMPTS[:5]:
    tokens = tokenizer(format_prompt(tokenizer, item), return_tensors="pt").to("cuda:0")
    with torch.inference_mode():
        a = adapter(**tokens).logits[0, -1].float()
        b = merged(**tokens).logits[0, -1].float()
    aid, bid = a.argmax().item(), b.argmax().item()
    print(json.dumps({"id": item["id"], "adapter_token": tokenizer.decode([aid]),
        "merged_token": tokenizer.decode([bid]), "cosine": round(torch.nn.functional.cosine_similarity(a, b, dim=0).item(), 7),
        "max_abs_diff": round((a - b).abs().max().item(), 6),
        "adapter_margin": round((a.topk(2).values[0] - a.topk(2).values[1]).item(), 6)}, ensure_ascii=False), flush=True)
