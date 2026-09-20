"""Merge Corrective SFT LoRA adapter into interim base model for DPO."""

import os
import shutil
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "outputs/ulm-1.7b-phase3-best-merged"
ADAPTER = "outputs/ulm-1.7b-phase4-corrective-sft"
OUTPUT_DIR = "outputs/ulm-1.7b-phase4-sft-interim"

print(f"Loading Base: {BASE_MODEL}")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
)

print(f"Loading Adapter: {ADAPTER}")
lora_model = PeftModel.from_pretrained(base_model, ADAPTER)
merged_model = lora_model.merge_and_unload()
merged_model.eval()

print(f"Saving Merged Interim Model to {OUTPUT_DIR}")
os.makedirs(OUTPUT_DIR, exist_ok=True)
merged_model.save_pretrained(OUTPUT_DIR, max_shard_size="5GB", safe_serialization=True)
tokenizer.save_pretrained(OUTPUT_DIR)

for f in ["chat_template.jinja", "generation_config.json"]:
    src = os.path.join(BASE_MODEL, f)
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(OUTPUT_DIR, f))

print("Interim Model Merge Complete!")
