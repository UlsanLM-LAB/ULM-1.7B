# Arm B+ experiment model-card material

Status: rejected. Name: ULM-4B instruction candidate (rejected).

Lineage: Qwen3.8-4B-Distill → Dialect Alignment v3 → Context Repair v1 Arm B → Instruction Recovery v1 Arm B → fresh-data continued SFT.

Research purpose: improve strict instruction following while preserving Ulsan dialect, factual knowledge and conversation. Existing Arm B remains the Live release candidate. No automatic deployment occurs.

Training: 2160 examples, 50% new instruction and 50% replay, completion-only LoRA continuation; three independent short arms. Initial adapter: /home/ubuntu/models/ULM-4B-Arm-B.

Measured results, rejection reasons, legacy/corrected/serving policies and limitations are in reports/ULM_4B_ARM_B_PLUS_REPORT.md and decision.json. Weights and original datasets are not uploaded to GitHub; base-model and source-data obligations remain inherited.
