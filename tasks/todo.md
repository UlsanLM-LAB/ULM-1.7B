# Phase 3 v3 Retraining Status & Handoff (2026-09-25)

## Current State
- [x] Preserved Codex's `reports/PHASE3_V3_SPEED_OPTIMIZATION.md`
- [x] Completed full 14-point failure diagnosis for Step 70 failure:
  - Root cause 1: Fatal loss masking bug (`assistant_only_loss=False` in TRL SFTTrainer left full user prompt, special tokens, and `<think>` tags unmasked in loss).
  - Root cause 2: Total absence of replay (0.0% factual and 0.0% memory/instruction replay in `data/ulsan_dialect_phase3_v3/train.jsonl`).
  - Root cause 3: Excessive template repetition in Categories C & D (openings repeated up to 177 times).
  - Root cause 4: High learning rate (5e-5) accelerated catastrophic forgetting.
- [x] Recorded diagnosis in `reports/PHASE3_V3_FAILURE_ANALYSIS.md`.
- [x] Generated decontaminated rebalanced mixture:
  - Path: `data/ulsan_dialect_phase3_v3_rebalanced/train.jsonl` (5,600 train, 400 val)
  - Composition: Dialect 45.04%, Factual replay 40.04%, Instruction/Memory replay 14.93%
  - Zero leakage against 110 holdout benchmark prompts.
- [x] Implemented pilot ablation pipeline: `scripts/run_pilot_ablations.py` (Pilot A, B, C, D).
- [x] Safely stopped active pilot execution upon user request (Pilot A stopped at step 3/35).
- [x] Preserved failed `checkpoint-70` at `/opt/dlami/nvme/phase3-v3-checkpoints/checkpoint-70`.
- [x] Revoked temporary SSH ingress rule (`106.101.136.35/32`).
- [x] Verified all GPU processes terminated and VRAM at 0 MiB.
- [x] EC2 instance `i-0f732bf7d1cc409b4` kept RUNNING.

## Next Action Tomorrow
1. Run the 4-way pilot ablation study (35 steps each, effective batch 32, ~6 min per candidate):
   - Command: `python scripts/run_pilot_ablations.py`
   - Pilot A: LR 5e-5, raw v3 dataset, assistant_only_loss=False (baseline reproduction)
   - Pilot B: LR 1.5e-5, raw v3 dataset, assistant_only_loss=False (lower LR alone)
   - Pilot C: LR 1.5e-5, rebalanced mixture (45/40/15), assistant_only_loss=False
   - Pilot D: LR 1.5e-5, rebalanced mixture (45/40/15), assistant_only_loss=True (strict assistant masking)
2. Compare pilot outcomes on 60-prompt preservation gate (`gate-60.jsonl`).
3. Select the safest passing candidate (expected: Pilot D) and initiate full Phase 3 v3 training from base `Qwen3.8-4B-Distill`.
4. Monitor quality gates at step 70 and step 140 (target: Factual >= 80%, Memory >= 85%, Instruction >= 95%, Dialect >= 60%).
# ULM-4B Phase3 v3 recovery (2026-09-25, 3-hour deadline)

- [x] Start EC2, obtain current IP, inspect disk/GPU/repository state.
- [x] Confirm root causes and prepare response-only loss plus replay mixtures.
- [x] Run at most two 25-step pilots on the same 40-prompt gate.
- [x] Train a fresh run from base; stop at failed step 70 (no step 140/210).
- [x] Select safer step-25 pilot, merge in FP32, evaluate 150+50, check 20/20 parity.
- [x] Update reports, commit/push code and small reports, remove temporary SSH rule, leave EC2 running.

## Review

The step-70 full run failed its dialect and instruction gate and stopped.
The safer step-25 B adapter scored factual 92%, memory 84%, instruction 100%,
and dialect 16% on regression-150; dialect holdout-50 was 8%. FP32 merge
passed 20/20 parity. Final verdict: FAIL against the dialect objective.
