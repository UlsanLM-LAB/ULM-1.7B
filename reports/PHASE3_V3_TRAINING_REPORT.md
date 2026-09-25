# ULM-4B Phase 3 v3 recovery training report

Date: 2026-09-25. AWS EC2 `i-0f732bf7d1cc409b4` (`ap-northeast-2`, NVIDIA L40S). The run began from the locally stored original Qwen3.8-4B-Distill model. The failed, unmasked step-70 checkpoint was excluded from resume and selection.

## Method

- Loss: final assistant completion and `<|im_end|>` only. Runtime training-batch assertions confirmed prompt labels are masked. Inference uses the matching Qwen non-thinking chat header.
- Candidates: A, LR 1.5e-5, 45/40/15 dialect/factual-general/memory-instruction samples; B, LR 1.0e-5, 50/35/15. Both use micro-batch 8, accumulation 4, effective batch 32, SDPA, gradient checkpointing, no packing, and the same 40-prompt pilot gate.
- Baseline on that gate: factual 11/12 (91.7%), memory 7/8 (87.5%), instruction 8/8 (100%), dialect 3/12 (25.0%). The 40-prompt gate is a fixed subset of the previously measured 60 prompts, using the same evaluator and decoding settings.
- Full training begins again at Base. Gates occur at step 70, 140, and at 210 only if the preceding gate shows improvement. Final regression-150 and dialect holdout-50 are run once on the selected checkpoint.

Evaluation caveat: the existing 50-item dialect file shares 12 prompts with
the selected 40-prompt gate and 25 with regression-150. It has zero exact
prompt overlap with training data, but is not fully independent of checkpoint
selection. The final report therefore also gives the score for its 38 items
outside the gate.

## Pilot results

| Candidate | Steps | Factual | Memory | Instruction | Dialect | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Base | 0 | 91.7% | 87.5% | 100.0% | 25.0% | Reference |
| A | 25 | 91.7% | 87.5% | 100.0% | 25.0% | Preservation held; no dialect gain |
| B | 25 | 91.7% | 87.5% | 100.0% | 25.0% | Selected: same scores, lower LR |

Neither pilot increased the 40-prompt dialect score. A trained for 114.42 seconds
and evaluated for 271.93 seconds; its average response length was 78.4 tokens.
B averaged 110.1 tokens per response and 9,706.5 ms per prompt, versus A's
6,729.1 ms. The full-run step-70 gate is required to establish whether more
updates improve dialect without degrading preservation.

## Full-run gates and selection

The selected B configuration restarted from Base with a 210-step ceiling and
an initial gate at 70. Training stopped at step 70 after that gate failed:

| Checkpoint | Factual | Memory | Instruction | Dialect | Avg tokens | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| B pilot step 25 | 91.7% | 87.5% | 100.0% | 25.0% | 110.1 | Best safe candidate |
| B full step 70 | 91.7% | 87.5% | 87.5% | 16.7% | 26.0 | Rejected |

The step-70 run took 453.9 seconds, including its 93.7-second gate. Its peak
reserved GPU memory was 16.381 GiB. There was no reason to continue to step
140: dialect fell 8.3 percentage points below Base and instruction adherence
fell 12.5 points. The step-70 optimizer state remains in the one latest EBS
checkpoint for audit and resumption metadata, but it is explicitly excluded
from final model selection. The step-25 pilot adapter is the best candidate
under the requested preservation-first tie rule. It is not claimed to meet
the dialect target.

The step-70 instruction miss was substantive: on a deliberately false
"Sejong as president" question, it fabricated another president and policies
instead of rejecting the premise. Its mean gate answer length collapsed from
the B pilot's 110.1 tokens to 26.0. On dialect items, three earlier hits were
lost and two new hits appeared, yielding a net decline from 3/12 to 2/12.

## Final regression comparison

| Model | Factual | Memory | Instruction | Dialect | Repetition | Malformed output | Avg latency | Avg tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 86.0% | 92.0% | 100.0% | 20.0% | 7/150 | 1/150 | 7,690 ms | 156.4 |
| Phase 3 v2 merged | 78.0% | 84.0% | 95.0% | 20.0% | 1/150 | 0/150 | 1,906 ms | 39.2 |
| Phase 3 v3 merged (FP32) | 92.0% | 84.0% | 100.0% | 16.0% | 8/150 | 0/150 | 5,126 ms | 115.1 |

The Base and v2 rows reuse their existing regression-150 results under the same evaluator and decoding policy. Malformed output means an empty final answer or Unicode replacement character; token-limit counts are reported separately below.

Token-limit counts were 49/150 for Base, 2/150 for v2, and 34/150 for v3.
V3 factual accuracy exceeded Base by 6 points and instruction accuracy matched
Base. Memory remained 8 points below Base and 1 point below the 85% target.
Dialect fell 4 points below Base on these 25 prompts. Repetition was 8/150,
one more than Base's 7/150. This does not meet the dialect objective.

| Acceptance target | Required | V3 result | Status |
| --- | ---: | ---: | --- |
| Factual | 80% | 92% | PASS |
| Memory | 85% | 84% | FAIL |
| Instruction | 95% | 100% | PASS |
| Dialect | 60% | 16% | FAIL |

There is no broad factual or instruction collapse in the selected pilot, but
memory remains 8 percentage points below Base. The model is **FAIL** for the
Phase 3 v3 objective because dialect deteriorated and two targets were missed.
Catastrophic forgetting: **No** under the preservation definition used for
this run (factual +6 points, instruction unchanged, memory -8 points versus
Base); the memory target is nonetheless missed.

## Dialect holdout, merge parity, storage, and verdict

The first BF16 merge passed only 5/20 greedy parity prompts despite matching
the adapter's first token on all five diagnostic prompts. First-token logit
cosine similarities were 0.9981–0.9994, but maximum absolute differences were
0.48–0.78; later tokens diverged. Adapter LoRA tensors are FP32. Re-merging
the base and adapter in FP32 and reloading the saved standalone merged model
passed **20/20 greedy parity prompts (100%)**. The final merged artifact is
therefore FP32 (about 16 GiB); final evaluation explicitly loads it in FP32.
The rejected BF16 merge is on disposable NVMe, while the final adapter and
FP32 merged model are on EBS. This precision change is a comparison caveat
against the existing BF16 Base and v2 benchmark rows.

The final dialect set scored **4/50 (8.0%)**. The 38 items outside the
checkpoint-selection gate scored **2/38 (5.3%)**; overlapping gate items
scored 2/12 (16.7%). These results reinforce the regression-150 finding that
the selected adapter has not acquired useful dialect behavior.

Final EBS artifacts:

- Adapter: `/home/ubuntu/ULM-1.7B/outputs/ulm-4b-phase3-v3-lora`
- Merged FP32: `/home/ubuntu/ULM-1.7B/outputs/ulm-4b-phase3-v3-best-merged`
- Best adapter backup: `outputs/ulm-4b-phase3-v3/best-checkpoint-25`
- Latest resumable but rejected: `outputs/ulm-4b-phase3-v3/latest-checkpoint-70`

**Final verdict: FAIL.** The artifacts are valid and the merge passes parity,
but the model should not replace Base or be used as a successful dialect
checkpoint. No Phase 4 work was started. The EC2 instance remains running.
