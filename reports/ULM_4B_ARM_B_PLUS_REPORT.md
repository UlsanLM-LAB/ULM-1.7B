# ULM-4B Arm B+ experiment — fresh data

Date: 2026-09-28

Verdict: **REJECT — original Arm B remains the release candidate**.

This is a new, explicitly requested three-arm experiment. Earlier instruction-recovery-v2 data, adapters and results are preserved. The branch and EC2 worktree are `ulm4b-arm-b-plus` and `/home/ubuntu/ulm4b-arm-b-plus`, based on main commit `60f4a926ae0006dbf3fa40c553dc19b222b1078a`. No main merge or Live deployment is performed.

## Before and after

Comparable regression uses the archived greedy policy with thinking disabled; context uses repetition_penalty=1.1 and no_repeat_ngram_size=3. Separate serving instruction rows apply both Live decoding controls to every instruction prompt. All four instruction scores must reach 90% for acceptance.

| Metric | Original Arm B | Best fully evaluated candidate | Delta |
| --- | ---: | ---: | ---: |
| instruction_legacy_pct | 65.0 | 85.0 | 20.0 |
| instruction_corrected_pct | 60.0 | 80.0 | 20.0 |
| instruction_serving_legacy_pct | 45.0 | 55.0 | 10.0 |
| instruction_serving_corrected_pct | 30.0 | 55.0 | 25.0 |
| factual_qa_pct | 84.0 | 86.0 | 2.0 |
| multi_turn_pct | 100.0 | 92.0 | -8.0 |
| generation_semantic | 0.9843 | 0.9842 | -0.0001 |
| generation_dialectness | 0.6589 | 0.6615 | 0.0026 |
| grammar_semantic | 0.9846 | 0.9777 | -0.0069 |
| grammar_dialectness | 0.8955 | 0.8975 | 0.002 |
| context_semantic | 0.4578 | 0.4576 | -0.0002 |
| context_repetition_count | 0 | 0 | 0 |
| context_malformed_count | 0 | 0 | 0 |
| comprehension_semantic | 0.9802 | 0.9821 | 0.0019 |
| identification_accuracy | 0.32 | 0.34 | 0.02 |
| synthetic_final_hidden_pct | 57.04 | 90.37 | 33.33 |
| synthetic_dev_pct | — | 50.37 | — |
| general_regression_repetition_count | 2 | 3 | 1 |

Best experimental checkpoint: `/home/ubuntu/ulm4b-arm-b-plus/outputs/instruction-recovery-v2/arm_c/checkpoint-60` (arm c, step 60).

Failed gates:

- instruction_legacy_pct=85.0 < 90
- instruction_corrected_pct=80.0 < 90
- instruction_serving_legacy_pct=55.0 < 90
- instruction_serving_corrected_pct=55.0 < 90
- multi_turn_pct=92.0 < 95
- grammar_semantic delta -0.0069 < -0.005
- General-language runaway increased above the original Arm B

Remaining strict instruction failures by type: `{'exact repetition count': 2, 'yes/no': 2}`. The saved full predictions distinguish extra blank lines in repeated output from factually wrong yes/no polarity; these remain model failures after legitimate rejection expressions are recognized.

## Arms, initialization and runtime

| Arm | LR | Optimizer steps | Runtime (seconds, including fast gates) | Best corrected instruction | Early stop |
| --- | ---: | ---: | ---: | ---: | --- |
| a | 2e-06 | 80 | 513.9 | 65.0 | 80-step cap |
| b | 4e-06 | 80 | 475.3 | 70.0 | 80-step cap |
| c | 7e-06 | 80 | 490.6 | 80.0 | general response runaway increased |

Initial adapter: `/home/ubuntu/models/ULM-4B-Arm-B`. Historical and release assets have identical weight hashes; the release path is loaded directly for every arm. `initial_load.json` and `INITIAL_TRAINABLE` log lines verify all 496 inherited LoRA tensors exactly match the release asset, only LoRA tensors require gradients, and `inference_mode=false`. The base is frozen.

GPU: NVIDIA L40S, 46068 MiB, g6e.xlarge in ap-northeast-2. Python environment: `/home/ubuntu/ULM-1.7B/.venv/bin/python`; torch 2.14.0+cu130, transformers 5.17.0, peft 0.20.0, trl 1.13.0. Existing reference kernels are reused; the runtime environment is unchanged.

Continued completion-only SFT: inherited LoRA r=8/alpha=16/dropout=.05 on 12 projections; AdamW, BF16, batch 8 × accumulation 2, max length 1024, cosine LR, 2 warmup steps, weight decay .01. Checkpoints and fast gates occur every 20 steps, at most 80 steps per arm. Actual serialized training arguments are saved with each arm.

LRs were adjusted to 2e-6/4e-6/7e-6 using the prior failed experiment as evidence: 2e-7 barely moved the adapter and did not improve corrected instruction. An additional general-language runaway gate rejects any increase over the fresh fast baseline.

## Dataset and leakage audit

Synthetic total: 1620. Categories: `{'exact': 240, 'yes_no': 240, 'repeat': 180, 'list': 240, 'json': 180, 'extraction': 120, 'false_premise': 300, 'stopping': 120}`. Splits: `{'train': 1080, 'dev': 270, 'hidden': 270}`. families 0–7 train, 8–9 dev, 10 monitor, 11 final hidden; final family is never monitored.

Training mixture: 2160 examples, `{'new_instruction': 1080, 'dialect': 540, 'factual': 270, 'context': 216, 'legacy_instruction': 54}`. New instruction 50%, dialect 25%, factual 12.5%, context 10%, legacy instruction 2.5%. Only generation/grammar/comprehension dialect replay is selected, together with natural context examples. The previous artificial anti-echo system hint is removed; 25% use the existing Live dialect system prompt.

Dialect target-token share: 32.00%; new instruction share: 34.97%. All 2160 completions supervise EOS. Maximum sequence length 213; truncations 0. A real SFTTrainer batch verifies ignored prompt labels and supervised completions/EOS before training.

All 1,620 synthetic prompts are unique; none is reused verbatim from the prior synthetic run. Exact benchmark prompt and shared normalized 24-character expression overlaps are zero. All 170 UlsanBench source sessions are excluded from replay. Train/dev/monitor/final template families are disjoint. Yes/no targets are balanced in each split. Required structural tokens such as 네/아니오, numbering and repetition counts are shared output grammar; benchmark names, claims and copied answer payloads are excluded. Four generated JSONL hashes match the independently generated AWS training inputs.

## Failure analysis and evaluator

Fresh original instruction scores: legacy 65.0%, corrected 60.0%. False-negative IDs: [134]; false-positive IDs: [82, 131]. All 20 predictions and their failure types are saved in `failure_analysis.json`.

Exact words, yes/no polarity, single-newline repetition, numbered output, extraction and valid JSON are checked separately. Rejection recognizes multiple explicit denial forms, while premise acceptance, fabricated continuations and repeated denial loops fail. Both legacy and corrected scores remain available; adversarial evaluator tests passed. Candidate predictions are recorded in `final_failure_analysis.json`.

A denial followed by invented chronology is not a valid factual correction. Gang Gam-chan lived in 948–1031; answers placing him in the 12th–13th centuries fail. [Korean Academy encyclopedia](https://encykorea.aks.ac.kr/Article/E0000954).

The native installed Transformers no-repeat processor was tested against both exact repetition targets. It blocks a required token in each target, including a token already present in the input prompt. `decoding_contract_audit.json` records the blocked positions. Thus these two exact targets are unreachable under the current global no-repeat policy, regardless of adapter preference. This proof concerns the benchmark prompt/template and does not change Live.

Failures are consistent with incomplete transfer of strict formatting, polarity and genuine premise denial, plus preservation risk in unconstrained general-language lists. These are observations and hypotheses, not isolated causal ablations. Original full regression has two general-language loops and one instruction loop; only identical shared categories are compared. Ten supplemental archived dialect probes are excluded; dialect preservation uses the identical 500-item UlsanBench.

## Decision and DPO condition

Selected release arm: None. DPO condition reached: True; DPO executed: False. No PPO/GRPO is run. Any DPO is limited to one separate small experiment after all three SFT arms and only when no SFT checkpoint reaches corrected strict instruction 90%.

Considered, not executed: keep the explicitly requested three-arm SFT experiment bounded. Strict serving repetition targets are prohibited by the existing native no-repeat processor; preference training cannot make prohibited tokens available. Saved SFT results remain available for a separately designed preference experiment.

Test a narrowly scoped decoding exception for explicit exact-output contracts on new held-out prompts, while retaining the existing loop controls for ordinary conversation. This is a proposal only; no Live policy is changed in this experiment.

## Artifacts and operations

Dataset summary, per-arm results, final comparison, decision, release config and model-card material are saved. Data, raw transcripts, optimizer states and weights remain on persistent EC2 EBS; original and candidate adapter/tokenizer hashes are backed up locally before shutdown. Git contains code, config and summaries only. Accepted naming/export occurs only after the gate decision; rejected weights keep their research checkpoint names.

Git push proof and the actual final EC2 state are recorded in `operations.json`. The required completion action is StopInstances, never TerminateInstances, after all GPU/train/evaluation processes exit and the original protected files are reverified.

## Known limitations

- The 20 instruction traps and declared UlsanBench subsets were monitored as requested; they are not untouched final benchmarks. No benchmark prompt or target was copied into synthetic training data.
- Final synthetic template family 11 is withheld from all training, dev evaluation and checkpoint monitoring; monitor items come only from family 10.
- Conservative rejection expressions plus manual prediction review are used. These are not universal semantic entailment checks.
- Serving-policy instruction scores apply the Live decoding values to benchmark prompts; they are not a test of the Live HTTP endpoint or its exact system prompt.
- The existing global no-repeat-ngram rule conflicts with verbatim repetition and prompt copying. Training cannot override forbidden logits. Live decoding is unchanged.
- UlsanBench embedding and dialectness values are existing evaluator proxies, not human judgments.
- Archived full baseline is reused only after all protected release files match their SHA256 manifest; fresh instruction and fast preservation baselines are also measured.
