# ULM-4B Arm B final low-LR experiment

**Rejected. Preserve original Arm B.** One run only; no full UlsanBench evaluation, further arms or preference training. Live and main were not changed.

## Run and data

- Started from immutable original Arm B (`fee03f872aaff55283be0413536e20b0930334333a20c6ee01cad63f57f43716`); 496 LoRA tensors verified exactly equal before training. Base frozen; inherited r8/alpha16/dropout0.05.
- LR **7e-7**, cosine, warmup2, AdamW, BF16, batch8 × accumulation2, completion-only loss, EOS supervision, max_length1024. **60 steps**, checkpoint20/40/60. Loss 0.6794; 960 examples consumed (39.54% of one epoch).
- Reused all 2,160 existing rows. Added **160 yes/no examples** (80 balanced contrastive pairs: arithmetic and supplied-record consistency) and **108 context replay** rows. Total2,428; context replay share10%→13.34%.
- Dev/hidden files unchanged. Added prompts audited against benchmarks and held-out prompts for normalized exact and24-character overlap; all170 held-out UlsanBench sessions excluded. No benchmark subjects/answers used in new yes/no examples.

## Checkpoint results

Instruction20, factual50 and multi-turn25 are evaluated in full. UlsanBench values below are fixed screening subsets (generation16, grammar8, context12), **not full benchmark scores**.

| Step | Legacy instruction | Corrected | Factual | Multi-turn | Gen semantic* | Gen dialectness* | Context semantic* | Regression repeats | UB repeats / malformed* |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Original | 65% | 60% | 84% | 100% | 0.9832 | 0.6375 | 0.4230 | 1 | 0 / 0 |
| 20 | 70% | 60% | 84% | 92% | 0.9832 | 0.6375 | 0.4123 | 2 | 0 / 0 |
| 40 | 75% | 60% | 84% | 96% | 0.9832 | 0.6375 | 0.4542 | 0 | 0 / 0 |
| 60 | 75% | 60% | 84% | 92% | 0.9832 | 0.6375 | 0.4402 | 0 | 0 / 0 |

*Subset metrics. Absolute requested thresholds were applied conservatively to this screen. Original Arm B itself has subset dialectness0.6375 and context0.4230, whereas its archived full scores are0.6589 and0.4578. Subset misses therefore do not establish full-benchmark failure. **Every candidate independently fails the complete instruction gate**, so this sampling limitation does not affect rejection or the decision to skip full evaluation.

Acceptance thresholds: legacy/corrected instruction≥90%, multi-turn≥95%, factual≥80%, generation semantic≥0.98/dialectness≥0.65, context semantic≥0.44, repetition/malformed=0. At most one passing checkpoint would receive full evaluation; **none passed, zero full evaluations performed**.

## Remaining failures

At step40, the eight corrected instruction misses comprise two numbered-list formatting errors, two repeat outputs with extra blank lines, two wrong yes/no polarity answers, and two false-premise responses. These are evaluation findings; benchmark content was not added to training. Legacy75% does not imply corrected75%: corrected scoring is60%.

## Verification and preservation

- Gate boundary/NaN/repetition/malformed and existing adversarial rubric tests:4 passed. Actual configuration and dataset/checkpoint hashes checked.
- Protected original/Live files:29 hash matches after completion. Training process ended; GPU memory0MiB.
- checkpoint20/40/60 and final adapter preserved on persistent EC2 EBS and in local `outputs/instruction-recovery-final-backup/adapter-backup.tar.gz`; all four local adapter hashes verified. Raw predictions and private datasets backed up locally and excluded from Git.
- Source/config/summary results are committed on `ulm4b-arm-b-plus`. AWS shutdown and push evidence: `instruction-recovery-final/operations.json`.
