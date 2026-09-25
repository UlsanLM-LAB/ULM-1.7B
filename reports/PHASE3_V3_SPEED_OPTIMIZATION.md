# Phase 3 v3 throughput optimization

## Recovery run timebox update (2026-09-25)

The recovery uses the measured micro-batch 8 / accumulation 4 configuration,
SDPA, gradient checkpointing, and no packing. The earlier micro-batch 16 trial
was slower (9.770 versus 6.157 seconds per step), so it was not repeated.
Intermediate scoring uses a fixed 40-prompt subset (12 factual, 8 memory,
8 instruction, 12 dialect) instead of the original 60 prompts. Pilot and
checkpoint gates use the same subset and decoding parameters. Base on this
subset: 91.7%, 87.5%, 100.0%, 25.0%, respectively. Only the selected best
checkpoint receives the full 150-prompt regression and 50-prompt dialect
holdout evaluation.

Measured pilot costs on the masked, mixed data: A trained 25 optimizer steps
in 114.42 seconds and scored 40 prompts in 271.93 seconds; B trained in
140.96 seconds and scored in 391.05 seconds. B's longer generated responses
(110.1 versus 78.4 tokens on average) explain most of its slower gate. These
measured gate times are used for the remaining deadline decisions.

The selected B full run stopped at step 70 after **453.9 seconds** including
the **93.7-second** gate; peak reserved VRAM was **16.381 GiB**. No step 140
or 210 was run. The final FP32 merged model averaged 5,126 ms and 115.1
tokens on regression-150, and 5,429 ms and 122.1 tokens on dialect-50.
FP32 was required to obtain 20/20 adapter-versus-merged greedy parity; a
BF16 merge passed only 5/20 because small adapter deltas were rounded during
the merge. The FP32 artifact is approximately 16 GiB on EBS.


Date: 2026-09-25. EC2 `i-0f732bf7d1cc409b4`, NVIDIA L40S, 46 GB.

## Storage and starting point

Root EBS gp3 `vol-07580a7ba8e092248` grew from 180 to 210 GiB. `growpart`
and online `resize2fs` completed; `df -hT /` showed 204 GiB usable and 35 GiB
free. No Phase 3 v3 training process or checkpoint existed on EBS or NVMe.
The first v3 run therefore starts from the original
`/home/ubuntu/models/Qwen3.8-4B-Distill` base model. Phase 3 v2 checkpoints
and merged models were retained.

## Comparable short workload

Each non-packing trial used the same first 96 v3 training examples, 3 optimizer
steps, effective batch 32, BF16, 2048 maximum length, LoRA rank 16/alpha 32,
and the existing cosine schedule configuration. Measured useful tokens exclude
batch padding; GPU utilization is the mean of 1-second `nvidia-smi` samples.
The training dataset has 8,997 examples: p50 68, p90 132, p99 699, maximum
794 tokens. 8,601 examples are at most 256 tokens.

| Trial | Micro | Accum | Checkpointing | sec/step | samples/sec | useful tokens/sec | mean GPU % | peak reserved GiB |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Baseline A | 2 | 16 | ON | 11.827 | 2.706 | 257.462 | 28.8 | 10.064 |
| B | 4 | 8 | ON | 6.920 | 4.624 | 440.002 | 56.6 | 10.924 |
| **C selected** | **8** | **4** | **ON** | **6.157** | **5.198** | **494.579** | **92.2** | **13.400** |
| D | 16 | 2 | ON | 9.770 | 3.275 | 311.681 | 97.1 | 18.533 |
| B/C/D | 4/8/16 | 8/4/2 | OFF | OOM | — | — | — | ~44.4 GiB at failure |
| C, length grouping | 8 | 4 | ON | 6.413 | 4.990 | 474.793 | 76.3 | 13.494 |
| C, two loader workers | 8 | 4 | ON | 6.255 | 5.116 | 486.836 | 90.7 | 13.398 |

The selected training-only speedup is **1.92×** (494.579 / 257.462 useful
tokens/sec). Its peak VRAM is below the 42 GB limit. GPU activation memory in
the hybrid GatedDeltaNet model made all tested checkpointing-OFF candidates
run out of memory, despite the much lower checkpointing-ON peaks.

## Sequence and kernel decisions

- Packing: **NO**. A disposable packed run reached 678.875 useful tokens/sec,
  but packs many original samples into each block, so it changes the meaning
  of effective batch 32. TRL 1.13 also warns that its padding-free BFD packing
  with SDPA may allow attention across sample boundaries. Packing speedup is
  **2.64× raw token throughput versus baseline A**, but it is ineligible for
  v3 training on this backend.
- Length-aware batching: **OFF**; measured 474.793 tokens/sec versus 494.579
  without grouping.
- Dynamic padding: **ON** through the standard SFT data collator.
- Tokenization cache: Hugging Face datasets preprocessing cache (default).
- DataLoader: `num_workers=0`, `pin_memory=True`, `persistent_workers=False`,
  `prefetch_factor=None`. Two workers were slower on this short dataset.
- Gradient checkpointing: **ON**.
- Attention backend: **SDPA**. Neither FlashAttention, flash-linear-attention,
  nor causal-conv1d is installed. No new kernel was installed for the hybrid
  Qwen3.5 architecture.

## Gate and completion

The intermediate gate uses 60 unchanged regression prompts: factual 20,
multi-turn memory 10, instruction/trap 10, dialect 20. The original v2 gate
used 100 prompts with 380.477 seconds of summed generation latency at
checkpoint 70. The new 60-prompt base evaluation took 414.599 seconds of
summed generation latency (2.12 seconds model load); its scores were factual
95%, memory 90%, instruction 100%, dialect 20%. This base run is not a fair
wall-clock comparison with the old trained-adapter gate because base outputs
are longer. The step-70 v3 adapter gate took **151.87 seconds wall time**
(148.919 seconds summed generation latency), a **2.55×** reduction in
generation time versus the old checkpoint-70 gate's 380.477 seconds.
The first v3 training run started at global step 0, with 564 planned steps.
At 20 steps, loss fell from 3.313 to 2.817 and GPU memory use was about
14.7 GB. Its step-70 gate failed: factual 70%, memory 50%, instruction 70%,
dialect 20% (base on the same 60 prompts: 95%, 90%, 100%, 20%). Because dialect
did not improve while preservation fell, training was interrupted after the
fully resumable step-70 checkpoint had reached both NVMe and EBS. The process
had displayed step 81; those later unsaved updates were not retained.

Selected micro batch: **8**. Selected gradient accumulation: **4**. Packing:
**NO**. Gradient checkpointing: **ON**. Attention backend: **SDPA**. Peak
short-trial VRAM: **13.4 GiB reserved**. New gate runtime: **151.87 s**. Old
gate runtime: **380.477 s**. A full-run projection at 564 steps and eight
gates is approximately **2.07× overall speedup**; the quality stop makes
this projection inapplicable to the first run's actual duration. The latest
resumable checkpoint is step 70; no passing best checkpoint exists and there
is no remaining runtime for this stopped run.
