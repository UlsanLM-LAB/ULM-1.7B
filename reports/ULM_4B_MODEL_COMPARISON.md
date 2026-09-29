# ULM-4B UlsanBench v2 Model Comparison Report

- **Date**: 2026-09-28
- **Benchmark**: UlsanBench v2 (500 items, held-out speaker-session disjoint test set)
- **Evaluation Environment**:
  - AWS Region: `ap-northeast-2`
  - EC2 Instance: `i-0f732bf7d1cc409b4` (`g6e.xlarge`)
  - GPU: NVIDIA L40S 46GB VRAM (Driver: 595.91.07, CUDA: 13.2)
  - Software: PyTorch 2.14.0+cu130, Transformers 5.17.0, PEFT 0.20.0
- **Evaluator**: Sentence embedding model (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`) + lexical marker and ending rule matchers.
- **Decoding Protocol**: Neutral deterministic evaluation (`do_sample=false`, `temperature=0`, `enable_thinking=false`). A supplementary **Context Guard** ablation applies `repetition_penalty=1.10` and `no_repeat_ngram_size=3` only to the context task; it is not a full deployed-mode rerun.

---

## 1. Executive Summary

This report documents the empirical benchmark comparing **ULM-4B Arm B** against its un-adapted base model (**Qwen3.8-4B-Distill Base**) and representative open-weights LLMs on the canonical 500-item UlsanBench v2 suite.

All models were evaluated on the **exact same hardware**, **identical prompts and system intent**, and **unchanged scoring pipeline**. No model-specific few-shot prompting or cherry-picked test splits were used.

### Key Findings
1. **Dialect Generation & Grammar Alignment**:
   - ULM-4B Arm B substantially outperforms its un-adapted base model on Ulsan dialect generation proxy score (**0.6591 vs 0.5629**, +0.0962) and grammar dialectness (**0.8909 vs 0.7790**, +0.1119).
   - In comparison, general open models such as Qwen2.5-3B (0.4537) and Qwen2.5-7B (0.5007) exhibit lower dialectness proxy scores and occasionally leak foreign tokens or standard Korean under dialect transformation instructions.
2. **Semantic Preservation**:
   - ULM-4B preserves remarkably high semantic similarity to references on both dialect generation (**0.9841**) and comprehension (**0.9802**), showing that domain adaptation did not corrupt the model's semantic fidelity.
3. **Identification Task**:
   - Dialect identification remains a challenging classification task across all zero-shot open models without few-shot examples: Llama-3.2-Bllossom-3B scored **0.33**, ULM-4B scored **0.30**, Qwen3.8 Base scored **0.28**, and Qwen2.5-3B scored **0.21**.
4. **Context & Repetition Mitigation**:
   - Under neutral greedy decoding without repetition penalties, ULM-4B recorded 3 repetitions on multi-turn dialogue context items (compared to 6 on Qwen3.8 Base). Under the context-only **Context Guard** ablation (`repetition_penalty=1.10`, `no_repeat_ngram_size=3`), context repetition is reduced to **0**.

---

## 2. Model Information & Specifications

| Model Identifier | Display Name | Base Model / Organization | Parameters | Quantization | Precision | Context Len |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `ulm-4b-arm-b-neutral` | **ULM-4B Arm B (Neutral)** | `empero-ai/Qwen3.8-4B-Distill` + LoRA | 4.21B | None | bfloat16 | 262,144 |
| `ulm-4b-arm-b-context-guard` | **ULM-4B Arm B (Context Guard)** | `empero-ai/Qwen3.8-4B-Distill` + LoRA | 4.21B | None | bfloat16 | 262,144 |
| `qwen3.8-4b-base` | **Qwen3.8-4B-Distill Base** | `empero-ai/Qwen3.8-4B-Distill` | 4.21B | None | bfloat16 | 262,144 |
| `qwen2.5-3b-instruct` | **Qwen2.5-3B-Instruct** | `Qwen/Qwen2.5-3B-Instruct` | 3.09B | None | bfloat16 | 32,768 |
| `qwen2.5-7b-instruct` | **Qwen2.5-7B-Instruct** | `Qwen/Qwen2.5-7B-Instruct` | 7.61B | None | bfloat16 | 131,072 |
| `llama-3.2-korean-bllossom-3b` | **Llama-3.2-Korean-Bllossom-3B** | `Bllossom/llama-3.2-Korean-Bllossom-3B` | 3.21B | None | bfloat16 | 131,072 |

---

## 3. Benchmark Results (UlsanBench v2 — 500 Items)

### 3.1 Main Comparison Table (Neutral Deterministic Decoding)

All scores below are measured under strict deterministic decoding (`do_sample=false`, `temperature=0`, `enable_thinking=false`, no repetition penalty).

| Model | Comprehension Sem. | Generation Sem. | Generation Dialect | Grammar Sem. | Grammar Dialect | Context Sem. | Identification Acc. |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ULM-4B Arm B** | **0.9802** | **0.9841** | **0.6591** | **0.9843** | **0.8909** | **0.4594** | 0.3000 |
| **Qwen3.8-4B-Distill Base** | 0.8220 | 0.8840 | 0.5629 | 0.9220 | 0.7790 | 0.3593 | 0.2800 |
| **Qwen2.5-3B-Instruct** | 0.8150 | 0.6888 | 0.4537 | 0.7554 | 0.6303 | 0.4186 | 0.2100 |
| **Qwen2.5-7B-Instruct** | 0.7193 | 0.7766 | 0.5007 | 0.8933 | 0.7241 | 0.3859 | 0.2700 |
| **Llama-3.2-Bllossom-3B** | 0.8228 | 0.8695 | 0.5548 | 0.9101 | 0.7530 | 0.3792 | **0.3300** |

*Note: Semantic similarity and dialectness proxy are calculated on a [0.0, 1.0] scale. In visualization assets, values are multiplied by 100 for display purposes.*

### 3.2 Context Guard vs Neutral Decoding for ULM-4B Arm B

For this supplementary ablation, ULM-4B repetition controls are applied only to the open-ended dialogue context task. This isolates their effect on context behavior and should not be read as a full production-serving benchmark.

| Decoding Setting | Context Sem. | Context Dialect | Context Repetitions | Context Malformed | Generation Dialect |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ULM-4B Neutral** (`penalty=1.0`) | 0.4594 | 0.2986 | 3 / 75 | 0 / 75 | 0.6591 |
| **ULM-4B Context Guard** (`penalty=1.10`, `ngram=3`, context only) | **0.4656** | **0.3073** | **0 / 75** | **0 / 75** | 0.6591 |

---

## 4. Runtime & Computational Efficiency

**2026-09-29 verification note:** Historical `total_tokens_generated` and throughput below counted padding after EOS and may be overstated. Values are retained as research records and are not validated v1.0 throughput claims. Corrected Arm B results are recorded separately in [v1.0 Stable Release](ULM_4B_V1_0_STABLE_RELEASE.md). Other models have not had throughput rerun for this release. `Avg Latency / Item` divides batch wall time by item count; it is not single-request latency or TTFT. Semantic/dialectness rubric and historical quality scores are unchanged.

Measured on NVIDIA L40S 46GB (`g6e.xlarge`), batch size 16 in bfloat16:

| Model | Total Inference Time (500 items) | Avg Latency / Item | Generation Throughput | Peak VRAM |
| :--- | :---: | :---: | :---: | :---: |
| **ULM-4B Arm B (Neutral)** | 92.52s | 185.0 ms | 174.9 tok/s | 9.27 GiB |
| **ULM-4B Arm B (Context Guard)** | 78.85s | 157.7 ms | 176.4 tok/s | 9.27 GiB |
| **Qwen3.8-4B-Distill Base** | 65.16s | 130.3 ms | 301.5 tok/s | 9.21 GiB |
| **Qwen2.5-3B-Instruct** | 51.68s | 103.4 ms | 471.8 tok/s | 5.99 GiB |
| **Qwen2.5-7B-Instruct** | 60.75s | 121.5 ms | 468.9 tok/s | 14.59 GiB |
| **Llama-3.2-Bllossom-3B** | 37.18s | 74.4 ms | 569.8 tok/s | 6.43 GiB |

*Inference throughput notes: Qwen3.8-4B-Distill uses a hybrid linear attention architecture which currently falls back to PyTorch reference delta rule kernels in our environment, resulting in lower raw token throughput compared to standard dense Transformers like Qwen2.5 or Llama-3.2.*

---

## 5. Detailed Task-by-Task Breakdown

### 5.1 Comprehension (100 items)
- **Task**: Translate spoken Ulsan dialect utterances into standard Korean.
- **Metric**: Semantic cosine similarity against standard reference.
- **Analysis**: ULM-4B Arm B scored **0.9802**, demonstrating near-perfect fidelity in comprehending dialect-specific colloquial phrasing. Base and comparison models scored between 0.719 and 0.823, occasionally failing to map dialect idioms.

### 5.2 Generation (150 items)
- **Task**: Convert standard Korean sentences into natural Ulsan dialect expressions.
- **Metrics**: Standard semantic preservation and dialectness proxy (composite of reference similarity + dialect marker presence + characteristic endings).
- **Analysis**: ULM-4B achieved **0.6591** dialectness while maintaining **0.9841** semantic similarity. Untuned base and general models predominantly generated standard Korean responses with minimal dialect endings.

### 5.3 Grammar (75 items)
- **Task**: Restore contextual dialect sentence endings (e.g., `-제`, `-나`, `-아이가`, `-데이`).
- **Metric**: Dialectness proxy & semantic similarity.
- **Analysis**: ULM-4B reached **0.8909** dialectness (Base: 0.7790, Bllossom: 0.7530, Qwen2.5-7B: 0.7241). ULM accurately applies appropriate sentence-final particles reflecting interpersonal stance.

### 5.4 Context (75 items)
- **Task**: Open-ended conversational turn in response to a conversational partner speaking Ulsan dialect.
- **Metrics**: Context semantic similarity, dialectness proxy, repetition count, malformed count.
- **Analysis**: ULM-4B scored **0.4594** semantic similarity in neutral mode (0.4656 with Context Guard), outperforming Qwen3.8 Base (0.3593) and other baselines. The Context Guard ablation suppressed the observed repetitive output loops on the 75 context items.

### 5.5 Identification (100 items)
- **Task**: Classify dialect utterance into `ULSAN`, `OTHER_GYEONGSANG`, or `STANDARD`.
- **Metric**: Exact-match accuracy against gold session label.
- **Analysis**: All zero-shot open models achieved modest accuracy between 21% and 33% (Llama-3.2-Bllossom-3B: 33%, ULM-4B: 30%, Qwen3.8 Base: 28%, Qwen2.5-7B: 27%, Qwen2.5-3B: 21%). Subtle acoustic-to-text nuances between regional variants within the Gyeongsang dialect group remain difficult for text-only zero-shot LLMs.

---

## 6. Fairness & Methodology Notes

1. **Unmodified Rubric**: Scoring uses the exact `score_rows` rubric from `scripts/evaluate_ulsanbench_v2.py`.
2. **Evaluator Independence**: Metric embeddings are generated by `paraphrase-multilingual-MiniLM-L12-v2`, an independent pretrained multilingual encoder not fine-tuned on ULM data.
3. **Official Chat Templates**: Every model used its official tokenizer chat template (`apply_chat_template`) with identical system prompt `"요청한 결과만 출력하세요. 설명, 머리말, 따옴표, 부가 설명을 추가하지 마세요."`.
4. **No Cherry-Picking**: The full 500-item split was evaluated end-to-end for all models. Raw prediction logs are retained on the benchmark EC2 volume and are not committed to Git.
5. **Metric Caveats**: Semantic similarity and dialectness proxy scores are automated evaluation proxies. They provide reproducible directional signals but do not replace human dialectal perception tests by native Ulsan residents.
