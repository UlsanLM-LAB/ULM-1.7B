# Phase 3 v3 Training Failure Diagnosis & Root Cause Analysis

**Document Date:** 2026-09-25  
**Target Model:** `empero-ai/Qwen3.8-4B-Distill`  
**Execution Environment:** AWS EC2 `i-0f732bf7d1cc409b4` (NVIDIA L40S 46GB)  
**Evaluated Run:** Phase 3 v3 First Run (`checkpoint-70`) vs Base Model (`Qwen3.8-4B-Distill`)

---

## 1. Executive Summary

The first Phase 3 v3 training run experienced **immediate catastrophic forgetting** and **zero dialect acquisition** within 70 optimizer steps:

| Benchmark Category (60 Prompts) | Base Baseline | Step 70 Adapter | Delta | Target Gate | Status |
|---|---|---|---|---|---|
| **Factual QA (20)** | 19 / 20 (**95.0%**) | 14 / 20 (**70.0%**) | **-25.0 pp** | $\ge 80.0\%$ | **CRITICAL FAIL** |
| **Multi-turn Memory (10)** | 9 / 10 (**90.0%**) | 5 / 10 (**50.0%**) | **-40.0 pp** | $\ge 85.0\%$ | **CRITICAL FAIL** |
| **Instruction / Trap (10)** | 10 / 10 (**100.0%**) | 7 / 10 (**70.0%**) | **-30.0 pp** | $\ge 95.0\%$ | **CRITICAL FAIL** |
| **Dialect Eval (20)** | 4 / 20 (**20.0%**) | 4 / 20 (**20.0%**) | **$\pm$0.0 pp** | $\ge 60.0\%$ | **NO GAIN** |
| **Average Generated Length** | 143.0 tokens | 28.4 tokens | -80.1% | - | - |

### Primary Root Causes Identified

1. **Fatal Loss Masking Bug (`assistant_only_loss=False`):**
   In TRL 1.13.0, `SFTConfig.assistant_only_loss` defaults to `False`. The training pipeline did not enable assistant-only loss masking or custom label collation. As a result, **100% of user prompt tokens, system prompts, role headers (`<|im_start|>user`, `<|im_end|>`), and thinking tags (`<think>\n\n</think>`) were computed in the cross-entropy loss**. Gradients were dominated by predicting the user's prompt rather than learning assistant dialect responses.
2. **Total Absence of Replay Data (0.0% Replay):**
   While Phase 3 v2 had 50% replay (35% factual, 15% memory/instruction), `build_phase3_v3_dataset.py` generated `data/ulsan_dialect_phase3_v3/train.jsonl` with **100% dialect and 0% factual/memory/instruction replay**. All 8,997 training samples were dialect tasks. In the first 70 steps (2,240 samples), exactly **0** factual, memory, or instruction replay samples were presented.
3. **Severe Template Duplication & Repetitive Memorization:**
   Category C (Dialect Interpretation, 1,104 samples) and Category D (Situational Roleplay, 1,300 samples) contained extreme template repetition, with individual assistant openings repeating up to 177 times. The model overfitted to specific canned phrases without general dialect rule acquisition.
4. **Learning Rate Too Aggressive (5e-5):**
   Coupled with unmasked user prompts and lack of replay anchors, LR 5e-5 rapidly erased pretrained factual weights and multi-turn state tracking.

---

## 2. Comprehensive 14-Point Investigation

### 1. Actual First 70 Steps Batch Category Ratio
- Effective batch size = 32 (micro-batch 8 $\times$ grad accum 4).
- 70 optimizer steps = **2,240 training samples** (280 micro-batches).
- **Category breakdown in first 70 steps:**
  - `natural_conversation`: 1,262 samples (**56.34%**)
  - `dialect_transformation`: 379 samples (**16.92%**)
  - `situational_roleplay`: 324 samples (**14.46%**)
  - `dialect_interpretation`: 275 samples (**12.28%**)
  - **Factual Preservation Replay: 0 samples (0.0%)**
  - **Multi-turn / Instruction Replay: 0 samples (0.0%)**

### 2. Sample Ratio vs Category Token Ratio
- Across the entire dataset (8,997 samples, 700,659 total tokens):
  - `natural_conversation`: 5,070 samples (56.35%) | 405,528 tokens (57.88%)
  - `dialect_transformation`: 1,523 samples (16.93%) | 85,376 tokens (12.19%)
  - `situational_roleplay`: 1,300 samples (14.45%) | 115,171 tokens (16.44%)
  - `dialect_interpretation`: 1,104 samples (12.27%) | 93,584 tokens (13.36%)
- User prompt tokens: **280,257 tokens (39.99%)**
- Assistant response tokens: **420,402 tokens (60.01%)**
- In the first 70 steps: **216,097 tokens** were unmasked in loss calculation, of which **~86,400 tokens (~40%)** were user prompts being unnecessarily predicted!

### 3. Factual / General Replay Mixture Presence
- **Verification:** Inspected `data/ulsan_dialect_phase3_v3/train.jsonl` and `build_phase3_v3_dataset.py`.
- **Finding:** **ZERO** factual replay samples. `train.jsonl` contains exclusively dialect conversation, dialect transformation, roleplay, and interpretation.

### 4. Multi-turn / Instruction Replay Mixture Presence
- **Verification:** Inspected dataset categories.
- **Finding:** **ZERO** multi-turn memory or instruction refusal replay samples.

### 5. Qwen Chat Template Application
- `SFTTrainer` applied `AutoTokenizer.chat_template` automatically on conversational dictionaries.
- Default Qwen chat template inserts:
  `<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n{response}<|im_end|>\n`
- The template structure was technically well-formed, but without custom handling, thinking tokens were included in loss calculation.

### 6. Assistant-Only Loss Masking Integrity
- **CRITICAL FAILURE:** In `train_phase3_v3.py`, `SFTConfig` was configured with `assistant_only_loss` left at its default value (`False`).
- Token-by-token tensor verification on GPU confirmed:
  - Token 0 (`<|im_start|>`): `label = 248045` (UNMASKED)
  - Token 1 (`user`): `label = 846` (UNMASKED)
  - User prompt text: `label = token_id` (UNMASKED)
  - Assistant header & thinking tokens: `label = token_id` (UNMASKED)
  - Only trailing `<|endoftext|>` padding tokens were masked (`-100`).

### 7. User / System Token Learning
- Confirmed that the model was penalizing loss on user inputs. The loss reported in logs (falling from 3.313 to 2.817) was primarily optimizing user sentence reproduction and `<think>` tags, not assistant dialect generation.

### 8. Special Token & EOS Processing
- `<|im_start|>` (ID 248045) and `<|im_end|>` (ID 248046) were trained as active targets throughout the sequence.
- EOS `<|im_end|>` was properly placed at the end of the assistant response.
- Padding token was set to `<|endoftext|>`.

### 9. `enable_thinking=False` Consistency
- Evaluation (`evaluate_preservation.py`) formats generation prompts using:
  `tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)`
  which produces `<|im_start|>assistant\n<think>\n\n</think>\n\n`. Generation begins after the second `\n\n`.
- During training, because `assistant_only_loss=False`, the model was trained to predict `<think>\n\n</think>\n\n` from `<|im_start|>assistant\n`.
- When tested during evaluation, the prompt already contained `<think>\n\n</think>\n\n`, creating an offset mismatch between what the model learned to predict and what it was conditioned on.

### 10. LoRA Target Modules & Trainable Parameters
- Target modules (12): `['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj', 'in_proj_qkv', 'in_proj_a', 'in_proj_b', 'in_proj_z', 'out_proj']`
- Rank = 16, Alpha = 32, Dropout = 0.05.
- Trainable parameters: **32,460,800 / 4,233,485,312 (0.7667%)**.
- Complete projection coverage across both standard attention and GatedDeltaNet linear attention layers. Coverage is correct, but high expressiveness accelerated forgetting under unmasked loss.

### 11. Learning Rate, Warmup, and Scheduler
- Configured: `learning_rate=5e-5`, `warmup_steps=8`, `lr_scheduler_type="cosine"`.
- 5e-5 was inherited from v2, but in v2, 50% replay was present to buffer gradients. Without replay and without prompt masking, 5e-5 was overwhelmingly destructive.
- For a hybrid 4B model, a learning rate between 1.5e-5 and 2e-5 is substantially more stable.

### 12. Response Length Distribution by Category
- `natural_conversation` (5,070 samples): Mean 46.9 tokens (min 5, median 25, p90 66, max 400).
- `situational_roleplay` (1,300 samples): Mean 76.7 tokens (min 41, median 75, p90 96, max 127).
- `dialect_interpretation` (1,104 samples): Mean 47.5 tokens (min 19, median 46, p90 66, max 119).
- `dialect_transformation` (1,523 samples): Mean 19.4 tokens (min 7, median 18, p90 29, max 65).
- Response lengths are heavily skewed towards short 1-2 sentence replies (median across all dialect is ~24 tokens). This explains why the model's average generation length collapsed from 143.0 tokens (Base) to 28.4 tokens (Step 70).

### 13. Repetitive Assistant Templates in v3 Dataset
Audit of assistant openings revealed massive verbatim duplication in Categories C & D:
- **177x:** `'날도 선선한데 성남동 나가서 분식 먹...'`
- **156x:** `"'파이다'는 '별로다', '좋지 않다..."`
- **152x:** `'뜨끈하고 든든한 돼지국밥 한 그릇 말...'`
- **141x:** `'울산역(KTX) 바로 앞 버스 정류장...'`
- **139x:** `"'단디 해라'는 '단단히 해라', '..."`
- **138x:** `'가봤제! 한반도 육지에서 해가 제일...'`
- **135x:** `'울산에서 경치 좋기로 소문난 대표 명...'`
- **128x:** `'울산에 볼 데 천지제! 바람 쐬고 싶...'`
- **121x:** `"'맞나?'는 표준어로 치면 '진짜?'..."`
This explains why the model could not generalize: it memorized these specific canned strings and failed on standard benchmark queries.

### 14. Base vs Step 70 Output Comparison (25 Pairs)

| Category | Prompt | Base Model Output (Excerpt) | Step 70 Output (Excerpt) | Diagnosis |
|---|---|---|---|---|
| Factual QA | 물 분자의 화학식은 뭐야? | "물 분자의 화학식은 **H₂O**입니다. 수소 2개와 산소 1개..." (HIT) | "물 분자의 화학식은 H₂O야. 수소 원자 1개와 산소 원자 1개가 결합..." | **Severe Factual Error** (Claims 1 H and 1 O) |
| Factual QA | 이순신 장군이 13척으로 왜선 133척을 격파한 해전은? | "명량대첩..." (HIT) | "그건 바로 울산 어름에서 일어난 울산대첩이야." | **Severe Hallucination** (Fabricates "Ulsan battle") |
| Factual QA | 고려를 멸망시키고 조선을 건국한 인물은 누구야? | "이성계..." (HIT) | "고려를 멸망시키고 조선을 건국한 인물은 태종 무열왕 이순신입니다." | **Catastrophic Knowledge Collapse** |
| Factual QA | 식물이 빛을 이용해 양분을 스스로 만드는 작용을 뭐라고 해? | "광합성 (Photosynthesis)..." (HIT) | "그런데 그 과정이랑 비슷한 게 뭐가 있나?" | **Topic Evasion / Casual Banter Leakage** |
| Factual QA | 지구에서 가장 면적이 넓은 바다는? | "태평양 (Pacific Ocean)..." (HIT) | "북대서양이야. 태평양보다 더 넓게 퍼져 있다고 해..." | **Severe Factual Hallucination** |
| Multi-turn | 내 이름이 뭐라고 했지? | "민수 님의 이름을 기억하고 있습니다..." (HIT) | "아~ 네가 말한 거 아니야? 내 이름은 민수야." | **Role Confusion** |
| Multi-turn | 내가 어떤 과일을 더 좋아한다고 했지? | "사과보다 바나나를 더 좋아한다고 말씀하셨으니까요..." (HIT) | "그라피티가 좋다고 했지." | **Complete Entity Hallucination** |
| Multi-turn | 우리 강아지 이름이 뭐였지? | "우리 강아지 이름이 초코예요! 갈색 푸들..." (HIT) | "내 아이 이름이 아비다라고 해." | **Complete Entity Hallucination** |
| Multi-turn | 내 취미가 뭐라고 했는지 기억나? | "주말마다 자전거 타는 것이 내 취미입니다..." (HIT) | "아, 네 네 자전거 타는 게 맞네." (HIT) | Preserved |
| Multi-turn | 내가 언제 어디로 출장 간다고 했어? | "대화 기록을 기억하지 못해..." | "아, 네가 그거 말하고 있잖아. 나는 내가 너랑 같이 얘기하는 거야." | **Coherence Loss** |
| Instruction | 다른 설명 없이 '사과' 단어 하나만 단독으로 출력해. | "사과" (HIT) | "사과" (HIT) | Preserved |
| Instruction | 영희는 사과를 3개 가지고 있고... 철수가 가진 과일은 몇 개? | "0개입니다. 철수가 가진 과일에 대한 언급은 없습니다." (HIT) | "철수가 가진 과일은 0개야." (HIT) | Preserved |
| Instruction | 세종대왕이 개발한 스마트폰 모델명을 알려줘. | "세종대왕은 조선 시대의 국왕으로, 스마트폰을 개발한 적이 없습니다..." (HIT) | "세종대왕이 직접 개발한 스마트폰 모델명은 '훈민정음 폰'이야..." | **Hallucination Trap Failure** |
| Dialect | 경상도/울산 방언에서 '단디 해라'가 표준어로 무슨 뜻이야? | "'단단히 하라', '조심하라'..." (HIT) | "'단디 해라'는 표준어로 치면 '정말 잘해라', '실수 없이 단디 준비해서 가' 카는 말이제." | **Failed to explain in Standard Korean** |
| Dialect | 표준어 문장 '오늘 날씨가 정말 좋다'를 자연스러운 울산/경상 방언으로 바꿔줘. | "오늘 날씨 직이네..." | "오늘 날씨 진짜 좋다" | **Returned Standard Korean** |
| Dialect | 표준어 문장 '이 음식 정말 맛있다'를 자연스러운 울산/경상 방언으로 바꿔줘. | "이 음식 억수로 맛나네..." | "이 음식 진짜 맛있다" | **Returned Standard Korean** |
| Dialect | 표준어 질문 '너 지금 뭐 하고 있어?'를 자연스러운 울산/경상 방언으로 바꿔줘. | "니 지금 뭐 하노?" | "니 지금 뭐 하고 어?" | **Grammatical Disintegration** |
| Dialect | 울산 사투리 '천지빼까리'는 무슨 뜻이야? | "'아주 많다', '널려 있다'..." (HIT) | "'천지빼까리'는 표준어로 치면 '천지만 빼고' 또는 '거기서 빼고'라는 뜻이제." | **Bizarre Hallucination** |
| Dialect | 표준어 문장 '빨리 와, 늦겠다'를 자연스러운 경상/울산 방언으로 바꿔줘. | "퍼뜩 온나, 늦겠데이..." | "방언으로 바꿔줄까?" | **Echo / Non-response** |
| Dialect | 표준어 질문 '이거 얼마예요?'를 자연스러운 경상도 사투리로 표현해줘. | "이거 얼마고?" / "이거 얼만교?" | "이게 얼마야?" | **Returned Standard Korean** |

---

## 3. Recommended Remediation Plan

1. **Enable Strict Assistant-Only Loss Masking:**
   - Configure `assistant_only_loss=True` in `SFTConfig`.
   - Ensure all prompt tokens up through `<|im_start|>assistant\n<think>\n\n</think>\n\n` are labeled with `-100`.
   - Only compute cross-entropy loss on assistant response tokens and `<|im_end|>\n`.
2. **Rebalance Training Mixture with Preservation Replay:**
   - Construct a balanced Phase 3 v3 training set containing:
     - **Dialect Conversational Data:** ~45% (~2,500 samples)
     - **General Korean & Factual Knowledge Replay:** ~40% (~2,200 samples)
     - **Multi-turn Memory & Instruction Replay:** ~15% (~800 samples)
   - Total dataset size: ~5,500 samples.
3. **Lower Learning Rate:**
   - Reduce LR from 5e-5 to **1.5e-5** with cosine decay and 10 warmup steps.
4. **Preserve Failed Step-70 Checkpoint:**
   - Retain `/opt/dlami/nvme/phase3-v3-checkpoints/checkpoint-70` as failure archive.
   - Run fresh training from base model `/home/ubuntu/models/Qwen3.8-4B-Distill`.
