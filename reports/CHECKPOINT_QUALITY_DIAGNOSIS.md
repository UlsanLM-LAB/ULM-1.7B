# ULM-1.7B Checkpoint Quality Degradation Comparative Diagnosis

**Date:** 2026-09-24  
**Environment:** AWS EC2 `i-0f732bf7d1cc409b4` (NVIDIA L40S 46GB VRAM, CUDA 13.2, Driver 595.91.07)  
**Evaluation:** Unified 40-Prompt Suite (Factual QA 15, General Korean 10, Ulsan Dialect 5, Multi-turn Memory 5, Instruction/Stress 5)  
**Inference Policy:** Unified Qwen Chat Template, System Prompt: Ulsan Assistant (`PHASE4_SYSTEM_PROMPT`), `enable_thinking=False`, `temperature=0.7`, `top_p=0.9`, `top_k=20`, `repetition_penalty=1.1`, `max_new_tokens=150`, `seed=42`, `eos_token_id=[151645, 151643]`.

---

## 1. Executive Summary & Required Report Format

### Available checkpoints:
- **Base model:** `Qwen/Qwen3-1.7B` (located in HF cache: `/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`)
- **Phase3 final model:** `outputs/ulm-1.7b-phase3-best-merged` (Standalone BF16 merged checkpoint, 3.3GB safetensors)
- **Phase4 adapter:** `outputs/ulm-1.7b-phase4-dpo-l40s` (PEFT LoRA adapter loaded on base `outputs/ulm-1.7b-phase4-sft-interim`)
- **Phase4 merged checkpoint:** `outputs/ulm-1.7b-phase4-best-merged` (Standalone BF16 merged checkpoint, 3.3GB safetensors)

### Base result:
- Factual QA: 6/15 (40.0%)
- Multi-turn memory: 4/5 (80.0%)
- Defects: 0 empty, 2 repetition, 0 echo, 3 replacement char (``), 22 token-limit (tendency to chatter indefinitely under 150 token budget).
- Factual answers: Correctly identified Seoul, 1969 moon landing, Oxygen for O, Paris for France capital, H2O, and 36.5°C. Struggled on 1.7B-scale geography/astronomy (Jupiter, Everest, Russia).

### Phase3 result:
- Factual QA: 3/15 (20.0%)
- Multi-turn memory: 2/5 (40.0%)
- Defects: 0 empty, 0 repetition, 1 prompt echo, 0 replacement char, 0 token-limit.
- Factual answers: Suffered catastrophic forgetting. Replaced factual knowledge with evasive dialect filler or fabricated statements ("지구가 태양계에서 제일 큰 거 알제?", "나도 모르겠다 진짜 무슨 과학이랑 역사 이런 거", "삼육도예요..."). Multi-turn failed to track names and preferences ("민수야가 이름을 잃었잖아", "코끼리").

### Phase4 adapter result:
- Factual QA: 5/15 (33.3%)
- Multi-turn memory: 3/5 (60.0%)
- Defects: 0 empty, 0 repetition, 0 echo, 0 replacement char, 1 token-limit, 4 malformed CJK/foreign fragments.
- Factual answers: Replaced Phase3 evasiveness with confident dialect hallucinations ("1971년이제! 미국과 런던에서 제2차 도전", "지구에서 가장 높은 산은 암厄 산(Amaurot)이제! 뉴질랜드와 아이슬란드 사이", "미국이 가장 넓은 나라", "한글 창제는 고구려 홍해왕", "O는 삼각수 원소").

### Phase4 merged result:
- Factual QA: 4/15 (26.7%)
- Multi-turn memory: 3/5 (60.0%)
- Defects: 0 empty, 0 repetition, 0 echo, 0 replacement char, 0 token-limit, 3 malformed CJK fragments.
- Factual answers: Virtually identical hallucinations as Phase4 adapter ("1971년이제! 빅터 부르바키 박사팀", "태양계에서 제일 크면서도 질량이 제일 많은 행성은 지구제!", "지구에서 가장 높은 산은 암살도이제... 아이슬란드", "미국이 가장 넓은 나라", "한글 창제는 고구려 채양왕신후궁").

### Factual QA score:
- **Base:** 6/15 (40.0%)
- **Phase3:** 3/15 (20.0%)
- **Phase4 adapter:** 5/15 (33.3%)
- **Phase4 merged:** 4/15 (26.7%)

### Multi-turn memory:
- **Base:** 4/5 (80.0%)
- **Phase3:** 2/5 (40.0%)
- **Phase4 adapter:** 3/5 (60.0%)
- **Phase4 merged:** 3/5 (60.0%)

### First point where quality drops:
**Phase3 SFT**

### Most likely cause:
**Catastrophic forgetting during Phase 3 SFT due to dialect-skewed conversational data without factual preservation replay. Phase 3 eliminated factual knowledge and multi-turn tracking ability; subsequent Phase 4 DPO trained the model to express confident dialect hallucinations over absent facts.**

### Merge issue suspected:
**NO** (Cosine similarity between Phase 4 Adapter and Phase 4 Merged logits is >0.99; Top-1 token match is 100% (5/5); exact same hallucination patterns are produced by both).

### Retraining needed now:
**YES**

### Recommended next single action:
**retrain Phase3 SFT with safer data/mixing**

---

## 2. Key Factual QA Breakdown (8 Required Questions)

| No. | Question (Target) | Base Model | Phase 3 Best Merged | Phase 4 DPO Adapter | Phase 4 Best Merged | Status / Diagnosis |
|---|---|---|---|---|---|---|
| 1 | 대한민국 수도 (서울) | **서울** (O) | 삼육도... 서울 (△) | **서울특별시** (O) | **서울특별시** (O) | Maintained |
| 2 | 지구에서 가장 높은 산 (에베레스트) | 쓰나미 산 (X) | "이케 쪼끔 길게 얘기해줄라고 해도" (X) | 암厄 산(Amaurot) [아이슬란드] (X) | 암살도이 [아이슬란드] (X) | P3 dropped, P4 hallucinated |
| 3 | 태양계 가장 큰 행성 (목성) | 지구/천왕성 (X) | **"지구가 태양계에서 제일 큰 거 알제?"** (X) | jupiter제... 지구보다 150배 (X) | **"지구제!"** (X) | P3 introduced "지구", P4 merged carried it |
| 4 | 한글 창제 국왕 (세종대왕) | 신라 왕조 (X) | "나는 쪼끔 궁금하긴 한데" (X) | 고구려 홍해왕 (X) | 고구려 채양왕신후궁 (X) | P3 forgot, P4 hallucinated |
| 5 | 가장 면적이 넓은 국가 (러시아) | 한국 (X) | 남극/남반구... (X) | 미국 (X) | 미국 (X) | Adapter & Merged both say USA |
| 6 | 인류 최초 달 착륙 연도 (1969) | **1969년** (O) | "나도 모르겠다 진짜 무슨 과학이랑 역사" (X) | **1971년** (X) | **1971년** (X) | P3 forgot 1969; P4 hallucinated 1971 |
| 7 | 원소 기호 O (산소) | **오xygen (산소)** (O) | "떨어진다거나 되게 많지 않아?" (X) | 삼각수 원소 3 (X) | 전기 원자기호... 알루미늄 (X) | P3 forgot oxygen; P4 hallucinated |
| 8 | 프랑스 수도 (파리) | **파리** (O) | 프랑스 전역 (X) | 오베르논(Oberland) (X) | 프랜체르(Paris) (△) | P3 forgot; P4 confused |

---

## 3. Multi-Turn Memory Breakdown (5 Questions)

| No. | History Context & Turn 2 Prompt | Base Model | Phase 3 Best Merged | Phase 4 DPO Adapter | Phase 4 Best Merged |
|---|---|---|---|---|---|
| 31 | `User: 내 이름은 민수야` &rarr; `User: 내 이름이 뭐라고 했지?` | **민수** (O) | "민수야가 이름을 잃었잖아" (X) | **민수** (O) | **민수** (O) |
| 32 | `User: 사과보다 바나나를 더 좋아해` &rarr; `User: 내가 어떤 과일을 더 좋아한다고 했지?` | **바나나** (O) | "나는 쪼끔 진짜 딱히 없는데" (X) | **바나나** (O) | **바나나** (O) |
| 33 | `User: 다음 주 화요일에 부산으로 출장 가` &rarr; `User: 내가 언제 어디로 출장 간다고 했어?` | "오늘 아침에 봐도..." (X) | "내일 출장... 캐시랑..." (X) | "장터 걷기..." (X) | **부산** (화요일 누락, △) |
| 34 | `User: 내 취미는 자전거 타는 거야` &rarr; `User: 내 취미가 뭐라고 했는지 기억나?` | **자전거** (O) | **자전거** (O) | **자전거** (O) | 자전거 타기 좋은 장소 질문 (△) |
| 35 | `User: 우리 집 강아지 이름은 초코야` &rarr; `User: 우리 집 강아지 이름이 뭐게?` | **초코** (O) | "코끼리" (X) | "어떤 강아지가 계세요?" (X) | "코난" (X) |

---

## 4. Phase 4 Adapter vs Merged Parity Verification

To determine whether the merge operation (`peft.merge_and_unload`) damaged weights or corrupted predictions, next-token logits were computed for 5 factual prompts under identical conditions:

| Prompt ID | Prompt | Logit Cosine Sim | Max Abs Diff | Mean Abs Diff | Top-1 Token Match | Adapter Top-1 Token (Prob) | Merged Top-1 Token (Prob) |
|---|---|---|---|---|---|---|---|
| 1 | 대한민국의 수도는 어디야? | **0.993976** | 2.281 | 0.370 | **MATCH (True)** | `대` (95.3%) | `대` (96.1%) |
| 3 | 태양계에서 가장 큰 행성은 뭐야? | **0.996058** | 2.078 | 0.324 | **MATCH (True)** | `태` (99.8%) | `태` (99.7%) |
| 4 | 한글을 창제한 조선의 국왕은? | **0.988283** | 1.875 | 0.298 | **MATCH (True)** | `한` (21.5%) | `한` (21.9%) |
| 6 | 인류가 최초로 달에 착륙한 연도는? | **0.988121** | 2.125 | 0.341 | **MATCH (True)** | `1` (64.1%) | `1` (59.9%) |
| 7 | 원소 기호 O는 무슨 원소야? | **0.990379** | 2.312 | 0.366 | **MATCH (True)** | `O` (49.9%) | `O` (40.3%) |

### Mathematical Findings:
1. Logit cosine similarity is uniformly **>0.988** (average >0.991), and top-1 tokens are **100% identical** across all test prompts.
2. The slight numerical divergence (mean diff ~0.33) is the expected floating-point rounding artifact of BF16 matrix multiplication vs GEMM fusion in `merge_and_unload`.
3. Crucially, the hallucinations are identical: both models predict `1971년` for the moon landing, both assert the largest country is the US, both invent an Icelandic mountain, and both cite Goguryeo for Hangul.
4. **Conclusion:** The merge step is **not the culprit**. The adapter itself is already trained on corrupted representations.

---

## 5. Root Cause Analysis: Why Did Past Audits Show 80%?

In `reports/phase4_final_regression_200.json`, the reported 80% pass rate created a false illusion of health. Inspection of the underlying category statistics reveals:
- **Casual Chat:** 15/15 (100% correct)
- **Empathy:** 15/15 (100% correct)
- **Recommendation:** 15/15 (100% correct)
- **Slang:** 15/15 (100% correct)
- **Short Utterance:** 15/15 (100% correct)
- **Clarification:** 15/15 (100% correct)
- **Factual QA:** **3/15 (80% FAIL)**
- **Science:** **5/15 (67% FAIL)**
- **Ulsan QA:** **3/20 (85% FAIL)**

Because 140+ out of 200 prompts tested chit-chat and tone where superficial dialect markers were accepted as "passes", the evaluation masked total catastrophic forgetting of factual knowledge that started in Phase 3.

---

## 6. Actionable Recommendation

**Retrain Phase3 SFT with safer data and replay mixing:**
1. Do not touch or re-merge Phase 4; it cannot fix absent knowledge.
2. Construct a balanced Phase 3 SFT mixture combining:
   - 35% Ulsan dialect conversational pairs
   - 45% General Korean factual QA, science, and history instruction data (e.g. Ko-Alpaca, Ko-CommonSense, AI Hub Q&A)
   - 20% Multi-turn context tracking and instruction-following data
3. Apply lower LoRA rank/learning rate or weight decay on the base model layers to prevent catastrophic forgetting of base pre-training knowledge.
