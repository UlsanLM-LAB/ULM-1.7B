# Phase4 merged checkpoint inference audit

2026-09-24 · EC2 `i-0f732bf7d1cc409b4` (L40S) · **CHECKPOINT QUALITY ISSUE**

## What was compared and changed

The production checkpoint was `/home/ubuntu/ULM-1.7B/outputs/ulm-1.7b-phase4-best-merged`, loaded with its own tokenizer in the Transformers 5.17 environment. Phase3, Phase4 SFT interim, and Phase4 merged copies have identical `config.json`, `tokenizer_config.json`, `chat_template.jinja`, and `generation_config.json` hashes. The tokenizer uses the checkpoint's Qwen chat template, with an explicit system/user/assistant history, `add_generation_prompt=True`, `enable_thinking=False`, attention mask, and `skip_special_tokens=True` on output. The checkpoint's EOS set remains in its generation config; the generation PAD is the tokenizer EOS. Overlong history is rejected before generation so role boundaries cannot be cut by token-tail truncation.

| Path | Before | Now |
| --- | --- | --- |
| Phase4 200-prompt evaluation | Phase4 system instruction; sample at temperature 0.7, top-p 0.9, top-k 20, repetition penalty 1.1, 150 new tokens | Reference policy |
| ULM-1.7B server | Same template, but 512-token default, no repetition penalty, tokenizer PAD, and always reported `stop` | Shared Phase4 policy, PAD=EOS, reports `length` when EOS is absent |
| ULM-chat | Browser settings were not passed to the provider; stored default system prompt differed from Phase4 | Settings are passed; an empty prompt uses the server's Phase4 instruction; legacy stored default is migrated |
| ULM-LIVE | Loaded the merged text checkpoint again under Transformers 4.57, with a tokenizer compatibility override, a different instruction, 64 tokens, and no repetition penalty | Calls the same ULM text server; Qwen3-TTS generation is unchanged |

The earlier reported 80% [Phase4 200-prompt score](phase4_final_regression_200.json) came from a **DPO adapter evaluation**, not from serving this merged checkpoint. The [merge verification](phase4_merge_verification.json) reported substantial similarity but not byte-identical output and recorded malformed text. This audit therefore tests the actual merged checkpoint directly.

## Real-model evidence

The raw checkpoint audit used eight prompt types (general, factual, casual, dialect, long, short, multi-turn, stress) under greedy, sampling without repetition penalty, and Phase4 sampling, with seed 42 reset per prompt. Greedy ended on EOS in **8/8**. Sampling without penalty ended on EOS in **7/8**; its long answer hit the token limit and contained `�`. Phase4 sampling ended on EOS in **7/8**; its long answer hit the token limit without `�`. These failures occur without any HTTP or UI layer. [Raw rows](phase4-inference-audit/raw-baseline.jsonl).

The 50-prompt raw regression reused the old Phase4 prompt source: general Korean 20, dialect 15, multi-turn 10, stress 5. Under the Phase4 settings, **50/50 ended on EOS**. The automated checks found **0 empty, echoed, repeated-phrase, replacement-character, token-limit, or abnormally long outputs** in this run. The checker covers structural defects, not factual correctness. [All 50 rows](phase4-inference-audit/regression-50.jsonl).

Manual review found clear factual errors in **6 of the first 10 factual questions**. Examples: highest mountain answered with a fabricated Icelandic place; largest planet answered Earth; Hangul's creator answered a fabricated Goguryeo figure; largest country answered the US; Moon landing year answered 1971; element `O` was not identified as oxygen. Some casual and multi-turn answers also leave the topic: a subway-transfer complaint received HTML/CSS advice. This is a quality failure despite clean EOS and Unicode statistics.

The real server ran once in a single process on loopback port 8000. `/health` returned `model_loaded=true`, `model_path=outputs/ulm-1.7b-phase4-best-merged`, `device=cuda:0`, and `dtype=bfloat16`. Ten direct HTTP/SSE requests and ten requests through the ULM-chat Next.js `/api/chat` proxy each produced a first token, `finish_reason=stop`, and `[DONE]`; neither path used mock mode. Both direct and proxied streams tolerated closing the client after the first token and answered a subsequent request. [Direct rows](phase4-inference-audit/backend-http-10.jsonl) · [Proxy rows](phase4_proxy_http_10.jsonl).

The proxy run still answered “Earth” to the largest-planet question and failed the two-turn “my name is Minsu” recall, instead introducing “Michael”. A real ULM-LIVE text-client call from the separate TTS environment received the backend health result and two generated answers; a Taehwagang walk request returned unrelated, fabricated imagery. These are generated outputs from the real Phase4 model. No audio or TTS-quality test was repeated in this audit.

## Decision

The inference divergence was real and has been consolidated in one text backend. The checked 50-sample Phase4 run no longer exhibited the original structural defects, but the raw baseline proves token-limit and broken-ending failures remain possible, and the regression/HTTP runs expose serious factual and conversational failures. The acceptance criterion for reliable answers is **not met**. Further inference-only filtering would conceal this; model/data work would be needed to resolve checkpoint quality, and retraining was expressly excluded from this task.
