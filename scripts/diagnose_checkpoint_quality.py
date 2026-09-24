"""ULM-1.7B Checkpoint Quality Degradation Comparative Diagnosis Suite.

Evaluates Base, Phase 3, Phase 4 DPO Adapter, and Phase 4 Merged models
under strictly unified inference conditions across 40 prompts.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

DEFAULT_SYSTEM_PROMPT = (
    "울산 지역어 대화 assistant로서 자연스럽고 일상적인 울산 사투리로 상대방과 친근하게 대화한다. "
    "자연스럽고 편안한 일상 울산 말투를 기본으로 구사한다."
)

CJK_REGEX = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff]")
REPETITION_REGEX = re.compile(r"(.{4,})\1{2,}")
REPEATED_PHRASE_REGEX = re.compile(r"(.{3,40}?)\1{2,}")

PROMPTS_40 = [
    # 1. Factual QA (15)
    {"id": 1, "category": "factual_qa", "prompt": "대한민국의 수도는 어디야?", "keywords": ["서울"]},
    {"id": 2, "category": "factual_qa", "prompt": "지구에서 가장 높은 산은?", "keywords": ["에베레스트"]},
    {"id": 3, "category": "factual_qa", "prompt": "태양계에서 가장 큰 행성은 뭐야?", "keywords": ["목성"]},
    {"id": 4, "category": "factual_qa", "prompt": "한글을 창제한 조선의 국왕은?", "keywords": ["세종"]},
    {"id": 5, "category": "factual_qa", "prompt": "세계에서 가장 면적이 넓은 국가는?", "keywords": ["러시아"]},
    {"id": 6, "category": "factual_qa", "prompt": "인류가 최초로 달에 착륙한 연도는?", "keywords": ["1969"]},
    {"id": 7, "category": "factual_qa", "prompt": "원소 기호 O는 무슨 원소야?", "keywords": ["산소"]},
    {"id": 8, "category": "factual_qa", "prompt": "프랑스의 수도는 어디야?", "keywords": ["파리"]},
    {"id": 9, "category": "factual_qa", "prompt": "임진왜란은 몇 년도에 일어났어?", "keywords": ["1592"]},
    {"id": 10, "category": "factual_qa", "prompt": "물 분자의 화학식은 뭐야?", "keywords": ["h2o", "H2O"]},
    {"id": 11, "category": "factual_qa", "prompt": "이순신 장군이 13척으로 왜선 133척을 격파한 해전은?", "keywords": ["명량"]},
    {"id": 12, "category": "factual_qa", "prompt": "미국의 초대 대통령은 누구야?", "keywords": ["워싱턴"]},
    {"id": 13, "category": "factual_qa", "prompt": "빛의 진공 속 속도는 초당 약 몇 km야?", "keywords": ["30만", "299,792", "300,000"]},
    {"id": 14, "category": "factual_qa", "prompt": "정상 사람 체온은 대략 몇 도야?", "keywords": ["36.5", "36"]},
    {"id": 15, "category": "factual_qa", "prompt": "삼국통일을 완성한 신라의 왕은 누구야?", "keywords": ["문무왕"]},

    # 2. General Korean Conversation (10)
    {"id": 16, "category": "general_conv", "prompt": "오늘 날씨가 참 좋네. 산책하기 딱 좋은 날씨야."},
    {"id": 17, "category": "general_conv", "prompt": "요즘 주말에 할 만한 재미있는 취미 하나 추천해 줄래?"},
    {"id": 18, "category": "general_conv", "prompt": "퇴근하고 집에 왔는데 너무 피곤하다."},
    {"id": 19, "category": "general_conv", "prompt": "친구 생일 선물로 뭐가 좋을까?"},
    {"id": 20, "category": "general_conv", "prompt": "주말에 영화 한 편 보고 싶은데 어떤 장르가 좋을까?"},
    {"id": 21, "category": "general_conv", "prompt": "스트레스 받을 때 어떻게 푸는 게 좋아?"},
    {"id": 22, "category": "general_conv", "prompt": "여행 가고 싶은데 바다랑 산 중에 어디가 나을까?"},
    {"id": 23, "category": "general_conv", "prompt": "아침에 일찍 일어나는 꿀팁 있어?"},
    {"id": 24, "category": "general_conv", "prompt": "건강을 위해서 매일 지키면 좋은 습관 알려줘."},
    {"id": 25, "category": "general_conv", "prompt": "오늘 점심 메뉴 고민인데 따뜻한 국물 요리 추천해줘."},

    # 3. Ulsan / Gyeongsang Dialect (5)
    {"id": 26, "category": "dialect", "prompt": "너 지금 뭐하노?"},
    {"id": 27, "category": "dialect", "prompt": "밥 묵었나?"},
    {"id": 28, "category": "dialect", "prompt": "와이리 늦게 왔노?"},
    {"id": 29, "category": "dialect", "prompt": "오늘 학교 끝나고 뭐 할 끼고?"},
    {"id": 30, "category": "dialect", "prompt": "태화강 국가정원 가봤나?"},

    # 4. Multi-turn Memory (5)
    {
        "id": 31,
        "category": "multi_turn",
        "prompt": "내 이름이 뭐라고 했지?",
        "history": [["내 이름은 민수야. 기억해 줘.", "알겠데이, 민수야! 꼭 기억하고 있을게."]],
        "memory_keywords": ["민수"],
    },
    {
        "id": 32,
        "category": "multi_turn",
        "prompt": "내가 어떤 과일을 더 좋아한다고 했지?",
        "history": [["나는 사과보다 바나나를 더 좋아해.", "바나나 달달하고 든든해서 맛있제!"]],
        "memory_keywords": ["바나나"],
    },
    {
        "id": 33,
        "category": "multi_turn",
        "prompt": "내가 언제 어디로 출장 간다고 했어?",
        "history": [["다음 주 화요일에 부산으로 출장 가.", "부산 출장 조심히 잘 댕겨온나!"]],
        "memory_keywords": ["화요일", "부산"],
    },
    {
        "id": 34,
        "category": "multi_turn",
        "prompt": "내 취미가 뭐라고 했는지 기억나?",
        "history": [["내 취미는 주말마다 자전거 타는 거야.", "주말에 자전거 타면 상쾌하고 기분 최고제!"]],
        "memory_keywords": ["자전거"],
    },
    {
        "id": 35,
        "category": "multi_turn",
        "prompt": "우리 집 강아지 이름이 뭐게?",
        "history": [["우리 집 강아지 이름은 초코야.", "초코라니 이름 진짜 귀엽네!"]],
        "memory_keywords": ["초코"],
    },

    # 5. Instruction / Repetition Stress (5)
    {"id": 36, "category": "stress", "prompt": "'사과'라는 단어만 정확히 한 번 출력해. 다른 설명이나 문장은 절대 붙이지 마."},
    {"id": 37, "category": "stress", "prompt": "1부터 5까지 숫자를 쉼표로 구분해서 차례대로 써줘. (예: 1, 2, 3, 4, 5)"},
    {"id": 38, "category": "stress", "prompt": "\"안녕하세요\"를 세 번 반복해서 써줘. 다른 군더더기 없이 문장만 출력해."},
    {"id": 39, "category": "stress", "prompt": "아래 문장에서 동물 이름만 골라내: '어제 공원에서 고양이와 참새를 보았다.'"},
    {"id": 40, "category": "stress", "prompt": "다음 질문에 '네' 또는 '아니오'로만 답해: 지구는 둥근가요?"},
]

# Prompts for logits verification (5 factual prompts)
LOGITS_VERIFY_IDS = [1, 3, 4, 6, 7]


def compute_file_hash(path: str | Path) -> str:
    p = Path(path)
    if not p.is_file():
        return "MISSING"
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def format_chat_prompt(tokenizer: Any, prompt_item: dict) -> str:
    messages = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]
    history = prompt_item.get("history", [])
    if history:
        for u, a in history:
            messages.append({"role": "user", "content": u})
            messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": prompt_item["prompt"]})

    template_kwargs = {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
    try:
        return tokenizer.apply_chat_template(messages, **template_kwargs)
    except TypeError:
        template_kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **template_kwargs)


def generate_single_response(
    model: Any,
    tokenizer: Any,
    prompt_item: dict,
    seed: int = 42,
    device: str = "cuda:0",
) -> dict:
    set_seed(seed)
    prompt_text = format_chat_prompt(tokenizer, prompt_item)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    input_len = input_ids.shape[1]

    eos_token_id = [151645, 151643]
    pad_token_id = 151643

    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
            top_k=20,
            repetition_penalty=1.1,
            do_sample=True,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    gen_tokens = outputs[0][input_len:].tolist()
    gen_token_count = len(gen_tokens)

    # Check termination
    ended_on_eos = False
    if gen_token_count > 0 and gen_tokens[-1] in eos_token_id:
        ended_on_eos = True
    elif gen_token_count < 150:
        ended_on_eos = True

    finish_reason = "stop" if ended_on_eos else "length"
    response = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

    return {
        "prompt_id": prompt_item["id"],
        "category": prompt_item["category"],
        "prompt": prompt_item["prompt"],
        "history": prompt_item.get("history", []),
        "response": response,
        "generated_token_count": gen_token_count,
        "ended_on_eos": ended_on_eos,
        "finish_reason": finish_reason,
        "latency_ms": latency_ms,
        "seed": seed,
    }


def get_first_token_logits(
    model: Any,
    tokenizer: Any,
    prompt_item: dict,
    device: str = "cuda:0",
) -> torch.Tensor:
    prompt_text = format_chat_prompt(tokenizer, prompt_item)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        # Logits for the next token (position after last input token)
        next_token_logits = outputs.logits[0, -1, :].to(torch.float32).cpu()
    return next_token_logits


def analyze_model_responses(rows: list[dict], prompt_items: list[dict]) -> dict:
    prompt_map = {item["id"]: item for item in prompt_items}
    total = len(rows)

    empty_count = 0
    repetition_count = 0
    prompt_echo_count = 0
    replacement_char_count = 0
    token_limit_count = 0
    malformed_cjk_count = 0

    factual_total = 0
    factual_hits = 0
    factual_details = []

    multi_turn_total = 0
    multi_turn_hits = 0
    multi_turn_details = []

    for row in rows:
        pid = row["prompt_id"]
        resp = row["response"]
        p_item = prompt_map[pid]
        p_text = row["prompt"]
        tokens = row["generated_token_count"]
        eos = row["ended_on_eos"]

        # 1. Empty output
        if not resp.strip():
            empty_count += 1

        # 2. Prompt echo
        clean_p = re.sub(r"[^\w\s]", "", p_text).strip()
        clean_r = re.sub(r"[^\w\s]", "", resp).strip()
        if len(clean_p) >= 4 and (clean_r == clean_p or (clean_p in clean_r and len(clean_r) < len(clean_p) * 1.25)):
            prompt_echo_count += 1

        # 3. Repetition
        if REPETITION_REGEX.search(resp) or REPEATED_PHRASE_REGEX.search(re.sub(r"\s+", " ", resp)):
            repetition_count += 1

        # 4. Replacement character
        if "\ufffd" in resp:
            replacement_char_count += 1

        # 5. Token limit termination
        if tokens >= 150 and not eos:
            token_limit_count += 1

        # 6. Malformed CJK / mixed language
        if CJK_REGEX.search(resp):
            malformed_cjk_count += 1

        # 7. Factual QA hit evaluation
        if p_item["category"] == "factual_qa":
            factual_total += 1
            kw_list = p_item.get("keywords", [])
            hit = any(kw.lower() in resp.lower() for kw in kw_list)
            if hit:
                factual_hits += 1
            factual_details.append({
                "id": pid,
                "prompt": p_text,
                "response": resp,
                "expected": kw_list,
                "hit": hit,
            })

        # 8. Multi-turn memory evaluation
        if p_item["category"] == "multi_turn":
            multi_turn_total += 1
            mem_kw = p_item.get("memory_keywords", [])
            # Must contain all memory keywords (or at least the key identifier)
            hit = all(kw.lower() in resp.lower() for kw in mem_kw)
            if hit:
                multi_turn_hits += 1
            multi_turn_details.append({
                "id": pid,
                "prompt": p_text,
                "response": resp,
                "memory_keywords": mem_kw,
                "hit": hit,
            })

    return {
        "total_prompts": total,
        "empty_count": empty_count,
        "repetition_count": repetition_count,
        "prompt_echo_count": prompt_echo_count,
        "replacement_char_count": replacement_char_count,
        "token_limit_count": token_limit_count,
        "malformed_cjk_count": malformed_cjk_count,
        "factual_qa": {
            "total": factual_total,
            "hits": factual_hits,
            "accuracy_pct": round(factual_hits / factual_total * 100, 1) if factual_total else 0,
            "details": factual_details,
        },
        "multi_turn": {
            "total": multi_turn_total,
            "hits": multi_turn_hits,
            "accuracy_pct": round(multi_turn_hits / multi_turn_total * 100, 1) if multi_turn_total else 0,
            "details": multi_turn_details,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Run ULM-1.7B comparative checkpoint diagnosis")
    parser.add_argument("--output-dir", default="reports/checkpoint-comparison")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models_config = {
        "base": {
            "type": "base",
            "model_path": "/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
            "base_model": "/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
            "adapter_path": None,
            "tokenizer_path": "/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        },
        "phase3": {
            "type": "merged",
            "model_path": "/home/ubuntu/ULM-1.7B/outputs/ulm-1.7b-phase3-best-merged",
            "base_model": "/home/ubuntu/ULM-1.7B/outputs/ulm-1.7b-phase3-best-merged",
            "adapter_path": None,
            "tokenizer_path": "/home/ubuntu/ULM-1.7B/outputs/ulm-1.7b-phase3-best-merged",
        },
        "phase4-adapter": {
            "type": "adapter",
            "model_path": "/home/ubuntu/ULM-1.7B/outputs/ulm-1.7b-phase4-dpo-l40s",
            "base_model": "/home/ubuntu/ULM-1.7B/outputs/ulm-1.7b-phase4-sft-interim",
            "adapter_path": "/home/ubuntu/ULM-1.7B/outputs/ulm-1.7b-phase4-dpo-l40s",
            "tokenizer_path": "/home/ubuntu/ULM-1.7B/outputs/ulm-1.7b-phase4-sft-interim",
        },
        "phase4-merged": {
            "type": "merged",
            "model_path": "/home/ubuntu/ULM-1.7B/outputs/ulm-1.7b-phase4-best-merged",
            "base_model": "/home/ubuntu/ULM-1.7B/outputs/ulm-1.7b-phase4-best-merged",
            "adapter_path": None,
            "tokenizer_path": "/home/ubuntu/ULM-1.7B/outputs/ulm-1.7b-phase4-best-merged",
        },
    }

    # Record model metadata
    model_metadata = {}
    for name, cfg in models_config.items():
        base_dir = Path(cfg["base_model"])
        tok_dir = Path(cfg["tokenizer_path"])
        adapter_dir = Path(cfg["adapter_path"]) if cfg["adapter_path"] else None

        meta = {
            "name": name,
            "type": cfg["type"],
            "model_path": cfg["model_path"],
            "base_model": cfg["base_model"],
            "adapter_path": cfg["adapter_path"],
            "tokenizer_path": cfg["tokenizer_path"],
            "config_hash": compute_file_hash(base_dir / "config.json"),
            "tokenizer_config_hash": compute_file_hash(tok_dir / "tokenizer_config.json"),
            "tokenizer_json_hash": compute_file_hash(tok_dir / "tokenizer.json"),
            "generation_config_hash": compute_file_hash(base_dir / "generation_config.json"),
            "generation_config": {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 20,
                "repetition_penalty": 1.1,
                "max_new_tokens": 150,
                "do_sample": True,
                "seed": 42,
                "enable_thinking": False,
                "eos_token_id": [151645, 151643],
                "pad_token_id": 151643,
            },
        }
        if adapter_dir:
            meta["adapter_config_hash"] = compute_file_hash(adapter_dir / "adapter_config.json")
        model_metadata[name] = meta

    with open(out_dir / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(model_metadata, f, indent=2, ensure_ascii=False)
    print("Saved model metadata to reports/checkpoint-comparison/model_metadata.json")

    # Storage for logits comparison
    adapter_logits_dict = {}
    merged_logits_dict = {}

    summary_results = {}

    for model_name, cfg in models_config.items():
        print(f"\n=======================================================")
        print(f"Loading Model: {model_name} ({cfg['type']})")
        print(f"Base: {cfg['base_model']}")
        if cfg['adapter_path']:
            print(f"Adapter: {cfg['adapter_path']}")
        print(f"=======================================================")

        tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer_path"], use_fast=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = 151643

        base_model = AutoModelForCausalLM.from_pretrained(
            cfg["base_model"],
            torch_dtype=torch.bfloat16,
            device_map=args.device,
        )

        if cfg["adapter_path"]:
            model = PeftModel.from_pretrained(base_model, cfg["adapter_path"])
        else:
            model = base_model

        model.eval()

        # If phase4-adapter or phase4-merged, extract logits on the 5 verify prompts
        if model_name in ("phase4-adapter", "phase4-merged"):
            print(f"Collecting first-token logits for {model_name} on {LOGITS_VERIFY_IDS}...")
            logits_store = adapter_logits_dict if model_name == "phase4-adapter" else merged_logits_dict
            for pid in LOGITS_VERIFY_IDS:
                p_item = next(p for p in PROMPTS_40 if p["id"] == pid)
                l = get_first_token_logits(model, tokenizer, p_item, device=args.device)
                logits_store[pid] = l

        # Run 40 prompts
        print(f"Running 40-prompt evaluation on {model_name}...")
        rows = []
        jsonl_path = out_dir / f"{model_name}.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as jf:
            for idx, p_item in enumerate(PROMPTS_40, start=1):
                res = generate_single_response(model, tokenizer, p_item, seed=42, device=args.device)
                res["model"] = model_name
                rows.append(res)
                jf.write(json.dumps(res, ensure_ascii=False) + "\n")
                if idx % 10 == 0 or idx == len(PROMPTS_40):
                    print(f"  [{idx}/{len(PROMPTS_40)}] {res['category']} -> {res['response'][:30]}...")

        print(f"Saved {len(rows)} responses to {jsonl_path}")

        # Analyze
        analysis = analyze_model_responses(rows, PROMPTS_40)
        summary_results[model_name] = analysis
        print(f"Analysis for {model_name}:")
        print(f"  Factual QA: {analysis['factual_qa']['hits']}/{analysis['factual_qa']['total']} ({analysis['factual_qa']['accuracy_pct']}%)")
        print(f"  Multi-turn Memory: {analysis['multi_turn']['hits']}/{analysis['multi_turn']['total']} ({analysis['multi_turn']['accuracy_pct']}%)")
        print(f"  Empty: {analysis['empty_count']}, Repetition: {analysis['repetition_count']}, TokenLimit: {analysis['token_limit_count']}")

        # Clean up GPU memory
        del model
        if cfg["adapter_path"]:
            del base_model
        del tokenizer
        torch.cuda.empty_cache()
        gc.collect()

    # Compare Logits Parity between Phase4-Adapter and Phase4-Merged
    print("\n=======================================================")
    print("Computing Logits Parity: Phase4-Adapter vs Phase4-Merged")
    print("=======================================================")
    parity_report = {}
    tokenizer = AutoTokenizer.from_pretrained(models_config["phase4-merged"]["tokenizer_path"])

    for pid in LOGITS_VERIFY_IDS:
        p_item = next(p for p in PROMPTS_40 if p["id"] == pid)
        l_ad = adapter_logits_dict[pid]
        l_me = merged_logits_dict[pid]

        abs_diff = torch.abs(l_ad - l_me)
        max_diff = float(abs_diff.max())
        mean_diff = float(abs_diff.mean())
        cos_sim = float(F.cosine_similarity(l_ad.unsqueeze(0), l_me.unsqueeze(0)).item())

        top5_ad_vals, top5_ad_idx = torch.topk(F.softmax(l_ad, dim=-1), k=5)
        top5_me_vals, top5_me_idx = torch.topk(F.softmax(l_me, dim=-1), k=5)

        top5_ad = [
            {"token_id": int(i), "token": tokenizer.decode([int(i)]), "prob": float(p)}
            for i, p in zip(top5_ad_idx, top5_ad_vals)
        ]
        top5_me = [
            {"token_id": int(i), "token": tokenizer.decode([int(i)]), "prob": float(p)}
            for i, p in zip(top5_me_idx, top5_me_vals)
        ]

        top1_match = top5_ad[0]["token_id"] == top5_me[0]["token_id"]

        parity_report[str(pid)] = {
            "prompt": p_item["prompt"],
            "max_abs_diff": max_diff,
            "mean_abs_diff": mean_diff,
            "cosine_similarity": cos_sim,
            "top1_match": top1_match,
            "adapter_top5": top5_ad,
            "merged_top5": top5_me,
        }
        print(f"Prompt {pid}: '{p_item['prompt']}'")
        print(f"  Cosine Sim: {cos_sim:.6f}, Max Diff: {max_diff:.6f}, Top1 Match: {top1_match}")
        print(f"  Adapter Top1: {repr(top5_ad[0]['token'])} ({top5_ad[0]['prob']:.4f})")
        print(f"  Merged  Top1: {repr(top5_me[0]['token'])} ({top5_me[0]['prob']:.4f})")

    parity_file = out_dir / "logits_parity.json"
    with open(parity_file, "w", encoding="utf-8") as f:
        json.dump(parity_report, f, indent=2, ensure_ascii=False)
    print(f"Saved logits parity report to {parity_file}")

    summary_file = out_dir / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=2, ensure_ascii=False)
    print(f"Saved summary comparison to {summary_file}")
    print("\nDiagnosis completed successfully!")


if __name__ == "__main__":
    main()
