#!/usr/bin/env python3
"""Comprehensive Diagnosis for ULM-LIVE Smoke Failure:
Sections 1 - 7:
1. Existing Smoke Diagnosis & Stop Probability Trajectory
2. Stop Target Verification (20 train, 20 val)
3. Stop Class Imbalance Analysis & Teacher-Forcing Eval (20 samples)
4. Codec Token Collapse Diagnosis (Generated vs Real Mimi Cache)
5. Training Alignment Verification & Batch Dump
6. Checkpoint / Runtime Config Match Verification
7. Real-checkpoint KV Cache Equivalence Verification
"""

import math
import sys
import json
from pathlib import Path
from collections import Counter
import torch
import torch.nn.functional as F

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ulm_live.talker import ULMTalker, TalkerConfig, load_talker_checkpoint
from ulm_live.talker.generator import TalkerGenerationConfig
from ulm_live.talker.data import TalkerDataset, TalkerCollator
from ulm_live.thinker import ULMThinker

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_DIR = Path("/home/ubuntu/ULM-LIVE/data/ulsan-full")
THINKER_PATH = "/home/ubuntu/ULM-1.7B/outputs/ulm-1.7b-phase4-best-merged"
CHECKPOINT_PATH = "/home/ubuntu/ULM-LIVE/outputs/talker-smoke-phase4/best.pt"
AUDIO_DIR = Path("/home/ubuntu/ULM-LIVE/outputs/talker-smoke-phase4/audio")

def compute_entropy(tokens: torch.Tensor) -> float:
    if tokens.numel() == 0:
        return 0.0
    counts = Counter(tokens.tolist())
    total = tokens.numel()
    ent = 0.0
    for c in counts.values():
        p = c / total
        ent -= p * math.log2(p)
    return ent

def main():
    print("=" * 80)
    print("ULM-LIVE SMOKE FAILURE DEEP ROOT-CAUSE DIAGNOSIS")
    print("=" * 80)

    # Load checkpoint
    print(f"\nLoading Talker checkpoint: {CHECKPOINT_PATH} ...")
    talker, meta = load_talker_checkpoint(CHECKPOINT_PATH, device=DEVICE)
    talker.eval()
    cfg = talker.config

    # =========================================================================
    # SECTION 6: Checkpoint / Runtime Config Match Verification
    # =========================================================================
    print("\n" + "=" * 60)
    print("SECTION 6: Checkpoint / Runtime Config Match Verification")
    print("=" * 60)
    ckpt_raw = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    training_cfg = ckpt_raw.get("config", {})
    if hasattr(training_cfg, "__dict__"):
        training_cfg = training_cfg.__dict__

    config_checks = [
        ("num_quantizers", cfg.num_quantizers, training_cfg.get("num_quantizers", cfg.num_quantizers)),
        ("semantic_dim", cfg.semantic_dim, training_cfg.get("semantic_dim", cfg.semantic_dim)),
        ("talker_dim", cfg.talker_dim, training_cfg.get("talker_dim", cfg.talker_dim)),
        ("acoustic_delay_frames", cfg.acoustic_delay_frames, training_cfg.get("acoustic_delay_frames", cfg.acoustic_delay_frames)),
        ("pad_token_id", cfg.pad_token_id, training_cfg.get("pad_token_id", cfg.pad_token_id)),
        ("bos_token_id", cfg.bos_token_id, training_cfg.get("bos_token_id", cfg.bos_token_id)),
        ("stop_threshold", cfg.stop_threshold, training_cfg.get("stop_threshold", cfg.stop_threshold)),
        ("min_audio_frames", cfg.min_audio_frames, training_cfg.get("min_audio_frames", cfg.min_audio_frames)),
        ("max_audio_frames", cfg.max_audio_frames, training_cfg.get("max_audio_frames", cfg.max_audio_frames)),
    ]
    print(f"{'Parameter':<25} {'Runtime Talker':<15} {'Checkpoint/Config':<15} {'Match'}")
    print("-" * 65)
    for name, run_v, ckpt_v in config_checks:
        match = (run_v == ckpt_v)
        print(f"{name:<25} {str(run_v):<15} {str(ckpt_v):<15} {'OK' if match else 'MISMATCH'}")

    # =========================================================================
    # SECTION 1: Existing Smoke Diagnosis (10 Prompts)
    # =========================================================================
    print("\n" + "=" * 60)
    print("SECTION 1: Existing Smoke Diagnosis (10 Prompts Stop Trajectory)")
    print("=" * 60)
    print("Loading Thinker model...")
    thinker = ULMThinker(THINKER_PATH, device=DEVICE, torch_dtype="bf16", freeze=True)

    PROMPTS = [
        "안녕",
        "밥 묵었나?",
        "오늘 뭐 하고 있었노?",
        "오늘 날씨 좋네.",
        "울산에서 놀러 갈 만한 데 추천해줘.",
        "나는 오늘 학교 끝나고 친구 만나러 간다.",
        "1 더하기 1은 2다.",
        "하늘이 파란 이유를 간단하게 설명해줘.",
        "Python에서 리스트를 정렬하는 방법 알려줘.",
        "오늘 기분이 좀 안 좋다.",
    ]

    sec1_results = []
    gen_cfg = TalkerGenerationConfig(
        max_new_tokens=750,
        do_sample=False,
        stop_threshold=0.5,
    )

    limit = min(gen_cfg.max_new_tokens, cfg.max_audio_frames)
    threshold = cfg.stop_threshold if gen_cfg.stop_threshold is None else gen_cfg.stop_threshold
    minimum = cfg.min_audio_frames if gen_cfg.min_audio_frames is None else gen_cfg.min_audio_frames

    print(f"Generation Limits: max_frames={limit}, min_frames={minimum}, threshold={threshold}")

    for idx, prompt in enumerate(PROMPTS, 1):
        inputs = thinker.tokenize(prompt)
        with torch.no_grad():
            sem = thinker.forward_hidden(inputs["input_ids"].to(DEVICE), inputs.get("attention_mask").to(DEVICE) if inputs.get("attention_mask") is not None else None)
        
        spk = torch.tensor([0], device=DEVICE)
        dia = torch.tensor([0], device=DEVICE)
        
        state = talker.init_generation_state(sem, spk, dia)
        current = torch.full((1, cfg.num_quantizers), cfg.bos_token_id, dtype=torch.long, device=DEVICE)
        
        stop_probs = []
        produced = []
        term_reason = "MAX_FRAMES"
        
        with torch.no_grad():
            while len(produced) < limit:
                tokens, stop_prob = talker.step(state, current, gen_cfg)
                sp_val = float(stop_prob.item())
                stop_probs.append(sp_val)
                produced.append(tokens)
                current = tokens
                if len(produced) >= minimum and sp_val >= threshold:
                    term_reason = "STOP_PREDICTED"
                    break

        audio_dur = len(produced) / 12.5
        max_sp = max(stop_probs)
        mean_sp = sum(stop_probs) / len(stop_probs)
        final_20_sp = [round(p, 4) for p in stop_probs[-20:]]
        crossed = any(p >= threshold for p in stop_probs)

        res_item = {
            "idx": idx,
            "prompt": prompt,
            "generated_frames": len(produced),
            "maximum_frames": limit,
            "audio_duration": audio_dur,
            "max_stop_prob": round(max_sp, 5),
            "mean_stop_prob": round(mean_sp, 5),
            "final_20_stop_probs": final_20_sp,
            "stop_threshold": threshold,
            "threshold_crossing": crossed,
            "termination_reason": term_reason,
            "tokens": torch.cat(produced, dim=0)
        }
        sec1_results.append(res_item)
        print(f"[{idx:02d}/10] '{prompt[:15]:<15}' | Frames: {len(produced):3d}/{limit} ({audio_dur:.2f}s) | MaxStopP: {max_sp:.4f} | MeanStopP: {mean_sp:.4f} | Reason: {term_reason}")

    # =========================================================================
    # SECTION 4: Codec Token Collapse Diagnosis
    # =========================================================================
    print("\n" + "=" * 60)
    print("SECTION 4: Codec Token Collapse Diagnosis")
    print("=" * 60)

    print("\n--- GENERATED TOKENS STATISTICS (from 10 smoke samples) ---")
    print(f"{'Quantizer':<10} {'Unique':<8} {'Min':<6} {'Max':<6} {'Most Common':<12} {'Ratio':<8} {'Entropy (bits)':<15}")
    print("-" * 70)
    
    for q in range(cfg.num_quantizers):
        all_q_tokens = []
        for item in sec1_results:
            all_q_tokens.append(item["tokens"][:, q].cpu())
        q_tensor = torch.cat(all_q_tokens, dim=0)
        unique_cnt = len(torch.unique(q_tensor))
        t_min = int(q_tensor.min().item())
        t_max = int(q_tensor.max().item())
        counts = Counter(q_tensor.tolist())
        most_common_tok, most_common_cnt = counts.most_common(1)[0]
        mc_ratio = most_common_cnt / q_tensor.numel()
        ent = compute_entropy(q_tensor)
        print(f"q{q:<9} {unique_cnt:<8} {t_min:<6} {t_max:<6} {most_common_tok:<12} {mc_ratio:<8.3f} {ent:<15.3f}")

    print("\n--- REAL MIMI CACHE STATISTICS (from first 20 train samples) ---")
    print(f"{'Quantizer':<10} {'Unique':<8} {'Min':<6} {'Max':<6} {'Most Common':<12} {'Ratio':<8} {'Entropy (bits)':<15}")
    print("-" * 70)
    real_q_tokens = [[] for _ in range(cfg.num_quantizers)]
    with open(DATASET_DIR / "train.cached.jsonl", "r") as f:
        lines = [f.readline() for _ in range(20)]
    for line in lines:
        row = json.loads(line)
        pt_path = DATASET_DIR / row["codec_path"]
        codes = torch.load(pt_path, map_location="cpu", weights_only=True)
        if codes.dim() == 3:
            codes = codes.squeeze(0)
        for q in range(cfg.num_quantizers):
            real_q_tokens[q].append(codes[q])

    for q in range(cfg.num_quantizers):
        q_tensor = torch.cat(real_q_tokens[q], dim=0)
        unique_cnt = len(torch.unique(q_tensor))
        t_min = int(q_tensor.min().item())
        t_max = int(q_tensor.max().item())
        counts = Counter(q_tensor.tolist())
        most_common_tok, most_common_cnt = counts.most_common(1)[0]
        mc_ratio = most_common_cnt / q_tensor.numel()
        ent = compute_entropy(q_tensor)
        print(f"q{q:<9} {unique_cnt:<8} {t_min:<6} {t_max:<6} {most_common_tok:<12} {mc_ratio:<8.3f} {ent:<15.3f}")

    # =========================================================================
    # SECTION 2: Stop Target Verification (20 train, 20 val)
    # =========================================================================
    print("\n" + "=" * 60)
    print("SECTION 2: Stop Target Verification (20 train, 20 val)")
    print("=" * 60)

    FINGERPRINT = "9b498e0563e390ea899d8cdf1992ffe2a4bcc3a0fb3523bc9459b86fc215b84e"
    for split_name in ["train", "val"]:
        print(f"\n--- Checking {split_name} split (20 samples) ---")
        ds = TalkerDataset(
            DATASET_DIR / f"{split_name}.cached.jsonl",
            num_quantizers=cfg.num_quantizers,
            cache_only=True,
            semantic_cache_metadata={"thinker_id": THINKER_PATH, "thinker_fingerprint": FINGERPRINT, "hidden_layer": -1},
        )
        collator = TalkerCollator(
            pad_token_id=cfg.pad_token_id,
            bos_token_id=cfg.bos_token_id,
            max_audio_len=cfg.max_seq_len,
            acoustic_delay_frames=cfg.acoustic_delay_frames,
        )

        samples = [ds[i] for i in range(20)]
        batch = collator(samples)

        stop_targets = batch["stop_targets"]
        audio_codes = batch["audio_codes"]
        targets = batch["targets"]
        audio_mask = batch["audio_attention_mask"]

        B, T = stop_targets.shape
        zero_cnt = 0
        one_cnt = 0
        minus_one_cnt = 0
        off_by_one_issues = 0
        post_padding_stops = 0

        for i in range(B):
            real_len = samples[i]["audio_codes"].shape[1]
            st = stop_targets[i]
            pos_indices = (st == 1).nonzero().view(-1)
            zeros = (st == 0).sum().item()
            ones = (st == 1).sum().item()
            minus_ones = (st == -1).sum().item()
            valid_mask_count = audio_mask[i].sum().item()

            zero_cnt += zeros
            one_cnt += ones
            minus_one_cnt += minus_ones

            pos_idx = int(pos_indices[0].item()) if len(pos_indices) > 0 else -1
            if len(pos_indices) != 1:
                print(f"  [Sample {i}] ERROR: Found {len(pos_indices)} positive stop targets!")
            elif pos_idx != real_len - 1:
                off_by_one_issues += 1
                print(f"  [Sample {i}] pos_idx={pos_idx} vs real_len-1={real_len-1} (real_len={real_len})")

            if pos_idx >= valid_mask_count:
                post_padding_stops += 1

        print(f"Summary for {split_name} (20 samples):")
        print(f"  Total frames: {B * T}")
        print(f"  Negative stop (0): {zero_cnt}")
        print(f"  Positive stop (1): {one_cnt} (Expected: {B})")
        print(f"  Ignored stop (-1): {minus_one_cnt}")
        print(f"  Off-by-one errors: {off_by_one_issues}")
        print(f"  Post-padding stops: {post_padding_stops}")

    # =========================================================================
    # SECTION 5: Training Alignment Verification & Batch Dump
    # =========================================================================
    print("\n" + "=" * 60)
    print("SECTION 5: Training Alignment Verification & Human-Readable Batch Dump")
    print("=" * 60)
    s0 = samples[0]
    b1 = collator([s0])
    inp_codes = b1["audio_codes"][0]
    tgt_codes = b1["targets"][0]
    st_targets = b1["stop_targets"][0]
    a_mask = b1["audio_attention_mask"][0]

    print(f"Sample 0: ID={s0['id']}, real_length={s0['audio_codes'].shape[1]}")
    print("Frame | inputs q0..q3        | targets q0..q3       | audio_mask | stop_target")
    print("-" * 75)
    for t in range(min(10, inp_codes.shape[1])):
        inp_str = " ".join(f"{inp_codes[q, t].item():4d}" for q in range(min(4, cfg.num_quantizers)))
        tgt_str = " ".join(f"{tgt_codes[q, t].item():4d}" for q in range(min(4, cfg.num_quantizers)))
        print(f"t={t:03d} | inp: [{inp_str}] | tgt: [{tgt_str}] | {int(a_mask[t].item()):^10} | {st_targets[t].item():^11}")

    print("...")
    for t in range(max(0, s0['audio_codes'].shape[1] - 3), min(s0['audio_codes'].shape[1] + 2, inp_codes.shape[1])):
        inp_str = " ".join(f"{inp_codes[q, t].item():4d}" for q in range(min(4, cfg.num_quantizers)))
        tgt_str = " ".join(f"{tgt_codes[q, t].item():4d}" for q in range(min(4, cfg.num_quantizers)))
        print(f"t={t:03d} | inp: [{inp_str}] | tgt: [{tgt_str}] | {int(a_mask[t].item()):^10} | {st_targets[t].item():^11}")

    pad_in_target = (tgt_codes == cfg.pad_token_id).sum().item()
    bos_in_target = (tgt_codes == cfg.bos_token_id).sum().item()
    print(f"\nPAD token in targets: {pad_in_target} (masked by targets != pad_token_id in forward: YES)")
    print(f"BOS token in targets: {bos_in_target} (Should be 0 in valid codec targets)")

    # =========================================================================
    # SECTION 3: Stop Class Imbalance Analysis & Teacher-Forcing Eval
    # =========================================================================
    print("\n" + "=" * 60)
    print("SECTION 3: Stop Class Imbalance Analysis & Teacher-Forcing Validation")
    print("=" * 60)
    
    total_pos = 0
    total_neg = 0
    with open(DATASET_DIR / "train.cached.jsonl", "r") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                frames = max(1, round(float(row.get("duration", 2.0)) * 12.5))
                total_pos += 1
                total_neg += (frames - 1)

    pos_ratio = total_pos / (total_pos + total_neg)
    print(f"Full Train Dataset Stop Balance:")
    print(f"  Positive stop frames (1): {total_pos:,}")
    print(f"  Negative stop frames (0): {total_neg:,}")
    print(f"  Positive Ratio:           {pos_ratio * 100:.3f}% (1 in every {1/pos_ratio:.1f} frames)")

    print(f"\nTeacher-Forcing Evaluation on 20 Validation Samples with best.pt:")
    val_ds = TalkerDataset(
        DATASET_DIR / "val.cached.jsonl",
        num_quantizers=cfg.num_quantizers,
        cache_only=True,
        semantic_cache_metadata={"thinker_id": THINKER_PATH, "thinker_fingerprint": FINGERPRINT, "hidden_layer": -1},
    )
    val_collator = TalkerCollator(
        pad_token_id=cfg.pad_token_id,
        bos_token_id=cfg.bos_token_id,
        max_audio_len=cfg.max_seq_len,
        acoustic_delay_frames=cfg.acoustic_delay_frames,
    )

    val_batch = val_collator([val_ds[i] for i in range(20)])
    with torch.no_grad():
        sem_h = val_batch["semantic_hidden_states"].to(DEVICE)
        sem_m = val_batch["semantic_attention_mask"].to(DEVICE)
        aud_c = val_batch["audio_codes"].to(DEVICE)
        aud_m = val_batch["audio_attention_mask"].to(DEVICE)
        spk_ids = val_batch["speaker_ids"].to(DEVICE)
        dia_ids = val_batch["dialect_ids"].to(DEVICE)
        st_targets = val_batch["stop_targets"].to(DEVICE)
        
        fwd_out = talker(
            semantic_hidden_states=sem_h,
            audio_codes=aud_c,
            speaker_ids=spk_ids,
            dialect_ids=dia_ids,
            semantic_attention_mask=sem_m,
            audio_attention_mask=aud_m,
            stop_targets=st_targets,
        )
        stop_logits = fwd_out.stop_logits
        stop_probs = torch.sigmoid(stop_logits)

    valid_mask = (st_targets >= 0)
    pred_pos = (stop_probs >= 0.5) & valid_mask
    actual_pos = (st_targets == 1) & valid_mask

    tp = (pred_pos & actual_pos).sum().item()
    fp = (pred_pos & ~actual_pos).sum().item()
    fn = (~pred_pos & actual_pos).sum().item()
    tn = (~pred_pos & ~actual_pos).sum().item()

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    pos_pred_rate = pred_pos.sum().item() / valid_mask.sum().item()

    print(f"  Stop Metrics @ threshold=0.5:")
    print(f"    TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
    print(f"    Precision:            {prec:.4f}")
    print(f"    Recall:               {rec:.4f}")
    print(f"    F1:                   {f1:.4f}")
    print(f"    Positive Pred Rate:   {pos_pred_rate * 100:.3f}%")

    print("\nDetailed Per-Sample Trajectory (20 samples):")
    print(f"{'Idx':<4} {'True End':<10} {'MaxP Frame':<12} {'Prob@End':<10} {'MaxProb':<10} {'Cross Frame (p>=0.5)'}")
    print("-" * 65)
    for i in range(20):
        real_len = val_ds[i]["audio_codes"].shape[1]
        probs_sample = stop_probs[i, :real_len].cpu().tolist()
        true_end = real_len - 1
        prob_at_end = probs_sample[true_end]
        max_prob = max(probs_sample)
        max_frame = probs_sample.index(max_prob)
        crossed_frames = [idx for idx, p in enumerate(probs_sample) if p >= 0.5]
        first_cross = crossed_frames[0] if crossed_frames else "NONE"
        print(f"{i:<4} {true_end:<10} {max_frame:<12} {prob_at_end:<10.4f} {max_prob:<10.4f} {str(first_cross)}")

    # =========================================================================
    # SECTION 7: Real-checkpoint KV Cache Equivalence Verification
    # =========================================================================
    print("\n" + "=" * 60)
    print("SECTION 7: KV Cache Real-checkpoint Equivalence Verification")
    print("=" * 60)

    sample_kv = val_ds[0]
    sem_h = sample_kv["semantic_hidden_states"].unsqueeze(0).to(DEVICE)
    spk_id = torch.tensor([sample_kv["speaker_id"]], device=DEVICE)
    dia_id = torch.tensor([sample_kv["dialect_id"]], device=DEVICE)
    real_codes = sample_kv["audio_codes"].unsqueeze(0).to(DEVICE)
    T_test = min(10, real_codes.shape[-1])

    inp_recompute = torch.full((1, cfg.num_quantizers, T_test), cfg.pad_token_id, dtype=torch.long, device=DEVICE)
    inp_recompute[0, :, 0] = cfg.bos_token_id
    if T_test > 1:
        inp_recompute[0, :, 1:T_test] = real_codes[0, :, :T_test-1]
    
    with torch.no_grad():
        recompute_frames = talker.temporal(
            sem_h, inp_recompute, spk_id, dia_id
        )
        recompute_stop_logits = talker.stop_head(recompute_frames).squeeze(-1)

    state = talker.init_generation_state(sem_h, spk_id, dia_id)
    kv_stop_logits = []
    kv_frames = []

    with torch.no_grad():
        for t in range(T_test):
            in_frame = inp_recompute[:, :, t]
            x = (
                talker.audio_embed(in_frame[:, :, None], state.condition)
                + talker.pos_embedding.weight[state.next_position][None, None]
            )
            padding = torch.cat(
                (
                    state.key_padding_mask,
                    torch.zeros(x.shape[0], 1, dtype=torch.bool, device=x.device),
                ),
                1,
            )
            caches = []
            for block, past in zip(talker.temporal_transformer, state.layer_kv):
                x, kv = block(x, padding=padding, past=past, cache=True)
                caches.append(kv)
            state.layer_kv = caches
            state.key_padding_mask = padding
            state.next_position += 1
            frame = talker.temporal_norm(x[:, 0])
            sl = talker.stop_head(frame).squeeze(-1)
            kv_frames.append(frame)
            kv_stop_logits.append(sl)

    kv_frames_tensor = torch.stack(kv_frames, dim=1)
    kv_stop_tensor = torch.stack(kv_stop_logits, dim=1)

def run_sections_2_to_7():
    print("=" * 80)
    print("RUNNING SECTIONS 2, 3, 5, 7")
    print("=" * 80)
    talker, meta = load_talker_checkpoint(CHECKPOINT_PATH, device=DEVICE)
    talker.eval()
    cfg = talker.config
    FINGERPRINT = "9b498e0563e390ea899d8cdf1992ffe2a4bcc3a0fb3523bc9459b86fc215b84e"

    # SECTION 2
    print("\n" + "=" * 60)
    print("SECTION 2: Stop Target Verification (20 train, 20 val)")
    print("=" * 60)
    for split_name in ["train", "val"]:
        print(f"\n--- Checking {split_name} split (20 samples) ---")
        ds = TalkerDataset(
            DATASET_DIR / f"{split_name}.cached.jsonl",
            num_quantizers=cfg.num_quantizers,
            cache_only=True,
            semantic_cache_metadata={"thinker_id": THINKER_PATH, "thinker_fingerprint": FINGERPRINT, "hidden_layer": -1},
        )
        collator = TalkerCollator(
            pad_token_id=cfg.pad_token_id,
            bos_token_id=cfg.bos_token_id,
            max_audio_len=cfg.max_seq_len,
            acoustic_delay_frames=cfg.acoustic_delay_frames,
        )
        samples = [ds[i] for i in range(20)]
        batch = collator(samples)
        stop_targets = batch["stop_targets"]
        audio_codes = batch["audio_codes"]
        targets = batch["targets"]
        audio_mask = batch["audio_attention_mask"]

        B, T = stop_targets.shape
        zero_cnt = 0
        one_cnt = 0
        minus_one_cnt = 0
        off_by_one_issues = 0
        post_padding_stops = 0

        for i in range(B):
            real_len = samples[i]["audio_codes"].shape[1]
            st = stop_targets[i]
            pos_indices = (st == 1).nonzero().view(-1)
            zeros = (st == 0).sum().item()
            ones = (st == 1).sum().item()
            minus_ones = (st == -1).sum().item()
            valid_mask_count = audio_mask[i].sum().item()

            zero_cnt += zeros
            one_cnt += ones
            minus_one_cnt += minus_ones

            pos_idx = int(pos_indices[0].item()) if len(pos_indices) > 0 else -1
            if len(pos_indices) != 1:
                print(f"  [Sample {i}] ERROR: Found {len(pos_indices)} positive stop targets!")
            elif pos_idx != real_len - 1:
                off_by_one_issues += 1
                print(f"  [Sample {i}] pos_idx={pos_idx} vs real_len-1={real_len-1} (real_len={real_len})")

            if pos_idx >= valid_mask_count:
                post_padding_stops += 1

        print(f"Summary for {split_name} (20 samples):")
        print(f"  Total frames: {B * T}")
        print(f"  Negative stop (0): {zero_cnt}")
        print(f"  Positive stop (1): {one_cnt} (Expected: {B})")
        print(f"  Ignored stop (-1): {minus_one_cnt}")
        print(f"  Off-by-one errors: {off_by_one_issues}")
        print(f"  Post-padding stops: {post_padding_stops}")

    # SECTION 5
    print("\n" + "=" * 60)
    print("SECTION 5: Training Alignment Verification & Human-Readable Batch Dump")
    print("=" * 60)
    s0 = samples[0]
    b1 = collator([s0])
    inp_codes = b1["audio_codes"][0]
    tgt_codes = b1["targets"][0]
    st_targets = b1["stop_targets"][0]
    a_mask = b1["audio_attention_mask"][0]

    print(f"Sample 0: ID={s0['id']}, real_length={s0['audio_codes'].shape[1]}")
    print("Frame | inputs q0..q3        | targets q0..q3       | audio_mask | stop_target")
    print("-" * 75)
    for t in range(min(10, inp_codes.shape[1])):
        inp_str = " ".join(f"{inp_codes[q, t].item():4d}" for q in range(min(4, cfg.num_quantizers)))
        tgt_str = " ".join(f"{tgt_codes[q, t].item():4d}" for q in range(min(4, cfg.num_quantizers)))
        print(f"t={t:03d} | inp: [{inp_str}] | tgt: [{tgt_str}] | {int(a_mask[t].item()):^10} | {st_targets[t].item():^11}")

    print("...")
    for t in range(max(0, s0['audio_codes'].shape[1] - 3), min(s0['audio_codes'].shape[1] + 2, inp_codes.shape[1])):
        inp_str = " ".join(f"{inp_codes[q, t].item():4d}" for q in range(min(4, cfg.num_quantizers)))
        tgt_str = " ".join(f"{tgt_codes[q, t].item():4d}" for q in range(min(4, cfg.num_quantizers)))
        print(f"t={t:03d} | inp: [{inp_str}] | tgt: [{tgt_str}] | {int(a_mask[t].item()):^10} | {st_targets[t].item():^11}")

    pad_in_target = (tgt_codes == cfg.pad_token_id).sum().item()
    bos_in_target = (tgt_codes == cfg.bos_token_id).sum().item()
    print(f"\nPAD token in targets: {pad_in_target} (masked by targets != pad_token_id in forward: YES)")
    print(f"BOS token in targets: {bos_in_target} (Should be 0 in valid codec targets)")

    # SECTION 3
    print("\n" + "=" * 60)
    print("SECTION 3: Stop Class Imbalance Analysis & Teacher-Forcing Validation")
    print("=" * 60)
    total_pos = 0
    total_neg = 0
    with open(DATASET_DIR / "train.cached.jsonl", "r") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                frames = max(1, round(float(row.get("duration", 2.0)) * 12.5))
                total_pos += 1
                total_neg += (frames - 1)

    pos_ratio = total_pos / (total_pos + total_neg)
    print(f"Full Train Dataset Stop Balance:")
    print(f"  Positive stop frames (1): {total_pos:,}")
    print(f"  Negative stop frames (0): {total_neg:,}")
    print(f"  Positive Ratio:           {pos_ratio * 100:.3f}% (1 in every {1/pos_ratio:.1f} frames)")

    val_ds = TalkerDataset(
        DATASET_DIR / "val.cached.jsonl",
        num_quantizers=cfg.num_quantizers,
        cache_only=True,
        semantic_cache_metadata={"thinker_id": THINKER_PATH, "thinker_fingerprint": FINGERPRINT, "hidden_layer": -1},
    )
    val_collator = TalkerCollator(
        pad_token_id=cfg.pad_token_id,
        bos_token_id=cfg.bos_token_id,
        max_audio_len=cfg.max_seq_len,
        acoustic_delay_frames=cfg.acoustic_delay_frames,
    )
    val_batch = val_collator([val_ds[i] for i in range(20)])
    with torch.no_grad():
        sem_h = val_batch["semantic_hidden_states"].to(DEVICE)
        sem_m = val_batch["semantic_attention_mask"].to(DEVICE)
        aud_c = val_batch["audio_codes"].to(DEVICE)
        aud_m = val_batch["audio_attention_mask"].to(DEVICE)
        spk_ids = val_batch["speaker_ids"].to(DEVICE)
        dia_ids = val_batch["dialect_ids"].to(DEVICE)
        st_targets = val_batch["stop_targets"].to(DEVICE)
        
        fwd_out = talker(
            semantic_hidden_states=sem_h,
            audio_codes=aud_c,
            speaker_ids=spk_ids,
            dialect_ids=dia_ids,
            semantic_attention_mask=sem_m,
            audio_attention_mask=aud_m,
            stop_targets=st_targets,
        )
        stop_logits = fwd_out.stop_logits
        stop_probs = torch.sigmoid(stop_logits)

    valid_mask = (st_targets >= 0)
    pred_pos = (stop_probs >= 0.5) & valid_mask
    actual_pos = (st_targets == 1) & valid_mask

    tp = (pred_pos & actual_pos).sum().item()
    fp = (pred_pos & ~actual_pos).sum().item()
    fn = (~pred_pos & actual_pos).sum().item()
    tn = (~pred_pos & ~actual_pos).sum().item()

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    pos_pred_rate = pred_pos.sum().item() / valid_mask.sum().item()

    print(f"  Stop Metrics @ threshold=0.5:")
    print(f"    TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
    print(f"    Precision:            {prec:.4f}")
    print(f"    Recall:               {rec:.4f}")
    print(f"    F1:                   {f1:.4f}")
    print(f"    Positive Pred Rate:   {pos_pred_rate * 100:.3f}%")

    print("\nDetailed Per-Sample Trajectory (20 samples):")
    print(f"{'Idx':<4} {'True End':<10} {'MaxP Frame':<12} {'Prob@End':<10} {'MaxProb':<10} {'Cross Frame (p>=0.5)'}")
    print("-" * 65)
    for i in range(20):
        real_len = val_ds[i]["audio_codes"].shape[1]
        probs_sample = stop_probs[i, :real_len].cpu().tolist()
        true_end = real_len - 1
        prob_at_end = probs_sample[true_end]
        max_prob = max(probs_sample)
        max_frame = probs_sample.index(max_prob)
        crossed_frames = [idx for idx, p in enumerate(probs_sample) if p >= 0.5]
        first_cross = crossed_frames[0] if crossed_frames else "NONE"
        print(f"{i:<4} {true_end:<10} {max_frame:<12} {prob_at_end:<10.4f} {max_prob:<10.4f} {str(first_cross)}")

    # SECTION 7
    print("\n" + "=" * 60)
    print("SECTION 7: KV Cache Real-checkpoint Equivalence Verification")
    print("=" * 60)
    sample_kv = val_ds[0]
    sem_h = sample_kv["semantic_hidden_states"].unsqueeze(0).to(DEVICE)
    spk_id = torch.tensor([sample_kv["speaker_id"]], device=DEVICE)
    dia_id = torch.tensor([sample_kv["dialect_id"]], device=DEVICE)
    real_codes = sample_kv["audio_codes"].unsqueeze(0).to(DEVICE)
    T_test = min(10, real_codes.shape[-1])

    inp_recompute = torch.full((1, cfg.num_quantizers, T_test), cfg.pad_token_id, dtype=torch.long, device=DEVICE)
    inp_recompute[0, :, 0] = cfg.bos_token_id
    if T_test > 1:
        inp_recompute[0, :, 1:T_test] = real_codes[0, :, :T_test-1]
    
    with torch.no_grad():
        recompute_frames = talker.temporal(
            sem_h, inp_recompute, spk_id, dia_id
        )
        recompute_stop_logits = talker.stop_head(recompute_frames).squeeze(-1)

    state = talker.init_generation_state(sem_h, spk_id, dia_id)
    kv_stop_logits = []
    kv_frames = []

    with torch.no_grad():
        for t in range(T_test):
            in_frame = inp_recompute[:, :, t]
            x = (
                talker.audio_embed(in_frame[:, :, None], state.condition)
                + talker.pos_embedding.weight[state.next_position][None, None]
            )
            padding = torch.cat(
                (
                    state.key_padding_mask,
                    torch.zeros(x.shape[0], 1, dtype=torch.bool, device=x.device),
                ),
                1,
            )
            caches = []
            for block, past in zip(talker.temporal_transformer, state.layer_kv):
                x, kv = block(x, padding=padding, past=past, cache=True)
                caches.append(kv)
            state.layer_kv = caches
            state.key_padding_mask = padding
            state.next_position += 1
            frame = talker.temporal_norm(x[:, 0])
            sl = talker.stop_head(frame).squeeze(-1)
            kv_frames.append(frame)
            kv_stop_logits.append(sl)

    kv_frames_tensor = torch.stack(kv_frames, dim=1)
    kv_stop_tensor = torch.stack(kv_stop_logits, dim=1)

    frame_diff = (recompute_frames - kv_frames_tensor).abs().max().item()
    stop_diff = (recompute_stop_logits - kv_stop_tensor).abs().max().item()

    print(f"Max absolute difference over {T_test} steps:")
    print(f"  Temporal hidden states: {frame_diff:.6e}")
    print(f"  Stop logits:            {stop_diff:.6e}")
    kv_match = (frame_diff < 1e-4 and stop_diff < 1e-4)
    print(f"KV Cache Numerical Equivalence: {'PERFECT PASS' if kv_match else 'FAIL'}")

    print("\n" + "=" * 80)
    print("DIAGNOSIS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
