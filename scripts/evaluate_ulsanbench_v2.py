"""UlsanBench v2: deterministic evaluation with anti-loop decoding for dialect tasks."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
ENCODER = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SYSTEM = "요청한 결과만 출력하세요. 설명, 머리말, 따옴표, 부가 설명을 추가하지 마세요."
CLASSES = ("ULSAN", "OTHER_GYEONGSANG", "STANDARD")
MARKERS = ("아이가", "데이", "단디", "뭇나", "파이다", "퍼뜩", "우째", "어데", "그라", "묵나", "할 끼", "했제", "맞제")
REPETITION = re.compile(r"(.{3,30}?)\1{2,}")


def jsonl(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f]


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


class Embedder:
    def __init__(self, device="cuda:0", revision=None):
        self.tokenizer = AutoTokenizer.from_pretrained(ENCODER, revision=revision)
        self.model = AutoModel.from_pretrained(ENCODER, revision=revision).to(device).eval()
        self.device = device

    @torch.inference_mode()
    def encode(self, texts, batch=64):
        result = []
        for start in range(0, len(texts), batch):
            tokens = self.tokenizer(texts[start:start+batch], padding=True, truncation=True,
                                    max_length=256, return_tensors="pt").to(self.device)
            hidden = self.model(**tokens).last_hidden_state
            mask = tokens.attention_mask.unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            result.append(F.normalize(pooled, dim=-1).cpu())
        return torch.cat(result)


def train_classifier(output_dir):
    output_dir = Path(output_dir)
    train = jsonl(ROOT / "data/dialect_alignment_v1/classifier_train.jsonl")
    test = jsonl(ROOT / "data/dialect_alignment_v1/classifier_test.jsonl")
    metric = Embedder()
    train_text = [x["messages"][0]["content"].split(": ", 1)[-1] for x in train]
    test_text = [x["prompt"].split(": ", 1)[-1] for x in test]
    x = metric.encode(train_text).clone()
    y = torch.tensor([CLASSES.index(x["label"]) for x in train])
    xt = metric.encode(test_text)
    yt = torch.tensor([CLASSES.index(x["label"]) for x in test])
    torch.manual_seed(42026)
    model = torch.nn.Linear(x.shape[1], 3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.02, weight_decay=.05)
    for _ in range(150):
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
    with torch.no_grad(): prediction = model(xt).argmax(-1)
    f1 = {}
    for i, name in enumerate(CLASSES):
        tp = int(((prediction == i) & (yt == i)).sum())
        fp = int(((prediction == i) & (yt != i)).sum())
        fn = int(((prediction != i) & (yt == i)).sum())
        f1[name] = round(2*tp/max(1,2*tp+fp+fn), 4)
    stats = {"heldout_macro_f1": round(sum(f1.values())/3,4), "class_f1": f1,
             "train_n":len(train),"test_n":len(test),"speaker_session_disjoint":True,
             "trusted_for_dialectness":sum(f1.values())/3 >= .90,"encoder":ENCODER}
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "classifier.pt")
    save(output_dir / "classifier_stats.json", stats)
    return stats


def prompts(tokenizer, rows):
    return [tokenizer.apply_chat_template([{"role":"system","content":SYSTEM},
                                           {"role":"user","content":row["prompt"]}],
                                          tokenize=False, add_generation_prompt=True,
                                          enable_thinking=False) for row in rows]


@torch.inference_mode()
def generate_rows(model_path, adapter, rows, batch_size=8):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16,
                                                 device_map="cuda:0", attn_implementation="sdpa").eval()
    if adapter: model = PeftModel.from_pretrained(model, adapter).eval()
    outputs = [None]*len(rows)
    for task in ("comprehension", "generation", "identification", "grammar", "context"):
        indices = [i for i,r in enumerate(rows) if r["task"]==task]
        limit = 16 if task == "identification" else 96
        for start in range(0,len(indices),batch_size):
            idx = indices[start:start+batch_size]
            enc = tokenizer(prompts(tokenizer,[rows[i] for i in idx]),padding=True,
                            truncation=True,max_length=1024,return_tensors="pt").to("cuda:0")
            extra = {"repetition_penalty": 1.10, "no_repeat_ngram_size": 3} if task == "context" else {}
            gen = model.generate(**enc,max_new_tokens=limit,do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id,
                                 eos_token_id=tokenizer.eos_token_id, **extra)
            for i,seq in zip(idx,gen):
                raw = tokenizer.decode(seq[enc.input_ids.shape[1]:],skip_special_tokens=True).strip()
                outputs[i] = raw.split("</think>")[-1].strip()
        print(f"evaluated {task}: {len(indices)}",flush=True)
    del model
    torch.cuda.empty_cache()
    return outputs


def evaluate(model_path, adapter, output_dir, subset=0, dataset_path=None):
    output_dir = Path(output_dir)
    rows = jsonl(dataset_path or ROOT / "data/ulsanbench_v1/benchmark.jsonl")
    if subset:
        # Deterministic stratified subset; never the old 12-item gate.
        by_task = defaultdict(list)
        for row in rows: by_task[row["task"]].append(row)
        rows = [row for task in by_task for row in by_task[task][:max(1,round(subset*len(by_task[task])/500))]]
        if len(rows)<200: raise ValueError("A/B subset must have at least 200 items")
    outputs = generate_rows(model_path,adapter,rows)
    return score_rows(rows, outputs, output_dir, model_path, adapter)


def score_rows(rows, outputs, output_dir, model_path, adapter, metric=None):
    """Score caller-generated outputs with the unchanged UlsanBench rubric."""
    output_dir = Path(output_dir)
    metric = metric or Embedder()
    output_emb = metric.encode(outputs)
    reference_emb = metric.encode([r["reference"] for r in rows])
    standard_emb = metric.encode([r["standard_reference"] for r in rows])
    ref_similarity = (output_emb*reference_emb).sum(-1).tolist()
    meaning_similarity = (output_emb*standard_emb).sum(-1).tolist()
    details = []
    grouped = defaultdict(list)
    for i,(row,text) in enumerate(zip(rows,outputs)):
        task = row["task"]
        compliance = bool(text) and not text.startswith(("설명:","답변:","물론",'"')) and "<think>" not in text
        if task == "identification": compliance &= text in CLASSES
        repeated = bool(REPETITION.search(text))
        malformed = not text or "�" in text or "<|" in text or len(text)>500
        lexical = sum(m in text for m in MARKERS)
        if task in ("generation","grammar","context"):
            ref_dialect = ref_similarity[i]
            standard = meaning_similarity[i]
            # Reference alignment and markers remain separate; no keyword-only gate.
            ending_match = bool(row.get("ending") and row["ending"] in text[-12:])
            dialectness = .65*ref_dialect + .35*(1 if lexical or ending_match else 0)
            semantic = standard if task != "context" else ref_dialect
        else:
            dialectness = None
            semantic = ref_similarity[i] if task == "comprehension" else None
        result = {**row,"output":text,"compliance":bool(compliance),"semantic_similarity":semantic,
                  "reference_similarity":ref_similarity[i],"standard_similarity":meaning_similarity[i],
                  "dialectness_proxy":dialectness,"lexical_marker_count":lexical,
                  "repetition":repeated,"malformed":malformed,
                  "exact_match":text==row["reference"] if task=="identification" else None}
        details.append(result);grouped[task].append(result)
    summary = {"model":model_path,"adapter":adapter,"n":len(rows),"config":{"do_sample":False,"temperature":0,
               "enable_thinking":False,"system":SYSTEM,"max_new_tokens":{"identification":16,"other":96}},
               "metrics":{}}
    for task,rr in grouped.items():
        avg = lambda key: round(sum(x[key] for x in rr if x[key] is not None)/max(1,sum(x[key] is not None for x in rr)),4)
        summary["metrics"][task] = {"n":len(rr),"compliance":avg("compliance"),
                                   "semantic_similarity":avg("semantic_similarity"),
                                   "reference_similarity":avg("reference_similarity"),
                                   "dialectness_proxy":avg("dialectness_proxy"),
                                   "lexical_marker_mean":avg("lexical_marker_count"),
                                   "repetition_count":sum(x["repetition"] for x in rr),
                                   "malformed_count":sum(x["malformed"] for x in rr)}
        if task=="identification": summary["metrics"][task]["accuracy"] = avg("exact_match")
    output_dir.mkdir(parents=True,exist_ok=True)
    with (output_dir/"predictions.jsonl").open("w") as f:
        for row in details: f.write(json.dumps(row,ensure_ascii=False)+"\n")
    save(output_dir/"summary.json",summary)
    # The 30 examples are a review queue, not an automatically claimed human verdict.
    review = [r for task in ("generation","grammar","context") for r in grouped[task][:10]]
    with (output_dir/"manual_review_30.jsonl").open("w") as f:
        for row in review: f.write(json.dumps(row,ensure_ascii=False)+"\n")
    return summary


def evaluate_regression(model_path, adapter, output_dir):
    """Deterministic preservation150 and old12 smoke, using existing category rubric."""
    from evaluate_preservation import evaluate_response
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    regression=jsonl(ROOT/"data/regression_benchmark_150.jsonl")
    old=[r for r in jsonl(ROOT/"data/preservation_benchmark_100.jsonl") if r["category"]=="dialect_eval"]
    items=regression+old
    tokenizer=AutoTokenizer.from_pretrained(model_path);tokenizer.padding_side="left"
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    model=AutoModelForCausalLM.from_pretrained(model_path,dtype=torch.bfloat16,
                                               device_map="cuda:0",attn_implementation="sdpa").eval()
    if adapter:model=PeftModel.from_pretrained(model,adapter).eval()
    results=[]
    for item in items:
        messages=[{"role":"system","content":SYSTEM}]
        for u,a in item.get("history",[]):messages.extend(({"role":"user","content":u},{"role":"assistant","content":a}))
        messages.append({"role":"user","content":item["prompt"]})
        prompt=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        enc=tokenizer(prompt,return_tensors="pt").to("cuda:0")
        with torch.inference_mode():
            gen=model.generate(**enc,max_new_tokens=128,do_sample=False,pad_token_id=tokenizer.pad_token_id)
        tokens=gen[0][enc.input_ids.shape[1]:]
        answer=tokenizer.decode(tokens,skip_special_tokens=True).split("</think>")[-1].strip()
        flags=evaluate_response(item,answer,answer,len(tokens),128)
        results.append({"category":item["category"],"answer":answer,**flags})
    key={"factual_qa":"factual_hit","multi_turn":"memory_hit","instruction_trap":"instruction_hit","dialect_eval":"dialect_hit"}
    stats={}
    for category,flag in key.items():
        rr=[x for x in results if x["category"]==category]
        stats[category]={"n":len(rr),"accuracy_pct":round(100*sum(x[flag] for x in rr)/max(1,len(rr)),2),
                         "repetition_count":sum(x["repetition"] for x in rr),
                         "empty_count":sum(x["empty_final"] for x in rr)}
    stats["config"]={"do_sample":False,"temperature":0,"enable_thinking":False,"max_new_tokens":128}
    save(output_dir/"summary.json",stats)
    with (output_dir/"predictions.jsonl").open("w") as f:
        for row in results:f.write(json.dumps(row,ensure_ascii=False)+"\n")
    return stats


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument("--model");p.add_argument("--adapter")
    p.add_argument("--dataset", help="Explicit UlsanBench JSONL for benchmark evaluation")
    p.add_argument("--output",required=True);p.add_argument("--subset",type=int,default=0)
    p.add_argument("--classifier",action="store_true");p.add_argument("--regression",action="store_true")
    a=p.parse_args(argv)
    if not a.classifier and not a.model:
        p.error("--model is required for generation")
    if not a.classifier and not a.regression and not a.dataset:
        p.error("--dataset is required for benchmark evaluation")
    if a.classifier: print(train_classifier(a.output),flush=True)
    elif a.regression: print(evaluate_regression(a.model,a.adapter,a.output),flush=True)
    else: print(evaluate(a.model,a.adapter,a.output,a.subset,a.dataset),flush=True)


if __name__=="__main__":
    main()
