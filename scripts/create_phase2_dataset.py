#!/usr/bin/env python3
"""Phase 2 dataset generation script using SequenceMatcher similarity heuristic."""

from __future__ import annotations

import difflib
import gzip
import json
from collections import Counter
from pathlib import Path


def generate_phase2_dataset(
    src_dir: str | Path = "data/ulsan_dialect_dense",
    out_dir: str | Path = "data/ulsan_dialect_phase2",
) -> None:
    src_path = Path(src_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    splits = ["train", "validation", "test"]

    for split in splits:
        file_path = src_path / f"{split}.jsonl.gz"
        if not file_path.exists():
            file_path = src_path / f"{split}.jsonl"

        out_file = out_path / f"{split}.jsonl"
        opener = gzip.open if file_path.suffix == ".gz" else open

        strength_counter = Counter()
        total_samples = 0

        with opener(file_path, "rt", encoding="utf-8") as fin, open(
            out_file, "w", encoding="utf-8"
        ) as fout:
            for line in fin:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("task") != "standard_to_dialect":
                    continue

                std = rec["standard_text"]
                dia = rec["dialect_text"]
                sim = difflib.SequenceMatcher(None, std, dia).ratio()

                if sim <= 0.90:
                    strength = 3
                elif sim <= 0.92:
                    strength = 2
                elif sim <= 0.95:
                    strength = 1
                else:
                    continue

                rec["dialect_strength"] = strength
                if "metadata" not in rec or not isinstance(rec["metadata"], dict):
                    rec["metadata"] = {}
                rec["metadata"]["phase2_similarity"] = sim
                rec["metadata"]["phase2_strength_heuristic"] = True

                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                strength_counter[strength] += 1
                total_samples += 1

        print(f"[{split}] Total samples: {total_samples}")
        print(f"  Strength 1: {strength_counter[1]}")
        print(f"  Strength 2: {strength_counter[2]}")
        print(f"  Strength 3: {strength_counter[3]}")


if __name__ == "__main__":
    generate_phase2_dataset()
