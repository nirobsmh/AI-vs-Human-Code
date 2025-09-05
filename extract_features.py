#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Feature extraction pipeline for ai_vs_human_code.jsonl
- Input: JSONL file with fields: problem_id, prompt, human_code, ai_code
- Output: features.csv + features.parquet
"""

import argparse
import json
import pandas as pd
from pathlib import Path

# Reuse the feature functions from earlier
from extract_features import extract_features_for_row  # 👈 make sure extract_features.py is in same folder


def load_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            # Each row produces TWO entries: human + ai
            if "human_code" in row:
                rows.append({
                    "problem_id": row["problem_id"],
                    "prompt": row["prompt"],
                    "code": row["human_code"],
                    "label": "human"
                })
            if "ai_code" in row:
                rows.append({
                    "problem_id": row["problem_id"],
                    "prompt": row["prompt"],
                    "code": row["ai_code"],
                    "label": "ai"
                })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="ai_vs_human_code.jsonl",
                    help="Input JSONL with problem_id, prompt, human_code, ai_code")
    ap.add_argument("--out_csv", type=str, default="features.csv")
    ap.add_argument("--out_parquet", type=str, default="features.parquet")
    ap.add_argument("--keep_text", action="store_true",
                    help="Include original code/prompt in output")
    args = ap.parse_args()

    # Load JSONL
    df = load_jsonl(args.input)
    print(f"Loaded {len(df)} rows from {args.input}")

    # Extract features
    feat_rows = []
    for _, row in df.iterrows():
        feats = extract_features_for_row(row["code"] or "")
        feats["problem_id"] = row["problem_id"]
        feats["label"] = row["label"]
        if args.keep_text:
            feats["prompt"] = row["prompt"]
            feats["code"] = row["code"]
        feat_rows.append(feats)

    out = pd.DataFrame(feat_rows)
    out.to_csv(args.out_csv, index=False)
    out.to_parquet(args.out_parquet, index=False)

    print(f"✅ Saved features to:\n  - {args.out_csv}\n  - {args.out_parquet}")
    print(f"Rows: {len(out)} | Cols: {len(out.columns)}")
    print(out.head(3))


if __name__ == "__main__":
    main()
