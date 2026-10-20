#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Embed code snippets from a CSV into a single vector per row.

Input CSV columns (at minimum):
  "code","generator","label","language"

Output CSV columns:
  id,generator,label,language,embedding
where `embedding` is a JSON string of a float list (e.g., "[0.1, -0.2, ...]").

Supports large files via chunked reading (pandas chunksize).
Default model: microsoft/codebert-base (768-dim).

Usage:
  python feature/embed_code.py \
    --in path/to/raw.csv \
    --out path/to/embeddings.csv \
    --model microsoft/codebert-base \
    --batch_size 64 \
    --max_len 512 \
    --chunksize 4096
"""

import json
import argparse
from pathlib import Path

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel


def mean_pool(last_hidden_state, attention_mask):
    # last_hidden_state: [B, T, H], attention_mask: [B, T]
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)  # [B, T, 1]
    summed = (last_hidden_state * mask).sum(dim=1)                  # [B, H]
    counts = mask.sum(dim=1).clamp(min=1e-9)                        # [B, 1]
    return summed / counts


def embed_batch(texts, tokenizer, model, device, max_len=512):
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        outputs = model(**enc)
        # Prefer mean pooling over [CLS] for robustness
        embs = mean_pool(outputs.last_hidden_state, enc["attention_mask"])
        embs = torch.nn.functional.normalize(embs, p=2, dim=1)  # L2 normalize (optional but common)
    return embs.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Input CSV path with columns code,generator,label,language")
    ap.add_argument("--out", dest="out_path", required=True, help="Output CSV path for embeddings")
    ap.add_argument("--model", default="microsoft/codebert-base", help="HF model name or path")
    ap.add_argument("--batch_size", type=int, default=64, help="Batch size for encoder")
    ap.add_argument("--max_len", type=int, default=512, help="Max tokens per snippet")
    ap.add_argument("--chunksize", type=int, default=4096, help="Pandas CSV read chunksize")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] loading model {args.model} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model)
    model.to(device)
    model.eval()

    # Prepare output CSV with header
    import csv
    with out_path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(["id", "generator", "label", "language", "embedding"])

    total_rows = 0
    chunk_idx = 0

    # Stream the input in chunks (handles huge files)
    for df in pd.read_csv(in_path, chunksize=args.chunksize):
        chunk_idx += 1
        # Ensure expected columns exist
        for col in ["code", "generator", "label", "language"]:
            if col not in df.columns:
                raise ValueError(f"Missing column '{col}' in input CSV.")
        # Synthesize id if absent
        if "id" not in df.columns:
            df["id"] = range(total_rows + 1, total_rows + 1 + len(df))

        codes = df["code"].astype(str).tolist()
        ids = df["id"].tolist()
        gens = df["generator"].astype(str).tolist()
        labels = df["label"].tolist()
        langs = df["language"].astype(str).tolist()

        # Batch over this chunk
        out_rows = []
        for start in range(0, len(codes), args.batch_size):
            batch_codes = codes[start:start + args.batch_size]
            embs = embed_batch(batch_codes, tokenizer, model, device, max_len=args.max_len)
            for i, e in enumerate(embs):
                out_rows.append([ids[start + i], gens[start + i], labels[start + i], langs[start + i], json.dumps(e.tolist())])

        # Append to output
        with out_path.open("a", encoding="utf-8", newline="") as fout:
            writer = csv.writer(fout)
            writer.writerows(out_rows)

        total_rows += len(df)
        print(f"[progress] chunk {chunk_idx}: embedded {len(df)} rows (total {total_rows})")

    print(f"[done] wrote embeddings CSV -> {out_path.resolve()}")

if __name__ == "__main__":
    main()
