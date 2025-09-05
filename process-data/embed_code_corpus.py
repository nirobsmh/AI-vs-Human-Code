#!/usr/bin/env python3
"""
Embed a large code corpus (CSV or Parquet) in batches using a local HuggingFace model.

- Streams input in chunks (no full-file memory load)
- Uses GPU if available (falls back to CPU)
- Handles very long snippets via token truncation (SentenceTransformers' max_seq_length)
- Saves outputs as Parquet shards with (row_id, code_hash, language, label, generator, embedding)
- Embeddings stored as float16 to reduce disk size
"""

import os
import sys
import math
import argparse
import hashlib
from typing import Iterable, Optional, List

import numpy as np
import pandas as pd

try:
    import torch
except Exception:
    torch = None

from sentence_transformers import SentenceTransformer
import pyarrow as pa
import pyarrow.parquet as pq


# -----------------------
# Utilities
# -----------------------

def sha1_hash(text: str) -> str:
    h = hashlib.sha1()
    h.update(text.encode("utf-8", errors="ignore"))
    return h.hexdigest()


def infer_reader(
    path: str,
    chunksize: int,
    text_col: str,
    schema_check: bool = True,
) -> Iterable[pd.DataFrame]:
    """
    Yield chunks of a CSV or Parquet file as pandas DataFrames.
    """
    lower = path.lower()
    if lower.endswith(".csv"):
        reader = pd.read_csv(path, chunksize=chunksize)
        for chunk in reader:
            if schema_check and text_col not in chunk.columns:
                raise ValueError(f"Column '{text_col}' not found in CSV.")
            yield chunk
    elif lower.endswith(".parquet") or lower.endswith(".pq"):
        # For Parquet, simulate chunking by reading entire table row groups progressively.
        # If the file is huge with many row groups, this will still be memory-friendly.
        parquet_file = pq.ParquetFile(path)
        cols = None
        for rg in range(parquet_file.num_row_groups):
            table = parquet_file.read_row_group(rg)
            df = table.to_pandas()
            if schema_check:
                if cols is None:
                    cols = df.columns
                elif list(df.columns) != list(cols):
                    raise ValueError("Inconsistent Parquet schema across row groups.")
                if text_col not in df.columns:
                    raise ValueError(f"Column '{text_col}' not found in Parquet.")
            # Further split into chunksize-like frames
            if len(df) <= chunksize:
                yield df
            else:
                for i in range(0, len(df), chunksize):
                    yield df.iloc[i:i+chunksize].copy()
    else:
        raise ValueError("Input must be .csv or .parquet")


def as_arrow_table(
    df_meta: pd.DataFrame,
    embeddings: np.ndarray,
) -> pa.Table:
    """
    Build a PyArrow Table with metadata + embedding (list<float16>).
    embeddings: shape (N, D), dtype float16 preferred.
    """
    # Ensure float16 for space
    if embeddings.dtype != np.float16:
        embeddings = embeddings.astype(np.float16)

    # Convert the 2D numpy array to Arrow ListArray
    # Build from a Python list of lists to keep it simple and robust
    emb_list = embeddings.tolist()
    emb_array = pa.array(emb_list, type=pa.list_(pa.float16()))

    data = {}
    for col in df_meta.columns:
        # Let Arrow infer types; ensure basic types are clean
        if pd.api.types.is_integer_dtype(df_meta[col]):
            data[col] = pa.array(df_meta[col].astype("int64").tolist())
        elif pd.api.types.is_float_dtype(df_meta[col]):
            data[col] = pa.array(df_meta[col].astype("float32").tolist())
        else:
            # strings / objects
            data[col] = pa.array(df_meta[col].astype(str).tolist())

    data["embedding"] = emb_array
    return pa.table(data)


def normalize_newlines(s: str) -> str:
    return s.replace("\r\n", "\n").replace("\r", "\n")


# -----------------------
# Main pipeline
# -----------------------

def main():
    p = argparse.ArgumentParser(
        description="Batch-generate local embeddings (code/text) from a large CSV/Parquet."
    )
    p.add_argument("--input", required=True, help="Path to input .csv or .parquet")
    p.add_argument("--out_dir", default="emb_shards_local", help="Directory to write shards")
    p.add_argument("--text_col", default="code", help="Column containing the code text")
    p.add_argument("--label_col", default="label", help="Optional label column name")
    p.add_argument("--lang_col", default="language", help="Optional language column name")
    p.add_argument("--gen_col", default="generator", help="Optional generator column name")
    p.add_argument("--chunksize", type=int, default=10000, help="Rows per streaming chunk")
    p.add_argument("--batch_size", type=int, default=256, help="Embedding batch size")
    p.add_argument("--model", default="jinaai/jina-embeddings-v2-base-code",
                   help="Sentence-Transformers model name")
    p.add_argument("--max_seq_length", type=int, default=1024,
                   help="Max tokens per sample (model-dependent). Longer code will be truncated.")
    p.add_argument("--start_shard", type=int, default=0,
                   help="Skip writing shards before this index (resume support).")
    p.add_argument("--normalize", action="store_true",
                   help="L2-normalize embeddings (recommended for retrieval, optional for classification)")
    p.add_argument("--dedup_within_shard", action="store_true",
                   help="Drop exact-duplicate code within each shard via SHA1.")
    p.add_argument("--float32", action="store_true",
                   help="Store embeddings as float32 instead of float16 (larger files).")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Device selection
    device = "cpu"
    if torch is not None and torch.cuda.is_available():
        device = "cuda"
    elif torch is not None and torch.backends.mps.is_available():
        device = "mps"  # Apple Silicon
    print(f"[info] Using device: {device}", file=sys.stderr)

    # Load model
    print(f"[info] Loading model: {args.model}", file=sys.stderr)
    model = SentenceTransformer(args.model, device=device)
    # Truncates long inputs (in tokens, not characters). Effective max depends on model tokenizer.
    model.max_seq_length = args.max_seq_length

    # Stream the input and write shards
    shard_idx = -1
    total_rows = 0
    total_written = 0

    for df in infer_reader(args.input, args.chunksize, text_col=args.text_col):
        shard_idx += 1
        if shard_idx < args.start_shard:
            # Skip until resume point
            total_rows += len(df)
            continue

        # Ensure required column
        if args.text_col not in df.columns:
            raise ValueError(f"Column '{args.text_col}' not found in input chunk.")

        # Light cleanup
        df[args.text_col] = df[args.text_col].astype(str).fillna("").map(normalize_newlines)

        # Deduplicate within shard (optional)
        if args.dedup_within_shard:
            if "code_hash" not in df.columns:
                df["code_hash"] = df[args.text_col].map(sha1_hash)
            before = len(df)
            df = df.drop_duplicates(subset=["code_hash"])
            print(f"[info] Shard {shard_idx}: dropped {before - len(df)} duplicates (within-shard).",
                  file=sys.stderr)
        else:
            if "code_hash" not in df.columns:
                df["code_hash"] = df[args.text_col].map(sha1_hash)

        # Build metadata frame
        meta_cols = ["row_id", "code_hash"]
        df_meta = pd.DataFrame()
        # Keep the original row index if present
        if "row_id" in df.columns:
            df_meta["row_id"] = df["row_id"].astype(str)
        else:
            # Use the dataframe index within this shard as a fallback
            df_meta["row_id"] = df.index.astype(str)

        df_meta["code_hash"] = df["code_hash"].astype(str)

        # Optional metadata
        if args.lang_col in df.columns:
            df_meta["language"] = df[args.lang_col].astype(str)
        else:
            df_meta["language"] = ""

        if args.label_col in df.columns:
            # Keep label numeric if possible
            try:
                df_meta["label"] = pd.to_numeric(df[args.label_col], errors="coerce").fillna(-1).astype(int)
            except Exception:
                df_meta["label"] = -1
        else:
            df_meta["label"] = -1

        if args.gen_col in df.columns:
            df_meta["generator"] = df[args.gen_col].astype(str)
        else:
            df_meta["generator"] = ""

        # Encode in batches
        texts: List[str] = df[args.text_col].tolist()
        embeddings_out: List[np.ndarray] = []

        # sentence-transformers can handle internal batching; but we chunk anyway
        for i in range(0, len(texts), args.batch_size):
            batch = texts[i:i + args.batch_size]
            embs = model.encode(
                batch,
                batch_size=min(args.batch_size, len(batch)),
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=args.normalize,
            )
            # Ensure numpy array
            if not isinstance(embs, np.ndarray):
                embs = np.asarray(embs)
            embeddings_out.append(embs)

        if not embeddings_out:
            print(f"[warn] Shard {shard_idx} produced no embeddings; skipping.", file=sys.stderr)
            continue

        embeddings = np.concatenate(embeddings_out, axis=0)
        if not args.float32:
            embeddings = embeddings.astype(np.float16)

        # Write shard
        table = as_arrow_table(df_meta.reset_index(drop=True), embeddings)
        out_path = os.path.join(args.out_dir, f"shard_{shard_idx:05d}.parquet")
        pq.write_table(table, out_path, compression="zstd")
        print(f"[done] Wrote {out_path} (rows={len(df_meta)}, dim={embeddings.shape[1]})", file=sys.stderr)

        total_rows += len(df)
        total_written += len(df_meta)

    print(f"[summary] processed_rows={total_rows} written_rows={total_written} shards={shard_idx+1}",
          file=sys.stderr)


if __name__ == "__main__":
    main()



# python embed_code_corpus.py \
#   --input task_a_training_set_1.csv \
#   --out_dir emb_shards_local \
#   --text_col code \
#   --label_col label \
#   --lang_col language \
#   --gen_col generator \
#   --chunksize 10000 \
#   --batch_size 32 \ 
#   --model jinaai/jina-embeddings-v2-base-code \
#   --max_seq_length 512 \ 
#   --normalize