#!/usr/bin/env python3
"""Prepare the held-out Python test set and apply robustness transforms.

Rebuilds the same 60/20/20 split used by train_hybrid.py, pulls raw code for
the 2,000-row test split, filters to Python, and writes transformed snippets.

Does NOT retrain models.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train-model"))
sys.path.insert(0, str(ROOT / "feature"))

from train_hybrid import load_aligned_data, make_splits  # noqa: E402
from code_transforms import (  # noqa: E402
    ast_normalize_code,
    black_format_code,
    remove_comments_and_docstrings,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--features-csv",
        default=str(
            ROOT / "feature/handcraft_feature_task_a_training_set_1_chunk_part001_10k.csv"
        ),
    )
    ap.add_argument(
        "--embeddings-csv",
        default=str(ROOT / "feature/embedding_feature_task_a_training_set_1_10k.csv"),
    )
    ap.add_argument(
        "--raw-csv",
        default=str(
            ROOT / "data/SemEval-2026-Task13/task_a/training-chunks/chunk_part001.csv"
        ),
        help="Raw CSV whose first N rows match the feature/embedding files",
    )
    ap.add_argument("--out-dir", default=str(ROOT / "hybrid-output"))
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--key", default="id")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data, _x_features, _x_embeddings, y = load_aligned_data(
        Path(args.features_csv), Path(args.embeddings_csv), args.key
    )
    _train_idx, _val_idx, test_idx, split_method = make_splits(
        data, y, args.random_state, "problem_id", row_wise=False
    )
    y_test = y[test_idx]

    raw_all = pd.read_csv(args.raw_csv, nrows=len(data)).copy()
    raw_all[args.key] = np.arange(1, len(raw_all) + 1).astype(str)
    raw_aligned = (
        raw_all.set_index(args.key).loc[data[args.key].astype(str)].reset_index()
    )
    raw_test = raw_aligned.iloc[test_idx].reset_index(drop=True)

    assert len(raw_test) == len(y_test)
    assert np.array_equal(raw_test["label"].to_numpy(), np.asarray(y_test))
    print(f"[info] split={split_method} | raw test rows={len(raw_test)}")

    python_names = {"python", "py", "python3"}
    python_mask = (
        raw_test["language"].astype(str).str.lower().isin(python_names)
    )
    python_test = raw_test.loc[python_mask].copy()
    python_indices = np.where(python_mask.to_numpy())[0]

    print(f"[info] python test samples: {len(python_test)} / {len(raw_test)}")
    print(python_test["label"].value_counts().sort_index())

    python_test["code_original"] = python_test["code"]
    python_test["code_black"] = python_test["code"].apply(black_format_code)
    python_test["code_ast"] = python_test["code"].apply(ast_normalize_code)
    python_test["code_no_comments"] = python_test["code"].apply(
        remove_comments_and_docstrings
    )

    summary = pd.DataFrame(
        {
            "Transformation": [
                "Black",
                "AST normalization",
                "Comments/docstrings removed",
            ],
            "Valid": [
                python_test["code_black"].notna().sum(),
                python_test["code_ast"].notna().sum(),
                python_test["code_no_comments"].notna().sum(),
            ],
            "Total": [len(python_test)] * 3,
        }
    )
    summary["Success Rate"] = summary["Valid"] / summary["Total"]
    print("\n" + summary.to_string(index=False))

    python_test.to_pickle(out_dir / "python_test_transformed.pkl")
    np.save(out_dir / "python_indices.npy", python_indices)
    summary.to_csv(out_dir / "transform_success_rates.csv", index=False)
    print(f"\n[done] wrote artifacts -> {out_dir.resolve()}")


if __name__ == "__main__":
    main()
