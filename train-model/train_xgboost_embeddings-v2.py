#!/usr/bin/env python3
"""
Train an XGBoost classifier on precomputed embeddings stored in a CSV,
then pick the decision threshold that maximizes macro-F1 (balanced across classes).

CSV schema expected:
    row_id, code_hash, language, label, generator, embedding

- `embedding` is a string like:
    "[-1.8692e-02 -1.0048e-02 ...]"   (space- or comma-separated)

Outputs:
- Model:        <out>.json
- Metrics:      <out>_metrics.json
- Threshold:    <out>_threshold.json
- Console:      prints metrics at 0.5 and at best macro-F1 threshold
"""

import argparse
import ast
import json
import os
from typing import Tuple

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split


# -------- parsing --------

def parse_vec(s: str) -> np.ndarray:
    """
    Parse an embedding string into a 1D float32 numpy array.
    Supports both space-separated and comma-separated formats.
    """
    if not isinstance(s, str):
        raise ValueError("embedding must be a string")
    s = s.strip()
    if s.startswith('"') and s.endswith('"'):
        s = s[1:-1]
    s = s.strip().strip("[] \n")

    # Fast path: space-separated numbers (common in your sample)
    arr = np.fromstring(s, sep=" ")
    if arr.size == 0:
        # Fallback to Python literal (comma-separated)
        arr = np.array(ast.literal_eval(s), dtype=float)
    return arr.astype(np.float32)


def load_csv(csv_path: str) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    df = pd.read_csv(csv_path)
    if "embedding" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must have 'embedding' and 'label' columns.")
    X = np.stack(df["embedding"].map(parse_vec).to_list())
    y = df["label"].astype(int).to_numpy()
    return X, y, df


# -------- threshold search (macro-F1) --------

def evaluate_at_threshold(y_true: np.ndarray, proba: np.ndarray, thr: float):
    pred = (proba >= thr).astype(int)
    return {
        "thr": float(thr),
        "macro_f1": f1_score(y_true, pred, average="macro"),
        "f1_pos": f1_score(y_true, pred, pos_label=1),
        "f1_neg": f1_score(y_true, pred, pos_label=0),
        "prec_pos": precision_score(y_true, pred, pos_label=1, zero_division=0),
        "rec_pos": recall_score(y_true, pred, pos_label=1, zero_division=0),
        "prec_neg": precision_score(y_true, 1 - pred, pos_label=1, zero_division=0),  # precision for class 0
        "rec_neg": recall_score(y_true, 1 - pred, pos_label=1, zero_division=0),      # recall for class 0
    }


def search_thresholds(y_true: np.ndarray, proba: np.ndarray):
    """
    Brute-force scan over candidate thresholds and return the one that maximizes macro-F1.
    Combines a grid and unique probabilities for robustness.
    """
    uniq = np.unique(np.round(proba, 6))
    grid = np.linspace(0.01, 0.99, 199)
    thrs = np.unique(np.clip(np.concatenate([uniq, grid]), 0, 1))

    best = None
    best_val = -1.0
    for thr in thrs:
        m = evaluate_at_threshold(y_true, proba, thr)
        val = m["macro_f1"]
        if val > best_val:
            best_val = val
            best = m
    # Fallback (shouldn't happen), return 0.5
    if best is None:
        best = evaluate_at_threshold(y_true, proba, 0.5)
    return best


def youdens_j_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Alternative balanced choice: argmax(TPR - FPR) on ROC."""
    fpr, tpr, thr = roc_curve(y_true, proba)
    j = tpr - fpr
    return float(thr[np.argmax(j)])


# -------- main --------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to embeddings CSV")
    ap.add_argument("--out", default="human_vs_ai_xgb", help="Output prefix (no extension)")
    ap.add_argument("--test_size", type=float, default=0.2, help="Validation split ratio")
    ap.add_argument("--random_state", type=int, default=42, help="Random seed")

    # XGBoost hyperparams (safe defaults; tune as needed)
    ap.add_argument("--n_estimators", type=int, default=800)
    ap.add_argument("--max_depth", type=int, default=6)
    ap.add_argument("--learning_rate", type=float, default=0.05)
    ap.add_argument("--subsample", type=float, default=0.9)
    ap.add_argument("--colsample_bytree", type=float, default=0.9)
    ap.add_argument("--reg_lambda", type=float, default=1.0)
    ap.add_argument("--n_jobs", type=int, default=8)
    args = ap.parse_args()

    print(f"[info] Loading {args.csv}")
    X, y, df = load_csv(args.csv)
    pos_rate = y.mean()
    print(f"[info] Data: X={X.shape}, positive_rate={pos_rate:.3f}")

    # Imbalance handling: classic heuristic
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    scale_pos_weight = n_neg / max(1, n_pos)
    print(f"[info] scale_pos_weight={scale_pos_weight:.3f}  (neg={n_neg}, pos={n_pos})")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.random_state
    )

    model = xgb.XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_lambda=args.reg_lambda,
        tree_method="hist",
        eval_metric="logloss",     # we'll compute AUC/PR-AUC ourselves
        n_jobs=args.n_jobs,
        scale_pos_weight=scale_pos_weight,
        random_state=args.random_state,
        use_label_encoder=False,
    )

    print("[info] Training XGBoost (no early stopping for compatibility)...")
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    proba = model.predict_proba(X_val)[:, 1]

    # Metrics at 0.5
    print("[info] Evaluating at default threshold = 0.5")
    metrics_05 = {
        "roc_auc": float(roc_auc_score(y_val, proba)),
        "pr_auc": float(average_precision_score(y_val, proba)),
        "threshold": 0.5,
        "report": classification_report(y_val, (proba >= 0.5).astype(int), output_dict=True),
        "cm": confusion_matrix(y_val, (proba >= 0.5).astype(int)).tolist(),
    }

    # Best threshold by macro-F1
    print("[info] Searching threshold that maximizes macro-F1")
    best = search_thresholds(y_val, proba)
    best_thr = best["thr"]
    pred_best = (proba >= best_thr).astype(int)

    metrics_best = {
        "roc_auc": float(roc_auc_score(y_val, proba)),
        "pr_auc": float(average_precision_score(y_val, proba)),
        "threshold": float(best_thr),
        "macro_f1": float(best["macro_f1"]),
        "f1_pos": float(best["f1_pos"]),
        "f1_neg": float(best["f1_neg"]),
        "prec_pos": float(best["prec_pos"]),
        "rec_pos": float(best["rec_pos"]),
        "prec_neg": float(best["prec_neg"]),
        "rec_neg": float(best["rec_neg"]),
        "report": classification_report(y_val, pred_best, output_dict=True),
        "cm": confusion_matrix(y_val, pred_best).tolist(),
    }

    # Save model & metrics
    out_model_json = f"{args.out}.json"
    model.save_model(out_model_json)
    print(f"[done] Saved model -> {out_model_json}")

    out_metrics = f"{args.out}_metrics.json"
    with open(out_metrics, "w") as f:
        json.dump(
            {
                "metrics@0.5": metrics_05,
                "metrics@best_macro_f1": metrics_best,
                "pos_rate": float(pos_rate),
                "params": {
                    "n_estimators": args.n_estimators,
                    "max_depth": args.max_depth,
                    "learning_rate": args.learning_rate,
                    "subsample": args.subsample,
                    "colsample_bytree": args.colsample_bytree,
                    "reg_lambda": args.reg_lambda,
                    "scale_pos_weight": scale_pos_weight,
                },
            },
            f,
            indent=2,
        )
    print(f"[done] Saved metrics -> {out_metrics}")

    out_thr = f"{args.out}_threshold.json"
    with open(out_thr, "w") as f:
        json.dump({"best_threshold_macro_f1": float(best_thr)}, f, indent=2)
    print(f"[done] Saved best threshold -> {out_thr}")

    # Pretty print summary
    def pp_block(title, m):
        print(f"\n=== {title} ===")
        print(f"ROC-AUC: {m['roc_auc']:.4f} | PR-AUC: {m['pr_auc']:.4f} | thr={m['threshold']:.3f}")
        print("Confusion Matrix [ [tn, fp], [fn, tp] ]:", m["cm"])
        cr = m["report"]
        for k in ("0", "1"):
            if k in cr:
                v = cr[k]
                print(f"class {k}: f1={v['f1-score']:.4f}, "
                      f"prec={v['precision']:.4f}, rec={v['recall']:.4f}")

    pp_block("Validation @ 0.5", metrics_05)
    pp_block(f"Validation @ best macro-F1 (thr={best_thr:.3f})", metrics_best)


if __name__ == "__main__":
    main()

#python train_xgboost_embeddings-v2.py --csv shard_00000.csv --out human_vs_ai_xgb_v2