#!/usr/bin/env python3
"""
Train an XGBoost classifier on precomputed embeddings stored in a CSV.

CSV schema expected:
    row_id, code_hash, language, label, generator, embedding

- `embedding` is a string like:
    "[-1.8692e-02 -1.0048e-02 ...]"   (space-separated)
  or a comma list; both are supported.

Outputs:
- XGBoost model:  <out>.json  (portable, can be loaded by XGBoost)
- Metrics JSON:   <out>_metrics.json
- Threshold JSON: <out>_threshold.json  (contains best threshold for F1)
"""

import argparse
import ast
import json
import os
from typing import Tuple

import numpy as np
import pandas as pd
from xgboost.callback import EarlyStopping


from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split

import xgboost as xgb


# ---------- parsing ----------

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

    # Fast path: space-separated numbers (as in your sample)
    arr = np.fromstring(s, sep=" ")
    if arr.size == 0:
        # Fallback to Python-literal (comma lists)
        arr = np.array(ast.literal_eval(s), dtype=float)
    return arr.astype(np.float32)


def load_csv(csv_path: str) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    df = pd.read_csv(csv_path)
    if "embedding" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must have 'embedding' and 'label' columns.")
    X = np.stack(df["embedding"].map(parse_vec).to_list())
    y = df["label"].astype(int).to_numpy()
    return X, y, df


# ---------- metrics & thresholding ----------

def evaluate(proba: np.ndarray, y_true: np.ndarray, threshold: float = 0.5):
    pred = (proba >= threshold).astype(int)

    roc = roc_auc_score(y_true, proba)
    pr_auc = average_precision_score(y_true, proba)

    report = classification_report(y_true, pred, output_dict=True)
    cm = confusion_matrix(y_true, pred).tolist()

    return {
        "roc_auc": float(roc),
        "pr_auc": float(pr_auc),
        "threshold": float(threshold),
        "classification_report": report,
        "confusion_matrix": cm,
    }


def find_best_threshold(proba: np.ndarray, y_true: np.ndarray, beta: float = 1.0):
    """
    Scan thresholds to maximize F_beta (default F1) for the positive class.
    Returns (best_threshold, best_f1).
    """
    precisions, recalls, ths = precision_recall_curve(y_true, proba)
    # precision_recall_curve returns thresholds of size N-1
    # compute F-beta for those points
    eps = 1e-12
    fbeta = (1 + beta**2) * (precisions * recalls) / (beta**2 * precisions + recalls + eps)
    idx = int(np.nanargmax(fbeta))
    # Map index back to threshold (clip if idx at last)
    best_thr = float(ths[max(0, min(idx, len(ths) - 1))]) if len(ths) > 0 else 0.5
    return best_thr, float(fbeta[idx] if idx < len(fbeta) else 0.0)


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to embeddings CSV")
    ap.add_argument("--out", default="human_vs_ai_xgb", help="Output prefix (no extension)")
    ap.add_argument("--test_size", type=float, default=0.2, help="Validation split ratio")
    ap.add_argument("--random_state", type=int, default=42, help="Random seed")
    # XGBoost core params (good starting point)
    ap.add_argument("--n_estimators", type=int, default=600)
    ap.add_argument("--max_depth", type=int, default=6)
    ap.add_argument("--learning_rate", type=float, default=0.05)
    ap.add_argument("--subsample", type=float, default=0.9)
    ap.add_argument("--colsample_bytree", type=float, default=0.9)
    ap.add_argument("--reg_lambda", type=float, default=1.0)
    ap.add_argument("--early_stopping_rounds", type=int, default=50)
    ap.add_argument("--n_jobs", type=int, default=8)
    args = ap.parse_args()

    print(f"[info] Loading {args.csv}")
    X, y, df = load_csv(args.csv)
    pos_rate = y.mean()
    print(f"[info] Data: X={X.shape}, positive_rate={pos_rate:.3f}")

    # Imbalance handling
    # scale_pos_weight = #neg / #pos (classic heuristic)
    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    scale_pos_weight = (n_neg / max(1, n_pos))
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
        tree_method="hist",       # fast & memory-efficient
        eval_metric="logloss",    # we'll compute AUC/PR-AUC ourselves
        n_jobs=args.n_jobs,
        scale_pos_weight=scale_pos_weight,
        random_state=args.random_state,
        use_label_encoder=False,
    )

    print("[info] Training XGBoost...")
    model.fit(
       X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    proba = model.predict_proba(X_val)[:, 1]

    print("[info] Evaluating at default threshold = 0.5")
    metrics_05 = evaluate(proba, y_val, threshold=0.5)

    print("[info] Searching best F1 threshold")
    best_thr, best_f1 = find_best_threshold(proba, y_val, beta=1.0)
    metrics_best = evaluate(proba, y_val, threshold=best_thr)

    # Save model & metrics
    out_model_json = f"{args.out}.json"
    model.save_model(out_model_json)
    print(f"[done] Saved model -> {out_model_json}")

    out_metrics = f"{args.out}_metrics.json"
    with open(out_metrics, "w") as f:
        json.dump(
            {
                "metrics@0.5": metrics_05,
                "metrics@best_f1": metrics_best,
                "best_threshold": best_thr,
                "best_f1": best_f1,
                "pos_rate": float(pos_rate),
                "params": {
                    "n_estimators": args.n_estimators,
                    "max_depth": args.max_depth,
                    "learning_rate": args.learning_rate,
                    "subsample": args.subsample,
                    "colsample_bytree": args.colsample_bytree,
                    "reg_lambda": args.reg_lambda,
                    "scale_pos_weight": scale_pos_weight,
                    "early_stopping_rounds": args.early_stopping_rounds,
                },
            },
            f,
            indent=2,
        )
    print(f"[done] Saved metrics -> {out_metrics}")

    out_thr = f"{args.out}_threshold.json"
    with open(out_thr, "w") as f:
        json.dump({"best_threshold": best_thr, "best_f1": best_f1}, f, indent=2)
    print(f"[done] Saved best threshold -> {out_thr}")

    # Pretty print summary
    def pp(m, title):
        print(f"\n=== {title} ===")
        print(f"ROC-AUC: {m['roc_auc']:.4f} | PR-AUC: {m['pr_auc']:.4f} | thr={m['threshold']:.3f}")
        cm = m["confusion_matrix"]
        print("Confusion Matrix [ [tn, fp], [fn, tp] ]:", cm)
        cr = m["classification_report"]
        for k in ("0", "1"):
            if k in cr:
                v = cr[k]
                print(f"class {k}: f1={v['f1-score']:.4f}, "
                      f"prec={v['precision']:.4f}, rec={v['recall']:.4f}")

    pp(metrics_05,  "Validation @ 0.5")
    pp(metrics_best, f"Validation @ best F1 (thr={best_thr:.3f})")


if __name__ == "__main__":
    main()

# pip install xgboost numpy pandas scikit-learn
# python train_xgboost_embeddings.py --csv your_embeddings.csv --out human_vs_ai_xgb
