#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train AI-vs-Human code classifier using ONLY handcrafted features.

Inputs:
- A CSV produced by your feature extractor with columns like:
  id,generator,label,language,n_lines,...,cyclomatic_complexity_est,problem_id

What this script does:
1) Loads the CSV and selects numeric feature columns.
2) Splits into train/validation (stratified).
3) Trains three sklearn models:
   - Logistic Regression (with StandardScaler)
   - Random Forest
   - HistGradientBoostingClassifier
4) Picks the BEST model by validation PR-AUC (then ROC-AUC as tiebreaker).
5) Selects a decision threshold on the validation set to maximize F1.
6) Saves artifacts:
   - best_model.joblib
   - feature_list.json
   - best_threshold.json
   - metrics.json

Usage:
  python train_handcrafted.py \
    --csv features.csv \
    --out_dir ./artifacts \
    --test_size 0.2 \
    --random_state 42
"""

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
)

import joblib


DEFAULT_FEATURES = [
    "n_lines","n_chars","avg_line_len","blank_ratio","comment_ratio","has_docstring",
    "avg_leading_spaces","avg_leading_tabs",
    "n_identifiers","avg_identifier_len","generic_identifier_ratio","snake_case_ratio","camel_case_ratio",
    "token_entropy","unique_identifier_ratio",
    "ast_depth","n_ifs","n_loops","n_funcs","n_imports","n_try","n_with","n_comprehensions","n_boolops",
    "cyclomatic_complexity_est",
]


def load_dataset(csv_path: Path, label_col: str = "label"):
    df = pd.read_csv(csv_path)
    # Coerce label to integers if needed
    if df[label_col].dtype == object:
        # Try to map "human"/"ai" to 0/1 if present; otherwise cast
        mapping = {"human": 0, "ai": 1, "Human": 0, "AI": 1}
        df[label_col] = df[label_col].map(mapping).fillna(df[label_col]).astype(int)
    else:
        df[label_col] = df[label_col].astype(int)

    # Choose the intersection of expected features and available columns
    features = [c for c in DEFAULT_FEATURES if c in df.columns]
    if not features:
        raise ValueError("No expected handcrafted features found in CSV.")

    X = df[features].copy()
    y = df[label_col].values.astype(int)

    # Basic clean-up: replace inf/nan
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    meta_cols = [c for c in ["id", "generator", "language", "problem_id"] if c in df.columns]
    meta = df[meta_cols] if meta_cols else None
    return X.values, y, features, meta


def pick_threshold_by_f1(y_true, y_prob):
    """Return threshold that maximizes F1 on (y_true, y_prob)."""
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    # precision_recall_curve returns len(thr)+1 points; skip the last (no threshold)
    f1 = 2 * (prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1] + 1e-12)
    idx = np.nanargmax(f1)
    best_thr = float(thr[idx])
    return best_thr, float(f1[idx]), float(prec[idx]), float(rec[idx])


def evaluate_at_threshold(y_true, y_prob, thr):
    y_pred = (y_prob >= thr).astype(int)
    cm = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return y_pred, cm, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to features CSV")
    ap.add_argument("--out_dir", required=True, help="Where to save model & metrics")
    ap.add_argument("--test_size", type=float, default=0.2, help="Validation split size")
    ap.add_argument("--random_state", type=int, default=42, help="Random seed")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X, y, feature_list, _ = load_dataset(csv_path)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    # ---- Define candidate models (pure sklearn, no extra deps) ----
    candidates = {
        "logreg": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=None)
        ),
        "rf": RandomForestClassifier(
            n_estimators=400, max_depth=None, min_samples_leaf=1,
            class_weight="balanced", random_state=args.random_state, n_jobs=-1
        ),
        "hgb": HistGradientBoostingClassifier(
            learning_rate=0.1, max_depth=None, max_bins=255,
            l2_regularization=0.0, early_stopping=True, random_state=args.random_state
        ),
    }

    results = {}
    best_key = None
    best_score = (-1.0, -1.0)  # (PR-AUC, ROC-AUC)

    for name, model in candidates.items():
        model.fit(X_train, y_train)
        val_prob = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else model.predict_proba(X_val)[:, 1]  # HGB has predict_proba
        pr_auc = float(average_precision_score(y_val, val_prob))
        roc = float(roc_auc_score(y_val, val_prob))

        thr, f1, p, r = pick_threshold_by_f1(y_val, val_prob)
        _, cm, rep = evaluate_at_threshold(y_val, val_prob, thr)

        results[name] = {
            "pr_auc": pr_auc,
            "roc_auc": roc,
            "best_threshold": thr,
            "f1_at_thr": f1,
            "precision_at_thr": p,
            "recall_at_thr": r,
            "confusion_matrix_at_thr": cm,
            "classification_report_at_thr": rep,
        }

        score_tuple = (pr_auc, roc)
        if score_tuple > best_score:
            best_score = score_tuple
            best_key = name

    assert best_key is not None
    best_model = candidates[best_key]
    best = results[best_key]

    # ---- Save artifacts ----
    joblib.dump(best_model, out_dir / "best_model.joblib")
    with open(out_dir / "feature_list.json", "w") as f:
        json.dump(feature_list, f, indent=2)
    with open(out_dir / "best_threshold.json", "w") as f:
        json.dump({"threshold": best["best_threshold"]}, f, indent=2)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"models": results, "selected_model": best_key}, f, indent=2)

    # ---- Human-readable summary ----
    print("\n=== Validation Metrics (handcrafted-only) ===")
    for k, v in results.items():
        print(f"- {k:6s} | PR-AUC={v['pr_auc']:.4f} | ROC-AUC={v['roc_auc']:.4f} | "
              f"F1@thr={v['f1_at_thr']:.4f} (thr={v['best_threshold']:.3f})")
    print(f"\n[best] model = {best_key}  |  PR-AUC={best['pr_auc']:.4f}  ROC-AUC={best['roc_auc']:.4f}")
    print(f"[done] Saved to: {out_dir.resolve()}")
    

if __name__ == "__main__":
    main()


#python train.py --csv ../../feature/handcraft_feature_task_a_training_set_1_chunk_part001_10k.csv --out_dir ./output --test_size 0.2 --random_state 42