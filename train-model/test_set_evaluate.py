    #!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate a saved handcrafted-only classifier on a features CSV.

Inputs:
- artifacts/ (from train_handcrafted.py):
    - best_model.joblib
    - best_threshold.json
    - feature_list.json
- test CSV with the same feature columns. If it also has `label`, full metrics are computed.
  If `label` is missing, we only output probabilities and predicted classes.

Outputs:
- Prints metrics (if labels present)
- Saves CSV with id, prob, pred, (label if available)
- Saves metrics.json

Usage:
  python evaluate_handcrafted.py \
    --csv test_features.csv \
    --artifacts_dir ./artifacts \
    --out_dir ./eval
"""

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

def load_features(csv_path: Path, feature_list):
    df = pd.read_csv(csv_path)
    X = df[feature_list].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    meta_cols = [c for c in ["id", "generator", "language", "problem_id", "label"] if c in df.columns]
    meta = df[meta_cols] if meta_cols else None
    y = df["label"].values.astype(int) if "label" in df.columns else None
    return X, y, meta, df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to features CSV to evaluate")
    ap.add_argument("--artifacts_dir", required=True, help="Directory with trained artifacts")
    ap.add_argument("--out_dir", required=True, help="Where to save predictions & metrics")
    args = ap.parse_args()

    artifacts = Path(args.artifacts_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    model = joblib.load(artifacts / "best_model.joblib")
    thr = json.load(open(artifacts / "best_threshold.json"))["threshold"]
    feature_list = json.load(open(artifacts / "feature_list.json"))

    X, y, meta, df = load_features(Path(args.csv), feature_list)

    # predict probabilities -> classes by threshold
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= float(thr)).astype(int)

    # save predictions
    out_pred = pd.DataFrame({
        "id": meta["id"] if meta is not None and "id" in meta.columns else np.arange(1, len(pred)+1),
        "prob_ai": proba,
        "pred_ai": pred
    })
    if y is not None:
        out_pred["label"] = y
    out_pred.to_csv(out_dir / "predictions.csv", index=False)

    metrics = {}
    if y is not None:
        # classification metrics at the chosen threshold
        metrics["accuracy"] = float(accuracy_score(y, pred))
        metrics["precision"] = float(precision_score(y, pred, zero_division=0))
        metrics["recall"] = float(recall_score(y, pred, zero_division=0))
        metrics["f1"] = float(f1_score(y, pred, zero_division=0))
        metrics["confusion_matrix"] = confusion_matrix(y, pred).tolist()
        metrics["classification_report"] = classification_report(y, pred, output_dict=True, zero_division=0)

        # threshold-independent metrics
        try:
            metrics["roc_auc"] = float(roc_auc_score(y, proba))
        except ValueError:
            metrics["roc_auc"] = None
        try:
            metrics["pr_auc"] = float(average_precision_score(y, proba))
        except ValueError:
            metrics["pr_auc"] = None

        # show best F1 achievable on this set (to sanity check threshold)
        prec, rec, thr_grid = precision_recall_curve(y, proba)
        f1_grid = 2 * (prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1] + 1e-12)
        best_idx = int(np.nanargmax(f1_grid)) if len(f1_grid) else 0
        metrics["best_f1_possible"] = float(f1_grid[best_idx]) if len(f1_grid) else None
        metrics["best_f1_threshold"] = float(thr_grid[best_idx]) if len(thr_grid) else None

        print("\n=== Evaluation (labels present) ===")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1      : {metrics['f1']:.4f} @ thr={float(thr):.3f}")
        print(f"Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f}")
        print(f"ROC-AUC : {metrics['roc_auc']}")
        print(f"PR-AUC  : {metrics['pr_auc']}")
        print(f"Best possible F1 on this set (by sweeping thr): {metrics['best_f1_possible']} at thr={metrics['best_f1_threshold']}")
        print("Confusion matrix [[tn, fp],[fn, tp]]:", metrics["confusion_matrix"])
    else:
        print("\n=== Evaluation (no labels) ===")
        print("Saved probabilities and predictions; supply labels to compute accuracy/F1/AUC.")

    # persist metrics
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[done] Wrote predictions -> {out_dir / 'predictions.csv'}")
    print(f"[done] Wrote metrics     -> {out_dir / 'metrics.json'}")

if __name__ == "__main__":
    main()


# 1) Train (already done earlier)
# python train_handcrafted.py \
#   --csv path/to/train_features.csv \
#   --out_dir ./artifacts

# 2) Evaluate on a separate test features file
# python test_set_evaluate.py \
#   --csv path/to/test_features.csv \
#   --artifacts_dir ./artifacts \
#   --out_dir ./eval_test
