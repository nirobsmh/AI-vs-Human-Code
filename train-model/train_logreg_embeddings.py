#!/usr/bin/env python3
"""
Train a Logistic Regression classifier on precomputed embeddings stored in a CSV.

CSV schema expected:
    row_id, code_hash, language, label, generator, embedding

- `embedding` is a string representation of a vector like:
    "[-1.8692e-02 -1.0048e-02 ...]"

Outputs:
- Saved sklearn Pipeline (StandardScaler + LogisticRegression) via joblib
- Evaluation report printed to stdout and saved as JSON next to the model
"""

import argparse
import ast
import json
import os
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight


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
    arr = np.fromstring(s, sep=" ")
    if arr.size == 0:
        try:
            arr = np.array(ast.literal_eval(s), dtype=float)
        except Exception as e:
            raise ValueError(f"Failed to parse embedding string: {s[:80]}...") from e
    return arr.astype(np.float32)


def load_data(csv_path: str):
    df = pd.read_csv(csv_path)
    if "embedding" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must have 'embedding' and 'label' columns.")
    X = np.stack(df["embedding"].map(parse_vec).to_list())
    y = df["label"].astype(int).to_numpy()
    return X, y, df


def train_and_eval(X, y, test_size=0.2, random_state=42,
                   C=1.0, solver="liblinear", max_iter=5000):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    classes = np.unique(y_train)
    cw_vals = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight = {int(c): float(w) for c, w in zip(classes, cw_vals)}

    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", LogisticRegression(
            max_iter=max_iter,
            class_weight=class_weight,
            solver=solver,
            C=C,
        ))
    ])

    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_val)[:, 1]
    pred = (proba >= 0.5).astype(int)

    report = classification_report(y_val, pred, output_dict=True)
    auc = roc_auc_score(y_val, proba)
    cm = confusion_matrix(y_val, pred).tolist()

    metrics = {
        "roc_auc": float(auc),
        "classification_report": report,
        "confusion_matrix": cm,
        "val_size": int(y_val.size),
        "train_size": int(y_train.size),
        "class_weight": class_weight,
        "C": C,
        "solver": solver,
        "max_iter": max_iter,
    }
    return pipe, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to embeddings CSV")
    ap.add_argument("--out", default="human_vs_ai_lr.joblib", help="Output model path (.joblib)")
    ap.add_argument("--test_size", type=float, default=0.2, help="Validation split ratio")
    ap.add_argument("--random_state", type=int, default=42, help="Random seed")
    ap.add_argument("--C", type=float, default=1.0, help="Inverse regularization strength")
    ap.add_argument("--solver", default="liblinear",
                    choices=["liblinear", "lbfgs", "saga", "newton-cg"],
                    help="LogReg solver")
    ap.add_argument("--max_iter", type=int, default=5000, help="Max iterations")
    args = ap.parse_args()

    print(f"[info] Loading {args.csv}")
    X, y, df = load_data(args.csv)
    print(f"[info] Data: X={X.shape}, y_pos_rate={y.mean():.3f}")

    print("[info] Training Logistic Regression...")
    model, metrics = train_and_eval(
        X, y,
        test_size=args.test_size,
        random_state=args.random_state,
        C=args.C,
        solver=args.solver,
        max_iter=args.max_iter,
    )

    joblib.dump(model, args.out)
    print(f"[done] Saved model -> {args.out}")

    metrics_path = os.path.splitext(args.out)[0] + "_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[done] Saved metrics -> {metrics_path}")

    print("\n=== Validation Summary ===")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print("Confusion Matrix [ [tn, fp], [fn, tp] ]:", metrics["confusion_matrix"])
    for k in ("0", "1"):
        if k in metrics["classification_report"]:
            v = metrics["classification_report"][k]
            print(f"class {k}: f1={v['f1-score']:.4f}, "
                  f"precision={v['precision']:.4f}, recall={v['recall']:.4f}")


if __name__ == "__main__":
    main()


# python train_logreg_embeddings.py --csv shard_00000.csv --out human_vs_ai_lr.joblib