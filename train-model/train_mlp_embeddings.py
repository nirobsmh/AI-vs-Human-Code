#!/usr/bin/env python3
"""
Train a non-linear MLP classifier on precomputed embeddings stored in a CSV,
then pick the decision threshold that maximizes macro-F1 (balanced across classes).

CSV schema expected:
    row_id, code_hash, language, label, generator, embedding

- `embedding` is a string like:
    "[-1.8692e-02 -1.0048e-02 ...]"   (space- or comma-separated)

Outputs:
- Model:        <out>.pt                (PyTorch state dict)
- Scaler:       <out>_scaler.json       (mean/std for standardization)
- Threshold:    <out>_threshold.json    (best macro-F1 threshold)
- Metrics:      <out>_metrics.json      (metrics at 0.5 and at best macro-F1)
"""

import argparse
import ast
import json
import os
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


# ---------- parsing ----------

def parse_vec(s: str) -> np.ndarray:
    if not isinstance(s, str):
        raise ValueError("embedding must be a string")
    s = s.strip()
    if s.startswith('"') and s.endswith('"'):
        s = s[1:-1]
    s = s.strip().strip("[] \n")
    arr = np.fromstring(s, sep=" ")
    if arr.size == 0:
        arr = np.array(ast.literal_eval(s), dtype=float)
    return arr.astype(np.float32)


def load_csv(csv_path: str) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    df = pd.read_csv(csv_path)
    if "embedding" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must have 'embedding' and 'label' columns.")
    X = np.stack(df["embedding"].map(parse_vec).to_list())
    y = df["label"].astype(int).to_numpy()
    return X, y, df


# ---------- threshold search (macro-F1) ----------

def evaluate_at_threshold(y_true: np.ndarray, proba: np.ndarray, thr: float):
    pred = (proba >= thr).astype(int)
    return {
        "thr": float(thr),
        "macro_f1": f1_score(y_true, pred, average="macro"),
        "f1_pos": f1_score(y_true, pred, pos_label=1),
        "f1_neg": f1_score(y_true, pred, pos_label=0),
        "prec_pos": precision_score(y_true, pred, pos_label=1, zero_division=0),
        "rec_pos": recall_score(y_true, pred, pos_label=1, zero_division=0),
        "prec_neg": precision_score(y_true, 1 - pred, pos_label=1, zero_division=0),
        "rec_neg": recall_score(y_true, 1 - pred, pos_label=1, zero_division=0),
        "cm": confusion_matrix(y_true, pred).tolist(),
        "report": classification_report(y_true, pred, output_dict=True),
    }


def search_best_macro_f1(y_true: np.ndarray, proba: np.ndarray):
    uniq = np.unique(np.round(proba, 6))
    grid = np.linspace(0.01, 0.99, 199)
    thrs = np.unique(np.clip(np.concatenate([uniq, grid]), 0, 1))

    best = None
    best_val = -1.0
    for thr in thrs:
        m = evaluate_at_threshold(y_true, proba, thr)
        if m["macro_f1"] > best_val:
            best_val = m["macro_f1"]
            best = m
    if best is None:
        best = evaluate_at_threshold(y_true, proba, 0.5)
    return best


# ---------- model ----------

class MLP(nn.Module):
    def __init__(self, dim_in: int, hidden: int = 512, pdrop: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim_in),
            nn.Linear(dim_in, hidden),
            nn.ReLU(),
            nn.Dropout(pdrop),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(pdrop),
            nn.Linear(hidden // 2, 1),   # binary logit
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # logits


def standardize_fit(X: np.ndarray):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def standardize_apply(X: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (X - mean) / std


# ---------- training ----------

def train_mlp(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    hidden: int = 512,
    pdrop: float = 0.2,
    lr: float = 3e-4,
    batch_size: int = 128,
    epochs: int = 30,
    patience: int = 5,
    device: str = "cpu",
):
    dim = X_train.shape[1]
    model = MLP(dim_in=dim, hidden=hidden, pdrop=pdrop).to(device)

    # Class-weighted loss for imbalance
    counts = np.bincount(y_train)
    # avoid div by zero
    w0 = float(counts.sum() / (2.0 * max(1, counts[0])))
    w1 = float(counts.sum() / (2.0 * max(1, counts[1])))
    class_weights = torch.tensor([w0, w1], dtype=torch.float32, device=device)

    def criterion(logits, target):
        # BCEWithLogitsLoss with class weights
        # weight w1 for positives, w0 for negatives
        weights = torch.where(target == 1, class_weights[1], class_weights[0])
        return nn.functional.binary_cross_entropy_with_logits(logits, target.float(), weight=weights)

    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long()),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long()),
        batch_size=256, shuffle=False
    )

    best_state = None
    best_val_loss = float("inf")
    best_epoch = -1
    no_improve = 0

    for epoch in range(1, epochs + 1):
        # train
        model.train()
        total = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total += loss.item() * xb.size(0)
        train_loss = total / len(train_loader.dataset)

        # val
        model.eval()
        with torch.no_grad():
            total = 0.0
            probs = []
            ys = []
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = nn.functional.binary_cross_entropy_with_logits(logits, yb.float())
                total += loss.item() * xb.size(0)
                probs.append(torch.sigmoid(logits).cpu().numpy())
                ys.append(yb.cpu().numpy())
            val_loss = total / len(val_loader.dataset)
            proba_val = np.concatenate(probs)
            y_val_np = np.concatenate(ys)

        print(f"[epoch {epoch:02d}] train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[early stop] best val loss {best_val_loss:.4f} at epoch {best_epoch}")
                break

    # Load best state
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final proba on val (for metrics)
    model.eval()
    with torch.no_grad():
        val_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long()),
            batch_size=256, shuffle=False
        )
        probs = []
        for xb, _ in val_loader:
            xb = xb.to(device)
            logits = model(xb)
            probs.append(torch.sigmoid(logits).cpu().numpy())
        proba_val = np.concatenate(probs)

    return model, proba_val


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to embeddings CSV")
    ap.add_argument("--out", default="human_vs_ai_mlp", help="Output prefix (no extension)")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--random_state", type=int, default=42)

    # MLP/Train params
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    print(f"[info] device={device}")

    print(f"[info] Loading {args.csv}")
    X, y, df = load_csv(args.csv)
    print(f"[info] Data: X={X.shape}, pos_rate={y.mean():.3f}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.random_state
    )

    # Feature standardization (store train mean/std)
    mean, std = standardize_fit(X_train)
    X_train_s = standardize_apply(X_train, mean, std)
    X_val_s = standardize_apply(X_val, mean, std)

    model, proba_val = train_mlp(
        X_train_s, y_train, X_val_s, y_val,
        hidden=args.hidden, pdrop=args.dropout,
        lr=args.lr, batch_size=args.batch_size,
        epochs=args.epochs, patience=args.patience,
        device=device
    )

    # Metrics @ 0.5
    pred05 = (proba_val >= 0.5).astype(int)
    metrics_05 = {
        "roc_auc": float(roc_auc_score(y_val, proba_val)),
        "pr_auc": float(average_precision_score(y_val, proba_val)),
        "threshold": 0.5,
        "report": classification_report(y_val, pred05, output_dict=True),
        "cm": confusion_matrix(y_val, pred05).tolist(),
    }

    # Best macro-F1 threshold
    best = search_best_macro_f1(y_val, proba_val)
    best_thr = float(best["thr"])
    metrics_best = {
        "roc_auc": float(roc_auc_score(y_val, proba_val)),
        "pr_auc": float(average_precision_score(y_val, proba_val)),
        "threshold": best_thr,
        "macro_f1": float(best["macro_f1"]),
        "f1_pos": float(best["f1_pos"]),
        "f1_neg": float(best["f1_neg"]),
        "prec_pos": float(best["prec_pos"]),
        "rec_pos": float(best["rec_pos"]),
        "prec_neg": float(best["prec_neg"]),
        "rec_neg": float(best["rec_neg"]),
        "report": best["report"],
        "cm": best["cm"],
    }

    # Save artifacts
    out_model = f"{args.out}.pt"
    torch.save(model.state_dict(), out_model)
    print(f"[done] Saved model -> {out_model}")

    scaler_path = f"{args.out}_scaler.json"
    with open(scaler_path, "w") as f:
        json.dump({"mean": mean.tolist(), "std": std.tolist()}, f)
    print(f"[done] Saved scaler -> {scaler_path}")

    thr_path = f"{args.out}_threshold.json"
    with open(thr_path, "w") as f:
        json.dump({"best_threshold_macro_f1": best_thr}, f, indent=2)
    print(f"[done] Saved best threshold -> {thr_path}")

    metrics_path = f"{args.out}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({"metrics@0.5": metrics_05, "metrics@best_macro_f1": metrics_best}, f, indent=2)
    print(f"[done] Saved metrics -> {metrics_path}")

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
