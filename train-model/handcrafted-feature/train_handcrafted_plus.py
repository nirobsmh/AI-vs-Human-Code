#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train AI-vs-Human classifier on HANDCRAFTED features with many models.

Models included:
- logreg (scaled), rf, hgb, svm (rbf, scaled), knn (scaled), mlp (scaled), extratrees
- xgb (if xgboost installed), lgbm (if lightgbm installed)

Selection metric: PR-AUC (then ROC-AUC tiebreaker).
Threshold: chosen to maximize F1 on validation (hold-out) or on out-of-fold (CV) probabilities.

Artifacts saved in --out_dir:
- best_model.joblib
- feature_list.json
- best_threshold.json
- metrics.json
- feature_importances.json (when available)
- val_predictions.csv (hold-out, if --save_preds)
- oof_predictions.csv (CV, if --save_preds)

Usage (hold-out):
  python train_handcrafted_plus.py --csv features.csv --out_dir ./artifacts --test_size 0.2 --random_state 42 --save_preds

Usage (k-fold CV, e.g., 5 folds):
  python train_handcrafted_plus.py --csv features.csv --out_dir ./artifacts --cv 5 --random_state 42 --save_preds
"""

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_recall_curve,
    classification_report, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# Optional external models
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False


DEFAULT_FEATURES = [
    "n_lines","n_chars","avg_line_len","blank_ratio","comment_ratio","has_docstring",
    "avg_leading_spaces","avg_leading_tabs","n_identifiers","avg_identifier_len",
    "generic_identifier_ratio","snake_case_ratio","camel_case_ratio",
    "token_entropy","unique_identifier_ratio","ast_depth","n_ifs","n_loops",
    "n_funcs","n_imports","n_try","n_with","n_comprehensions","n_boolops",
    "cyclomatic_complexity_est",
]


# ---------------- Utilities ----------------

def load_dataset(csv_path: Path, label_col="label"):
    df = pd.read_csv(csv_path)
    # Coerce label to 0/1
    if label_col in df.columns:
        if df[label_col].dtype == object:
            mapping = {"human": 0, "ai": 1, "Human": 0, "AI": 1}
            df[label_col] = df[label_col].map(mapping).fillna(df[label_col]).astype(int)
        else:
            df[label_col] = df[label_col].astype(int)

    features = [c for c in DEFAULT_FEATURES if c in df.columns]
    if not features:
        raise ValueError("No expected handcrafted features found in CSV.")

    X = df[features].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    y = df[label_col].values.astype(int) if label_col in df.columns else None
    meta_cols = [c for c in ["id", "generator", "language", "problem_id", label_col] if c in df.columns]
    meta = df[meta_cols] if meta_cols else None
    # Also return the full df to access ids if needed
    return X, y, features, meta, df


def pr_auc(y_true, y_prob):
    return float(average_precision_score(y_true, y_prob))

def roc_auc(y_true, y_prob):
    return float(roc_auc_score(y_true, y_prob))

def pick_threshold_by_f1(y_true, y_prob):
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    if len(thr) == 0:  # corner case: single class
        return 0.5, 0.0, 0.0, 0.0
    f1 = 2 * (prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1] + 1e-12)
    idx = int(np.nanargmax(f1))
    return float(thr[idx]), float(f1[idx]), float(prec[idx]), float(rec[idx])

def eval_at_threshold(y_true, y_prob, thr):
    y_pred = (y_prob >= thr).astype(int)
    cm = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return y_pred, cm, report


def build_candidates(seed):
    models = {
        "logreg": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=-1)
        ),
        "rf": RandomForestClassifier(
            n_estimators=500, max_depth=None, min_samples_leaf=1,
            class_weight="balanced", random_state=seed, n_jobs=-1
        ),
        "hgb": HistGradientBoostingClassifier(
            learning_rate=0.08, max_depth=None, max_bins=255,
            early_stopping=True, random_state=seed
        ),
        "svm": make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=seed)
        ),
        "knn": make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=15)
        ),
        "mlp": make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(256, 128), activation="relu",
                          max_iter=600, random_state=seed)
        ),
        "extratrees": ExtraTreesClassifier(
            n_estimators=600, max_depth=None, min_samples_leaf=1,
            class_weight="balanced", random_state=seed, n_jobs=-1
        ),
    }
    if HAS_XGB:
        models["xgb"] = XGBClassifier(
            n_estimators=700, max_depth=8, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            random_state=seed, n_jobs=-1, eval_metric="logloss",
            tree_method="hist"
        )
    if HAS_LGBM:
        models["lgbm"] = LGBMClassifier(
            n_estimators=1000, learning_rate=0.05, subsample=0.9,
            colsample_bytree=0.9, reg_lambda=0.0, random_state=seed, n_jobs=-1
        )
    return models


def choose_best(results_dict):
    # results_dict[name] = {"pr_auc":..., "roc_auc":..., "val_prob":..., "model":...}
    best_key = None
    best_tuple = (-1.0, -1.0)
    for k, v in results_dict.items():
        cand = (v["pr_auc"], v["roc_auc"])
        if cand > best_tuple:
            best_tuple = cand
            best_key = k
    return best_key


def maybe_feature_importance(model, feature_list):
    info = None
    base = model
    if hasattr(model, "steps"):  # pipeline -> get final estimator
        base = model.steps[-1][1]
    if hasattr(base, "feature_importances_"):
        vals = getattr(base, "feature_importances_")
        info = {f: float(v) for f, v in zip(feature_list, vals)}
    elif hasattr(base, "coef_"):  # linear models
        try:
            coef = np.ravel(base.coef_)
            info = {f: float(v) for f, v in zip(feature_list, coef)}
        except Exception:
            pass
    return info


# ---------------- Training (hold-out or CV) ----------------

def train_holdout(X, y, features, ids_series, seed, test_size, out_dir, save_preds: bool):
    candidates = build_candidates(seed)
    results = {}
    Xtr, Xval, ytr, yval, id_tr, id_val = train_test_split(
        X, y, ids_series, test_size=test_size, random_state=seed, stratify=y
    )
    for name, model in candidates.items():
        model.fit(Xtr, ytr)
        val_prob = model.predict_proba(Xval)[:, 1]
        results[name] = {
            "model": model,
            "val_prob": val_prob,
            "yval": yval,
            "pr_auc": pr_auc(yval, val_prob),
            "roc_auc": roc_auc(yval, val_prob),
            "ids": id_val,
        }
    best_key = choose_best(results)
    best = results[best_key]
    thr, f1, p, r = pick_threshold_by_f1(best["yval"], best["val_prob"])
    y_pred, cm, rep = eval_at_threshold(best["yval"], best["val_prob"], thr)

    # Save validation predictions if requested
    if save_preds:
        val_df = pd.DataFrame({
            "id": best["ids"].values if hasattr(best["ids"], "values") else best["ids"],
            "prob_ai": best["val_prob"],
            "pred_ai": y_pred,
            "label": best["yval"],
            "model": best_key
        })
        val_df.to_csv(out_dir / "val_predictions.csv", index=False)

    summary = {
        "per_model": {
            k: {
                "pr_auc": v["pr_auc"],
                "roc_auc": v["roc_auc"],
            } for k, v in results.items()
        },
        "selected": best_key,
        "threshold": thr,
        "f1_at_thr": f1,
        "precision_at_thr": p,
        "recall_at_thr": r,
        "confusion_matrix_at_thr": cm,
        "classification_report_at_thr": rep,
    }
    return results[best_key]["model"], thr, summary


def train_cv(X, y, features, ids_series, seed, n_folds, out_dir, save_preds: bool):
    candidates = build_candidates(seed)
    n = len(y)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    # Store per-model OOF probs and fold index used for each row
    oof_probs = {name: np.full(n, np.nan) for name in candidates}
    oof_fold  = {name: np.full(n, -1, dtype=int) for name in candidates}
    per_model_scores = {name: [] for name in candidates}

    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        Xtr, Xval, ytr, yval = X[tr], X[va], y[tr], y[va]
        for name, model in candidates.items():
            model.fit(Xtr, ytr)
            val_prob = model.predict_proba(Xval)[:, 1]
            oof_probs[name][va] = val_prob
            oof_fold[name][va] = fold
            per_model_scores[name].append({
                "fold": fold,
                "pr_auc": pr_auc(yval, val_prob),
                "roc_auc": roc_auc(yval, val_prob),
            })

    # Aggregate and select best on full OOF
    agg = {}
    for name in candidates:
        # rows validated once → no averaging needed
        pr = pr_auc(y, oof_probs[name][~np.isnan(oof_probs[name])])
        ro = roc_auc(y, oof_probs[name][~np.isnan(oof_probs[name])])
        agg[name] = {"pr_auc": pr, "roc_auc": ro, "folds": per_model_scores[name]}

    best_key = choose_best(agg)
    best_model = build_candidates(seed)[best_key]
    best_model.fit(X, y)

    # Threshold from OOF
    valid_mask = ~np.isnan(oof_probs[best_key])
    best_thr, f1, p, r = pick_threshold_by_f1(y[valid_mask], oof_probs[best_key][valid_mask])
    y_pred, cm, rep = eval_at_threshold(y[valid_mask], oof_probs[best_key][valid_mask], best_thr)

    # Save OOF predictions if requested
    if save_preds:
        ids = ids_series.values if hasattr(ids_series, "values") else ids_series
        oof_df = pd.DataFrame({
            "id": ids[valid_mask],
            "prob_ai": oof_probs[best_key][valid_mask],
            "pred_ai": y_pred,
            "label": y[valid_mask],
            "fold": oof_fold[best_key][valid_mask],
            "model": best_key,
        })
        oof_df.to_csv(out_dir / "oof_predictions.csv", index=False)

    summary = {
        "per_model": agg,
        "selected": best_key,
        "threshold": best_thr,
        "f1_at_thr": f1,
        "precision_at_thr": p,
        "recall_at_thr": r,
        "confusion_matrix_at_thr": cm,
        "classification_report_at_thr": rep,
    }
    return best_model, best_thr, summary


# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to features CSV")
    ap.add_argument("--out_dir", required=True, help="Output directory for artifacts")
    ap.add_argument("--test_size", type=float, default=0.2, help="Hold-out validation size")
    ap.add_argument("--cv", type=int, default=0, help="If >0, use StratifiedKFold with this many folds")
    ap.add_argument("--random_state", type=int, default=42, help="Random seed")
    ap.add_argument("--save_preds", action="store_true", help="If set, save validation/OOF predictions CSV")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    X, y, features, meta, df = load_dataset(Path(args.csv))

    # Build an id series for saving preds; if no 'id' column, synthesize 1..N
    if meta is not None and "id" in meta.columns:
        ids_series = meta["id"]
    else:
        ids_series = pd.Series(np.arange(1, (len(y) if y is not None else len(df)) + 1), name="id")

    if args.cv and args.cv > 1:
        if y is None:
            raise ValueError("CV requires labels; 'label' column not found.")
        best_model, best_thr, summary = train_cv(X, y, features, ids_series, args.random_state, args.cv, out_dir, args.save_preds)
    else:
        if y is None:
            raise ValueError("Hold-out validation requires labels; 'label' column not found.")
        best_model, best_thr, summary = train_holdout(X, y, features, ids_series, args.random_state, args.test_size, out_dir, args.save_preds)

    # Save artifacts
    joblib.dump(best_model, out_dir / "best_model.joblib")
    with open(out_dir / "feature_list.json", "w") as f:
        json.dump(features, f, indent=2)
    with open(out_dir / "best_threshold.json", "w") as f:
        json.dump({"threshold": float(best_thr)}, f, indent=2)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Feature importance (if available)
    fi = maybe_feature_importance(best_model, features)
    if fi is not None:
        with open(out_dir / "feature_importances.json", "w") as f:
            json.dump(fi, f, indent=2)

    # Console summary
    print("\n=== Validation Metrics (handcrafted-only) ===")
    for name, vals in summary["per_model"].items():
        if "folds" in vals:  # CV case
            prs = [v["pr_auc"] for v in vals["folds"]]
            rocs = [v["roc_auc"] for v in vals["folds"]]
            print(f"- {name:10s} | OOF PR-AUC={summary['per_model'][name]['pr_auc']:.4f} "
                  f"| OOF ROC-AUC={summary['per_model'][name]['roc_auc']:.4f} "
                  f"| folds PR-AUC mean={np.mean(prs):.4f}±{np.std(prs):.4f}")
        else:
            print(f"- {name:10s} | PR-AUC={vals['pr_auc']:.4f} | ROC-AUC={vals['roc_auc']:.4f}")

    print(f"\n[best] model = {summary['selected']}")
    print(f"[thr ] best threshold = {summary['threshold']:.3f} "
          f"(F1={summary['f1_at_thr']:.4f}, P={summary['precision_at_thr']:.4f}, R={summary['recall_at_thr']:.4f})")
    print(f"[done] Artifacts saved to: {out_dir.resolve()}")

if __name__ == "__main__":
    main()


# Examples

# Hold-out + save validation predictions

# python train_handcrafted_plus.py \
#   --csv path/to/features.csv \
#   --out_dir ./artifacts \
#   --test_size 0.2 \
#   --random_state 42 \
#   --save_preds
# -> artifacts/val_predictions.csv


# 5-fold CV + save OOF predictions

# python train_handcrafted_plus.py \
#   --csv path/to/features.csv \
#   --out_dir ./artifacts_cv5 \
#   --cv 5 \
#   --random_state 42 \
#   --save_preds
# -> artifacts_cv5/oof_predictions.csv (with 'fold' column)