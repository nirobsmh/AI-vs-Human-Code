#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train classifiers on precomputed embeddings.

Input CSV:
  id,generator,label,language,embedding
Where `embedding` is a JSON array string, e.g., "[0.01, -0.02, ...]".

Models:
- logreg (scaled)
- svm_rbf (scaled, calibrated for probabilities)
- rf
- hgb
- extratrees
- (optional) xgb, lgbm if installed

Selection metric: PR-AUC (tiebreaker ROC-AUC)
Threshold: tuned to maximize F1 on validation or on OOF (for CV)
Artifacts:
- best_model.joblib
- feature_list.json (just ["embedding"])
- best_threshold.json
- metrics.json
- (optional) val_predictions.csv / oof_predictions.csv

Usage (hold-out):
  python train_embeddings.py --csv embeddings.csv --out_dir ./artifacts_emb --test_size 0.2 --save_preds

Usage (k-fold CV):
  python train_embeddings.py --csv embeddings.csv --out_dir ./artifacts_emb_cv --cv 5 --save_preds
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
from sklearn.calibration import CalibratedClassifierCV

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


def load_embeddings(csv_path: Path, label_col: str = "label"):
    df = pd.read_csv(csv_path)
    if label_col not in df.columns:
        raise ValueError("Embeddings CSV must contain a 'label' column for supervised training.")

    # Coerce labels to 0/1 if needed
    if df[label_col].dtype == object:
        mapping = {"human": 0, "ai": 1, "Human": 0, "AI": 1}
        df[label_col] = df[label_col].map(mapping).fillna(df[label_col]).astype(int)
    else:
        df[label_col] = df[label_col].astype(int)

    # Parse JSON embeddings
    emb = df["embedding"].apply(lambda s: np.array(json.loads(s), dtype=np.float32))
    dim = int(emb.iloc[0].shape[0]) if len(emb) else 0
    X = np.vstack(emb.values)  # [N, D]
    y = df[label_col].values.astype(int)

    # Optional: defensive clean-up (should be normalized already)
    X = np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    ids = df["id"].values if "id" in df.columns else np.arange(1, len(df) + 1)
    return X, y, ids, dim


def pr_auc(y_true, y_prob):
    return float(average_precision_score(y_true, y_prob))

def roc_auc(y_true, y_prob):
    return float(roc_auc_score(y_true, y_prob))

def pick_threshold_by_f1(y_true, y_prob):
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    if len(thr) == 0:
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
    # SVM needs probabilities; we'll calibrate a Linear SVC via CalibratedClassifierCV or just use RBF SVC with probability=True.
    # RBF can be slower; try it first and switch if too slow.
    models = {
        "logreg": make_pipeline(
            StandardScaler(with_mean=False),  # embeddings are already normalized; keep with_mean=False for sparse safety
            LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=-1)
        ),
        "svm_rbf": make_pipeline(
            StandardScaler(with_mean=False),
            SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=seed)
        ),
        "rf": RandomForestClassifier(
            n_estimators=500, max_depth=None, min_samples_leaf=1,
            class_weight="balanced", random_state=seed, n_jobs=-1
        ),
        "hgb": HistGradientBoostingClassifier(
            learning_rate=0.08, max_depth=None, max_bins=255,
            early_stopping=True, random_state=seed
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
            random_state=seed, n_jobs=-1, eval_metric="logloss", tree_method="hist"
        )
    if HAS_LGBM:
        models["lgbm"] = LGBMClassifier(
            n_estimators=1000, learning_rate=0.05, subsample=0.9,
            colsample_bytree=0.9, reg_lambda=0.0, random_state=seed, n_jobs=-1
        )
    return models


def choose_best(results_dict):
    best_key = None
    best_tuple = (-1.0, -1.0)
    for k, v in results_dict.items():
        cand = (v["pr_auc"], v["roc_auc"])
        if cand > best_tuple:
            best_tuple = cand
            best_key = k
    return best_key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Embeddings CSV (id,generator,label,language,embedding)")
    ap.add_argument("--out_dir", required=True, help="Artifacts directory")
    ap.add_argument("--test_size", type=float, default=0.2, help="Hold-out validation size")
    ap.add_argument("--cv", type=int, default=0, help="If >0, use StratifiedKFold with this many folds")
    ap.add_argument("--random_state", type=int, default=42, help="Random seed")
    ap.add_argument("--save_preds", action="store_true", help="Save validation/OOF predictions CSV")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    X, y, ids, dim = load_embeddings(Path(args.csv))
    print(f"[info] Loaded embeddings: N={len(y)}, dim={dim}")

    candidates = build_candidates(args.random_state)

    if args.cv and args.cv > 1:
        # Stratified K-Fold CV with out-of-fold predictions for model selection/threshold tuning
        skf = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.random_state)
        oof = {name: np.full(len(y), np.nan, dtype=float) for name in candidates}
        oof_fold = {name: np.full(len(y), -1, dtype=int) for name in candidates}
        fold_scores = {name: [] for name in candidates}

        for fold, (tr, va) in enumerate(skf.split(X, y), 1):
            Xtr, Xval, ytr, yval = X[tr], X[va], y[tr], y[va]
            for name, model in candidates.items():
                model.fit(Xtr, ytr)
                val_prob = model.predict_proba(Xval)[:, 1]
                oof[name][va] = val_prob
                oof_fold[name][va] = fold
                fold_scores[name].append({
                    "fold": fold,
                    "pr_auc": pr_auc(yval, val_prob),
                    "roc_auc": roc_auc(yval, val_prob),
                })
            print(f"[progress] fold {fold}/{args.cv} done")

        # Pick best by OOF PR-AUC
        summary = {"per_model": {}}
        for name in candidates:
            mask = ~np.isnan(oof[name])
            pr = pr_auc(y[mask], oof[name][mask])
            ro = roc_auc(y[mask], oof[name][mask])
            summary["per_model"][name] = {
                "pr_auc": pr, "roc_auc": ro, "folds": fold_scores[name]
            }
        best_key = choose_best(summary["per_model"])
        best_model = candidates[best_key].fit(X, y)

        # Threshold from OOF
        mask = ~np.isnan(oof[best_key])
        thr, f1, p, r = pick_threshold_by_f1(y[mask], oof[best_key][mask])
        y_pred, cm, rep = eval_at_threshold(y[mask], oof[best_key][mask], thr)

        summary.update({
            "selected": best_key,
            "threshold": float(thr),
            "f1_at_thr": float(f1),
            "precision_at_thr": float(p),
            "recall_at_thr": float(r),
            "confusion_matrix_at_thr": cm,
            "classification_report_at_thr": rep,
        })

        if args.save_preds:
            oof_df = pd.DataFrame({
                "id": ids[mask],
                "prob_ai": oof[best_key][mask],
                "pred_ai": y_pred,
                "label": y[mask],
                "fold": oof_fold[best_key][mask],
                "model": best_key,
            })
            oof_df.to_csv(out_dir / "oof_predictions.csv", index=False)

    else:
        # Hold-out validation
        Xtr, Xval, ytr, yval, id_tr, id_val = train_test_split(
            X, y, ids, test_size=args.test_size, random_state=args.random_state, stratify=y
        )
        results = {}
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

        summary = {
            "per_model": {
                k: {"pr_auc": v["pr_auc"], "roc_auc": v["roc_auc"]}
                for k, v in results.items()
            },
            "selected": best_key,
            "threshold": float(thr),
            "f1_at_thr": float(f1),
            "precision_at_thr": float(p),
            "recall_at_thr": float(r),
            "confusion_matrix_at_thr": cm,
            "classification_report_at_thr": rep,
        }

        best_model = results[best_key]["model"]

        if args.save_preds:
            val_df = pd.DataFrame({
                "id": best["ids"],
                "prob_ai": best["val_prob"],
                "pred_ai": y_pred,
                "label": best["yval"],
                "model": best_key,
            })
            val_df.to_csv(out_dir / "val_predictions.csv", index=False)

    # Save artifacts
    joblib.dump(best_model, out_dir / "best_model.joblib")
    with open(out_dir / "feature_list.json", "w") as f:
        json.dump(["embedding"], f, indent=2)
    with open(out_dir / "best_threshold.json", "w") as f:
        json.dump({"threshold": summary["threshold"]}, f, indent=2)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Console summary
    print("\n=== Validation Metrics (embeddings-only) ===")
    for name, vals in summary["per_model"].items():
        if "folds" in vals:
            prs = [v["pr_auc"] for v in vals["folds"]]
            rocs = [v["roc_auc"] for v in vals["folds"]]
            print(f"- {name:10s} | OOF PR-AUC={vals['pr_auc']:.4f} | OOF ROC-AUC={vals['roc_auc']:.4f} "
                  f"| folds PR-AUC mean={np.mean(prs):.4f}±{np.std(prs):.4f}")
        else:
            print(f"- {name:10s} | PR-AUC={vals['pr_auc']:.4f} | ROC-AUC={vals['roc_auc']:.4f}")

    print(f"\n[best] model = {summary['selected']}")
    print(f"[thr ] best threshold = {summary['threshold']:.3f} "
          f"(F1={summary['f1_at_thr']:.4f}, P={summary['precision_at_thr']:.4f}, R={summary['recall_at_thr']:.4f})")
    print(f"[done] Artifacts -> {out_dir.resolve()}")

if __name__ == "__main__":
    main()


# python train_embeddings.py \
#   --csv ../../feature/embedding_feature_task_a_training_set_1_10k.csv \ 
#   --out_dir ./artifacts_emb \
#   --test_size 0.2 \
#   --random_state 42 \
#   --save_preds