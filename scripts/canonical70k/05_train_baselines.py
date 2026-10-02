#!/usr/bin/env python3
"""05 — Train RF (features) and LR (embeddings) baselines on frozen 70k."""

from __future__ import annotations

import argparse

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from pathlib import Path
import sys

SYS_DIR = Path(__file__).resolve().parent
if str(SYS_DIR) not in sys.path:
    sys.path.insert(0, str(SYS_DIR))

from common import (
    best_threshold,
    ensure_dirs,
    evaluate_probs,
    load_config,
    load_xy,
    refuse_overwrite,
    resolve_path,
    save_json,
)


def build_models(cfg):
    rf_cfg = cfg["models"]["random_forest"]
    lr_cfg = cfg["models"]["logistic_regression"]
    feature_model = RandomForestClassifier(
        n_estimators=int(rf_cfg["n_estimators"]),
        random_state=int(rf_cfg["random_state"]),
        n_jobs=int(rf_cfg["n_jobs"]),
        class_weight=rf_cfg["class_weight"],
    )
    embedding_model = LogisticRegression(
        max_iter=int(lr_cfg["max_iter"]),
        random_state=int(lr_cfg["random_state"]),
        n_jobs=int(lr_cfg["n_jobs"]),
        class_weight=lr_cfg["class_weight"],
    )
    return feature_model, embedding_model


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)
    models_dir = resolve_path(cfg, "models_dir")
    results_dir = resolve_path(cfg, "results_dir")

    out_paths = {
        "feature_model": models_dir / "feature_model.joblib",
        "embedding_model": models_dir / "embedding_model.joblib",
        "thresholds": models_dir / "baseline_thresholds.json",
        "metrics": results_dir / "baseline_test_metrics.csv",
    }
    for p in out_paths.values():
        refuse_overwrite(p, args.force)

    Xtr_f, Xtr_e, ytr = load_xy(cfg, "train")
    Xva_f, Xva_e, yva = load_xy(cfg, "val")
    Xte_f, Xte_e, yte = load_xy(cfg, "test")

    feature_model, embedding_model = build_models(cfg)
    print(f"[info] fitting RF on {Xtr_f.shape}")
    feature_model.fit(Xtr_f, ytr)
    print(f"[info] fitting LR on {Xtr_e.shape}")
    embedding_model.fit(Xtr_e, ytr)

    val_pf = feature_model.predict_proba(Xva_f)[:, 1]
    val_pe = embedding_model.predict_proba(Xva_e)[:, 1]
    thr_f = best_threshold(yva, val_pf)
    thr_e = best_threshold(yva, val_pe)

    test_pf = feature_model.predict_proba(Xte_f)[:, 1]
    test_pe = embedding_model.predict_proba(Xte_e)[:, 1]
    rows = [
        evaluate_probs("Features (RF)", yte, test_pf, thr_f),
        evaluate_probs("CodeBERT (LR)", yte, test_pe, thr_e),
    ]
    metrics = pd.DataFrame(rows)
    print("\n=== BASELINE TEST METRICS ===")
    print(metrics.round(4).to_string(index=False))

    joblib.dump(feature_model, out_paths["feature_model"])
    joblib.dump(embedding_model, out_paths["embedding_model"])
    save_json(
        out_paths["thresholds"],
        {
            "feature_threshold": thr_f,
            "embedding_threshold": thr_e,
            "val_feature_pr_auc": float(
                __import__("sklearn.metrics", fromlist=["average_precision_score"]).average_precision_score(
                    yva, val_pf
                )
            ),
            "val_embedding_pr_auc": float(
                __import__("sklearn.metrics", fromlist=["average_precision_score"]).average_precision_score(
                    yva, val_pe
                )
            ),
        },
    )
    metrics.to_csv(out_paths["metrics"], index=False)
    print(f"[done] models -> {models_dir}")


if __name__ == "__main__":
    main()
