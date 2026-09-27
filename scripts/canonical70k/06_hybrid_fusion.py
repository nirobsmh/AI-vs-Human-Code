#!/usr/bin/env python3
"""06 — Late-fusion hybrid search on validation; evaluate on frozen test."""

from __future__ import annotations

import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

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
    require_file,
    resolve_path,
    save_json,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)
    models_dir = resolve_path(cfg, "models_dir")
    results_dir = resolve_path(cfg, "results_dir")

    feature_model_path = models_dir / "feature_model.joblib"
    embedding_model_path = models_dir / "embedding_model.joblib"
    require_file(feature_model_path, "Run 05_train_baselines.py")
    require_file(embedding_model_path, "Run 05_train_baselines.py")

    out_config = models_dir / "hybrid_config.json"
    out_search = results_dir / "hybrid_validation_search.csv"
    out_test = results_dir / "hybrid_test_results.csv"
    for p in (out_config, out_search, out_test):
        refuse_overwrite(p, args.force)

    feature_model = joblib.load(feature_model_path)
    embedding_model = joblib.load(embedding_model_path)
    _, _, _ = load_xy(cfg, "train")  # ensure artifacts exist
    Xva_f, Xva_e, yva = load_xy(cfg, "val")
    Xte_f, Xte_e, yte = load_xy(cfg, "test")

    val_pf = feature_model.predict_proba(Xva_f)[:, 1]
    val_pe = embedding_model.predict_proba(Xva_e)[:, 1]
    step = float(cfg["models"]["hybrid"]["alpha_grid_step"])

    rows = []
    for alpha in np.arange(0.0, 1.0 + 1e-12, step):
        p = alpha * val_pf + (1 - alpha) * val_pe
        thr = best_threshold(yva, p)
        pred = (p >= thr).astype(int)
        rows.append(
            {
                "alpha_feature": float(alpha),
                "alpha_embedding": float(1 - alpha),
                "roc_auc": float(roc_auc_score(yva, p)),
                "pr_auc": float(average_precision_score(yva, p)),
                "best_f1": float(f1_score(yva, pred, zero_division=0)),
                "threshold": float(thr),
            }
        )
    search = pd.DataFrame(rows).sort_values(["pr_auc", "best_f1"], ascending=False)
    best = search.iloc[0]
    best_alpha = float(best["alpha_feature"])
    hybrid_thr = float(best["threshold"])

    feature_thr = best_threshold(yva, val_pf)
    embedding_thr = best_threshold(yva, val_pe)

    test_pf = feature_model.predict_proba(Xte_f)[:, 1]
    test_pe = embedding_model.predict_proba(Xte_e)[:, 1]
    test_ph = best_alpha * test_pf + (1 - best_alpha) * test_pe

    final = pd.DataFrame(
        [
            evaluate_probs("Features (RF)", yte, test_pf, feature_thr),
            evaluate_probs("CodeBERT (LR)", yte, test_pe, embedding_thr),
            evaluate_probs("Hybrid Fusion", yte, test_ph, hybrid_thr),
        ]
    )

    print("\n=== TOP VALIDATION FUSION CONFIGS ===")
    print(search.head(10).round(6).to_string(index=False))
    print("\n=== SELECTED HYBRID CONFIGURATION ===")
    print(f"Feature weight   : {best_alpha:.3f}")
    print(f"Embedding weight : {1-best_alpha:.3f}")
    print(f"Threshold        : {hybrid_thr:.3f}")
    print("\n=== FINAL TEST RESULTS ===")
    print(final.round(6).to_string(index=False))

    search.to_csv(out_search, index=False)
    final.to_csv(out_test, index=False)
    save_json(
        out_config,
        {
            "feature_weight": best_alpha,
            "embedding_weight": 1 - best_alpha,
            "hybrid_threshold": hybrid_thr,
            "feature_threshold": feature_thr,
            "embedding_threshold": embedding_thr,
            "val_pr_auc": float(best["pr_auc"]),
            "val_roc_auc": float(best["roc_auc"]),
            "val_f1": float(best["best_f1"]),
        },
    )
    print(f"[done] hybrid artifacts -> {results_dir}")


if __name__ == "__main__":
    main()
