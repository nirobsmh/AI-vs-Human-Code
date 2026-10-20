#!/usr/bin/env python3
"""Stage 4 — Train RF + CodeBERT LR and validation-selected hybrid on frozen 70k.

Uses only Stage-1/2/3 artifacts. Does not resample or regenerate features/embeddings.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

SYS_DIR = Path(__file__).resolve().parent
if str(SYS_DIR) not in sys.path:
    sys.path.insert(0, str(SYS_DIR))

from common import (  # noqa: E402
    ensure_dirs,
    load_config,
    refuse_overwrite,
    resolve_path,
    save_json,
    sha256_file,
)

THRESHOLD_GRID = np.arange(0.05, 0.951, 0.005)
ALPHA_GRID = np.arange(0.0, 1.0 + 1e-12, 0.025)
BOOTSTRAP_SEED = 42
BOOTSTRAP_N = 1000


def sha256_npy(path: Path) -> str:
    return sha256_file(path)


def find_best_threshold(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Maximize F1 on validation only. Never call with test labels for selection."""
    best_thr, best_f1 = 0.5, -1.0
    for thr in THRESHOLD_GRID:
        pred = (probs >= thr).astype(int)
        score = f1_score(y_true, pred, zero_division=0)
        if score > best_f1:
            best_f1, best_thr = float(score), float(thr)
    return best_thr


def assert_probs(name: str, probs: np.ndarray, n: int) -> None:
    if len(probs) != n:
        raise AssertionError(f"{name}: length {len(probs)} != {n}")
    if not np.isfinite(probs).all():
        raise AssertionError(f"{name}: non-finite probabilities")
    if np.any(probs < 0.0) or np.any(probs > 1.0):
        raise AssertionError(f"{name}: probabilities outside [0,1]")


def full_metrics(name: str, y_true: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    pred = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "Model": name,
        "ROC_AUC": float(roc_auc_score(y_true, probs)),
        "PR_AUC": float(average_precision_score(y_true, probs)),
        "Accuracy": float(accuracy_score(y_true, pred)),
        "Precision": float(precision_score(y_true, pred, zero_division=0)),
        "Recall": float(recall_score(y_true, pred, zero_division=0)),
        "F1": float(f1_score(y_true, pred, zero_division=0)),
        "Threshold": float(threshold),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def load_split_arrays(features_dir: Path, embeddings_dir: Path, manifests_dir: Path, split_out: str, manifest_key: str, expected_n: int):
    Xf = np.load(features_dir / f"X_{split_out}.npy")
    y = np.load(features_dir / f"y_{split_out}.npy").astype(int)
    ids_f = np.load(features_dir / f"ids_{split_out}.npy", allow_pickle=True).astype(str)
    Xe = np.load(embeddings_dir / f"X_{split_out}_codebert.npy")
    ids_e = np.load(embeddings_dir / f"ids_{split_out}.npy", allow_pickle=True).astype(str)
    manifest = pd.read_csv(manifests_dir / f"{manifest_key}.csv")
    ids_m = manifest["id"].astype(str).to_numpy()
    y_m = manifest["label"].astype(int).to_numpy()

    if len(manifest) != expected_n:
        raise AssertionError(f"{split_out}: manifest n={len(manifest)} != {expected_n}")
    if Xf.shape != (expected_n, 25):
        raise AssertionError(f"{split_out}: features shape {Xf.shape}")
    if Xe.shape != (expected_n, 768):
        raise AssertionError(f"{split_out}: embeddings shape {Xe.shape}")
    if y.shape != (expected_n,):
        raise AssertionError(f"{split_out}: y shape {y.shape}")

    if not np.array_equal(ids_f, ids_e):
        raise AssertionError(f"{split_out}: feature IDs != embedding IDs")
    if not np.array_equal(ids_f, ids_m):
        raise AssertionError(f"{split_out}: feature IDs != manifest IDs")
    if not np.array_equal(y, y_m):
        raise AssertionError(f"{split_out}: feature labels != manifest labels")

    return Xf, Xe, y, ids_f


def bootstrap_cis(y_true, probs, threshold, seed=BOOTSTRAP_SEED, n_boot=BOOTSTRAP_N):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    roc_vals, pr_vals, f1_vals = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_b = y_true[idx]
        p_b = probs[idx]
        if np.unique(y_b).size < 2:
            continue
        roc_vals.append(roc_auc_score(y_b, p_b))
        pr_vals.append(average_precision_score(y_b, p_b))
        pred = (p_b >= threshold).astype(int)
        f1_vals.append(f1_score(y_b, pred, zero_division=0))

    def pct(vals, est):
        arr = np.asarray(vals, dtype=float)
        return {
            "Estimate": float(est),
            "CI_Lower": float(np.percentile(arr, 2.5)),
            "CI_Upper": float(np.percentile(arr, 97.5)),
            "n_valid_resamples": int(len(arr)),
        }

    pred_full = (probs >= threshold).astype(int)
    return {
        "ROC_AUC": pct(roc_vals, roc_auc_score(y_true, probs)),
        "PR_AUC": pct(pr_vals, average_precision_score(y_true, probs)),
        "F1": pct(f1_vals, f1_score(y_true, pred_full, zero_division=0)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)
    features_dir = resolve_path(cfg, "features_dir")
    embeddings_dir = resolve_path(cfg, "embeddings_dir")
    manifests_dir = resolve_path(cfg, "manifests_dir")
    models_dir = resolve_path(cfg, "models_dir")
    results_dir = resolve_path(cfg, "results_dir")

    out = {
        "rf_model": models_dir / "rf_features.joblib",
        "lr_model": models_dir / "lr_codebert.joblib",
        "val_rf": results_dir / "val_rf_prob.npy",
        "test_rf": results_dir / "test_rf_prob.npy",
        "val_cb": results_dir / "val_codebert_prob.npy",
        "test_cb": results_dir / "test_codebert_prob.npy",
        "val_hy": results_dir / "val_hybrid_prob.npy",
        "test_hy": results_dir / "test_hybrid_prob.npy",
        "alpha_search": results_dir / "hybrid_alpha_search.csv",
        "selected": results_dir / "selected_configuration.json",
        "val_results": results_dir / "main_validation_results.csv",
        "test_results": results_dir / "main_test_results.csv",
        "cis": results_dir / "main_test_confidence_intervals.csv",
        "meta": results_dir / "stage4_meta.json",
    }
    for path in out.values():
        refuse_overwrite(path, args.force)

    # ------------------------------------------------------------------
    # Alignment assertions
    # ------------------------------------------------------------------
    Xtr_f, Xtr_e, ytr, ids_tr = load_split_arrays(
        features_dir, embeddings_dir, manifests_dir, "train", "train", 50000
    )
    Xva_f, Xva_e, yva, ids_va = load_split_arrays(
        features_dir, embeddings_dir, manifests_dir, "validation", "val", 10000
    )
    Xte_f, Xte_e, yte, ids_te = load_split_arrays(
        features_dir, embeddings_dir, manifests_dir, "test", "test", 10000
    )
    print("ALIGNMENT: PASS")

    # ------------------------------------------------------------------
    # A. Random Forest (features)
    # ------------------------------------------------------------------
    rf_params = {
        "n_estimators": 300,
        "random_state": 42,
        "n_jobs": -1,
        "class_weight": "balanced",
    }
    # Prefer config if present, but keep development defaults above as source of truth.
    rf_cfg = cfg.get("models", {}).get("random_forest", {})
    for k in rf_params:
        if k in rf_cfg:
            rf_params[k] = rf_cfg[k]
    rf_params["n_estimators"] = int(rf_params["n_estimators"])
    rf_params["random_state"] = int(rf_params["random_state"])
    rf_params["n_jobs"] = int(rf_params["n_jobs"])

    print("\nRF PARAMETERS")
    for k, v in rf_params.items():
        print(f"  {k}: {v}")

    rf = RandomForestClassifier(**rf_params)
    rf.fit(Xtr_f, ytr)
    # Integrity: threshold/alpha selection uses validation only.
    val_rf = rf.predict_proba(Xva_f)[:, 1]
    test_rf = rf.predict_proba(Xte_f)[:, 1]
    assert_probs("val_rf", val_rf, 10000)
    assert_probs("test_rf", test_rf, 10000)
    rf_threshold = find_best_threshold(yva, val_rf)  # validation labels only

    # ------------------------------------------------------------------
    # B. Logistic Regression (CodeBERT)
    # ------------------------------------------------------------------
    lr_params = {
        "max_iter": 2000,
        "random_state": 42,
        "class_weight": "balanced",
        "n_jobs": -1,
    }
    lr_cfg = cfg.get("models", {}).get("logistic_regression", {})
    for k in lr_params:
        if k in lr_cfg:
            lr_params[k] = lr_cfg[k]
    lr_params["max_iter"] = int(lr_params["max_iter"])
    lr_params["random_state"] = int(lr_params["random_state"])
    lr_params["n_jobs"] = int(lr_params["n_jobs"])

    print("\nCODEBERT LR PARAMETERS")
    for k, v in lr_params.items():
        print(f"  {k}: {v}")

    lr = LogisticRegression(**lr_params)
    lr.fit(Xtr_e, ytr)
    val_cb = lr.predict_proba(Xva_e)[:, 1]
    test_cb = lr.predict_proba(Xte_e)[:, 1]
    assert_probs("val_codebert", val_cb, 10000)
    assert_probs("test_codebert", test_cb, 10000)
    codebert_threshold = find_best_threshold(yva, val_cb)  # validation only

    # ------------------------------------------------------------------
    # C. Hybrid fusion — alpha/threshold on validation only
    # ------------------------------------------------------------------
    alpha_rows = []
    for alpha in ALPHA_GRID:
        # Validation probabilities/labels only for selection:
        p_val = alpha * val_rf + (1.0 - alpha) * val_cb
        thr = find_best_threshold(yva, p_val)
        pred = (p_val >= thr).astype(int)
        alpha_rows.append(
            {
                "alpha_feature": float(alpha),
                "alpha_embedding": float(1.0 - alpha),
                "ROC_AUC": float(roc_auc_score(yva, p_val)),
                "PR_AUC": float(average_precision_score(yva, p_val)),
                "best_F1": float(f1_score(yva, pred, zero_division=0)),
                "threshold": float(thr),
            }
        )
    alpha_search = pd.DataFrame(alpha_rows).sort_values(
        ["PR_AUC", "best_F1"], ascending=False
    ).reset_index(drop=True)
    best = alpha_search.iloc[0]
    hybrid_alpha_feature = float(best["alpha_feature"])
    hybrid_alpha_embedding = float(best["alpha_embedding"])
    hybrid_threshold = float(best["threshold"])

    val_hybrid = hybrid_alpha_feature * val_rf + hybrid_alpha_embedding * val_cb
    test_hybrid = hybrid_alpha_feature * test_rf + hybrid_alpha_embedding * test_cb
    assert_probs("val_hybrid", val_hybrid, 10000)
    assert_probs("test_hybrid", test_hybrid, 10000)

    selected = {
        "rf_threshold": rf_threshold,
        "codebert_threshold": codebert_threshold,
        "hybrid_alpha_feature": hybrid_alpha_feature,
        "hybrid_alpha_embedding": hybrid_alpha_embedding,
        "hybrid_threshold": hybrid_threshold,
    }

    print("\nSELECTED CONFIGURATION")
    print(f"RF threshold: {rf_threshold:.3f}")
    print(f"CodeBERT threshold: {codebert_threshold:.3f}")
    print(f"Hybrid feature weight: {hybrid_alpha_feature:.3f}")
    print(f"Hybrid embedding weight: {hybrid_alpha_embedding:.3f}")
    print(f"Hybrid threshold: {hybrid_threshold:.3f}")

    # ------------------------------------------------------------------
    # D/E. Validation + test evaluation
    # ------------------------------------------------------------------
    val_results = pd.DataFrame(
        [
            full_metrics("Features (RF)", yva, val_rf, rf_threshold),
            full_metrics("CodeBERT (LR)", yva, val_cb, codebert_threshold),
            full_metrics("Hybrid Fusion", yva, val_hybrid, hybrid_threshold),
        ]
    )
    test_results = pd.DataFrame(
        [
            full_metrics("Features (RF)", yte, test_rf, rf_threshold),
            full_metrics("CodeBERT (LR)", yte, test_cb, codebert_threshold),
            full_metrics("Hybrid Fusion", yte, test_hybrid, hybrid_threshold),
        ]
    )

    print("\nVALIDATION RESULTS")
    print(val_results.round(6).to_string(index=False))
    print("\nFINAL TEST RESULTS")
    print(test_results.round(6).to_string(index=False))

    # ------------------------------------------------------------------
    # F. Bootstrap CIs on test only (after selection finished)
    # ------------------------------------------------------------------
    ci_rows = []
    for model_name, probs, thr in [
        ("Features (RF)", test_rf, rf_threshold),
        ("CodeBERT (LR)", test_cb, codebert_threshold),
        ("Hybrid Fusion", test_hybrid, hybrid_threshold),
    ]:
        cis = bootstrap_cis(yte, probs, thr)
        for metric, vals in cis.items():
            ci_rows.append(
                {
                    "Model": model_name,
                    "Metric": metric,
                    "Estimate": vals["Estimate"],
                    "CI_Lower": vals["CI_Lower"],
                    "CI_Upper": vals["CI_Upper"],
                }
            )
    ci_df = pd.DataFrame(ci_rows)
    print("\nBOOTSTRAP 95% CIs")
    print(ci_df.round(6).to_string(index=False))

    # ------------------------------------------------------------------
    # Persist artifacts
    # ------------------------------------------------------------------
    joblib.dump(rf, out["rf_model"])
    joblib.dump(lr, out["lr_model"])
    np.save(out["val_rf"], val_rf.astype(np.float64))
    np.save(out["test_rf"], test_rf.astype(np.float64))
    np.save(out["val_cb"], val_cb.astype(np.float64))
    np.save(out["test_cb"], test_cb.astype(np.float64))
    np.save(out["val_hy"], val_hybrid.astype(np.float64))
    np.save(out["test_hy"], test_hybrid.astype(np.float64))
    alpha_search.to_csv(out["alpha_search"], index=False)
    save_json(out["selected"], selected)
    val_results.to_csv(out["val_results"], index=False)
    test_results.to_csv(out["test_results"], index=False)
    ci_df.to_csv(out["cis"], index=False)

    # ------------------------------------------------------------------
    # H. Model reload check
    # ------------------------------------------------------------------
    rf2 = joblib.load(out["rf_model"])
    lr2 = joblib.load(out["lr_model"])
    reload_val_rf = rf2.predict_proba(Xva_f)[:, 1]
    reload_test_rf = rf2.predict_proba(Xte_f)[:, 1]
    reload_val_cb = lr2.predict_proba(Xva_e)[:, 1]
    reload_test_cb = lr2.predict_proba(Xte_e)[:, 1]
    ok = (
        np.allclose(reload_val_rf, val_rf, rtol=1e-7, atol=1e-7)
        and np.allclose(reload_test_rf, test_rf, rtol=1e-7, atol=1e-7)
        and np.allclose(reload_val_cb, val_cb, rtol=1e-7, atol=1e-7)
        and np.allclose(reload_test_cb, test_cb, rtol=1e-7, atol=1e-7)
    )
    if not ok:
        raise AssertionError("MODEL RELOAD CHECK: FAIL")
    print("\nMODEL RELOAD CHECK: PASS")

    # ------------------------------------------------------------------
    # G. Metadata
    # ------------------------------------------------------------------
    meta = {
        "stage": 4,
        "python_version": sys.version,
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "random_seed": 42,
        "threshold_search_grid": {
            "start": 0.05,
            "stop": 0.95,
            "step": 0.005,
        },
        "alpha_search_grid": {
            "start": 0.0,
            "stop": 1.0,
            "step": 0.025,
        },
        "rf_parameters": rf_params,
        "lr_parameters": lr_params,
        "selected_configuration": selected,
        "bootstrap": {"seed": BOOTSTRAP_SEED, "n_resamples": BOOTSTRAP_N},
        "manifest_fingerprints": {
            name: sha256_file(manifests_dir / f"{name}.csv")
            for name in ("train", "val", "test")
        },
        "feature_fingerprints": {
            f"X_{s}.npy": sha256_npy(features_dir / f"X_{s}.npy")
            for s in ("train", "validation", "test")
        },
        "embedding_fingerprints": {
            f"X_{s}_codebert.npy": sha256_npy(embeddings_dir / f"X_{s}_codebert.npy")
            for s in ("train", "validation", "test")
        },
        "output_file_hashes": {k: sha256_file(v) for k, v in out.items() if v.exists()},
        "alignment": {
            "train_ids_match": True,
            "validation_ids_match": True,
            "test_ids_match": True,
            "rows": {"train": 50000, "validation": 10000, "test": 10000},
            "feature_dim": 25,
            "embedding_dim": 768,
        },
        "integrity": {
            "test_unused_during_selection": True,
            "model_reload_check": True,
        },
    }
    save_json(out["meta"], meta)

    print("\nOutput paths")
    for k, v in out.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
