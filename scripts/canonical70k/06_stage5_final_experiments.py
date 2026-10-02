#!/usr/bin/env python3
"""Stage 5 — Final experiments on frozen canonical70k (no Stage-4 retrain)."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

SYS_DIR = Path(__file__).resolve().parent
if str(SYS_DIR) not in sys.path:
    sys.path.insert(0, str(SYS_DIR))

from common import (  # noqa: E402
    ensure_dirs,
    is_python_series,
    load_config,
    load_json,
    load_manifest,
    refuse_overwrite,
    resolve_path,
    save_json,
    sha256_file,
)

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "feature"))
from extract_handcraft_features import FEATURE_FIELDS, features_for_code  # noqa: E402
from embed_code import embed_batch  # noqa: E402
from code_transforms import (  # noqa: E402
    ast_normalize_code,
    black_format_code,
    remove_comments_only,
)

EXPECTED_CONFIG = {
    "rf_threshold": 0.530,
    "codebert_threshold": 0.380,
    "hybrid_alpha_feature": 0.875,
    "hybrid_alpha_embedding": 0.125,
    "hybrid_threshold": 0.570,
}
SURFACE_FIELDS = [
    "n_lines",
    "n_chars",
    "avg_line_len",
    "blank_ratio",
    "comment_ratio",
    "avg_leading_spaces",
    "avg_leading_tabs",
    "token_entropy",
]
AST_FIELDS = [
    "has_docstring",
    "n_identifiers",
    "avg_identifier_len",
    "generic_identifier_ratio",
    "snake_case_ratio",
    "camel_case_ratio",
    "unique_identifier_ratio",
    "ast_depth",
    "n_ifs",
    "n_loops",
    "n_funcs",
    "n_imports",
    "n_try",
    "n_with",
    "n_comprehensions",
    "n_boolops",
    "cyclomatic_complexity_est",
]
LENGTH_FIELDS = ["n_lines", "n_chars"]
THRESHOLD_GRID = np.arange(0.05, 0.951, 0.005)
RF_PARAMS = dict(n_estimators=300, random_state=42, n_jobs=-1, class_weight="balanced")
BOOT_SEED = 42
BOOT_N = 1000


def parse_ok(code: str) -> bool:
    try:
        ast.parse(str(code))
        return True
    except Exception:
        return False


def find_best_threshold(y_true, probs) -> float:
    best_thr, best_f1 = 0.5, -1.0
    for thr in THRESHOLD_GRID:
        pred = (probs >= thr).astype(int)
        score = f1_score(y_true, pred, zero_division=0)
        if score > best_f1:
            best_f1, best_thr = float(score), float(thr)
    return best_thr


def full_metrics(y_true, probs, thr) -> dict:
    pred = (probs >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "ROC_AUC": float(roc_auc_score(y_true, probs)),
        "PR_AUC": float(average_precision_score(y_true, probs)),
        "Accuracy": float(accuracy_score(y_true, pred)),
        "Precision": float(precision_score(y_true, pred, zero_division=0)),
        "Recall": float(recall_score(y_true, pred, zero_division=0)),
        "F1": float(f1_score(y_true, pred, zero_division=0)),
        "Threshold": float(thr),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def verify_selected_config(path: Path) -> dict:
    cfg = load_json(path)
    for key, expected in EXPECTED_CONFIG.items():
        got = float(cfg[key])
        if abs(got - expected) > 1e-6:
            raise AssertionError(
                f"selected_configuration.json {key}={got} != expected {expected}"
            )
    return {
        "rf_threshold": float(cfg["rf_threshold"]),
        "codebert_threshold": float(cfg["codebert_threshold"]),
        "hybrid_alpha_feature": float(cfg["hybrid_alpha_feature"]),
        "hybrid_alpha_embedding": float(cfg["hybrid_alpha_embedding"]),
        "hybrid_threshold": float(cfg["hybrid_threshold"]),
    }


def paired_bootstrap_hybrid_vs_rf(yte, test_rf, test_hy, rf_thr, hy_thr) -> pd.DataFrame:
    rng = np.random.default_rng(BOOT_SEED)
    n = len(yte)
    obs = {
        "ROC_AUC": roc_auc_score(yte, test_hy) - roc_auc_score(yte, test_rf),
        "PR_AUC": average_precision_score(yte, test_hy)
        - average_precision_score(yte, test_rf),
        "F1": f1_score(yte, (test_hy >= hy_thr).astype(int), zero_division=0)
        - f1_score(yte, (test_rf >= rf_thr).astype(int), zero_division=0),
    }
    deltas = {"ROC_AUC": [], "PR_AUC": [], "F1": []}
    for _ in range(BOOT_N):
        idx = rng.integers(0, n, size=n)
        yb = yte[idx]
        rf_b = test_rf[idx]
        hy_b = test_hy[idx]
        if np.unique(yb).size >= 2:
            deltas["ROC_AUC"].append(roc_auc_score(yb, hy_b) - roc_auc_score(yb, rf_b))
            deltas["PR_AUC"].append(
                average_precision_score(yb, hy_b) - average_precision_score(yb, rf_b)
            )
        deltas["F1"].append(
            f1_score(yb, (hy_b >= hy_thr).astype(int), zero_division=0)
            - f1_score(yb, (rf_b >= rf_thr).astype(int), zero_division=0)
        )
    rows = []
    for metric, vals in deltas.items():
        arr = np.asarray(vals, dtype=float)
        rows.append(
            {
                "Metric": metric,
                "Observed_Difference": float(obs[metric]),
                "CI_Lower": float(np.percentile(arr, 2.5)),
                "CI_Upper": float(np.percentile(arr, 97.5)),
            }
        )
    return pd.DataFrame(rows)


def python_masks(cfg, manifests):
    masks = {}
    for split, key in (("train", "train"), ("validation", "val"), ("test", "test")):
        df = manifests[key]
        masks[split] = is_python_series(df["language"], cfg).to_numpy()
    return masks


def codes_to_features(codes) -> np.ndarray:
    rows = []
    for code in codes:
        feats = features_for_code(code)
        rows.append([feats[name] for name in FEATURE_FIELDS])
    return np.asarray(rows, dtype=np.float32)


def run_feature_ablation(cfg, features_dir, manifests, results_dir) -> pd.DataFrame:
    if len(FEATURE_FIELDS) != 25:
        raise AssertionError(f"Expected 25 FEATURE_FIELDS, got {len(FEATURE_FIELDS)}")
    if set(SURFACE_FIELDS) | set(AST_FIELDS) != set(FEATURE_FIELDS):
        raise AssertionError("Surface/AST feature partition does not cover FEATURE_FIELDS")
    if set(SURFACE_FIELDS) & set(AST_FIELDS):
        raise AssertionError("Surface/AST feature partitions overlap")
    if len(SURFACE_FIELDS) != 8 or len(AST_FIELDS) != 17:
        raise AssertionError("Expected 8 surface + 17 AST features")
    surface_idx = [FEATURE_FIELDS.index(x) for x in SURFACE_FIELDS]
    ast_idx = [FEATURE_FIELDS.index(x) for x in AST_FIELDS]
    length_idx = [FEATURE_FIELDS.index(x) for x in LENGTH_FIELDS]

    Xtr = np.load(features_dir / "X_train.npy")
    ytr = np.load(features_dir / "y_train.npy").astype(int)
    Xva = np.load(features_dir / "X_validation.npy")
    yva = np.load(features_dir / "y_validation.npy").astype(int)
    Xte = np.load(features_dir / "X_test.npy")
    yte = np.load(features_dir / "y_test.npy").astype(int)

    masks = python_masks(cfg, manifests)
    Xtr_p, ytr_p = Xtr[masks["train"]], ytr[masks["train"]]
    Xva_p, yva_p = Xva[masks["validation"]], yva[masks["validation"]]
    Xte_p, yte_p = Xte[masks["test"]], yte[masks["test"]]
    n_counts = {
        "N_train": int(len(ytr_p)),
        "N_validation": int(len(yva_p)),
        "N_test": int(len(yte_p)),
    }

    rows = []

    def eval_rf(name, idx=None):
        model = RandomForestClassifier(**RF_PARAMS)
        tr = Xtr_p if idx is None else Xtr_p[:, idx]
        va = Xva_p if idx is None else Xva_p[:, idx]
        te = Xte_p if idx is None else Xte_p[:, idx]
        model.fit(tr, ytr_p)
        thr = find_best_threshold(yva_p, model.predict_proba(va)[:, 1])
        m = full_metrics(yte_p, model.predict_proba(te)[:, 1], thr)
        rows.append({"Model": name, **m, **n_counts})

    eval_rf("RF - All 25")
    eval_rf("RF - Surface 8", surface_idx)
    eval_rf("RF - AST 17", ast_idx)
    eval_rf("RF - Length only", length_idx)

    # Parse-failure-only baseline
    def parse_score(code):
        return 0.0 if parse_ok(code) else 1.0

    test_codes = manifests["test"].loc[masks["test"], "code"].astype(str)
    scores = test_codes.map(parse_score).to_numpy(dtype=float)
    pred = (scores >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(yte_p, pred, labels=[0, 1]).ravel()
    rows.append(
        {
            "Model": "Parse failure only",
            "ROC_AUC": float(roc_auc_score(yte_p, scores)),
            "PR_AUC": float(average_precision_score(yte_p, scores)),
            "Accuracy": float(accuracy_score(yte_p, pred)),
            "Precision": float(precision_score(yte_p, pred, zero_division=0)),
            "Recall": float(recall_score(yte_p, pred, zero_division=0)),
            "F1": float(f1_score(yte_p, pred, zero_division=0)),
            "Threshold": 0.5,
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
            **n_counts,
        }
    )
    return pd.DataFrame(rows)


def run_parseable_control(cfg, features_dir, manifests, results_dir):
    Xtr = np.load(features_dir / "X_train.npy")
    ytr = np.load(features_dir / "y_train.npy").astype(int)
    Xva = np.load(features_dir / "X_validation.npy")
    yva = np.load(features_dir / "y_validation.npy").astype(int)
    Xte = np.load(features_dir / "X_test.npy")
    yte = np.load(features_dir / "y_test.npy").astype(int)

    count_rows = []
    masks = {}
    for split, key, X, y in (
        ("train", "train", Xtr, ytr),
        ("validation", "val", Xva, yva),
        ("test", "test", Xte, yte),
    ):
        df = manifests[key]
        py = is_python_series(df["language"], cfg).to_numpy()
        ok = df["code"].astype(str).map(parse_ok).to_numpy()
        mask = py & ok
        masks[split] = mask
        for label in (0, 1):
            count_rows.append(
                {
                    "Split": split,
                    "Label": "Human" if label == 0 else "AI",
                    "N": int(((y == label) & mask).sum()),
                }
            )
    counts = pd.DataFrame(count_rows)
    counts.to_csv(results_dir / "parseability_by_split_and_label.csv", index=False)

    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(Xtr[masks["train"]], ytr[masks["train"]])
    thr = find_best_threshold(
        yva[masks["validation"]],
        model.predict_proba(Xva[masks["validation"]])[:, 1],
    )
    m = full_metrics(
        yte[masks["test"]],
        model.predict_proba(Xte[masks["test"]])[:, 1],
        thr,
    )
    result = pd.DataFrame(
        [
            {
                "Model": "RF - Parseable Python only",
                **m,
                "N_train": int(masks["train"].sum()),
                "N_validation": int(masks["validation"].sum()),
                "N_test": int(masks["test"].sum()),
            }
        ]
    )
    result.to_csv(results_dir / "parseable_python_results.csv", index=False)
    return counts, result, masks["test"]


def metric_pair(y, p_orig, p_trans, thr) -> dict:
    o = full_metrics(y, p_orig, thr)
    t = full_metrics(y, p_trans, thr)
    o_pred = (p_orig >= thr).astype(int)
    t_pred = (p_trans >= thr).astype(int)
    return {
        "Original_ROC_AUC": o["ROC_AUC"],
        "Transformed_ROC_AUC": t["ROC_AUC"],
        "Delta_ROC_AUC": t["ROC_AUC"] - o["ROC_AUC"],
        "Original_PR_AUC": o["PR_AUC"],
        "Transformed_PR_AUC": t["PR_AUC"],
        "Delta_PR_AUC": t["PR_AUC"] - o["PR_AUC"],
        "Original_Accuracy": o["Accuracy"],
        "Transformed_Accuracy": t["Accuracy"],
        "Delta_Accuracy": t["Accuracy"] - o["Accuracy"],
        "Original_Precision": o["Precision"],
        "Transformed_Precision": t["Precision"],
        "Original_Recall": o["Recall"],
        "Transformed_Recall": t["Recall"],
        "Original_F1": o["F1"],
        "Transformed_F1": t["F1"],
        "Delta_F1": t["F1"] - o["F1"],
        "Original_FP": o["FP"],
        "Transformed_FP": t["FP"],
        "Original_FN": o["FN"],
        "Transformed_FN": t["FN"],
        "Prediction_Flip_Rate": float(np.mean(o_pred != t_pred)),
    }


def paired_delta_bootstrap(y, p_orig, p_trans, thr, seed=BOOT_SEED, n_boot=BOOT_N):
    rng = np.random.default_rng(seed)
    n = len(y)
    obs = metric_pair(y, p_orig, p_trans, thr)
    buckets = {"Delta_ROC_AUC": [], "Delta_PR_AUC": [], "Delta_F1": []}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb, ob, tb = y[idx], p_orig[idx], p_trans[idx]
        if np.unique(yb).size >= 2:
            buckets["Delta_ROC_AUC"].append(
                roc_auc_score(yb, tb) - roc_auc_score(yb, ob)
            )
            buckets["Delta_PR_AUC"].append(
                average_precision_score(yb, tb) - average_precision_score(yb, ob)
            )
        buckets["Delta_F1"].append(
            f1_score(yb, (tb >= thr).astype(int), zero_division=0)
            - f1_score(yb, (ob >= thr).astype(int), zero_division=0)
        )
    rows = []
    for metric, vals in buckets.items():
        arr = np.asarray(vals, dtype=float)
        rows.append(
            {
                "Metric": metric,
                "Observed_Difference": float(obs[metric]),
                "CI_Lower": float(np.percentile(arr, 2.5)),
                "CI_Upper": float(np.percentile(arr, 97.5)),
            }
        )
    return rows, obs


def run_robustness(
    cfg,
    manifests,
    features_dir,
    embeddings_dir,
    models_dir,
    results_dir,
    selected,
    test_rf_prob,
    test_cb_prob,
    test_hy_prob,
):
    test_df = manifests["test"].reset_index(drop=True)
    py = is_python_series(test_df["language"], cfg).to_numpy()
    ok = test_df["code"].astype(str).map(parse_ok).to_numpy()
    cohort_mask = py & ok
    cohort = test_df.loc[cohort_mask].copy().reset_index(drop=True)
    global_idx = np.where(cohort_mask)[0]
    print(f"total N: {len(cohort)}")
    print(f"human N: {int((cohort['label']==0).sum())}")
    print(f"AI N: {int((cohort['label']==1).sum())}")

    cohort["code_black"] = cohort["code"].map(black_format_code)
    cohort["code_ast"] = cohort["code"].map(ast_normalize_code)
    cohort["code_comments_removed"] = cohort["code"].map(remove_comments_only)

    success_rows = []
    for col, name in [
        ("code_black", "Black formatting"),
        ("code_ast", "AST normalization"),
        ("code_comments_removed", "Comments removed"),
    ]:
        # Successful transforms must remain parseable.
        ok_mask = cohort[col].notna()
        if ok_mask.any():
            still_parseable = cohort.loc[ok_mask, col].astype(str).map(parse_ok)
            if not bool(still_parseable.all()):
                raise AssertionError(f"{name}: transformed code not all parseable")
        for label, label_name in ((0, "Human"), (1, "AI")):
            g = cohort[cohort["label"] == label]
            valid = int(g[col].notna().sum())
            success_rows.append(
                {
                    "Transformation": name,
                    "Label": label_name,
                    "Valid": valid,
                    "Total": len(g),
                    "Success_Rate": valid / len(g) if len(g) else np.nan,
                }
            )
        print(col, int(cohort[col].notna().sum()), "/", len(cohort))
    success_df = pd.DataFrame(success_rows)
    success_df.to_csv(results_dir / "robustness_transform_success.csv", index=False)

    common = (
        cohort["code_black"].notna()
        & cohort["code_ast"].notna()
        & cohort["code_comments_removed"].notna()
    )
    print(f"original cohort N: {len(cohort)}")
    print(f"final common N: {int(common.sum())}")
    eval_df = cohort.loc[common].reset_index(drop=True)
    eval_global = global_idx[np.where(common.to_numpy())[0]]
    labels = eval_df["label"].astype(int).to_numpy()
    ids = eval_df["id"].astype(str).to_numpy()

    # Load Stage-4 models (no retrain)
    rf = joblib.load(models_dir / "rf_features.joblib")
    lr = joblib.load(models_dir / "lr_codebert.joblib")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = cfg.get("codebert", {}).get("model", "microsoft/codebert-base")
    max_len = int(cfg.get("codebert", {}).get("max_length", 512))
    batch_size = int(cfg.get("codebert", {}).get("batch_size", 32))
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoder = AutoModel.from_pretrained(model_name)
    encoder.to(device)
    encoder.eval()

    orig_rf = test_rf_prob[eval_global]
    orig_cb = test_cb_prob[eval_global]
    orig_hy = test_hy_prob[eval_global]

    transforms = [
        ("code_black", "Black formatting"),
        ("code_ast", "AST normalization"),
        ("code_comments_removed", "Comments removed"),
    ]
    result_rows = []
    ci_rows = []

    for col, name in transforms:
        codes = eval_df[col].astype(str).tolist()
        # Features
        Xf = []
        for code in tqdm(codes, desc=f"features-{name}", leave=False):
            feats = features_for_code(code)
            Xf.append([feats[f] for f in FEATURE_FIELDS])
        Xf = np.asarray(Xf, dtype=np.float32)
        if not np.isfinite(Xf).all():
            raise AssertionError(f"{name}: non-finite features")

        # Embeddings
        chunks = []
        for start in tqdm(
            range(0, len(codes), batch_size), desc=f"embed-{name}", leave=False
        ):
            batch = codes[start : start + batch_size]
            chunks.append(
                embed_batch(batch, tokenizer, encoder, device, max_len=max_len).astype(
                    np.float32
                )
            )
        Xe = np.vstack(chunks)
        if Xe.shape != (len(codes), 768):
            raise AssertionError(f"{name}: bad embedding shape {Xe.shape}")

        p_rf = rf.predict_proba(Xf)[:, 1]
        p_cb = lr.predict_proba(Xe)[:, 1]
        p_hy = (
            selected["hybrid_alpha_feature"] * p_rf
            + selected["hybrid_alpha_embedding"] * p_cb
        )

        for model_name_, o_prob, t_prob, thr in [
            ("Features (RF)", orig_rf, p_rf, selected["rf_threshold"]),
            ("CodeBERT (LR)", orig_cb, p_cb, selected["codebert_threshold"]),
            ("Hybrid Fusion", orig_hy, p_hy, selected["hybrid_threshold"]),
        ]:
            pair = metric_pair(labels, o_prob, t_prob, thr)
            result_rows.append(
                {"Transformation": name, "N": len(labels), "Model": model_name_, **pair}
            )
            boot_rows, _ = paired_delta_bootstrap(labels, o_prob, t_prob, thr)
            for br in boot_rows:
                ci_rows.append(
                    {
                        "Transformation": name,
                        "Model": model_name_,
                        **br,
                    }
                )

        # ID sanity
        if not np.array_equal(ids, eval_df["id"].astype(str).to_numpy()):
            raise AssertionError("robustness ID order changed")

    results = pd.DataFrame(result_rows)
    cis = pd.DataFrame(ci_rows)
    results.to_csv(results_dir / "robustness_fixed_cohort.csv", index=False)
    cis.to_csv(results_dir / "robustness_delta_confidence_intervals.csv", index=False)
    return {
        "cohort_n": int(len(cohort)),
        "common_n": int(len(eval_df)),
        "human_n": int((cohort["label"] == 0).sum()),
        "ai_n": int((cohort["label"] == 1).sum()),
        "results": results,
        "cis": cis,
        "success": success_df,
    }


def build_final_summary(results_dir: Path, manifests_dir: Path) -> dict:
    def read_csv(name):
        path = results_dir / name
        return pd.read_csv(path).to_dict(orient="records") if path.exists() else {}

    def read_json(name):
        path = results_dir / name
        return load_json(path) if path.exists() else {}

    summary = {
        "dataset": load_json(manifests_dir / "split_meta.json")
        if (manifests_dir / "split_meta.json").exists()
        else {},
        "split_audit": read_json("split_audit.json"),
        "main_validation_results": read_csv("main_validation_results.csv"),
        "main_test_results": read_csv("main_test_results.csv"),
        "main_test_confidence_intervals": read_csv("main_test_confidence_intervals.csv"),
        "hybrid_vs_rf_paired_bootstrap": read_csv("hybrid_vs_rf_paired_bootstrap.csv"),
        "selected_configuration": read_json("selected_configuration.json"),
        "python_feature_ablation": read_csv("python_feature_ablation.csv"),
        "parseability": read_csv("parseability_by_split_and_label.csv"),
        "parseable_python_control": read_csv("parseable_python_results.csv"),
        "robustness": read_csv("robustness_fixed_cohort.csv"),
        "robustness_confidence_intervals": read_csv(
            "robustness_delta_confidence_intervals.csv"
        ),
    }
    return summary


def assert_summary_matches(summary: dict, results_dir: Path):
    mapping = {
        "main_test_results": "main_test_results.csv",
        "hybrid_vs_rf_paired_bootstrap": "hybrid_vs_rf_paired_bootstrap.csv",
        "python_feature_ablation": "python_feature_ablation.csv",
        "parseable_python_control": "parseable_python_results.csv",
        "robustness": "robustness_fixed_cohort.csv",
    }
    for key, fname in mapping.items():
        src = pd.read_csv(results_dir / fname)
        got = pd.DataFrame(summary[key])
        # Compare numeric columns with tolerance
        for col in src.columns:
            if col not in got.columns:
                raise AssertionError(f"summary missing column {col} for {key}")
            if pd.api.types.is_numeric_dtype(src[col]):
                if not np.allclose(
                    got[col].to_numpy(dtype=float),
                    src[col].to_numpy(dtype=float),
                    equal_nan=True,
                ):
                    raise AssertionError(f"summary mismatch {key}.{col}")
            else:
                if not (got[col].astype(str).to_numpy() == src[col].astype(str).to_numpy()).all():
                    raise AssertionError(f"summary mismatch {key}.{col}")


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

    out_files = [
        results_dir / "hybrid_vs_rf_paired_bootstrap.csv",
        results_dir / "python_feature_ablation.csv",
        results_dir / "parseable_python_results.csv",
        results_dir / "parseability_by_split_and_label.csv",
        results_dir / "robustness_fixed_cohort.csv",
        results_dir / "robustness_transform_success.csv",
        results_dir / "robustness_delta_confidence_intervals.csv",
        results_dir / "final_summary.json",
        results_dir / "stage5_meta.json",
    ]
    for path in out_files:
        refuse_overwrite(path, args.force)

    # Integrity: Stage-4 selected config
    selected = verify_selected_config(results_dir / "selected_configuration.json")

    # Fingerprints from Stage-4 meta / Stage artifacts
    stage4_meta = load_json(results_dir / "stage4_meta.json")
    stage1_fps = {
        row["file"].replace(".csv", ""): row["sha256"]
        for row in load_json(results_dir / "manifest_fingerprints.json")
        if row["file"].endswith(".csv") and row["file"] != "all_ids.csv"
    }
    for name in ("train", "val", "test"):
        current = sha256_file(manifests_dir / f"{name}.csv")
        expected = stage4_meta["manifest_fingerprints"][name]
        if current != expected:
            raise AssertionError(f"manifest fingerprint changed for {name}")
        if stage1_fps.get(name) and current != stage1_fps[name]:
            raise AssertionError(f"Stage-1 fingerprint mismatch for {name}")
    for s in ("train", "validation", "test"):
        feat = sha256_file(features_dir / f"X_{s}.npy")
        if feat != stage4_meta["feature_fingerprints"][f"X_{s}.npy"]:
            raise AssertionError(f"feature fingerprint changed for {s}")
        emb = sha256_file(embeddings_dir / f"X_{s}_codebert.npy")
        if emb != stage4_meta["embedding_fingerprints"][f"X_{s}_codebert.npy"]:
            raise AssertionError(f"embedding fingerprint changed for {s}")

    features_meta = load_json(features_dir / "features_meta.json")
    codebert_meta = load_json(embeddings_dir / "codebert_meta.json")
    feat_hashes = features_meta.get("npy_sha256", {})
    emb_hashes = codebert_meta.get("embedding_sha256", {})
    for s in ("train", "validation", "test"):
        key = f"X_{s}.npy"
        if key in feat_hashes and sha256_file(features_dir / key) != feat_hashes[key]:
            raise AssertionError(f"Stage-2 feature hash mismatch: {key}")
        ekey = f"X_{s}_codebert.npy"
        if ekey in emb_hashes and sha256_file(embeddings_dir / ekey) != emb_hashes[ekey]:
            raise AssertionError(f"Stage-3 embedding hash mismatch: {ekey}")

    # Probability vectors unchanged vs Stage-4 recorded hashes
    for label, fname in (
        ("test_rf", "test_rf_prob.npy"),
        ("test_cb", "test_codebert_prob.npy"),
        ("test_hy", "test_hybrid_prob.npy"),
    ):
        got = sha256_file(results_dir / fname)
        expected = stage4_meta["output_file_hashes"][label]
        if got != expected:
            raise AssertionError(f"Stage-4 probability hash changed: {fname}")

    # Selected thresholds/alpha unchanged
    for k, v in selected.items():
        if abs(float(stage4_meta["selected_configuration"][k]) - v) > 1e-12:
            raise AssertionError(f"selected {k} drifted from stage4_meta")
    if not stage4_meta["integrity"]["test_unused_during_selection"]:
        raise AssertionError("Stage-4 reported test leakage during selection")
    # Reload Stage-4 models/probs unchanged
    rf = joblib.load(models_dir / "rf_features.joblib")
    lr = joblib.load(models_dir / "lr_codebert.joblib")
    test_rf = np.load(results_dir / "test_rf_prob.npy")
    test_cb = np.load(results_dir / "test_codebert_prob.npy")
    test_hy = np.load(results_dir / "test_hybrid_prob.npy")
    Xte_f = np.load(features_dir / "X_test.npy")
    Xte_e = np.load(embeddings_dir / "X_test_codebert.npy")
    yte = np.load(features_dir / "y_test.npy").astype(int)
    if not np.allclose(rf.predict_proba(Xte_f)[:, 1], test_rf, rtol=1e-7, atol=1e-7):
        raise AssertionError("Stage-4 RF probabilities changed")
    if not np.allclose(lr.predict_proba(Xte_e)[:, 1], test_cb, rtol=1e-7, atol=1e-7):
        raise AssertionError("Stage-4 CodeBERT probabilities changed")
    expected_hy = (
        selected["hybrid_alpha_feature"] * test_rf
        + selected["hybrid_alpha_embedding"] * test_cb
    )
    if not np.allclose(expected_hy, test_hy, rtol=1e-7, atol=1e-7):
        raise AssertionError("Stage-4 hybrid probabilities inconsistent with alpha")

    manifests = {
        "train": load_manifest(cfg, "train"),
        "val": load_manifest(cfg, "val"),
        "test": load_manifest(cfg, "test"),
    }

    # A. Paired bootstrap Hybrid vs RF
    paired = paired_bootstrap_hybrid_vs_rf(
        yte,
        test_rf,
        test_hy,
        selected["rf_threshold"],
        selected["hybrid_threshold"],
    )
    paired.to_csv(results_dir / "hybrid_vs_rf_paired_bootstrap.csv", index=False)

    # B. Python feature ablation
    ablation = run_feature_ablation(cfg, features_dir, manifests, results_dir)
    ablation.to_csv(results_dir / "python_feature_ablation.csv", index=False)

    # C. Parseable-Python control
    parse_counts, parse_result, _ = run_parseable_control(
        cfg, features_dir, manifests, results_dir
    )

    # D. Robustness
    rob = run_robustness(
        cfg,
        manifests,
        features_dir,
        embeddings_dir,
        models_dir,
        results_dir,
        selected,
        test_rf,
        test_cb,
        test_hy,
    )

    # E. Final summary from artifacts
    summary = build_final_summary(results_dir, manifests_dir)
    save_json(results_dir / "final_summary.json", summary)
    assert_summary_matches(summary, results_dir)

    # Finite checks on key metrics
    for df in (paired, ablation, parse_result, rob["results"], rob["cis"]):
        num = df.select_dtypes(include=[np.number])
        if not np.isfinite(num.to_numpy()).all():
            raise AssertionError("non-finite metric values found")

    save_json(
        results_dir / "stage5_meta.json",
        {
            "stage": 5,
            "selected_configuration_verified": selected,
            "no_stage4_retrain": True,
            "robustness_cohort": {
                "n": rob["cohort_n"],
                "common_n": rob["common_n"],
                "human": rob["human_n"],
                "ai": rob["ai_n"],
            },
        },
    )

    print("FINAL INTEGRITY: PASS")
    print("\nPAIRED HYBRID VS RF BOOTSTRAP")
    print(paired.round(6).to_string(index=False))
    print("\nPYTHON FEATURE ABLATION")
    print(ablation.round(6).to_string(index=False))
    print("\nPARSEABILITY COUNTS")
    print(parse_counts.to_string(index=False))
    print("\nPARSEABLE-PYTHON CONTROL")
    print(parse_result.round(6).to_string(index=False))
    print("\nROBUSTNESS COHORT")
    print(f"total N: {rob['cohort_n']}")
    print(f"human N: {rob['human_n']}")
    print(f"AI N: {rob['ai_n']}")
    print(f"final common N: {rob['common_n']}")
    print("\nROBUSTNESS RESULTS")
    cols = [
        "Transformation",
        "N",
        "Model",
        "Original_ROC_AUC",
        "Transformed_ROC_AUC",
        "Delta_ROC_AUC",
        "Original_PR_AUC",
        "Transformed_PR_AUC",
        "Delta_PR_AUC",
        "Original_F1",
        "Transformed_F1",
        "Delta_F1",
        "Prediction_Flip_Rate",
    ]
    print(rob["results"][cols].round(6).to_string(index=False))
    print("\nROBUSTNESS DELTA 95% CIs")
    print(rob["cis"].round(6).to_string(index=False))
    print(f"\nfinal_summary.json path: {results_dir / 'final_summary.json'}")
    print(f"README path: {resolve_path(cfg, 'experiment_dir') / 'README.md'}")


if __name__ == "__main__":
    main()
