#!/usr/bin/env python3
"""09 — Robustness on fixed parseable-Python test cohort (no retrain).

Transforms: Black, AST normalization, comments-only removal.
All transforms evaluated on the same successful common subset.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
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

from common import (
    ensure_dirs,
    is_python_series,
    load_all_manifests,
    load_config,
    load_json,
    load_xy,
    refuse_overwrite,
    require_file,
    resolve_path,
    save_json,
)

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "feature"))
from code_to_arrays import code_to_features  # noqa: E402
from code_transforms import (  # noqa: E402
    ast_normalize_code,
    black_format_code,
    remove_comments_only,
)
from embed_code import embed_batch  # noqa: E402


def parse_ok(code: str) -> bool:
    try:
        ast.parse(str(code))
        return True
    except Exception:
        return False


def metric_dict(y_true, probs, thr):
    pred = (probs >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "ROC_AUC": roc_auc_score(y_true, probs),
        "PR_AUC": average_precision_score(y_true, probs),
        "Accuracy": accuracy_score(y_true, pred),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred, zero_division=0),
        "FP": int(fp),
        "FN": int(fn),
    }


def extract_features(codes):
    rows, good = [], []
    for code in tqdm(codes, desc="features", leave=False):
        try:
            x = code_to_features(code)
            if x.shape != (25,) or not np.all(np.isfinite(x)):
                raise ValueError("bad")
            rows.append(x)
            good.append(True)
        except Exception:
            rows.append(np.zeros(25, dtype=np.float32))
            good.append(False)
    return np.vstack(rows), np.asarray(good, dtype=bool)


def extract_embeddings(codes, tokenizer, model, device, batch_size, max_len):
    chunks = []
    for start in tqdm(range(0, len(codes), batch_size), desc="embeddings", leave=False):
        batch = codes[start : start + batch_size]
        chunks.append(embed_batch(batch, tokenizer, model, device, max_len=max_len))
    return np.vstack(chunks).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--batch-size", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)
    cols = cfg["columns"]
    models_dir = resolve_path(cfg, "models_dir")
    results_dir = resolve_path(cfg, "results_dir")

    feature_model_path = models_dir / "feature_model.joblib"
    embedding_model_path = models_dir / "embedding_model.joblib"
    hybrid_cfg_path = models_dir / "hybrid_config.json"
    require_file(feature_model_path, "Run 05_train_baselines.py")
    require_file(embedding_model_path, "Run 05_train_baselines.py")
    require_file(hybrid_cfg_path, "Run 06_hybrid_fusion.py")

    out_full = results_dir / "robustness_fixed_parseable_full.csv"
    out_compact = results_dir / "robustness_fixed_parseable_compact.csv"
    out_meta = results_dir / "robustness_fixed_parseable_meta.json"
    for p in (out_full, out_compact, out_meta):
        refuse_overwrite(p, args.force)

    feature_model = joblib.load(feature_model_path)
    embedding_model = joblib.load(embedding_model_path)
    hybrid_cfg = load_json(hybrid_cfg_path)
    best_alpha = float(hybrid_cfg["feature_weight"])
    feature_thr = float(hybrid_cfg["feature_threshold"])
    embedding_thr = float(hybrid_cfg["embedding_threshold"])
    hybrid_thr = float(hybrid_cfg["hybrid_threshold"])

    manifests = load_all_manifests(cfg)
    test_df = manifests["test"].reset_index(drop=True)
    Xte_f, Xte_e, yte = load_xy(cfg, "test")

    py = is_python_series(test_df[cols["language"]], cfg)
    parseable = test_df[cols["code"]].apply(parse_ok)
    fixed_mask = (py & parseable).to_numpy()
    fixed_test = test_df.loc[fixed_mask].copy().reset_index(drop=True)
    fixed_global = np.where(fixed_mask)[0]
    print("Fixed cohort:", len(fixed_test))
    print(fixed_test[cols["label"]].value_counts().sort_index())

    fixed_test["code_black"] = fixed_test[cols["code"]].apply(black_format_code)
    fixed_test["code_ast"] = fixed_test[cols["code"]].apply(ast_normalize_code)
    fixed_test["code_comments_removed"] = fixed_test[cols["code"]].apply(
        remove_comments_only
    )
    for col in ("code_black", "code_ast", "code_comments_removed"):
        print(col, fixed_test[col].notna().sum(), "/", len(fixed_test))

    common = (
        fixed_test["code_black"].notna()
        & fixed_test["code_ast"].notna()
        & fixed_test["code_comments_removed"].notna()
    )
    print("Common successful N:", int(common.sum()))

    test_pf = feature_model.predict_proba(Xte_f)[:, 1]
    test_pe = embedding_model.predict_proba(Xte_e)[:, 1]
    test_ph = best_alpha * test_pf + (1 - best_alpha) * test_pe

    device = "cuda" if torch.cuda.is_available() else "cpu"
    emb_cfg = cfg["embeddings"]
    tokenizer = AutoTokenizer.from_pretrained(emb_cfg["model_name"])
    model = AutoModel.from_pretrained(emb_cfg["model_name"])
    model.to(device)
    model.eval()
    batch_size = args.batch_size or int(emb_cfg["batch_size"])

    all_rows = []
    transforms = [
        ("code_black", "Black formatting"),
        ("code_ast", "AST normalization"),
        ("code_comments_removed", "Comments removed"),
    ]
    for col, name in transforms:
        codes = fixed_test.loc[common, col].astype(str).tolist()
        labels = fixed_test.loc[common, cols["label"]].astype(int).to_numpy()
        local = np.where(common.to_numpy())[0]
        global_pos = fixed_global[local]

        Xf, ok = extract_features(codes)
        if not ok.all():
            codes = [c for c, g in zip(codes, ok) if g]
            labels = labels[ok]
            global_pos = global_pos[ok]
            Xf = Xf[ok]
        Xe = extract_embeddings(
            codes, tokenizer, model, device, batch_size, int(emb_cfg["max_len"])
        )

        t_pf = feature_model.predict_proba(Xf)[:, 1]
        t_pe = embedding_model.predict_proba(Xe)[:, 1]
        t_ph = best_alpha * t_pf + (1 - best_alpha) * t_pe

        for model_name, o_prob, t_prob, thr in [
            ("Features (RF)", test_pf[global_pos], t_pf, feature_thr),
            ("CodeBERT (LR)", test_pe[global_pos], t_pe, embedding_thr),
            ("Hybrid Fusion", test_ph[global_pos], t_ph, hybrid_thr),
        ]:
            o = metric_dict(labels, o_prob, thr)
            t = metric_dict(labels, t_prob, thr)
            flip = float(
                np.mean((o_prob >= thr).astype(int) != (t_prob >= thr).astype(int))
            )
            all_rows.append(
                {
                    "Transformation": name,
                    "N": len(labels),
                    "Model": model_name,
                    "Original_ROC_AUC": o["ROC_AUC"],
                    "Transformed_ROC_AUC": t["ROC_AUC"],
                    "Delta_ROC_AUC": t["ROC_AUC"] - o["ROC_AUC"],
                    "Original_PR_AUC": o["PR_AUC"],
                    "Transformed_PR_AUC": t["PR_AUC"],
                    "Delta_PR_AUC": t["PR_AUC"] - o["PR_AUC"],
                    "Original_F1": o["F1"],
                    "Transformed_F1": t["F1"],
                    "Delta_F1": t["F1"] - o["F1"],
                    "Prediction_Flip_Rate": flip,
                }
            )

    full = pd.DataFrame(all_rows)
    compact_cols = [
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
    compact = full[compact_cols].copy()
    print("\n=== FIXED-COHORT ROBUSTNESS ===")
    print(compact.round(4).to_string(index=False))
    full.to_csv(out_full, index=False)
    compact.to_csv(out_compact, index=False)
    save_json(
        out_meta,
        {
            "fixed_n": int(len(fixed_test)),
            "common_n": int(common.sum()),
            "label_counts": fixed_test.loc[common, cols["label"]]
            .value_counts()
            .sort_index()
            .to_dict(),
            "best_alpha": best_alpha,
        },
    )
    print(f"[done] {out_compact}")


if __name__ == "__main__":
    main()
