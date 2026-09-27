#!/usr/bin/env python3
"""07 — Feature ablation on Python-only rows of the frozen 70k split."""

from __future__ import annotations

import argparse
import ast

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from pathlib import Path
import sys

SYS_DIR = Path(__file__).resolve().parent
if str(SYS_DIR) not in sys.path:
    sys.path.insert(0, str(SYS_DIR))

from common import (
    best_threshold,
    ensure_dirs,
    evaluate_probs,
    is_python_series,
    load_all_manifests,
    load_config,
    load_xy,
    refuse_overwrite,
    require_file,
    resolve_path,
)

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


def parse_flag(code: str) -> int:
    try:
        ast.parse(str(code))
        return 1
    except Exception:
        return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)
    cols = cfg["columns"]
    out = resolve_path(cfg, "results_dir") / "python_feature_ablation.csv"
    refuse_overwrite(out, args.force)

    model_path = resolve_path(cfg, "models_dir") / "feature_model.joblib"
    require_file(model_path, "Run 05_train_baselines.py")
    feature_model = joblib.load(model_path)

    # Feature names from saved npz
    feat_npz = np.load(resolve_path(cfg, "features_dir") / "train_features.npz")
    feature_names = [str(x) for x in feat_npz["feature_names"].tolist()]
    surface_idx = [feature_names.index(x) for x in SURFACE_FIELDS]
    ast_idx = [feature_names.index(x) for x in AST_FIELDS]
    length_idx = [feature_names.index(x) for x in LENGTH_FIELDS]

    manifests = load_all_manifests(cfg)
    Xtr, _, ytr = load_xy(cfg, "train")
    Xva, _, yva = load_xy(cfg, "val")
    Xte, _, yte = load_xy(cfg, "test")

    train_py = is_python_series(manifests["train"][cols["language"]], cfg).to_numpy()
    val_py = is_python_series(manifests["val"][cols["language"]], cfg).to_numpy()
    test_py = is_python_series(manifests["test"][cols["language"]], cfg).to_numpy()
    print("Python train/val/test:", train_py.sum(), val_py.sum(), test_py.sum())

    Xtr_p, ytr_p = Xtr[train_py], ytr[train_py]
    Xva_p, yva_p = Xva[val_py], yva[val_py]
    Xte_p, yte_p = Xte[test_py], yte[test_py]

    rows = []

    def run_rf(name, idx=None):
        model = clone(feature_model)
        tr = Xtr_p if idx is None else Xtr_p[:, idx]
        va = Xva_p if idx is None else Xva_p[:, idx]
        te = Xte_p if idx is None else Xte_p[:, idx]
        model.fit(tr, ytr_p)
        thr = best_threshold(yva_p, model.predict_proba(va)[:, 1])
        rows.append(evaluate_probs(name, yte_p, model.predict_proba(te)[:, 1], thr))

    run_rf("RF - All 25")
    run_rf("RF - Surface 8", surface_idx)
    run_rf("RF - AST 17", ast_idx)
    run_rf("RF - Length only", length_idx)

    # Parse failure only baseline
    val_parse = (
        manifests["val"].loc[val_py, cols["code"]].apply(parse_flag).to_numpy()
    )
    test_parse = (
        manifests["test"].loc[test_py, cols["code"]].apply(parse_flag).to_numpy()
    )
    thr_parse = best_threshold(yva_p, (1 - val_parse).astype(float))
    rows.append(
        evaluate_probs(
            "Parse failure only",
            yte_p,
            (1 - test_parse).astype(float),
            thr_parse,
        )
    )

    result = pd.DataFrame(rows)
    print(result.round(4).to_string(index=False))
    result.to_csv(out, index=False)
    print(f"[done] {out}")


if __name__ == "__main__":
    main()
