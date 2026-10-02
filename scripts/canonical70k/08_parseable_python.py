#!/usr/bin/env python3
"""08 — Retrain/evaluate RF on parseable-Python-only subset of frozen 70k."""

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
    save_json,
)


def parse_ok(code: str) -> bool:
    try:
        ast.parse(str(code))
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)
    cols = cfg["columns"]
    results = resolve_path(cfg, "results_dir")
    out_csv = results / "parseable_python_rf.csv"
    out_meta = results / "parseable_python_meta.json"
    for p in (out_csv, out_meta):
        refuse_overwrite(p, args.force)

    model_path = resolve_path(cfg, "models_dir") / "feature_model.joblib"
    require_file(model_path, "Run 05_train_baselines.py")
    feature_model = joblib.load(model_path)

    manifests = load_all_manifests(cfg)
    Xtr, _, ytr = load_xy(cfg, "train")
    Xva, _, yva = load_xy(cfg, "val")
    Xte, _, yte = load_xy(cfg, "test")

    masks = {}
    counts = {}
    for split, X, y, df in [
        ("train", Xtr, ytr, manifests["train"]),
        ("val", Xva, yva, manifests["val"]),
        ("test", Xte, yte, manifests["test"]),
    ]:
        py = is_python_series(df[cols["language"]], cfg).to_numpy()
        parseable = df[cols["code"]].apply(parse_ok).to_numpy()
        mask = py & parseable
        masks[split] = mask
        counts[split] = {
            "n": int(mask.sum()),
            "human": int(((y == 0) & mask).sum()),
            "ai": int(((y == 1) & mask).sum()),
        }
        print(f"Parseable Python {split}: {counts[split]}")

    model = clone(feature_model)
    model.fit(Xtr[masks["train"]], ytr[masks["train"]])
    thr = best_threshold(
        yva[masks["val"]], model.predict_proba(Xva[masks["val"]])[:, 1]
    )
    probs = model.predict_proba(Xte[masks["test"]])[:, 1]
    row = evaluate_probs(
        "RF - Parseable Python only", yte[masks["test"]], probs, thr
    )
    result = pd.DataFrame([row])
    print(result.round(4).to_string(index=False))
    result.to_csv(out_csv, index=False)
    save_json(out_meta, {"counts": counts, "threshold": thr})
    print(f"[done] {out_csv}")


if __name__ == "__main__":
    main()
