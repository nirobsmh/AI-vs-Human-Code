#!/usr/bin/env python3
"""Stage 2 — Handcrafted feature extraction for frozen canonical70k manifests.

Uses extract_handcraft_features.FEATURE_FIELDS and features_for_code() exactly.
Does not resample, reorder, or modify manifests. Does not train models or embed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

SYS_DIR = Path(__file__).resolve().parent
if str(SYS_DIR) not in sys.path:
    sys.path.insert(0, str(SYS_DIR))

from common import (  # noqa: E402
    ensure_dirs,
    load_config,
    load_manifest,
    refuse_overwrite,
    resolve_path,
    save_json,
    sha256_file,
)

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "feature"))
from extract_handcraft_features import FEATURE_FIELDS, features_for_code  # noqa: E402

EXPECTED_FEATURE_NAMES = [
    "n_lines",
    "n_chars",
    "avg_line_len",
    "blank_ratio",
    "comment_ratio",
    "has_docstring",
    "avg_leading_spaces",
    "avg_leading_tabs",
    "n_identifiers",
    "avg_identifier_len",
    "generic_identifier_ratio",
    "snake_case_ratio",
    "camel_case_ratio",
    "token_entropy",
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

SPLIT_ALIAS = {
    "train": "train",
    "validation": "val",
    "test": "test",
}
OUTPUT_NAME = {
    "train": "train",
    "val": "validation",
    "test": "test",
}
EXPECTED_SHAPES = {
    "train": (50000, 25),
    "validation": (10000, 25),
    "test": (10000, 25),
}


def extract_matrix(codes: list[str]) -> np.ndarray:
    """Build (N, 25) float32 matrix using features_for_code + FEATURE_FIELDS order."""
    rows = []
    for code in tqdm(codes, desc="features", leave=False):
        feats = features_for_code(code)
        rows.append([feats[name] for name in FEATURE_FIELDS])
    X = np.asarray(rows, dtype=np.float32)
    return X


def audit_features(X: np.ndarray, split_name: str) -> pd.DataFrame:
    rows = []
    for j, name in enumerate(FEATURE_FIELDS):
        col = X[:, j]
        rows.append(
            {
                "split": split_name,
                "feature": name,
                "min": float(np.min(col)),
                "max": float(np.max(col)),
                "mean": float(np.mean(col)),
                "std": float(np.std(col)),
                "median": float(np.median(col)),
                "zero_fraction": float(np.mean(col == 0)),
            }
        )
    return pd.DataFrame(rows)


def recompute_check(df: pd.DataFrame, X: np.ndarray, rng: np.random.Generator, n: int = 10):
    idxs = rng.choice(len(df), size=min(n, len(df)), replace=False)
    for i in idxs:
        code = str(df.iloc[int(i)]["code"])
        feats = features_for_code(code)
        vec = np.asarray([feats[name] for name in FEATURE_FIELDS], dtype=np.float32)
        if not np.allclose(vec, X[int(i)], rtol=0.0, atol=0.0, equal_nan=False):
            # Allow tiny float32 casting diffs only via allclose with strict tiny atol
            if not np.allclose(vec, X[int(i)], rtol=1e-6, atol=1e-6, equal_nan=False):
                raise AssertionError(
                    f"Recomputation mismatch at row {i}: "
                    f"id={df.iloc[int(i)]['id']}"
                )
    return sorted(int(i) for i in idxs)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--seed", type=int, default=42, help="Seed for 10-row recomputation sample")
    args = ap.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)
    features_dir = resolve_path(cfg, "features_dir")
    results_dir = resolve_path(cfg, "results_dir")
    manifests_dir = resolve_path(cfg, "manifests_dir")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Strict: FEATURE_FIELDS must match the paper/experiment expected order exactly.
    if list(FEATURE_FIELDS) != EXPECTED_FEATURE_NAMES:
        raise AssertionError(
            "FEATURE_FIELDS does not exactly match expected Stage-2 feature order.\n"
            f"FEATURE_FIELDS={list(FEATURE_FIELDS)}\n"
            f"EXPECTED={EXPECTED_FEATURE_NAMES}"
        )
    if len(FEATURE_FIELDS) != 25:
        raise AssertionError(f"Expected 25 features, got {len(FEATURE_FIELDS)}")

    out_paths = {
        "feature_names": features_dir / "feature_names.json",
        "ids_train": features_dir / "ids_train.npy",
        "ids_validation": features_dir / "ids_validation.npy",
        "ids_test": features_dir / "ids_test.npy",
        "X_train": features_dir / "X_train.npy",
        "X_validation": features_dir / "X_validation.npy",
        "X_test": features_dir / "X_test.npy",
        "y_train": features_dir / "y_train.npy",
        "y_validation": features_dir / "y_validation.npy",
        "y_test": features_dir / "y_test.npy",
        "audit": results_dir / "feature_audit.csv",
        "meta": features_dir / "features_meta.json",
    }
    for path in out_paths.values():
        refuse_overwrite(path, args.force)

    rng = np.random.default_rng(args.seed)
    audit_frames = []
    label_distributions = {}
    alignment_ok = True
    recompute_ok = True
    recompute_rows = {}
    checksums = {}
    shapes = {}
    dtypes = {}
    finite_ok = {}

    # Manifest fingerprints for metadata alignment.
    manifest_fingerprints = {}
    for split_file in ("train.csv", "val.csv", "test.csv", "split_meta.json"):
        path = manifests_dir / split_file
        if path.exists():
            manifest_fingerprints[split_file] = sha256_file(path)

    for manifest_key in ("train", "val", "test"):
        out_key = OUTPUT_NAME[manifest_key]
        df = load_manifest(cfg, manifest_key)
        expected_shape = EXPECTED_SHAPES[out_key]
        if len(df) != expected_shape[0]:
            raise AssertionError(
                f"{manifest_key} manifest has {len(df)} rows; expected {expected_shape[0]}"
            )

        codes = df["code"].astype(str).tolist()
        y_manifest = df["label"].astype(int).to_numpy()
        ids = df["id"].astype(str).to_numpy()

        X = extract_matrix(codes)

        # Strict validations (no silent nan/inf rewriting).
        if X.shape != expected_shape:
            raise AssertionError(f"{out_key} shape {X.shape} != {expected_shape}")
        if X.dtype != np.float32:
            raise AssertionError(f"{out_key} dtype {X.dtype} != float32")
        if X.shape[0] != len(df):
            alignment_ok = False
            raise AssertionError("Feature row count != manifest row count")
        if X.shape[1] != 25:
            raise AssertionError(f"Feature dim {X.shape[1]} != 25")
        if not np.array_equal(y_manifest, y_manifest):  # noqa: placeholder
            pass
        y = y_manifest.astype(np.int64)
        if not np.array_equal(y, df["label"].astype(int).to_numpy()):
            alignment_ok = False
            raise AssertionError("Saved labels do not match manifest labels in order")
        if not np.isfinite(X).all():
            finite_ok[out_key] = False
            bad = int((~np.isfinite(X)).sum())
            raise AssertionError(f"{out_key} contains {bad} non-finite values")
        finite_ok[out_key] = True

        # 10-row recomputation check
        try:
            recompute_rows[out_key] = recompute_check(df, X, rng, n=10)
        except AssertionError:
            recompute_ok = False
            raise

        x_path = out_paths[f"X_{out_key}"]
        y_path = out_paths[f"y_{out_key}"]
        id_path = out_paths[f"ids_{out_key}"]
        np.save(x_path, X)
        np.save(y_path, y)
        np.save(id_path, ids)

        # Reload and verify label alignment on disk
        y_loaded = np.load(y_path)
        if not np.array_equal(y_loaded, df["label"].astype(int).to_numpy()):
            alignment_ok = False
            raise AssertionError(f"Disk y_{out_key}.npy does not match manifest labels")
        ids_loaded = np.load(id_path, allow_pickle=True)
        if not np.array_equal(ids_loaded.astype(str), ids.astype(str)):
            alignment_ok = False
            raise AssertionError(f"Disk ids_{out_key}.npy does not match manifest IDs")

        shapes[f"X_{out_key}"] = list(X.shape)
        shapes[f"y_{out_key}"] = list(y.shape)
        dtypes[f"X_{out_key}"] = str(X.dtype)
        dtypes[f"y_{out_key}"] = str(y.dtype)
        checksums[f"X_{out_key}.npy"] = sha256_file(x_path)
        checksums[f"y_{out_key}.npy"] = sha256_file(y_path)
        checksums[f"ids_{out_key}.npy"] = sha256_file(id_path)
        label_distributions[out_key] = {
            "0": int((y == 0).sum()),
            "1": int((y == 1).sum()),
        }
        audit_frames.append(audit_features(X, out_key))

    # Save feature names
    with out_paths["feature_names"].open("w") as f:
        json.dump(list(FEATURE_FIELDS), f, indent=2)
    checksums["feature_names.json"] = sha256_file(out_paths["feature_names"])

    audit = pd.concat(audit_frames, ignore_index=True)
    audit.to_csv(out_paths["audit"], index=False)

    meta = {
        "stage": 2,
        "feature_names": list(FEATURE_FIELDS),
        "shapes": shapes,
        "dtypes": dtypes,
        "label_distributions": label_distributions,
        "finite_value_checks": finite_ok,
        "alignment_check": alignment_ok,
        "recompute_check": {
            "passed": recompute_ok,
            "rows_checked": recompute_rows,
            "seed": args.seed,
        },
        "manifest_fingerprints": manifest_fingerprints,
        "npy_sha256": checksums,
        "extractor": {
            "module": "feature/extract_handcraft_features.py",
            "function": "features_for_code",
            "feature_fields_source": "FEATURE_FIELDS",
            "safe_parse_altered": False,
        },
        "output_paths": {k: str(v) for k, v in out_paths.items()},
    }
    save_json(out_paths["meta"], meta)

    # Final console output (only requested fields)
    print("matrix shapes:")
    for k, v in shapes.items():
        print(f"  {k}: {tuple(v)}")
    print("label distributions:")
    for k, v in label_distributions.items():
        print(f"  {k}: {v}")
    print("finite-value checks:")
    for k, v in finite_ok.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    print(f"alignment-check status: {'PASS' if alignment_ok else 'FAIL'}")
    print(f"10-row recomputation-check status: {'PASS' if recompute_ok else 'FAIL'}")
    print("output paths:")
    for k, v in out_paths.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
