#!/usr/bin/env python3
"""02 — Complete Stage-1 audit of the frozen canonical70k split."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SYS_DIR = Path(__file__).resolve().parent
if str(SYS_DIR) not in sys.path:
    sys.path.insert(0, str(SYS_DIR))

from common import (
    assert_disjoint,
    ensure_dirs,
    is_python_series,
    load_all_manifests,
    load_config,
    load_json,
    resolve_path,
    save_json,
    sha256_file,
)


def parse_ok(code: str) -> bool:
    try:
        ast.parse(str(code))
        return True
    except Exception:
        return False


def overlap_count(a: set, b: set) -> int:
    return len(a & b)


def main():
    cfg = load_config()
    ensure_dirs(cfg)
    cols = cfg["columns"]
    results = resolve_path(cfg, "results_dir")
    manifests_dir = resolve_path(cfg, "manifests_dir")
    results.mkdir(parents=True, exist_ok=True)

    splits = load_all_manifests(cfg)

    # ------------------------------------------------------------------
    # Assertions: exact sizes + disjoint IDs / normalized hashes
    # ------------------------------------------------------------------
    assert len(splits["train"]) == 50000, len(splits["train"])
    assert len(splits["val"]) == 10000, len(splits["val"])
    assert len(splits["test"]) == 10000, len(splits["test"])

    id_sets = {n: set(df["id"].astype(str)) for n, df in splits.items()}
    norm_sets = {n: set(df["normalized_hash"].astype(str)) for n, df in splits.items()}
    exact_sets = {n: set(df["exact_hash"].astype(str)) for n, df in splits.items()}
    assert_disjoint(id_sets, "IDs")
    assert_disjoint(norm_sets, "normalized hashes")
    assert_disjoint(exact_sets, "exact hashes")
    print("[assert] sizes 50k/10k/10k OK; IDs and hashes disjoint across splits")

    # ------------------------------------------------------------------
    # Class counts
    # ------------------------------------------------------------------
    class_rows = []
    for name, df in splits.items():
        class_rows.append(
            {
                "Split": name,
                "Total": len(df),
                "Human": int((df["label"] == 0).sum()),
                "AI": int((df["label"] == 1).sum()),
                "Unique_IDs": int(df["id"].nunique()),
                "Unique_Exact": int(df["exact_hash"].nunique()),
                "Unique_Normalized": int(df["normalized_hash"].nunique()),
                "Languages": int(df["language"].nunique()),
                "Generators": int(df["generator"].nunique()),
            }
        )
    class_summary = pd.DataFrame(class_rows)
    print("\n======================================")
    print("CLASS COUNTS BY SPLIT")
    print("======================================")
    print(class_summary.to_string(index=False))
    class_summary.to_csv(results / "dataset_split_summary.csv", index=False)

    combined = []
    for name, df in splits.items():
        tmp = df.copy()
        tmp["_split"] = name
        combined.append(tmp)
    all_df = pd.concat(combined, ignore_index=True)

    # ------------------------------------------------------------------
    # Language counts
    # ------------------------------------------------------------------
    language_table = pd.crosstab(all_df["language"], all_df["_split"])
    language_table["Total"] = language_table.sum(axis=1)
    language_table = language_table.sort_values("Total", ascending=False)
    print("\n======================================")
    print("LANGUAGE COUNTS BY SPLIT")
    print("======================================")
    print(language_table)
    language_table.to_csv(results / "language_distribution.csv")

    language_label = pd.crosstab(all_df["language"], all_df["label"]).rename(
        columns={0: "Human", 1: "AI"}
    )
    language_label["Total"] = language_label.sum(axis=1)
    print("\n======================================")
    print("LANGUAGE × LABEL")
    print("======================================")
    print(language_label)
    language_label.to_csv(results / "language_label_distribution.csv")

    # ------------------------------------------------------------------
    # Generator counts (reported, not used as stratum)
    # ------------------------------------------------------------------
    generator_table = pd.crosstab(all_df["generator"], all_df["_split"])
    generator_table["Total"] = generator_table.sum(axis=1)
    generator_table = generator_table.sort_values("Total", ascending=False)
    print("\n======================================")
    print("GENERATOR COUNTS BY SPLIT")
    print("======================================")
    print(generator_table.to_string())
    generator_table.to_csv(results / "generator_distribution.csv")

    # ------------------------------------------------------------------
    # Overlaps
    # ------------------------------------------------------------------
    print("\n======================================")
    print("ID OVERLAP BETWEEN SPLITS")
    print("======================================")
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        print(f"{a} ↔ {b}: {overlap_count(id_sets[a], id_sets[b])} shared IDs")

    print("\n======================================")
    print("EXACT CODE OVERLAP")
    print("======================================")
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        print(
            f"{a} ↔ {b}: {overlap_count(exact_sets[a], exact_sets[b])} "
            "exact duplicate snippets"
        )

    print("\n======================================")
    print("NORMALIZED CODE OVERLAP")
    print("======================================")
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        print(
            f"{a} ↔ {b}: {overlap_count(norm_sets[a], norm_sets[b])} "
            "normalized duplicates"
        )

    # ------------------------------------------------------------------
    # Within-split duplicates
    # ------------------------------------------------------------------
    dup_rows = []
    for name, df in splits.items():
        dup_rows.append(
            {
                "Split": name,
                "Rows": len(df),
                "Exact_Duplicate_Rows": int(len(df) - df["exact_hash"].nunique()),
                "Normalized_Duplicate_Rows": int(
                    len(df) - df["normalized_hash"].nunique()
                ),
            }
        )
    dup_summary = pd.DataFrame(dup_rows)
    print("\n======================================")
    print("WITHIN-SPLIT DUPLICATES")
    print("======================================")
    print(dup_summary.to_string(index=False))
    dup_summary.to_csv(results / "duplicate_summary.csv", index=False)

    # ------------------------------------------------------------------
    # Conflicting labels inside the frozen split
    # ------------------------------------------------------------------
    label_conflicts = (
        all_df.groupby("normalized_hash")["label"].nunique()
    )
    n_conflicts = int((label_conflicts > 1).sum())
    print("\n======================================")
    print("LABEL CONFLICTS (FROZEN SPLIT)")
    print("======================================")
    print(f"Normalized code appearing with both labels: {n_conflicts}")

    # ------------------------------------------------------------------
    # Python AST parseability by label and split
    # ------------------------------------------------------------------
    parse_rows = []
    for name, df in splits.items():
        py = df[is_python_series(df["language"], cfg)].copy()
        py["_parse_ok"] = py["code"].apply(parse_ok)
        for label in (0, 1):
            group = py[py["label"] == label]
            total = len(group)
            valid = int(group["_parse_ok"].sum()) if total else 0
            parse_rows.append(
                {
                    "Split": name,
                    "Label": "Human" if label == 0 else "AI",
                    "Parseable": valid,
                    "Total": total,
                    "Parse_Rate": (valid / total) if total else np.nan,
                }
            )
    parse_summary = pd.DataFrame(parse_rows)
    print("\n======================================")
    print("PYTHON AST PARSEABILITY BY LABEL AND SPLIT")
    print("======================================")
    print(parse_summary.round(4).to_string(index=False))
    parse_summary.to_csv(results / "python_parse_validity.csv", index=False)

    # ------------------------------------------------------------------
    # Code length by label
    # ------------------------------------------------------------------
    length_df = all_df[["label", "code"]].copy()
    length_df["chars"] = length_df["code"].astype(str).str.len()
    length_df["lines"] = length_df["code"].astype(str).str.count("\n") + 1
    length_stats = (
        length_df.groupby("label")[["chars", "lines"]]
        .agg(["count", "mean", "median", "std"])
    )
    print("\n======================================")
    print("CODE LENGTH BY LABEL")
    print("======================================")
    print(length_stats)
    length_stats.to_csv(results / "code_length_by_label.csv")

    # ------------------------------------------------------------------
    # Manifest SHA-256 fingerprints
    # ------------------------------------------------------------------
    fingerprint_rows = []
    for name in ("train", "val", "test", "all_ids", "split_meta"):
        path = manifests_dir / (f"{name}.csv" if name != "split_meta" else "split_meta.json")
        if not path.exists():
            continue
        digest = sha256_file(path)
        fingerprint_rows.append(
            {
                "file": path.name,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": digest,
            }
        )
        print(f"[fingerprint] {path.name}: {digest}")
    fingerprints = pd.DataFrame(fingerprint_rows)
    fingerprints.to_csv(results / "manifest_fingerprints.csv", index=False)
    save_json(results / "manifest_fingerprints.json", fingerprint_rows)

    # ------------------------------------------------------------------
    # Summary JSON
    # ------------------------------------------------------------------
    meta_path = manifests_dir / "split_meta.json"
    meta = load_json(meta_path) if meta_path.exists() else {}
    audit = {
        "assertions_passed": True,
        "sizes": {k: len(v) for k, v in splits.items()},
        "id_overlaps": {
            f"{a}_{b}": overlap_count(id_sets[a], id_sets[b])
            for a, b in [("train", "val"), ("train", "test"), ("val", "test")]
        },
        "exact_overlaps": {
            f"{a}_{b}": overlap_count(exact_sets[a], exact_sets[b])
            for a, b in [("train", "val"), ("train", "test"), ("val", "test")]
        },
        "normalized_overlaps": {
            f"{a}_{b}": overlap_count(norm_sets[a], norm_sets[b])
            for a, b in [("train", "val"), ("train", "test"), ("val", "test")]
        },
        "label_conflicts_in_frozen_split": n_conflicts,
        "manifest_fingerprints": fingerprint_rows,
        "split_meta": meta,
    }
    save_json(results / "split_audit.json", audit)

    print("\n======================================")
    print("STAGE 1 AUDIT COMPLETE")
    print("======================================")
    print(f"Artifacts -> {results}")


if __name__ == "__main__":
    main()
