#!/usr/bin/env python3
"""01 — Build the frozen canonical 70k split (Stage 1 only).

Pipeline:
  1. Load the FULL original Task A training CSV.
  2. Hash codes (exact + normalized).
  3. Report conflicting-label normalized duplicates (do not silently resolve).
  4. Drop exact-code duplicates, then normalized-code duplicates.
  5. Stratified sample 70k by label+language (preserve source proportions).
  6. Split 50k / 10k / 10k with the same strata (seed 42).
  7. Freeze manifests; refuse overwrite unless --force.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SYS_DIR = Path(__file__).resolve().parent
if str(SYS_DIR) not in sys.path:
    sys.path.insert(0, str(SYS_DIR))

from common import (
    assert_disjoint,
    ensure_dirs,
    exact_code_hash,
    load_config,
    load_source_benchmark,
    make_stratum,
    normalized_code_hash,
    refuse_overwrite,
    resolve_path,
    save_json,
)


MANIFEST_COLUMNS = [
    "id",
    "source_index",
    "label",
    "language",
    "generator",
    "code",
    "exact_hash",
    "normalized_hash",
    "split",
]


def report_conflicting_labels(
    df: pd.DataFrame,
    label_col: str,
    out_path: Path,
) -> pd.DataFrame:
    """Find normalized hashes that appear with more than one label. Report; do not resolve."""
    grouped = (
        df.groupby("normalized_hash")
        .agg(
            n_rows=("normalized_hash", "size"),
            n_labels=(label_col, "nunique"),
            labels=(label_col, lambda s: sorted(set(int(x) for x in s))),
            source_indices=("source_index", lambda s: sorted(int(x) for x in s)[:20]),
        )
        .reset_index()
    )
    conflicts = grouped[grouped["n_labels"] > 1].copy()
    conflicts.to_csv(out_path, index=False)
    print(
        f"[warn] conflicting-label normalized hashes: {len(conflicts)} "
        f"(report -> {out_path})"
    )
    if len(conflicts):
        print(conflicts.head(10).to_string(index=False))
    return conflicts


def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop exact then normalized duplicates, keeping the earliest source_index."""
    before = len(df)
    df = df.sort_values("source_index", kind="mergesort")

    after_exact = df.drop_duplicates(subset=["exact_hash"], keep="first")
    n_exact_removed = before - len(after_exact)

    after_norm = after_exact.drop_duplicates(subset=["normalized_hash"], keep="first")
    n_norm_removed = len(after_exact) - len(after_norm)

    stats = {
        "rows_before_dedup": int(before),
        "exact_duplicates_removed": int(n_exact_removed),
        "normalized_duplicates_removed": int(n_norm_removed),
        "rows_after_dedup": int(len(after_norm)),
    }
    print(
        f"[info] dedup: {before} -> {len(after_exact)} after exact "
        f"(-{n_exact_removed}); -> {len(after_norm)} after normalized "
        f"(-{n_norm_removed})"
    )
    return after_norm.reset_index(drop=True), stats


def stratified_split(
    df: pd.DataFrame,
    stratum: pd.Series,
    train_n: int,
    val_n: int,
    test_n: int,
    seed: int,
) -> dict[str, pd.DataFrame]:
    total = train_n + val_n + test_n
    if len(df) < total:
        raise ValueError(f"Only {len(df)} deduplicated rows available; need {total}")

    # Draw a 70k pool preserving stratum proportions, then split 50/10/10.
    pool_idx, _ = train_test_split(
        np.arange(len(df)),
        train_size=total,
        random_state=seed,
        stratify=stratum,
    )
    pool = df.iloc[pool_idx].reset_index(drop=True)
    pool_stratum = stratum.iloc[pool_idx].reset_index(drop=True)

    train_idx, temp_idx = train_test_split(
        np.arange(len(pool)),
        train_size=train_n,
        random_state=seed,
        stratify=pool_stratum,
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=test_n,
        random_state=seed,
        stratify=pool_stratum.iloc[temp_idx],
    )

    splits = {
        "train": pool.iloc[train_idx].copy(),
        "val": pool.iloc[val_idx].copy(),
        "test": pool.iloc[test_idx].copy(),
    }
    if len(splits["train"]) != train_n or len(splits["val"]) != val_n or len(splits["test"]) != test_n:
        raise AssertionError(
            "Split sizes mismatch: "
            f"train={len(splits['train'])}, val={len(splits['val'])}, "
            f"test={len(splits['test'])}"
        )
    return splits


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--force",
        action="store_true",
        help="Overwrite frozen manifests (invalidates the canonical experiment)",
    )
    args = ap.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)
    cols = cfg["columns"]
    manifests_dir = resolve_path(cfg, "manifests_dir")
    results_dir = resolve_path(cfg, "results_dir")
    results_dir.mkdir(parents=True, exist_ok=True)

    out_files = {
        "train": manifests_dir / "train.csv",
        "val": manifests_dir / "val.csv",
        "test": manifests_dir / "test.csv",
        "meta": manifests_dir / "split_meta.json",
        "ids": manifests_dir / "all_ids.csv",
        "conflicts": results_dir / "conflicting_label_duplicates.csv",
        "dedup_stats": results_dir / "dedup_stats.json",
    }
    for key, path in out_files.items():
        if key in {"conflicts", "dedup_stats"}:
            continue
        refuse_overwrite(path, args.force)

    df = load_source_benchmark(cfg)
    source_rows = len(df)
    print(f"[info] source rows: {source_rows}")

    print("[info] hashing codes (exact + normalized)...")
    df["exact_hash"] = df[cols["code"]].map(exact_code_hash)
    df["normalized_hash"] = df[cols["code"]].map(normalized_code_hash)

    conflicts = report_conflicting_labels(
        df, cols["label"], out_files["conflicts"]
    )
    conflict_hashes = set(conflicts["normalized_hash"].tolist())
    if conflict_hashes:
        before = len(df)
        df = df[~df["normalized_hash"].isin(conflict_hashes)].copy()
        print(
            f"[info] excluded {before - len(df)} rows belonging to "
            f"{len(conflict_hashes)} conflicting-label hashes "
            "(reported, not silently resolved)"
        )

    df, dedup_stats = deduplicate(df)
    dedup_stats.update(
        {
            "source_rows": int(source_rows),
            "conflicting_normalized_hashes": int(len(conflict_hashes)),
            "rows_after_conflict_filter": int(dedup_stats["rows_before_dedup"]),
            "rows_excluded_due_to_label_conflicts": int(
                source_rows - dedup_stats["rows_before_dedup"]
            ),
        }
    )
    save_json(out_files["dedup_stats"], dedup_stats)

    seed = int(cfg["random_seed"])
    train_n = int(cfg["split"]["train_size"])
    val_n = int(cfg["split"]["val_size"])
    test_n = int(cfg["split"]["test_size"])
    total = int(cfg["split"]["total_size"])
    assert train_n + val_n + test_n == total

    stratum = make_stratum(df, cols["label"], cols["language"])
    print("[info] stratum counts (label|||language) after dedup:")
    print(stratum.value_counts().sort_index().to_string())

    splits = stratified_split(df, stratum, train_n, val_n, test_n, seed)

    # Hard assertions requested for Stage 1.
    assert len(splits["train"]) == 50000
    assert len(splits["val"]) == 10000
    assert len(splits["test"]) == 10000

    id_sets = {name: set(frame[cols["id"]].astype(str)) for name, frame in splits.items()}
    norm_sets = {
        name: set(frame["normalized_hash"].astype(str)) for name, frame in splits.items()
    }
    assert_disjoint(id_sets, "IDs")
    assert_disjoint(norm_sets, "normalized hashes")

    # Also assert exact hashes disjoint (stronger than required, useful sanity).
    exact_sets = {
        name: set(frame["exact_hash"].astype(str)) for name, frame in splits.items()
    }
    assert_disjoint(exact_sets, "exact hashes")

    all_id_rows = []
    for name, frame in splits.items():
        out = pd.DataFrame(
            {
                "id": frame[cols["id"]].astype(str),
                "source_index": frame["source_index"].astype(int),
                "label": frame[cols["label"]].astype(int),
                "language": frame[cols["language"]].astype(str),
                "generator": frame[cols["generator"]].astype(str),
                "code": frame[cols["code"]].astype(str),
                "exact_hash": frame["exact_hash"].astype(str),
                "normalized_hash": frame["normalized_hash"].astype(str),
                "split": name,
            }
        )
        assert list(out.columns) == MANIFEST_COLUMNS
        out.to_csv(out_files[name], index=False)
        all_id_rows.append(
            out[["id", "source_index", "split", "label", "exact_hash", "normalized_hash"]]
        )
        print(
            f"[ok] {name}: n={len(out)} "
            f"human={(out['label']==0).sum()} ai={(out['label']==1).sum()} "
            f"langs={out['language'].nunique()} gens={out['generator'].nunique()}"
        )

    all_ids = pd.concat(all_id_rows, ignore_index=True)
    all_ids.to_csv(out_files["ids"], index=False)

    meta = {
        "experiment": cfg["experiment_name"],
        "stage": 1,
        "random_seed": seed,
        "source_csv": cfg["paths"]["source_csv"],
        "source_rows": source_rows,
        "stratify_by": ["label", "language"],
        "generator_used_as_stratum": False,
        "total_size": total,
        "train_size": train_n,
        "val_size": val_n,
        "test_size": test_n,
        "dedup_stats": dedup_stats,
        "conflicting_label_hashes": int(len(conflict_hashes)),
        "label_counts": {
            name: frame[cols["label"]].value_counts().sort_index().to_dict()
            for name, frame in splits.items()
        },
        "language_counts": {
            name: frame[cols["language"]].value_counts().to_dict()
            for name, frame in splits.items()
        },
        "frozen": True,
        "note": (
            "Canonical Stage-1 split. Do not regenerate without --force. "
            "Downstream stages must reuse these manifests."
        ),
    }
    save_json(out_files["meta"], meta)
    print(f"[done] frozen manifests -> {manifests_dir}")


if __name__ == "__main__":
    main()
