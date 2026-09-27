#!/usr/bin/env python3
"""10 — Export a single paper-facing results bundle from frozen artifacts."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from pathlib import Path
import sys

SYS_DIR = Path(__file__).resolve().parent
if str(SYS_DIR) not in sys.path:
    sys.path.insert(0, str(SYS_DIR))

from common import ensure_dirs, load_config, load_json, require_file, resolve_path, save_json


def maybe_copy(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def main():
    cfg = load_config()
    ensure_dirs(cfg)
    results = resolve_path(cfg, "results_dir")
    models = resolve_path(cfg, "models_dir")
    manifests = resolve_path(cfg, "manifests_dir")
    export_dir = results / "paper_export"
    export_dir.mkdir(parents=True, exist_ok=True)

    catalog = []
    wanted = [
        (manifests / "split_meta.json", "split_meta.json"),
        (results / "dataset_split_summary.csv", "dataset_split_summary.csv"),
        (results / "python_parse_validity.csv", "python_parse_validity.csv"),
        (results / "split_audit.json", "split_audit.json"),
        (results / "baseline_test_metrics.csv", "baseline_test_metrics.csv"),
        (results / "hybrid_test_results.csv", "hybrid_test_results.csv"),
        (results / "hybrid_validation_search.csv", "hybrid_validation_search.csv"),
        (models / "hybrid_config.json", "hybrid_config.json"),
        (results / "python_feature_ablation.csv", "python_feature_ablation.csv"),
        (results / "parseable_python_rf.csv", "parseable_python_rf.csv"),
        (
            results / "robustness_fixed_parseable_compact.csv",
            "robustness_fixed_parseable_compact.csv",
        ),
        (
            results / "robustness_fixed_parseable_meta.json",
            "robustness_fixed_parseable_meta.json",
        ),
    ]

    for src, name in wanted:
        ok = maybe_copy(src, export_dir / name)
        catalog.append({"file": name, "present": ok, "source": str(src)})
        print(("OK  " if ok else "MISS"), name)

    # Build a compact markdown summary if core tables exist.
    lines = ["# Canonical 70k — paper export", ""]
    hybrid = models / "hybrid_config.json"
    if hybrid.exists():
        h = load_json(hybrid)
        lines += [
            "## Hybrid configuration",
            f"- Feature weight: `{h.get('feature_weight')}`",
            f"- Embedding weight: `{h.get('embedding_weight')}`",
            f"- Hybrid threshold: `{h.get('hybrid_threshold')}`",
            "",
        ]
    for label, path in [
        ("Baseline test", results / "baseline_test_metrics.csv"),
        ("Hybrid test", results / "hybrid_test_results.csv"),
        ("Feature ablation", results / "python_feature_ablation.csv"),
        ("Parseable Python RF", results / "parseable_python_rf.csv"),
        ("Robustness (fixed cohort)", results / "robustness_fixed_parseable_compact.csv"),
    ]:
        if path.exists():
            df = pd.read_csv(path)
            lines += [f"## {label}", "```", df.round(4).to_string(index=False), "```", ""]

    (export_dir / "SUMMARY.md").write_text("\n".join(lines))
    save_json(export_dir / "catalog.json", catalog)
    print(f"[done] export -> {export_dir}")


if __name__ == "__main__":
    main()
