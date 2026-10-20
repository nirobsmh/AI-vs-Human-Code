#!/usr/bin/env python3
"""Shared helpers for the canonical70k pipeline (Stage 1+)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import yaml

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parents[1]

CONFIG_PATH = REPO_ROOT / "experiments" / "canonical70k" / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> Dict[str, Any]:
    with path.open() as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(path.resolve())
    cfg["_repo_root"] = str(REPO_ROOT)
    return cfg


def resolve_path(cfg: Dict[str, Any], key: str) -> Path:
    rel = Path(cfg["paths"][key])
    if rel.is_absolute():
        return rel
    return (REPO_ROOT / rel).resolve()


def ensure_dirs(cfg: Dict[str, Any]) -> None:
    for key in (
        "experiment_dir",
        "manifests_dir",
        "features_dir",
        "embeddings_dir",
        "models_dir",
        "results_dir",
        "logs_dir",
    ):
        resolve_path(cfg, key).mkdir(parents=True, exist_ok=True)


def require_file(path: Path, hint: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required artifact: {path}\nHint: {hint}")


def refuse_overwrite(path: Path, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(
            f"Refusing to overwrite frozen artifact: {path}\n"
            "Pass --force only if you intentionally invalidate the canonical experiment."
        )


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def exact_code_hash(code: str) -> str:
    """SHA-256 of the raw code string (exact bytes as Unicode)."""
    return sha256_text(str(code))


def normalize_code(code: str) -> str:
    """Normalize line endings and trailing whitespace; strip outer whitespace."""
    text = str(code).replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def normalized_code_hash(code: str) -> str:
    return sha256_text(normalize_code(code))


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def load_source_benchmark(cfg: Dict[str, Any]) -> pd.DataFrame:
    """Load the full original Task A training CSV (not a pre-sampled subset)."""
    source = Path(cfg["_repo_root"]) / cfg["paths"]["source_csv"]
    if not source.exists():
        raise FileNotFoundError(source)

    cols = cfg["columns"]
    required = [cols["code"], cols["label"], cols["language"], cols["generator"]]
    print(f"[info] loading full benchmark: {source}")
    df = pd.read_csv(source)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Source CSV missing columns: {missing}")

    df = df.reset_index(drop=True)
    df["source_index"] = np.arange(len(df), dtype=np.int64)
    # Preserve any existing id if present; otherwise synthesize from source row order.
    if cols["id"] in df.columns and df[cols["id"]].notna().all():
        df[cols["id"]] = df[cols["id"]].astype(str)
    else:
        df[cols["id"]] = df["source_index"].astype(str)

    df[cols["label"]] = df[cols["label"]].astype(int)
    df[cols["language"]] = df[cols["language"]].astype(str)
    df[cols["generator"]] = df[cols["generator"]].astype(str)
    df[cols["code"]] = df[cols["code"]].astype(str)
    return df


def load_manifest(cfg: Dict[str, Any], split: str) -> pd.DataFrame:
    path = resolve_path(cfg, "manifests_dir") / f"{split}.csv"
    require_file(path, "Run 01_build_split.py first.")
    return pd.read_csv(path)


def load_all_manifests(cfg: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    return {
        "train": load_manifest(cfg, "train"),
        "val": load_manifest(cfg, "val"),
        "test": load_manifest(cfg, "test"),
    }


def is_python_series(series: pd.Series, cfg: Dict[str, Any]) -> pd.Series:
    names = {s.lower() for s in cfg["python_languages"]}
    return series.astype(str).str.lower().isin(names)


def make_stratum(df: pd.DataFrame, label_col: str, language_col: str) -> pd.Series:
    return df[label_col].astype(str) + "|||" + df[language_col].astype(str)


def assert_disjoint(sets: Dict[str, set], what: str) -> None:
    names = list(sets)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            shared = sets[a] & sets[b]
            if shared:
                raise AssertionError(
                    f"{what} not disjoint across splits: "
                    f"{a} ∩ {b} has {len(shared)} values "
                    f"(example={next(iter(shared))!r})"
                )


def feature_matrix_path(cfg: Dict[str, Any], split: str) -> Path:
    return resolve_path(cfg, "features_dir") / f"{split}_features.npz"


def embedding_matrix_path(cfg: Dict[str, Any], split: str) -> Path:
    return resolve_path(cfg, "embeddings_dir") / f"{split}_embeddings.npz"


def load_xy(cfg: Dict[str, Any], split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return X_features, X_embeddings, y for a split (Stage 2+)."""
    feat_path = feature_matrix_path(cfg, split)
    emb_path = embedding_matrix_path(cfg, split)
    require_file(feat_path, "Run 03_extract_features.py")
    require_file(emb_path, "Run 04_embed_codebert.py")
    feat = np.load(feat_path)
    emb = np.load(emb_path)
    Xf = feat["X"]
    Xe = emb["X"]
    yf = feat["y"].astype(int)
    ye = emb["y"].astype(int)
    if not np.array_equal(yf, ye):
        raise ValueError(f"Label mismatch between features and embeddings for {split}")
    if len(Xf) != len(Xe):
        raise ValueError(
            f"Row count mismatch for {split}: features={len(Xf)} embeddings={len(Xe)}"
        )
    return Xf, Xe, yf


def best_threshold(y_true, probs) -> float:
    from sklearn.metrics import f1_score

    best_thr, best_f1 = 0.5, -1.0
    for thr in np.arange(0.05, 0.951, 0.005):
        pred = (probs >= thr).astype(int)
        score = f1_score(y_true, pred, zero_division=0)
        if score > best_f1:
            best_f1, best_thr = float(score), float(thr)
    return best_thr


def evaluate_probs(name: str, y_true, probs, threshold: float) -> Dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    pred = (probs >= threshold).astype(int)
    return {
        "Model": name,
        "ROC_AUC": float(roc_auc_score(y_true, probs)),
        "PR_AUC": float(average_precision_score(y_true, probs)),
        "Accuracy": float(accuracy_score(y_true, pred)),
        "Precision": float(precision_score(y_true, pred, zero_division=0)),
        "Recall": float(recall_score(y_true, pred, zero_division=0)),
        "F1": float(f1_score(y_true, pred, zero_division=0)),
        "Threshold": float(threshold),
        "N": int(len(y_true)),
    }
