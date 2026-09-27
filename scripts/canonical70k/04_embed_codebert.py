#!/usr/bin/env python3
"""Stage 3 — Frozen CodeBERT embeddings for canonical70k (resumable).

Reuses feature/embed_code.py mean_pool + embed_batch exactly.
Does not train models. Does not alter manifests or Stage-2 features.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import transformers
from numpy.lib.format import open_memmap
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

SYS_DIR = Path(__file__).resolve().parent
if str(SYS_DIR) not in sys.path:
    sys.path.insert(0, str(SYS_DIR))

from common import (  # noqa: E402
    ensure_dirs,
    load_config,
    load_manifest,
    resolve_path,
    save_json,
    sha256_file,
)

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "feature"))
from embed_code import embed_batch  # noqa: E402

SPLIT_MAP = {
    "train": ("train", 50000),
    "validation": ("val", 10000),
    "test": ("test", 10000),
}


def atomic_write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def load_progress(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def select_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def flush_memmap(arr) -> None:
    if isinstance(arr, np.memmap):
        arr.flush()


def embed_split(
    *,
    cfg,
    manifest_key: str,
    out_key: str,
    n_rows: int,
    embeddings_dir: Path,
    features_dir: Path,
    manifests_dir: Path,
    tokenizer,
    model,
    device: str,
    batch_size: int,
    max_len: int,
    model_name: str,
    dim: int,
    force: bool,
) -> None:
    df = load_manifest(cfg, manifest_key)
    if len(df) != n_rows:
        raise AssertionError(f"{out_key}: manifest has {len(df)} rows, expected {n_rows}")

    codes = df["code"].astype(str).tolist()
    ids = df["id"].astype(str).to_numpy()

    stage2_ids_path = features_dir / f"ids_{out_key}.npy"
    if not stage2_ids_path.exists():
        raise FileNotFoundError(f"Missing Stage-2 IDs: {stage2_ids_path}")
    stage2_ids = np.load(stage2_ids_path, allow_pickle=True).astype(str)
    if not np.array_equal(ids, stage2_ids):
        raise AssertionError(
            f"{out_key}: manifest IDs do not exactly match Stage-2 feature IDs"
        )

    manifest_sha = sha256_file(manifests_dir / f"{manifest_key}.csv")
    x_path = embeddings_dir / f"X_{out_key}_codebert.npy"
    ids_path = embeddings_dir / f"ids_{out_key}.npy"
    progress_path = embeddings_dir / f"progress_{out_key}.json"

    expected = {
        "manifest_sha256": manifest_sha,
        "total_rows": n_rows,
        "embedding_dim": dim,
        "model": model_name,
        "max_length": max_len,
        "pooling": "mean",
        "l2_normalize": True,
    }

    if force:
        for path in (x_path, ids_path, progress_path):
            if path.exists():
                path.unlink()

    progress = load_progress(progress_path)
    if progress is not None:
        for key, value in expected.items():
            if progress.get(key) != value:
                raise RuntimeError(
                    f"Refuse to resume {out_key}: {key} mismatch "
                    f"(got {progress.get(key)!r}, expected {value!r})"
                )
        if not x_path.exists() or not ids_path.exists():
            raise RuntimeError(f"Refuse to resume {out_key}: missing matrix/IDs")
        saved_ids = np.load(ids_path, allow_pickle=True).astype(str)
        if not np.array_equal(saved_ids, ids):
            raise RuntimeError(f"Refuse to resume {out_key}: saved IDs mismatch")
        existing = np.load(x_path, mmap_mode="r")
        if existing.shape != (n_rows, dim):
            raise RuntimeError(
                f"Refuse to resume {out_key}: shape {existing.shape} != {(n_rows, dim)}"
            )
        del existing
        start = int(progress["next_row"])
        if start < 0 or start > n_rows:
            raise RuntimeError(f"Refuse to resume {out_key}: invalid next_row={start}")
        if start == n_rows:
            print(f"[info] {out_key}: already complete")
            return
        print(f"[info] {out_key}: resuming at {start}/{n_rows}")
        arr = open_memmap(x_path, mode="r+", dtype=np.float32, shape=(n_rows, dim))
    else:
        if x_path.exists():
            raise RuntimeError(
                f"{x_path} exists without trusted progress. Use --force to regenerate."
            )
        print(f"[info] {out_key}: creating {x_path}")
        arr = open_memmap(x_path, mode="w+", dtype=np.float32, shape=(n_rows, dim))
        np.save(ids_path, ids)
        start = 0
        atomic_write_json(progress_path, {**expected, "next_row": 0})

    pbar = tqdm(total=n_rows, initial=start, desc=f"embed-{out_key}")
    row = start
    try:
        while row < n_rows:
            end = min(row + batch_size, n_rows)
            try:
                embs = embed_batch(
                    codes[row:end], tokenizer, model, device, max_len=max_len
                )
            except torch.cuda.OutOfMemoryError as exc:
                raise RuntimeError(
                    "CUDA out of memory during CodeBERT embedding. "
                    "Reduce codebert.batch_size in experiments/canonical70k/config.yaml "
                    f"(current={batch_size}) and resume."
                ) from exc
            embs = np.asarray(embs, dtype=np.float32)
            if embs.shape != (end - row, dim):
                raise AssertionError(f"Unexpected batch shape {embs.shape}")
            arr[row:end] = embs
            flush_memmap(arr)
            n_done = end - row
            row = end
            atomic_write_json(progress_path, {**expected, "next_row": row})
            pbar.update(n_done)
    finally:
        pbar.close()
        flush_memmap(arr)
        del arr

    progress = load_progress(progress_path)
    if progress is None or int(progress["next_row"]) != n_rows:
        raise RuntimeError(f"{out_key} incomplete: {progress}")


def validate_split(
    *,
    out_key: str,
    n_rows: int,
    dim: int,
    embeddings_dir: Path,
    features_dir: Path,
) -> dict:
    x_path = embeddings_dir / f"X_{out_key}_codebert.npy"
    ids_path = embeddings_dir / f"ids_{out_key}.npy"
    progress_path = embeddings_dir / f"progress_{out_key}.json"

    progress = load_progress(progress_path)
    if progress is None or int(progress.get("next_row", -1)) != n_rows:
        raise AssertionError(
            f"{out_key}: incomplete matrix "
            f"(progress next_row={None if progress is None else progress.get('next_row')})"
        )

    X = np.load(x_path)
    ids = np.load(ids_path, allow_pickle=True).astype(str)
    stage2_ids = np.load(features_dir / f"ids_{out_key}.npy", allow_pickle=True).astype(str)

    assert X.shape == (n_rows, dim), X.shape
    assert X.dtype == np.float32, X.dtype
    assert np.isfinite(X).all(), f"{out_key}: non-finite values present"
    assert np.array_equal(ids, stage2_ids), f"{out_key}: IDs != Stage-2 IDs"

    norms = np.linalg.norm(X, axis=1)
    zero_rows = int(np.sum(np.all(X == 0, axis=1)))
    if zero_rows:
        raise AssertionError(f"{out_key}: found {zero_rows} all-zero embedding rows")
    if not np.allclose(norms, 1.0, rtol=1e-3, atol=1e-3):
        raise AssertionError(
            f"{out_key}: L2 norms not ~1 (min={norms.min():.6f}, max={norms.max():.6f})"
        )

    return {
        "shape": list(X.shape),
        "dtype": str(X.dtype),
        "finite": True,
        "zero_rows": zero_rows,
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std()),
        "id_feature_alignment": True,
        "sha256": sha256_file(x_path),
        "ids_sha256": sha256_file(ids_path),
    }


def recompute_check(
    *,
    cfg,
    out_key: str,
    manifest_key: str,
    n_rows: int,
    embeddings_dir: Path,
    tokenizer,
    model,
    device: str,
    max_len: int,
    seed: int = 42,
    n_samples: int = 10,
) -> dict:
    df = load_manifest(cfg, manifest_key)
    X = np.load(embeddings_dir / f"X_{out_key}_codebert.npy")
    rng = np.random.default_rng(seed)
    idxs = np.sort(rng.choice(n_rows, size=min(n_samples, n_rows), replace=False))
    codes = [str(df.iloc[int(i)]["code"]) for i in idxs]
    recomputed = np.asarray(
        embed_batch(codes, tokenizer, model, device, max_len=max_len),
        dtype=np.float32,
    )
    stored = X[idxs]
    ok = bool(np.allclose(recomputed, stored, rtol=1e-5, atol=1e-5))
    if not ok:
        max_abs = float(np.max(np.abs(recomputed - stored)))
        raise AssertionError(f"{out_key}: recomputation FAIL (max abs={max_abs})")
    return {"passed": True, "rows": [int(i) for i in idxs], "seed": seed}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--split", choices=["train", "validation", "test", "all"], default="all"
    )
    args = ap.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)
    embeddings_dir = resolve_path(cfg, "embeddings_dir")
    features_dir = resolve_path(cfg, "features_dir")
    manifests_dir = resolve_path(cfg, "manifests_dir")

    cb = cfg.get("codebert", {})
    model_name = cb.get("model", "microsoft/codebert-base")
    max_len = int(cb.get("max_length", 512))
    batch_size = int(cb.get("batch_size", 32))
    dim = int(cb.get("dim", 768))
    if cb.get("pooling", "mean") != "mean" or cb.get("l2_normalize", True) is not True:
        raise ValueError("Stage 3 requires pooling=mean and l2_normalize=true")

    device = select_device()
    print(f"[info] loading {model_name} on {device}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    splits = list(SPLIT_MAP.keys()) if args.split == "all" else [args.split]
    for out_key in splits:
        manifest_key, n_rows = SPLIT_MAP[out_key]
        embed_split(
            cfg=cfg,
            manifest_key=manifest_key,
            out_key=out_key,
            n_rows=n_rows,
            embeddings_dir=embeddings_dir,
            features_dir=features_dir,
            manifests_dir=manifests_dir,
            tokenizer=tokenizer,
            model=model,
            device=device,
            batch_size=batch_size,
            max_len=max_len,
            model_name=model_name,
            dim=dim,
            force=args.force,
        )

    stats = {}
    recompute = {}
    for out_key in splits:
        manifest_key, n_rows = SPLIT_MAP[out_key]
        stats[out_key] = validate_split(
            out_key=out_key,
            n_rows=n_rows,
            dim=dim,
            embeddings_dir=embeddings_dir,
            features_dir=features_dir,
        )
        recompute[out_key] = recompute_check(
            cfg=cfg,
            out_key=out_key,
            manifest_key=manifest_key,
            n_rows=n_rows,
            embeddings_dir=embeddings_dir,
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_len=max_len,
        )

    meta = {
        "stage": 3,
        "model_name": model_name,
        "tokenizer_name": model_name,
        "hidden_size": dim,
        "max_length": max_len,
        "pooling_strategy": "attention_mask_mean",
        "normalization": "l2_p2_dim1",
        "dtype": "float32",
        "batch_size": batch_size,
        "device": device,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "numpy_version": np.__version__,
        "manifest_sha256": {
            f"{k}.csv": sha256_file(manifests_dir / f"{k}.csv")
            for k in ("train", "val", "test")
            if (manifests_dir / f"{k}.csv").exists()
        },
        "embedding_sha256": {f"X_{k}_codebert.npy": stats[k]["sha256"] for k in stats},
        "shapes": {k: stats[k]["shape"] for k in stats},
        "l2_norm_statistics": {
            k: {
                "min": stats[k]["norm_min"],
                "max": stats[k]["norm_max"],
                "mean": stats[k]["norm_mean"],
                "std": stats[k]["norm_std"],
            }
            for k in stats
        },
        "recomputation_check": recompute,
        "id_feature_alignment": {k: stats[k]["id_feature_alignment"] for k in stats},
        "finite_checks": {k: stats[k]["finite"] for k in stats},
    }
    meta_path = embeddings_dir / "codebert_meta.json"
    save_json(meta_path, meta)

    print(f"device: {device}")
    print(f"batch size: {batch_size}")
    for name in ("train", "validation", "test"):
        if name in stats:
            print(f"{name} shape: {tuple(stats[name]['shape'])}")
    print("finite checks:")
    for name, st in stats.items():
        print(f"  {name}: {'PASS' if st['finite'] else 'FAIL'}")
    print("norm statistics:")
    for name, st in stats.items():
        print(
            f"  {name}: min={st['norm_min']:.6f} max={st['norm_max']:.6f} "
            f"mean={st['norm_mean']:.6f} std={st['norm_std']:.6f}"
        )
    print("ID/feature alignment check:")
    for name, st in stats.items():
        print(f"  {name}: {'PASS' if st['id_feature_alignment'] else 'FAIL'}")
    print("10-row-per-split recomputation check:")
    for name, rc in recompute.items():
        print(f"  {name}: {'PASS' if rc['passed'] else 'FAIL'}")
    print("embedding file paths:")
    for name in stats:
        print(f"  {embeddings_dir / f'X_{name}_codebert.npy'}")
        print(f"  {embeddings_dir / f'ids_{name}.npy'}")
    print(f"metadata path: {meta_path}")


if __name__ == "__main__":
    main()
