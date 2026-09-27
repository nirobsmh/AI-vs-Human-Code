#!/usr/bin/env python3
"""Convert raw source code into the same arrays used for hybrid training.

Produces:
  X_features   — shape (N, 25) via extract_handcraft_features.features_for_code
  X_embeddings — shape (N, 768) via embed_code (CodeBERT mean-pool + L2 normalize)

Use these for robustness re-featurization so transformed snippets stay
consistent with the original training pipelines.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

_FEATURE_DIR = Path(__file__).resolve().parent
if str(_FEATURE_DIR) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_FEATURE_DIR))

from extract_handcraft_features import FEATURE_FIELDS, features_for_code  # noqa: E402
from embed_code import embed_batch  # noqa: E402


DEFAULT_EMBEDDING_MODEL = "microsoft/codebert-base"
DEFAULT_MAX_LEN = 512


def code_to_features(code: str) -> np.ndarray:
    """Return a single (25,) float32 feature vector."""
    feats = features_for_code(code or "")
    return np.asarray([feats[name] for name in FEATURE_FIELDS], dtype=np.float32)


def codes_to_features(codes: Sequence[str]) -> np.ndarray:
    """Return an (N, 25) float32 feature matrix."""
    if not codes:
        return np.zeros((0, len(FEATURE_FIELDS)), dtype=np.float32)
    rows = [code_to_features(code) for code in codes]
    return np.vstack(rows)


class CodeBertEmbedder:
    """Lazy CodeBERT embedder matching feature/embed_code.py."""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        max_len: int = DEFAULT_MAX_LEN,
        device: Optional[str] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

    def embed_texts(
        self, texts: Sequence[str], batch_size: int = 16
    ) -> np.ndarray:
        """Return an (N, 768) float32 embedding matrix."""
        if not texts:
            return np.zeros((0, 768), dtype=np.float32)

        chunks: List[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = [str(t) if t is not None else "" for t in texts[start : start + batch_size]]
            embs = embed_batch(
                batch,
                self.tokenizer,
                self.model,
                self.device,
                max_len=self.max_len,
            )
            chunks.append(embs.astype(np.float32, copy=False))
        return np.vstack(chunks)

    def embed_one(self, code: str) -> np.ndarray:
        """Return a single (768,) float32 embedding."""
        return self.embed_texts([code], batch_size=1)[0]


def codes_to_embeddings(
    codes: Sequence[str],
    embedder: Optional[CodeBertEmbedder] = None,
    batch_size: int = 16,
) -> np.ndarray:
    """Convenience wrapper around CodeBertEmbedder.embed_texts."""
    if embedder is None:
        embedder = CodeBertEmbedder()
    return embedder.embed_texts(codes, batch_size=batch_size)


__all__ = [
    "FEATURE_FIELDS",
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_MAX_LEN",
    "code_to_features",
    "codes_to_features",
    "CodeBertEmbedder",
    "codes_to_embeddings",
]
