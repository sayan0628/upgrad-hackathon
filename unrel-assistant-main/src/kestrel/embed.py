"""Embedders behind one interface (design.md §1.4.1 row 3: one embedding per request, reused by dense retrieval)."""
from __future__ import annotations

import asyncio
import hashlib
import re
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    dim: int

    async def embed(self, texts: list[str]) -> np.ndarray: ...   # L2-normalised rows, float32


class SentenceTransformerEmbedder:
    """Local open-weight model. Defaults to the small model already cached on this machine; bge-* models are
    drop-in via `model_name` (benchmark on our own data before choosing, see design.md §0.3)."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", local_only: bool = True):
        self.model_name, self._local_only, self._model = model_name, local_only, None
        self.dim = 384

    def _load(self):
        if self._model is None:
            import os
            if self._local_only:
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device="cpu")
            get_dim = getattr(self._model, "get_embedding_dimension", None) or self._model.get_sentence_embedding_dimension
            self.dim = get_dim()
        return self._model

    def warm(self) -> None:
        self._load().encode(["warm-up"], normalize_embeddings=True)

    async def embed(self, texts: list[str]) -> np.ndarray:
        m = self._load()
        out = await asyncio.to_thread(lambda: m.encode(texts, normalize_embeddings=True, convert_to_numpy=True))
        return out.astype(np.float32)


class HashEmbedder:
    """Deterministic feature-hashing embedder for tests and offline use. Cosine reflects word overlap only."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    async def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                h = int(hashlib.md5(w.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0
        n = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(n == 0, 1.0, n)
