"""Rerankers. The cross-encoder is the cost driver (design.md R-1): on this CPU-only machine 8 short pairs take
~150 ms and 16 take ~350 ms, so the dev default pool is 8; the 40 ms budget assumes a GPU."""
from __future__ import annotations

import asyncio
import math
import pathlib
from typing import Protocol

from .tokenize import tokenize

CROSS_ENCODER_DEFAULT = "cross-encoder/ms-marco-MiniLM-L-6-v2"
ONNX_RERANKER_DIR = pathlib.Path(__file__).resolve().parents[3] / "models" / "reranker"   # exported by tools/export_reranker_onnx.py


class Reranker(Protocol):
    async def score(self, query: str, texts: list[str]) -> list[float]: ...   # each in [0, 1]


class CrossEncoderReranker:
    def __init__(self, model_name: str = CROSS_ENCODER_DEFAULT, local_only: bool = True,
                 batch_size: int = 16, max_length: int = 128):
        self.model_name, self._local_only, self.batch_size, self.max_length = model_name, local_only, batch_size, max_length
        self._model = None

    def _load(self):
        if self._model is None:
            import os
            if self._local_only:
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name, device="cpu", max_length=self.max_length)
        return self._model

    def warm(self) -> None:
        self._load().predict([("warm", "up")])

    async def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        m = self._load()
        logits = await asyncio.to_thread(lambda: m.predict([(query, t) for t in texts], batch_size=self.batch_size))
        return [1 / (1 + math.exp(-float(x))) for x in logits]


class OnnxCrossEncoderReranker:
    """Same model and output as CrossEncoderReranker (sigmoid of the raw logit), run with onnxruntime on CPU.

    Needs the artefacts from tools/export_reranker_onnx.py (models/reranker/). `intra_op_threads` defaults to 2
    to match the torch thread setting on this 2-physical-core box (kestrel.runtime.configure_torch)."""

    def __init__(self, model_dir: str | pathlib.Path = ONNX_RERANKER_DIR, model_file: str = "model_fp32.onnx",
                 batch_size: int = 16, max_length: int = 128, intra_op_threads: int = 2):
        self.model_dir, self.model_file = pathlib.Path(model_dir), model_file
        self.batch_size, self.max_length, self.intra_op_threads = batch_size, max_length, intra_op_threads
        self._sess = None
        self._tok = None
        self._inputs: list[str] = []

    def _load(self):
        if self._sess is None:
            import onnxruntime as ort
            from transformers import AutoTokenizer
            so = ort.SessionOptions()
            so.intra_op_num_threads = self.intra_op_threads
            so.inter_op_num_threads = 1
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            tok = AutoTokenizer.from_pretrained(str(self.model_dir / "tokenizer"), local_files_only=True)
            sess = ort.InferenceSession(str(self.model_dir / self.model_file), so, providers=["CPUExecutionProvider"])
            self._inputs = [i.name for i in sess.get_inputs()]
            self._tok, self._sess = tok, sess
        return self._sess

    def warm(self) -> None:
        self._infer("warm", ["up"])

    def _infer(self, query: str, texts: list[str]) -> list[float]:
        sess = self._load()
        out: list[float] = []
        for i in range(0, len(texts), self.batch_size):
            chunk = texts[i:i + self.batch_size]
            enc = self._tok([query] * len(chunk), chunk, padding=True, truncation="longest_first",
                            max_length=self.max_length, return_tensors="np")
            feed = {k: enc[k] for k in self._inputs}
            logits = sess.run(None, feed)[0].reshape(-1)
            out.extend(float(x) for x in logits)
        return out

    async def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        logits = await asyncio.to_thread(self._infer, query, texts)
        return [1 / (1 + math.exp(-x)) for x in logits]


class LexicalReranker:
    """Deterministic stand-in for tests: fraction of query content tokens present in the passage."""

    async def score(self, query: str, texts: list[str]) -> list[float]:
        q = set(tokenize(query))
        if not q:
            return [0.0] * len(texts)
        return [len(q & set(tokenize(t))) / len(q) for t in texts]
