"""Ingestion (design.md §1.3.5, simplified): redact PII, chunk, embed, store, build the tenant index, swap it in."""
from __future__ import annotations

import hashlib
import re

from ..embed import Embedder
from ..plugins import PiiRedactor, PreContext
from ..stores import InMemoryStateStore
from .index import Chunk, IndexRegistry, TenantIndex


def chunk_text(text: str, words: int = 70, overlap: int = 10) -> list[str]:
    """Sentence-aligned windows of ~`words` words with `overlap` words carried over (design: 512 tok / 64 overlap;
    smaller here so CPU reranking stays affordable)."""
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n{2,}", text) if s.strip()]
    chunks, cur = [], []
    for s in sents:
        cur.append(s)
        if sum(len(x.split()) for x in cur) >= words:
            chunks.append(" ".join(cur))
            tail = " ".join(" ".join(cur).split()[-overlap:]) if overlap else ""
            cur = [tail] if tail else []
    if cur and (not chunks or " ".join(cur) != chunks[-1][-len(" ".join(cur)):]):
        chunks.append(" ".join(cur))
    return [c for c in chunks if c.strip()]


async def ingest_texts(tenant: str, docs: dict[str, str], embedder: Embedder, registry: IndexRegistry,
                       state: InMemoryStateStore, words: int = 70, overlap: int = 10) -> int:
    pii, chunks = PiiRedactor(), []
    for source, text in docs.items():
        ctx = PreContext(tenant, text)
        pii.run(ctx)                                    # zero raw PII persistence: only redacted text is stored
        for piece in chunk_text(ctx.text, words, overlap):
            cid = hashlib.sha256(f"{tenant}\x00{piece}".encode()).hexdigest()[:32]
            chunks.append(Chunk(cid, tenant, piece, source))
            await state.put_chunk(tenant, cid, piece, {"source": source})
    emb = await embedder.embed([c.text for c in chunks]) if chunks else None
    registry.set(tenant, TenantIndex.build(chunks, emb if emb is not None else __import__("numpy").zeros((0, embedder.dim), "float32")))
    return len(chunks)
