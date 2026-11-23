import asyncio

import numpy as np
import pytest

from kestrel.embed import HashEmbedder
from kestrel.retrieval import HybridRetriever, IndexRegistry, LexicalReranker, chunk_text, ingest_texts, tokenize
from kestrel.retrieval.index import BM25
from kestrel.retrieval.retriever import rrf
from kestrel.sample_data import DEMO_DOCS
from kestrel.stores import InMemoryStateStore


def test_tokenizer_preserves_identifiers_and_normalises_them():
    assert tokenize("Is SKU99 under clause 4.2 or lot 44?") == ["sku-099", "under", "clause", "4.2", "lot-00044"]
    assert tokenize("SKU-099") == tokenize("sku 99") == ["sku-099"]
    assert tokenize("Certificates are retained") == ["certificate", "retain"]


def test_bm25_exact_identifier_beats_semantic_noise():
    docs = [tokenize(t) for t in ("SKU-099 is a flask", "SKU-100 is a mug", "flasks keep drinks hot")]
    r = BM25(docs).search(tokenize("SKU-099"), 3)
    assert r[0][0] == 0 and len(r) == 1                       # only the exact identifier matches


def test_rrf_fuses_and_dedupes():
    out = rrf([[1, 2, 3], [3, 1, 4]], k=60)
    assert [d for d, _ in out][:2] in ([1, 3], [3, 1]) and len({d for d, _ in out}) == 4


def test_chunker_is_sentence_aligned_with_overlap():
    text = " ".join(f"Sentence number {i} has a few words in it." for i in range(30))
    cs = chunk_text(text, words=30, overlap=5)
    assert len(cs) > 2 and all(c.rstrip().endswith(".") for c in cs)
    assert len(set(cs)) == len(cs)


async def _setup(tenants=("tenant_a",), docs=DEMO_DOCS):
    emb, reg, st = HashEmbedder(), IndexRegistry(), InMemoryStateStore()
    for t in tenants:
        await ingest_texts(t, docs, emb, reg, st, words=40, overlap=5)
    return emb, reg, st, HybridRetriever(reg, LexicalReranker(), pool=8)


async def test_retrieves_the_right_chunk_for_identifier_and_semantic_queries():
    emb, reg, st, r = await _setup()
    q = "What is SKU-099?"
    res = await r.retrieve("tenant_a", q, (await emb.embed([q]))[0])
    assert not res.abstain and "SKU-099" in res.chunks[0].chunk.text
    assert set(res.stage_ms) >= {"bm25", "dense", "rrf", "rerank"} and len(res.chunks) <= 4
    q = "how fast must a recall notice be sent"
    res = await r.retrieve("tenant_a", q, (await emb.embed([q]))[0])
    assert "24 hours" in res.chunks[0].chunk.text


async def test_unanswerable_query_abstains():
    emb, reg, st, r = await _setup()
    q = "capital of Mars"
    res = await r.retrieve("tenant_a", q, (await emb.embed([q]))[0])
    assert res.abstain and res.top_score < 0.42


async def test_tenant_isolation_in_retrieval():
    emb, reg, st, r = await _setup(tenants=("tenant_a",))
    q = "What is SKU-099?"
    res = await r.retrieve("tenant_b", q, (await emb.embed([q]))[0])         # tenant_b has no index
    assert res.chunks == [] and res.abstain
    await ingest_texts("tenant_b", {"x.md": "SKU-099 is a secret prototype of tenant b only."}, emb, reg, st)
    res_a = await r.retrieve("tenant_a", q, (await emb.embed([q]))[0])
    assert all(c.chunk.tenant_id == "tenant_a" and "secret prototype" not in c.chunk.text for c in res_a.chunks)


async def test_ingestion_redacts_pii_before_storing_or_indexing():
    emb, reg, st = HashEmbedder(), IndexRegistry(), InMemoryStateStore()
    await ingest_texts("tenant_a", {"c.md": "Contact jane.doe@example.com or call (415) 555-0132 about recalls."}, emb, reg, st)
    text = reg.get("tenant_a").chunks[0].text
    assert "example.com" not in text and "555-0132" not in text and "[EMAIL]" in text
    stored = await st.get_chunk("tenant_a", reg.get("tenant_a").chunks[0].chunk_id)
    assert "jane.doe" not in stored["text"]


async def test_index_swap_bumps_version():
    emb, reg, st, r = await _setup()
    v1 = reg.get("tenant_a").version
    await ingest_texts("tenant_a", DEMO_DOCS, emb, reg, st)
    assert reg.get("tenant_a").version == v1 + 1


@pytest.mark.slow
async def test_real_local_models_end_to_end():
    """Uses the cached MiniLM embedder + cross-encoder (offline). Skipped if they are not in the HF cache."""
    pytest.importorskip("sentence_transformers")
    from kestrel.embed import SentenceTransformerEmbedder
    from kestrel.retrieval import CrossEncoderReranker
    try:
        emb, rr = SentenceTransformerEmbedder(), CrossEncoderReranker()
        emb.warm(); rr.warm()
    except Exception as e:                                                      # noqa: BLE001
        pytest.skip(f"local models unavailable: {type(e).__name__}")
    reg, st = IndexRegistry(), InMemoryStateStore()
    await ingest_texts("tenant_a", DEMO_DOCS, emb, reg, st, words=40, overlap=5)
    r = HybridRetriever(reg, rr, pool=8)
    q = "How long must supplier certificates be kept?"
    res = await r.retrieve("tenant_a", q, (await emb.embed([q]))[0])
    assert not res.abstain and "seven years" in res.chunks[0].chunk.text and res.top_score > 0.8
    q = "What is the cafeteria menu on Fridays?"
    res = await r.retrieve("tenant_a", q, (await emb.embed([q]))[0])
    assert res.abstain, f"unanswerable query scored {res.top_score:.3f}"
