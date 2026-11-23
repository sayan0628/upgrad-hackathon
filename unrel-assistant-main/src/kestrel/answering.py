"""Extractive answers, prompt composition, and the inline grounding check (design.md §1.2.8, §1.4.2, FR-10).

The inline verifier here is lexical (content-word support). The design's NLI cross-encoder replaces it; the
`grounding_min` threshold is provisional and must be calibrated on the golden set (KPI-24)."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .retrieval.retriever import ScoredChunk
from .retrieval.tokenize import tokenize

_SENT = re.compile(r"(?<=[.!?])\s+")
_CITE = re.compile(r"\[\d+\]")

SYSTEM_PROMPT = ("You answer strictly from the numbered context. Cite sources like [1]. If the context does not "
                 "contain the answer, say you cannot find it. Be concise: at most three sentences.")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT.split(text) if s.strip()]


def sentence_overlap(query: str, sentence: str) -> float:
    q = set(tokenize(query))
    return len(q & set(tokenize(sentence))) / len(q) if q else 0.0


def best_sentence(query: str, chunks: list[ScoredChunk]) -> tuple[str, ScoredChunk, float] | None:
    best: tuple[str, ScoredChunk, float] | None = None
    for c in chunks:
        for s in split_sentences(c.chunk.text):
            ov = sentence_overlap(query, s)
            if best is None or ov > best[2]:
                best = (s, c, ov)
    return best


@dataclass(frozen=True)
class Extractive:
    quote: str
    chunk: ScoredChunk
    overlap: float


def extractive_answer(query: str, chunks: list[ScoredChunk], min_overlap: float = 0.5) -> Extractive | None:
    """A verbatim sentence from the top chunks that covers >= half of the query's content tokens."""
    b = best_sentence(query, chunks[:2])
    if b is None or b[2] < min_overlap:
        return None
    quote, chunk, ov = b
    assert quote in chunk.chunk.text            # verbatim by construction: the verification for this path
    return Extractive(quote, chunk, ov)


def compress(query: str, text: str, max_words: int = 90) -> str:
    sents = split_sentences(text)
    ranked = sorted(range(len(sents)), key=lambda i: -sentence_overlap(query, sents[i]))
    keep, n = set(), 0
    for i in ranked:
        w = len(sents[i].split())
        if n + w > max_words and keep:
            break
        keep.add(i)
        n += w
    return " ".join(sents[i] for i in sorted(keep))


def build_messages(query: str, chunks: list[ScoredChunk], history: list[dict] | None = None) -> list[dict[str, str]]:
    ctx = "\n".join(f"[{i}] {compress(query, c.chunk.text)}" for i, c in enumerate(chunks, 1))
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {query}"}]


def grounding_score(answer: str, chunk_texts: list[str]) -> float:
    """Fraction of the answer's content tokens that appear in the cited chunks (1.0 for an empty answer)."""
    toks = tokenize(_CITE.sub(" ", answer))
    if not toks:
        return 1.0
    support = set()
    for t in chunk_texts:
        support.update(tokenize(t))
    return sum(1 for t in toks if t in support) / len(toks)
