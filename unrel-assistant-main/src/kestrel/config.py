"""Settings from environment variables only. Secrets are never defaulted, logged or written to files."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    openrouter_api_key: str | None
    openrouter_base_url: str
    generation_model: str | None
    fallback_model: str | None
    judge_model: str | None
    max_output_tokens: int = 96
    # Measured (Phase 0): reasoning off cuts Haiku 5.5 TTFT p50 from ~1.2 s to ~0.7 s.
    generation_extra_body: dict | None = None
    # Retrieval / answering knobs (design.md §1.4.2). Thresholds are provisional until calibrated on the golden set.
    abstain_threshold: float = 0.42
    extractive_threshold: float = 0.80
    extractive_fast_path: bool = True
    rerank_pool: int = 4            # CPU default (measured: pool 4 = pool 8 on hit@4, half the latency); the design's 40 -> 16-20 assumes a GPU reranker
    grounding_min: float = 0.60

    @staticmethod
    def from_env() -> "Settings":
        e = os.environ.get
        return Settings(
            openrouter_api_key=e("OPENROUTER_API_KEY") or None,
            openrouter_base_url=e("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            generation_model=e("GENERATION_MODEL") or "anthropic/claude-haiku-5.5",
            fallback_model=e("FALLBACK_MODEL") or "anthropic/claude-haiku-4.5",
            judge_model=e("JUDGE_MODEL") or "anthropic/claude-sonnet-5.5",
            generation_extra_body={"reasoning": {"enabled": False}},
        )

    def __repr__(self) -> str:  # never leak the key through logging/reprs
        return (f"Settings(openrouter_api_key={'set' if self.openrouter_api_key else 'unset'}, "
                f"generation_model={self.generation_model!r})")
