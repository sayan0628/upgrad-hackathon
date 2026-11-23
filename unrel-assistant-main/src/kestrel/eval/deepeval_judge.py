"""DeepEval judge for KPI-20..24, routed through OpenRouter (design.md D-4, D-21, §1.2.11).

* Telemetry is switched off before deepeval is imported: nothing about our answers goes to Confident AI.
* The judge is the pinned `judge_model` (default anthropic/claude-sonnet-5.5), temperature 0, with a per-run budget
  (max calls and max output tokens). The budget is enforced in code and raises `JudgeBudgetExceeded`; it does not
  fall back to any other model.
* The API key is only ever placed in the Authorization header; errors never include it.
"""
from __future__ import annotations

import asyncio
import os
import threading
from dataclasses import dataclass
from typing import Any

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "1")   # must be set before deepeval is imported

import httpx  # noqa: E402
from deepeval.models.base_model import DeepEvalBaseLLM  # noqa: E402

from ..config import Settings  # noqa: E402
from ..contracts import ErrorCode, PlatformError  # noqa: E402


class JudgeBudgetExceeded(PlatformError):
    def __init__(self, what: str):
        super().__init__(ErrorCode.UPSTREAM_RATE_LIMITED, f"judge budget exceeded: {what}")


@dataclass
class JudgeBudget:
    max_calls: int
    max_output_tokens: int = 512          # per judge call; verdicts are short JSON
    calls: int = 0
    output_tokens: int = 0
    prompt_chars: int = 0

    def reserve(self) -> None:
        if self.calls >= self.max_calls:
            raise JudgeBudgetExceeded(f"{self.max_calls} calls")

    def record(self, usage: dict[str, Any] | None, prompt: str) -> None:
        self.calls += 1
        self.prompt_chars += len(prompt)
        if usage:
            self.output_tokens += int(usage.get("completion_tokens", 0))


class OpenRouterJudge(DeepEvalBaseLLM):
    """DeepEval custom model. `generate` returns the judge's text; DeepEval parses the JSON verdict itself."""

    def __init__(self, settings: Settings, budget: JudgeBudget, *, http: httpx.Client | None = None,
                 model: str | None = None):
        if not settings.openrouter_api_key:
            raise PlatformError(ErrorCode.INVALID_REQUEST, "OPENROUTER_API_KEY is not set")
        self._key = settings.openrouter_api_key
        self._url = settings.openrouter_base_url.rstrip("/") + "/chat/completions"
        self._model_id = model or settings.judge_model
        self.budget = budget
        self._http = http or httpx.Client(timeout=httpx.Timeout(60.0, connect=5.0))
        self._lock = threading.Lock()
        super().__init__(self._model_id)

    # DeepEvalBaseLLM contract
    def load_model(self, *args, **kwargs):
        return self

    def get_model_name(self, *args, **kwargs) -> str:
        return self._model_id

    def generate(self, prompt: str, schema: Any = None, **kwargs: Any) -> str:
        with self._lock:                                 # the budget is shared; deepeval may call in threads
            self.budget.reserve()
            self.budget.record(None, prompt)             # reserve the call before sending it
        body = {"model": self._model_id, "temperature": 0, "max_tokens": self.budget.max_output_tokens,
                "messages": [{"role": "user", "content": prompt}], "usage": {"include": True}}
        headers = {"Authorization": f"Bearer {self._key}", "Content-Type": "application/json", "X-Title": "kestrel-judge"}
        try:
            r = self._http.post(self._url, json=body, headers=headers)
        except httpx.HTTPError as e:
            raise PlatformError(ErrorCode.INVALID_REQUEST, f"judge transport error: {type(e).__name__}") from None
        if r.status_code == 429:
            raise PlatformError(ErrorCode.UPSTREAM_RATE_LIMITED, "judge rate limited", {"retry_after": r.headers.get("retry-after")})
        if r.status_code >= 400:
            raise PlatformError(ErrorCode.INVALID_REQUEST, f"judge provider error {r.status_code}",
                                {"detail": r.text[:200].replace(self._key, "***")})
        data = r.json()
        usage = data.get("usage") or {}
        with self._lock:
            self.budget.output_tokens += int(usage.get("completion_tokens", 0))
        return data["choices"][0]["message"]["content"] or ""

    async def a_generate(self, prompt: str, schema: Any = None, **kwargs: Any) -> str:
        return await asyncio.to_thread(self.generate, prompt, schema, **kwargs)


def build_metrics(judge: OpenRouterJudge, thresholds: dict[str, float] | None = None) -> dict[str, Any]:
    """The DeepEval metrics that map to KPI-20..22 and the per-response hallucination proxy for KPI-24.

    Mapping caveats (reported in the validators, not hidden):
    * KPI-20 faithfulness  -> FaithfulnessMetric (aggregate).
    * KPI-21 answer relevancy -> AnswerRelevancyMetric.
    * KPI-22 context entity recall -> NOT a built-in DeepEval metric; ContextualRecallMetric is a proxy until a custom
      entity-recall GEval metric is written.
    * KPI-24 claim-level unsupported-claim rate -> NOT covered by a built-in; HallucinationMetric is per response.
    """
    from deepeval.metrics import AnswerRelevancyMetric, ContextualRecallMetric, FaithfulnessMetric, HallucinationMetric
    # HallucinationMetric (deepeval ≥ 4) passes at score >= threshold, and 1 = no hallucination. KPI-24 allows
    # <= 0.5% violations, so the minimum passing score is 0.995 (NOT 0.005, which the older API used).
    t = {"faithfulness": 0.92, "answer_relevancy": 0.90, "contextual_recall": 0.85, "hallucination": 0.995, **(thresholds or {})}
    return {
        "KPI-20 faithfulness": FaithfulnessMetric(threshold=t["faithfulness"], model=judge, include_reason=False, async_mode=False),
        "KPI-21 answer relevancy": AnswerRelevancyMetric(threshold=t["answer_relevancy"], model=judge, include_reason=False, async_mode=False),
        "KPI-22 contextual recall (proxy)": ContextualRecallMetric(threshold=t["contextual_recall"], model=judge, include_reason=False, async_mode=False),
        "KPI-24 hallucination (per response proxy)": HallucinationMetric(threshold=t["hallucination"], model=judge, include_reason=False, async_mode=False),
    }
