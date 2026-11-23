"""DeepEval judge adapter (KPI-20..24) without any network: httpx.MockTransport stands in for OpenRouter."""
import json
import os

import httpx
import pytest

from kestrel.config import Settings
from kestrel.contracts import ErrorCode, PlatformError
from kestrel.eval.deepeval_judge import JudgeBudget, JudgeBudgetExceeded, OpenRouterJudge, build_metrics

KEY = "sk-or-v1-FAKEKEYFORTESTS000000"
SETTINGS = Settings(openrouter_api_key=KEY, openrouter_base_url="https://or.test/api/v1", generation_model="gen",
                    fallback_model=None, judge_model="anthropic/claude-sonnet-5.5")


def judge_with(handler, max_calls=5):
    http = httpx.Client(transport=httpx.MockTransport(handler))
    return OpenRouterJudge(SETTINGS, JudgeBudget(max_calls=max_calls), http=http)


def ok(text="{\"verdict\": \"yes\"}", completion_tokens=7):
    def handler(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": text}}],
                                         "usage": {"completion_tokens": completion_tokens}})
    return handler


def test_telemetry_is_opted_out_before_deepeval_is_used():
    assert os.environ.get("DEEPEVAL_TELEMETRY_OPT_OUT") == "1"


def test_request_is_pinned_to_the_judge_model_at_temperature_zero_with_a_token_cap():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        seen["auth"] = request.headers["authorization"]
        return ok()(request)

    j = judge_with(handler)
    assert j.generate("Is the claim supported?") == "{\"verdict\": \"yes\"}"
    b = seen["body"]
    assert b["model"] == "anthropic/claude-sonnet-5.5" and b["temperature"] == 0 and b["max_tokens"] == 512
    assert seen["auth"] == f"Bearer {KEY}"


def test_budget_stops_the_run_before_another_paid_call_is_made():
    calls = []

    def handler(request):
        calls.append(1)
        return ok()(request)

    j = judge_with(handler, max_calls=2)
    j.generate("a"); j.generate("b")
    with pytest.raises(JudgeBudgetExceeded):
        j.generate("c")
    assert len(calls) == 2                     # the third call never reached the network
    assert j.budget.calls == 2 and j.budget.output_tokens == 14


def test_rate_limit_is_surfaced_as_upstream_429_and_not_retried_silently():
    j = judge_with(lambda r: httpx.Response(429, headers={"retry-after": "3"}))
    with pytest.raises(PlatformError) as e:
        j.generate("x")
    assert e.value.code == ErrorCode.UPSTREAM_RATE_LIMITED


def test_provider_errors_never_echo_the_key():
    j = judge_with(lambda r: httpx.Response(401, text=f"bad key {KEY}"))
    with pytest.raises(PlatformError) as e:
        j.generate("x")
    assert KEY not in json.dumps(e.value.to_dict())


def test_transport_errors_do_not_leak_the_key_either():
    def handler(request):
        raise httpx.ConnectError(f"refused for {KEY}")
    j = judge_with(handler)
    with pytest.raises(PlatformError) as e:
        j.generate("x")
    assert KEY not in json.dumps(e.value.to_dict())


def test_metric_set_maps_to_the_kpis_and_uses_our_judge():
    j = judge_with(ok(), max_calls=0)              # constructing metrics must not spend anything
    m = build_metrics(j)
    assert set(m) == {"KPI-20 faithfulness", "KPI-21 answer relevancy",
                      "KPI-22 contextual recall (proxy)", "KPI-24 hallucination (per response proxy)"}
    assert m["KPI-20 faithfulness"].threshold == 0.92 and m["KPI-21 answer relevancy"].threshold == 0.90
    assert m["KPI-24 hallucination (per response proxy)"].threshold == 0.995   # min passing score, deepeval >= 4
    assert j.budget.calls == 0


def test_key_is_required():
    with pytest.raises(PlatformError):
        OpenRouterJudge(Settings(openrouter_api_key=None, openrouter_base_url="", generation_model=None,
                                 fallback_model=None, judge_model=None), JudgeBudget(max_calls=1))
