"""Unit tests for the workbench HTTP client helpers (workbench/client.py). No network: fixed bytes and MockTransport."""
import os
import sys

import httpx
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "workbench"))

import client  # noqa: E402
from client import (  # noqa: E402
    SSEParser, api_headers, error_message, parse_sse_bytes, percentile, run_streaming_query, validate_ids,
)

SSE_BODY = (
    b": keep-alive comment\n"
    b"event: meta\ndata: {\"trace_id\": \"abc\", \"route\": \"HYBRID_RAG\", \"path\": \"llm\"}\n\n"
    b"event: token\ndata: {\"t\": \"Seven \"}\n\n"
    b"event: token\ndata: {\"t\": \"years.\"}\n\n"
    b"event: final\ndata: {\"status\": \"OK\", \"answer\": \"Seven years.\", \"citations\": [], \"grounding_score\": 0.91}\n\n"
)


def test_parse_complete_body_in_order():
    events = parse_sse_bytes(SSE_BODY)
    assert [e.event for e in events] == ["meta", "token", "token", "final"]
    assert events[1].data == {"t": "Seven "}
    assert events[3].data["grounding_score"] == 0.91


def test_parser_handles_events_split_across_chunks():
    parser = SSEParser()
    got = []
    for i in range(0, len(SSE_BODY), 7):          # 7-byte chunks split lines and events everywhere
        got += parser.feed(SSE_BODY[i:i + 7], now_ms=float(i))
    assert [e.event for e in got] == ["meta", "token", "token", "final"]
    # The meta event ends with the blank line in the chunk starting at offset 91, so that chunk time is its arrival.
    assert got[0].arrived_ms == 91.0


def test_parser_accepts_crlf_and_multiline_data():
    body = b"event: token\r\ndata: {\"t\":\r\ndata: \"x\"}\r\n\r\n"
    events = parse_sse_bytes(body)
    assert len(events) == 1 and events[0].event == "token"
    assert events[0].data == {"t": "x"}


def test_unnamed_event_defaults_to_message_and_bad_json_is_kept_raw():
    events = parse_sse_bytes(b"data: not json\n\n")
    assert events[0].event == "message"
    assert events[0].data == {"raw": "not json"}


def test_incomplete_event_is_not_emitted_until_blank_line():
    parser = SSEParser()
    assert parser.feed(b"event: token\ndata: {\"t\": \"a\"}\n") == []
    assert len(parser.feed(b"\n")) == 1


def test_api_headers_carry_tenant_and_user_only_no_role():
    assert api_headers("tenant_acme") == {"X-Tenant-Id": "tenant_acme"}
    assert api_headers("tenant_acme", "user_demo") == {"X-Tenant-Id": "tenant_acme", "X-User-Id": "user_demo"}


def test_validate_ids_follows_the_contract_patterns():
    assert validate_ids("tenant_acme", "user_demo") == []
    assert validate_ids("acme", "user_demo")          # missing tenant_ prefix
    assert validate_ids("tenant_acme", "demo")        # missing user_ prefix
    assert validate_ids("tenant_" + "x" * 65, "user_demo")   # too long


@pytest.mark.parametrize("status,body,needle", [
    (403, {"error": {"message": "header and body tenant differ"}}, "403"),
    (404, None, "404"),
    (422, {"error": {"message": "invalid request body"}}, "422"),
    (429, None, "Rate limited"),
    (500, None, "500 Server error"),
])
def test_error_message_is_readable(status, body, needle):
    assert needle in error_message(status, body)


def test_error_message_for_exceptions():
    req = httpx.Request("POST", "http://127.0.0.1:1/x")
    assert "unreachable" in error_message(None, exc=httpx.ConnectError("refused", request=req))
    assert "Timed out" in error_message(None, exc=httpx.ReadTimeout("slow", request=req))


def test_percentile_nearest_rank():
    values = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert percentile(values, 50) == 30.0
    assert percentile(values, 95) == 50.0
    assert percentile([], 50) is None
    assert percentile([7.0], 95) == 7.0


def test_run_streaming_query_consumes_events_and_applies_retract():
    body = (
        b"event: meta\ndata: {\"trace_id\": \"t\", \"route\": \"HYBRID_RAG\", \"path\": \"llm\"}\n\n"
        b"event: token\ndata: {\"t\": \"Wrong \"}\n\n"
        b"event: retract\ndata: {\"reason\": \"ungrounded\", \"grounding_score\": 0.2}\n\n"
        b"event: final\ndata: {\"status\": \"OK\", \"answer\": \"Right.\", \"citations\": [], \"grounding_score\": 0.9}\n\n"
    )
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["accept"] = request.headers.get("accept")
        seen["tenant"] = request.headers.get("x-tenant-id")
        return httpx.Response(200, headers={"X-Path": "llm", "Content-Type": "text/event-stream"}, content=body)

    http = httpx.Client(transport=httpx.MockTransport(handler))
    payload = {"tenant_id": "tenant_acme", "user_id": "user_demo", "query": "q"}
    arrivals = []
    result = run_streaming_query(http, "http://gw.test", payload, "tenant_acme", None,
                                 on_event=lambda ev, r: arrivals.append(ev.event))
    assert seen == {"accept": "text/event-stream", "tenant": "tenant_acme"}
    assert result.path == "llm" and result.problem is None
    assert result.answer_text == "Right."         # final.answer replaces the retracted tokens
    assert result.final["grounding_score"] == 0.9
    assert arrivals == ["meta", "token", "retract", "final"]
    assert result.ttfb_ms is not None


def test_run_streaming_query_reports_http_error_without_raising():
    def handler(request):
        return httpx.Response(403, json={"error": {"message": "header and body tenant differ"}})

    http = httpx.Client(transport=httpx.MockTransport(handler))
    result = run_streaming_query(http, "http://gw.test", {}, "tenant_acme", "user")
    assert result.status_code == 403
    assert "403" in result.problem and result.final is None


def test_run_streaming_query_reports_unreachable_server_without_raising():
    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    http = httpx.Client(transport=httpx.MockTransport(handler))
    result = run_streaming_query(http, "http://gw.test", {}, "tenant_acme", "user")
    assert "unreachable" in result.problem


def test_timeouts_are_sensible():
    assert client.READ_TIMEOUT == 60.0
    assert client.CONNECT_TIMEOUT <= 10.0
