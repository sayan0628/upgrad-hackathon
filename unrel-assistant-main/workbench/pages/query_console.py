"""Query console: JSON mode and live streaming mode against POST /api/v1/agent/query."""
import time

import httpx
import streamlit as st

from client import api_headers, error_message, run_streaming_query, safe_json, validate_ids
from common import current_settings, shared_client


def _short(ev) -> str:
    d = ev.data
    if not isinstance(d, dict):
        return str(d)[:120]
    if ev.event == "token":
        return str(d.get("t", ""))
    if ev.event == "final":
        return f"status={d.get('status')} route={d.get('route', '')}"
    return ", ".join(f"{k}={str(v)[:40]}" for k, v in list(d.items())[:4])


def _render_final(final: dict) -> None:
    st.subheader("Final answer")
    st.markdown(final.get("answer") or "_(empty answer)_")
    c1, c2, c3 = st.columns(3)
    c1.metric("Status", str(final.get("status", "-")))
    score = final.get("grounding_score")
    c2.metric("Grounding score", "-" if score is None else f"{score:.2f}")
    c3.metric("Trace id", str(final.get("trace_id", "-"))[:12])
    st.markdown("**Citations**")
    citations = final.get("citations") or []
    if citations:
        st.dataframe(citations, hide_index=True, width="stretch")
    else:
        st.caption("No citations.")
    if final.get("approval_id"):
        st.info(f"Approval pending: `{final['approval_id']}` (see the Approvals page).")
    if final.get("remediation"):
        st.caption(f"Remediation: {final['remediation']}")


st.header("Query console")
st.caption("Sends a query to the gateway. Tenant and user come from the sidebar.")

with st.form("query_form"):
    query = st.text_area("Query", value="What is the maximum issue age for the 20-year Term Life policy?", height=90)
    c1, c2 = st.columns(2)
    session_id = c1.text_input("Session id (UUIDv4, optional)", value="",
                               help="Leave empty to send no session id.")
    mode = c2.radio("Mode", ["Streaming (text/event-stream)", "JSON (application/json)"], horizontal=True)
    submitted = st.form_submit_button("Send", type="primary")

cfg = current_settings()
problems = validate_ids(cfg["tenant"], cfg["user"])

if not submitted:
    st.info("Enter a query and press Send.")
elif problems:
    for p in problems:
        st.error(p)
elif not query.strip():
    st.error("Query must not be empty.")
else:
    body = {"tenant_id": cfg["tenant"], "user_id": cfg["user"], "query": query}
    if session_id.strip():
        body["session_id"] = session_id.strip()

    if mode.startswith("Streaming"):
        st.subheader("Live events")
        status_box = st.empty()
        events_box = st.empty()
        answer_box = st.empty()
        rows = []

        def on_event(ev, result):
            rows.append({"event": ev.event, "arrived (ms)": round(ev.arrived_ms, 1), "detail": _short(ev)})
            events_box.dataframe(rows, hide_index=True, width="stretch")
            answer_box.markdown("**Answer so far:**\n\n" + (result.answer_text or "_(waiting for tokens)_"))

        result = run_streaming_query(shared_client(), cfg["base_url"], body, cfg["tenant"],
                                     cfg["user"], on_event=on_event)
        if result.problem:
            status_box.error(result.problem)
        elif result.error:
            status_box.error(f"Server error event: {result.error.get('message', result.error)}")
        if result.path or result.status_code:
            ttft = f"first token at {result.ttfb_ms:.1f} ms" if result.ttfb_ms is not None else "no tokens"
            st.write(f"X-Path: `{result.path}`  |  HTTP {result.status_code}  |  {ttft}")
        if result.final:
            _render_final(result.final)
        elif not result.problem and not result.error:
            status_box.warning("Stream ended without a final event.")
        st.caption(f"Total client time: {result.total_ms:.1f} ms")
    else:
        t0 = time.perf_counter()
        try:
            headers = {**api_headers(cfg["tenant"]), "Accept": "application/json"}
            resp = shared_client().post(cfg["base_url"] + "/api/v1/agent/query", json=body, headers=headers)
            elapsed = (time.perf_counter() - t0) * 1000
            data = safe_json(resp)
            st.write(f"X-Path: `{resp.headers.get('X-Path')}`  |  HTTP {resp.status_code}  |  {elapsed:.1f} ms")
            if resp.status_code in (200, 202) and isinstance(data, dict):
                _render_final(data)
                with st.expander("Raw JSON"):
                    st.json(data)
            else:
                st.error(error_message(resp.status_code, data))
                with st.expander("Raw error body"):
                    st.json(data)
        except (httpx.HTTPError, OSError) as exc:
            st.error(error_message(None, exc=exc))
