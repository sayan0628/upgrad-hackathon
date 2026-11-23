"""Health & latency: /healthz, /readyz (with per-path counters) and a small client-side latency sample."""
import httpx
import streamlit as st

from client import error_message, percentile, run_streaming_query, safe_json
from common import current_settings, shared_client

# Ten golden questions, embedded (not read from disk). Taken from data/insurance/golden/golden_set.jsonl.
SAMPLE_QUESTIONS = [
    "What term lengths does Acme Mutual Term Life offer?",
    "What is the multi-policy discount on auto insurance?",
    "How long does a customer have to appeal a claim denial?",
    "Who must approve a claim of $25,000?",
    "A traveller paid a $4,000 non-refundable trip cost and cancels 5 days before departure under the Cancel For Any Reason upgrade. How much is reimbursed?",
    "A payee is owed a $15,000 claim payout and presents a driver's license that expired 6 months ago. Can payment proceed on that document?",
    "A claim is referred to the Special Investigations Unit on day 1. What is the longest the investigation should take, and what clock normally applies to decisions?",
    "Summarize the overview rules in the Term Life Insurance Guide.",
    "What are the key points about filing claims in the Health Claims Guide?",
    "What is the coverage limit for an RV policy?",
]


def _fmt(value):
    return None if value is None else round(value, 1)


def _get(path: str):
    """GET a health endpoint. Returns (status, body, problem). A 503 body is still returned (readyz uses it)."""
    cfg = current_settings()
    try:
        resp = shared_client().get(cfg["base_url"] + path)
    except (httpx.HTTPError, OSError) as exc:
        return None, None, error_message(None, exc=exc)
    body = safe_json(resp)
    problem = None if resp.status_code == 200 else error_message(resp.status_code, body)
    return resp.status_code, body, problem


st.header("Health & latency")
st.caption("Liveness and readiness of the gateway, plus a small client-side latency sample.")

if st.button("Refresh health"):
    st.session_state.pop("health_cache", None)

if "health_cache" not in st.session_state:
    st.session_state["health_cache"] = {"healthz": _get("/healthz"), "readyz": _get("/readyz")}

health = st.session_state["health_cache"]
c1, c2 = st.columns(2)
with c1:
    st.subheader("/healthz")
    code, body, problem = health["healthz"]
    if problem:
        st.error(problem)
    else:
        st.success(f"Live: {body.get('status', 'ok') if isinstance(body, dict) else body}")
with c2:
    st.subheader("/readyz")
    code, body, problem = health["readyz"]
    if body is None:
        st.error(problem)
    else:
        if isinstance(body, dict) and body.get("ready"):
            st.success("Ready")
        else:
            st.warning(f"Not ready (HTTP {code})")
        if isinstance(body, dict) and body.get("checks"):
            st.json(body["checks"])
        paths = (body or {}).get("paths") if isinstance(body, dict) else None
        if paths:
            m1, m2 = st.columns(2)
            m1.metric("Fast path requests", paths.get("fast", 0))
            m2.metric("LLM path requests", paths.get("llm", 0))

st.subheader("Latency sample (client side)")
n = st.slider("Queries to run", min_value=1, max_value=len(SAMPLE_QUESTIONS), value=5,
              help="Runs sequentially in the streaming mode, so the times include network and streaming.")
if st.button("Run latency sample", type="primary"):
    cfg = current_settings()
    rows, timings = [], {"fast": [], "llm": []}
    progress = st.progress(0.0, text="Running...")
    for i, question in enumerate(SAMPLE_QUESTIONS[:n]):
        body = {"tenant_id": cfg["tenant"], "user_id": cfg["user"], "query": question}
        result = run_streaming_query(shared_client(), cfg["base_url"], body, cfg["tenant"], cfg["user"])
        path = result.path or "unknown"
        if result.problem or result.error:
            rows.append({"#": i + 1, "path": path, "status": "error",
                         "total ms": None, "first token ms": None,
                         "detail": result.problem or str(result.error.get("message", ""))})
        else:
            timings.setdefault(path, []).append(result.total_ms)
            rows.append({"#": i + 1, "path": path, "status": result.final.get("status", "-"),
                         "total ms": round(result.total_ms, 1),
                         "first token ms": None if result.ttfb_ms is None else round(result.ttfb_ms, 1),
                         "detail": ""})
        progress.progress((i + 1) / n, text=f"{i + 1} of {n}")
    progress.empty()
    st.dataframe(rows, hide_index=True, width="stretch")

    summary = []
    for path in ("fast", "llm"):
        values = timings.get(path, [])
        summary.append({"X-Path": path, "n": len(values),
                        "p50 ms": _fmt(percentile(values, 50)), "p95 ms": _fmt(percentile(values, 95))})
    st.markdown("**Per-path latency (client total time to final event)**")
    st.dataframe(summary, hide_index=True, width="stretch")
    st.caption("Percentiles use nearest rank. With small n the p95 is just the largest value; treat it as indicative.")
