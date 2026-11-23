# Developer workbench

A local Streamlit checking UI for the platform. It is a client of the running gateway over HTTP
(design D-9). It imports no kestrel code and never needs `OPENROUTER_API_KEY`.

## Start the gateway

```bash
uvicorn kestrel.runtime:app --port 8000
```

## Start the workbench

```bash
pip install -e ".[workbench]"        # streamlit is an optional extra
streamlit run workbench/app.py
```

Open the URL Streamlit prints (default http://localhost:8501). The gateway URL defaults to
`http://127.0.0.1:8000` and can be changed in the sidebar.

## Pages

- **Query console**: JSON mode or live streaming (SSE) mode, with X-Path, event arrival times, citations and grounding score.
- **Approvals**: reviewer queue with approve / reject and the repaired args. Needs the `reviewer` role.
- **Validators**: one quick run per page open, polled only while running. Shows the last 10 full runs.
  Shows "validators API not available on this server" if the endpoints are missing.
- **Health & latency**: `/healthz`, `/readyz` with per-path counters, and a small client-side p50/p95 sample.

The sidebar sets `X-Tenant-Id` and `X-User-Id`. There is no role header or role check (design D-25). Identity is
caller-declared and is not a security control (design D-22).
