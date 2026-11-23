"""Validators (design.md §2.7, Tab 7).

On opening the page one quick run is started (guarded by st.session_state so reruns do not start another).
The run is polled only while it is QUEUED or RUNNING; once it is finished the page does one full rerun,
which drops the polling. Leaving the page cancels an unfinished run (app.py clears 'val_run' on navigation).
"""
import streamlit as st

from common import request_json, show_problem

ACTIVE = ("QUEUED", "RUNNING")
NOT_AVAILABLE = "validators API not available on this server (the endpoint returned an error such as 404 or 500)."
UNAVAILABLE_STATUS = (404, 500, 502, 503)


def _problem_text(status: int | None, problem: str | None) -> str | None:
    if problem and status in UNAVAILABLE_STATUS:
        return NOT_AVAILABLE
    return problem


def _start_run(profile: str) -> None:
    """POST a validators run and remember it in session_state. Re-decorates polling on the next rerun."""
    status, data, problem = request_json("POST", "/api/v1/validators/runs", body={"profile": profile})
    st.session_state["val_run"] = {
        "run_id": None if problem else (data or {}).get("run_id"),
        "problem": _problem_text(status, problem),
        "profile": profile,
    }
    st.session_state["val_last_status"] = None


def _render_verdict(v: dict) -> None:
    verdict = v.get("verdict", "?")
    icon = {"PASS": "🟢", "FAIL": "🔴", "WARN": "🟡", "SKIPPED": "⚪", "UNTRUSTED": "🟣"}.get(verdict, "")
    st.markdown(f"### {icon} {v.get('validator_id', '?')}  **{verdict}**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Checks passed", f"{v.get('checks_passed', 0)} / {v.get('checks_total', 0)}")
    c2.metric("Seeded faults caught", f"{v.get('seeded_faults_caught', 0)} / {v.get('seeded_faults_total', 0)}")
    duration = v.get("duration_ms")
    c3.metric("Duration", "-" if duration is None else f"{duration / 1000:.1f} s")
    cost = v.get("est_cost_usd")
    c4.metric("Tokens in / out", f"{v.get('llm_tokens_in', 0)} / {v.get('llm_tokens_out', 0)}",
              help="Estimated cost USD: " + ("-" if cost is None else f"{cost:.4f}"))
    metrics = v.get("metrics") or []
    if metrics:
        st.dataframe([{
            "metric": m.get("name"), "value": m.get("value"), "comparator": m.get("comparator"),
            "threshold": m.get("threshold"), "n": m.get("n"),
            "95% CI": "-" if not m.get("ci95") else f"[{m['ci95'][0]:.3f}, {m['ci95'][1]:.3f}]",
        } for m in metrics], hide_index=True, width="stretch")
    findings = v.get("findings") or []
    if findings:
        with st.expander(f"Findings ({len(findings)})"):
            for f in findings:
                st.markdown(f"- **{f.get('severity', '?')}** {f.get('summary', '')}  trace: `{f.get('trace_id') or '-'}`")


GOLDEN_VALIDATOR = "V4"   # routing and retrieval: runs the insurance golden sets (design §2.7 V4)


def _render_golden(results: list[dict], active: bool) -> None:
    """Golden dataset evaluation: shown as soon as the V4 validator (which runs the golden sets) reports."""
    st.subheader("Golden dataset evaluation")
    v4 = next((r for r in results if r.get("validator_id") == GOLDEN_VALIDATOR), None)
    if v4 is None:
        st.info("Running the golden datasets (routing, retrieval, abstention) ..." if active
                else "No golden dataset evaluation in this run.")
        return
    icon = {"PASS": "🟢", "FAIL": "🔴", "WARN": "🟡", "SKIPPED": "⚪", "UNTRUSTED": "🟣"}.get(v4.get("verdict"), "")
    st.markdown(f"{icon} **{v4.get('verdict', '?')}**  |  checks {v4.get('checks_passed', 0)} / {v4.get('checks_total', 0)}"
                "  |  indicative in quick profile (fake embedder and reranker, sampled sets)")
    metrics = v4.get("metrics") or []
    if metrics:
        st.dataframe([{
            "metric": m.get("name"), "value": m.get("value"), "comparator": m.get("comparator"),
            "threshold": m.get("threshold"), "n": m.get("n"),
            "pass": "" if m.get("value") is None else ("yes" if _meets(m) else "no"),
        } for m in metrics], hide_index=True, width="stretch")
    for f in v4.get("findings") or []:
        st.markdown(f"- **{f.get('severity', '?')}** {f.get('summary', '')}")


def _meets(m: dict) -> bool:
    op, v, t = m.get("comparator"), m.get("value"), m.get("threshold")
    return {"<=": v <= t, ">=": v >= t, "==": v == t}.get(op, False)


def _render_run(run_id: str) -> None:
    status, run, problem = request_json("GET", f"/api/v1/validators/runs/{run_id}")
    if problem:
        st.session_state["val_last_status"] = "ERROR"      # stops polling after a failed GET
        show_problem(_problem_text(status, problem))
        return
    run = run or {}
    st.session_state["val_last_status"] = run.get("status")
    st.markdown(f"**Run** `{run_id}`  |  profile **{run.get('profile', '?')}**  |  status **{run.get('status', '?')}**")
    if run.get("status") in ACTIVE:
        st.info("Running. Results appear per validator as each one finishes.")
    elif run.get("status") == "CANCELLED":
        st.caption("This run was cancelled.")
    results = run.get("results") or []
    _render_golden(results, active=run.get("status") in ACTIVE)
    st.divider()
    for v in results:
        if v.get("validator_id") == GOLDEN_VALIDATOR:
            continue                                   # already shown above
        _render_verdict(v)
        st.divider()


st.header("Validators")
st.caption("A quick run starts when this page opens, in a worker on the server. Results are indicative. "
           "Only Full runs can change a release gate.")

# One quick run per page open (app.py clears 'val_run' when the user navigates away).
if "val_run" not in st.session_state:
    _start_run("quick")

val = st.session_state["val_run"]

controls = st.columns([1, 1, 3])
if controls[0].button("Re-run (quick)"):
    if val.get("run_id"):
        request_json("DELETE", f"/api/v1/validators/runs/{val['run_id']}")   # best effort: cancel the previous run
    _start_run("quick")
    st.rerun()

with controls[1].popover("Run full"):
    st.warning("A full run includes the 50K-VU load and soak tests. It can take hours and spends model budget. "
               "The API does not give an estimate.")
    if st.button("Confirm full run"):
        _start_run("full")
        st.rerun()

if val.get("problem"):
    show_problem(val["problem"])
elif val.get("run_id"):
    last = st.session_state.get("val_last_status")
    polling = last is None or last in ACTIVE      # first render (unknown) or still running

    def _run_body():
        _render_run(val["run_id"])
        if polling and st.session_state.get("val_last_status") not in ACTIVE:
            st.rerun()                             # finished: one full rerun drops the polling interval

    if polling:
        st.fragment(run_every=2)(_run_body)()
    else:
        _run_body()

st.subheader("Last full-run history")
status_h, hist, problem_h = request_json("GET", "/api/v1/validators/runs", params={"profile": "full", "limit": 10})
if problem_h:
    st.info(NOT_AVAILABLE) if status_h in UNAVAILABLE_STATUS else st.error(problem_h)
else:
    runs = hist.get("runs", []) if isinstance(hist, dict) else (hist or [])
    if not runs:
        st.caption("No full runs recorded yet.")
    else:
        st.dataframe([{
            "run_id": r.get("run_id"), "status": r.get("status"), "started": r.get("started_at"),
            "finished": r.get("finished_at"),
            "verdicts": ", ".join(f"{x.get('validator_id')}={x.get('verdict')}" for x in r.get("results") or []),
        } for r in runs], hide_index=True, width="stretch")
