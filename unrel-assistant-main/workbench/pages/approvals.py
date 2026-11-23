"""Approvals: reviewer queue (GET /api/v1/approvals) with approve / reject and the repaired args."""
import streamlit as st

from common import request_json, show_problem

st.header("Approvals")
st.caption("Tier 3 changes waiting for a reviewer. The reviewer must differ from the requester; the server enforces it.")

if st.button("Refresh queue"):
    st.session_state.pop("approvals_cache", None)

if "approvals_cache" not in st.session_state:
    status, data, problem = request_json("GET", "/api/v1/approvals", params={"status": "PENDING"})
    st.session_state["approvals_cache"] = {"status": status, "data": data, "problem": problem}

cache = st.session_state["approvals_cache"]
if show_problem(cache["problem"], cache["status"]):
    with st.expander("Raw response"):
        st.json(cache["data"] or {})
else:
    items = (cache["data"] or {}).get("approvals", [])
    st.write(f"{len(items)} pending approval(s) for this tenant.")
    for item in items:
        aid = item.get("approval_id", "?")
        with st.container(border=True):
            st.markdown(f"**{item.get('tool', '?')}**  |  id `{aid}`  |  requested by `{item.get('requested_by')}`")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("Args (redacted)")
                st.json(item.get("args_redacted") or {})
            with c2:
                st.markdown("Repaired by the harness")
                repaired = item.get("repaired") or []
                if repaired:
                    st.json(repaired)
                else:
                    st.caption("Nothing repaired.")
            reason = st.text_input("Reason", key=f"reason_{aid}", max_chars=300)
            b1, b2, _ = st.columns([1, 1, 4])
            if b1.button("Approve", key=f"approve_{aid}", type="primary"):
                status, data, problem = request_json("POST", f"/api/v1/approvals/{aid}/approve",
                                                     body={"reason": reason}, with_user=True)
                if problem:
                    st.error(problem)
                else:
                    st.success(f"Approved: {data.get('status', 'APPROVED')}")
                    st.json(data)
            if b2.button("Reject", key=f"reject_{aid}"):
                status, data, problem = request_json("POST", f"/api/v1/approvals/{aid}/reject",
                                                     body={"reason": reason}, with_user=True)
                if problem:
                    st.error(problem)
                else:
                    st.success(f"Rejected: {data.get('status', 'REJECTED')}")
                    st.json(data)
    if not items:
        st.info("The queue is empty.")
