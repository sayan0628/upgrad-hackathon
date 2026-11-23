"""Developer workbench entry point (design.md §2.6, D-9, D-19).

Run:  streamlit run workbench/app.py
A client of the running gateway over HTTP. Page files are run by st.navigation only when selected.
"""
import os
import sys

import streamlit as st

# Make sibling modules (client.py, common.py) importable however Streamlit was launched.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import request_json, sidebar  # noqa: E402

st.set_page_config(page_title="kestrel workbench", layout="wide")

PAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pages")

pages = [
    st.Page(os.path.join(PAGES_DIR, "query_console.py"), title="Query console", url_path="query", default=True),
    st.Page(os.path.join(PAGES_DIR, "approvals.py"), title="Approvals", url_path="approvals"),
    st.Page(os.path.join(PAGES_DIR, "validators.py"), title="Validators", url_path="validators"),
    st.Page(os.path.join(PAGES_DIR, "health_latency.py"), title="Health & latency", url_path="health"),
]

sidebar()
current = st.navigation(pages)

# Leaving the Validators page cancels its unfinished run and clears the one-run-per-open guard.
previous = st.session_state.get("_active_page")
if previous != current.url_path:
    if previous == "validators" and "val_run" in st.session_state:
        run_id = st.session_state["val_run"].get("run_id")
        if run_id and st.session_state.get("val_last_status") in ("QUEUED", "RUNNING", None):
            request_json("DELETE", f"/api/v1/validators/runs/{run_id}")
        for key in ("val_run", "val_last_status"):
            st.session_state.pop(key, None)
    st.session_state["_active_page"] = current.url_path

current.run()
