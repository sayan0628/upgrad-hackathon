"""Shared Streamlit pieces for every workbench page: the cached httpx client, the sidebar, and JSON calls.

Imports no kestrel code (design D-9). Never reads or displays API keys.
"""
from __future__ import annotations

from typing import Any

import httpx
import streamlit as st

from client import (
    CONNECT_TIMEOUT, DEFAULT_BASE_URL, DEFAULT_TENANT, DEFAULT_USER, READ_TIMEOUT,
    api_headers, error_message, safe_json, validate_ids,
)


@st.cache_resource
def shared_client() -> httpx.Client:
    """One httpx client for the whole Streamlit process (the only cached resource)."""
    return httpx.Client(timeout=httpx.Timeout(READ_TIMEOUT, connect=CONNECT_TIMEOUT))


def init_state() -> None:
    st.session_state.setdefault("base_url", DEFAULT_BASE_URL)
    st.session_state.setdefault("tenant", DEFAULT_TENANT)
    st.session_state.setdefault("user", DEFAULT_USER)


def sidebar() -> dict[str, str]:
    """Connection and persona settings, shown on every page. Returns the current settings."""
    init_state()
    with st.sidebar:
        st.markdown("**Connection**")
        st.text_input("Gateway URL", key="base_url", help="Running kestrel gateway, e.g. http://127.0.0.1:8000")
        st.markdown("**Identity**")
        st.text_input("Tenant (X-Tenant-Id)", key="tenant")
        st.text_input("User id (X-User-Id)", key="user")
        problems = validate_ids(st.session_state.tenant, st.session_state.user)
        for p in problems:
            st.warning(p)
    return current_settings()


def current_settings() -> dict[str, str]:
    init_state()
    return {
        "base_url": st.session_state.base_url.rstrip("/"),
        "tenant": st.session_state.tenant,
        "user": st.session_state.user,
    }


def request_json(method: str, path: str, *, body: Any = None,
                 params: dict | None = None, with_user: bool = False) -> tuple[int | None, Any, str | None]:
    """Call the API and return (status, body, problem).

    problem is a readable message for connection errors and non-2xx responses; body is the parsed JSON.
    Never raises.
    """
    cfg = current_settings()
    headers = api_headers(cfg["tenant"], cfg["user"] if with_user else None)
    try:
        resp = shared_client().request(method, cfg["base_url"] + path, json=body, params=params, headers=headers)
    except (httpx.HTTPError, OSError) as exc:
        return None, None, error_message(None, exc=exc)
    data = safe_json(resp)
    if 200 <= resp.status_code < 300:
        return resp.status_code, data, None
    return resp.status_code, data, error_message(resp.status_code, data)


def show_problem(problem: str | None, status: int | None = None) -> bool:
    """Render a readable error if there is one. Returns True when an error was shown."""
    if problem:
        if status == 404:
            st.info("validators API not available on this server" if "validators" in problem.lower() else problem)
        else:
            st.error(problem)
        return True
    return False
