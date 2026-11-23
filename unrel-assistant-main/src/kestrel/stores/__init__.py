"""Storage interfaces and in-memory reference backends (design.md §1.3.6, D-24).

The in-memory classes reproduce the *behaviour* the design relies on (tenant scoping, TTL + jitter,
single-flight, semantic tier, approval lifecycle, per-tenant hash-chained audit). They do not reproduce
latency, Redis-cluster or Postgres-RLS enforcement, or memory behaviour at scale.
"""
from .memory import (ApprovalStatus, CacheHit, InMemoryCache, InMemoryStateStore, normalise_query)

__all__ = ["ApprovalStatus", "CacheHit", "InMemoryCache", "InMemoryStateStore", "normalise_query"]
