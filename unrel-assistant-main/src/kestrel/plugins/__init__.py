"""Deterministic pre-execution plugins (design.md ADR-006). Fail-closed, ordered by priority."""
from .base import PreContext, PrePipeline, PrePlugin
from .injection import InjectionShield
from .pii import PiiRedactor

__all__ = ["PreContext", "PrePipeline", "PrePlugin", "PiiRedactor", "InjectionShield"]
