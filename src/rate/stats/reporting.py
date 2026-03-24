"""Reporting helpers for group-level outputs."""

from __future__ import annotations

from typing import Any, Mapping


def summarize_group_stats(result: Mapping[str, Any]) -> dict[str, Any]:
    """Return a lightweight report payload for group-level outputs."""
    return {
        "status": result.get("status", "unknown"),
        "message": result.get("message", ""),
    }
