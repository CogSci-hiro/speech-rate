"""Minimal wrappers around MNE cluster/permutation routines."""

from __future__ import annotations

from typing import Any


def run_cluster_permutation(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
    """Return a placeholder result for future MNE-based group statistics."""
    return {
        "status": "placeholder",
        "message": "MNE cluster permutation integration will be implemented in a later pass.",
    }
