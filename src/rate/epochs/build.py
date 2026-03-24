"""Epoch building placeholders."""

from __future__ import annotations

from typing import Any, Mapping


def build_epoch_plan(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return a minimal description of the intended epoching plan."""
    epochs_cfg = dict(config.get("epochs") or {})
    return {
        "stage": "epochs",
        "tmin_s": epochs_cfg.get("tmin_s"),
        "tmax_s": epochs_cfg.get("tmax_s"),
        "event_column": epochs_cfg.get("event_column"),
    }
