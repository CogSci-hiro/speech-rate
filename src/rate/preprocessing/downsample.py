"""Resampling helpers for preprocessing."""

from __future__ import annotations

from typing import Any, Mapping


def downsample_raw(raw, config: Mapping[str, Any]) -> None:
    """Downsample a raw object using MNE resampling."""
    preprocessing_cfg = dict(config.get("preprocessing") or {})
    downsample_cfg = dict(preprocessing_cfg.get("downsample") or {})
    target_sfreq = downsample_cfg.get("sfreq_hz", preprocessing_cfg.get("downsample_hz"))
    if target_sfreq is None:
        return

    raw.resample(float(target_sfreq), npad="auto", verbose="ERROR")
