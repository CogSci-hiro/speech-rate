"""Band-pass filtering for preprocessing."""

from __future__ import annotations

from typing import Any, Mapping


def apply_bandpass_filter(raw, config: Mapping[str, Any]) -> None:
    """Apply a config-driven EEG band-pass filter."""
    preprocessing_cfg = dict(config.get("preprocessing") or {})
    filter_cfg = dict(preprocessing_cfg.get("filter") or {})

    l_freq = filter_cfg.get("l_freq_hz", preprocessing_cfg.get("highpass_hz"))
    h_freq = filter_cfg.get("h_freq_hz", preprocessing_cfg.get("lowpass_hz"))
    if l_freq is None and h_freq is None:
        return

    raw.filter(
        l_freq=float(l_freq) if l_freq is not None else None,
        h_freq=float(h_freq) if h_freq is not None else None,
        picks="eeg",
        verbose="ERROR",
    )
