"""Sliding-window helpers for FOOOF mass-univariate analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class WindowDefinition:
    """Definition of one deterministic epoch-relative time window."""

    start_sample: int
    stop_sample: int
    start_s: float
    stop_s: float
    center_s: float


def make_sliding_windows(times: np.ndarray, sfreq: float, config: Mapping[str, Any]) -> list[WindowDefinition]:
    """Create sliding windows from config and an epoch time axis.

    Parameters
    ----------
    times
        Epoch-relative time vector in seconds.
    sfreq
        Sampling frequency in Hz.
    config
        Resolved repository config containing ``fooof.windows``.

    Returns
    -------
    list of WindowDefinition
        Window definitions ordered by increasing start time.
    """
    fooof_cfg = dict(config.get("fooof") or {})
    windows_cfg = dict(fooof_cfg.get("windows") or {})
    if times.ndim != 1 or times.size == 0:
        raise ValueError("Epoch times must be a non-empty 1D array.")
    if sfreq <= 0:
        raise ValueError("Sampling frequency must be positive.")

    epoch_start_s = float(times[0])
    epoch_stop_s = float(times[-1]) + (1.0 / float(sfreq))

    start_s = float(windows_cfg.get("start_s", epoch_start_s))
    stop_s = float(windows_cfg.get("stop_s", epoch_stop_s))
    length_s = float(windows_cfg.get("length_s", fooof_cfg.get("window_s", 1.0)))
    step_s = float(windows_cfg.get("step_s", fooof_cfg.get("step_s", length_s)))

    if length_s <= 0 or step_s <= 0:
        raise ValueError("FOOOF windows length_s and step_s must both be positive.")
    if stop_s <= start_s:
        raise ValueError("FOOOF windows stop_s must be greater than start_s.")

    length_samples = max(int(round(length_s * float(sfreq))), 1)
    step_samples = max(int(round(step_s * float(sfreq))), 1)

    epoch_start_sample = 0
    epoch_stop_sample = int(times.size)
    requested_start_sample = int(round((start_s - epoch_start_s) * float(sfreq)))
    requested_stop_sample = int(round((stop_s - epoch_start_s) * float(sfreq)))

    first_start = max(requested_start_sample, epoch_start_sample)
    last_stop = min(requested_stop_sample, epoch_stop_sample)
    if last_stop - first_start < length_samples:
        raise ValueError("FOOOF sliding-window settings do not leave room for any complete window.")

    windows: list[WindowDefinition] = []
    start_sample = first_start
    while start_sample + length_samples <= last_stop:
        stop_sample = start_sample + length_samples
        start_time = epoch_start_s + (start_sample / float(sfreq))
        stop_time = epoch_start_s + (stop_sample / float(sfreq))
        windows.append(
            WindowDefinition(
                start_sample=start_sample,
                stop_sample=stop_sample,
                start_s=float(start_time),
                stop_s=float(stop_time),
                center_s=float(start_time + ((stop_time - start_time) / 2.0)),
            )
        )
        start_sample += step_samples

    if not windows:
        raise ValueError("No valid sliding windows were generated from the configured FOOOF window settings.")
    return windows


def iter_windows(times: np.ndarray, sfreq: float, config: Mapping[str, Any]) -> list[WindowDefinition]:
    """Backward-compatible alias returning generated windows."""
    return make_sliding_windows(times, sfreq, config)


def windows_to_records(windows: list[WindowDefinition]) -> list[dict[str, Any]]:
    """Serialize windows into plain dictionaries for JSON output."""
    return [asdict(window) for window in windows]
