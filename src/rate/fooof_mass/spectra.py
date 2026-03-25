"""Spectral estimation helpers for sliding-window FOOOF analyses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from rate.fooof_mass.windows import WindowDefinition


@dataclass(frozen=True)
class WindowSpectraResult:
    """Windowed PSD tensor and frequency axis."""

    frequencies: np.ndarray
    psd: np.ndarray
    method: str


def compute_window_spectra(
    epochs,
    windows: list[WindowDefinition],
    config: Mapping[str, Any],
) -> WindowSpectraResult:
    """Compute PSDs for each epoch, channel, and sliding window.

    Returns
    -------
    WindowSpectraResult
        PSD tensor with shape ``(n_epochs, n_channels, n_windows, n_freqs)``.
    """
    fooof_cfg = dict(config.get("fooof") or {})
    spectrum_cfg = dict(fooof_cfg.get("spectrum") or {})
    method = str(spectrum_cfg.get("method", spectrum_cfg.get("spectrum_method", "welch"))).lower()
    if method != "welch":
        raise ValueError(f"Unsupported fooof spectrum method `{method}`. Only `welch` is implemented.")

    import mne

    data = epochs.get_data(copy=True)
    sfreq = float(epochs.info["sfreq"])
    fit_range = spectrum_cfg.get("fit_range_hz", fooof_cfg.get("freq_range_hz", [1.0, 40.0]))
    fmin, fmax = float(fit_range[0]), float(fit_range[1])
    n_per_seg_s = float(spectrum_cfg.get("welch_n_per_seg_s", fooof_cfg.get("window_s", 1.0)))
    overlap_fraction = float(spectrum_cfg.get("welch_overlap", 0.5))
    n_per_seg = max(int(round(n_per_seg_s * sfreq)), 1)
    n_overlap = int(round(n_per_seg * overlap_fraction))
    n_overlap = min(n_overlap, n_per_seg - 1) if n_per_seg > 1 else 0

    psd_windows: list[np.ndarray] = []
    freqs: np.ndarray | None = None
    for window in windows:
        window_data = data[:, :, window.start_sample : window.stop_sample]
        window_psd, window_freqs = mne.time_frequency.psd_array_welch(
            window_data,
            sfreq=sfreq,
            fmin=fmin,
            fmax=fmax,
            n_per_seg=min(n_per_seg, window_data.shape[-1]),
            n_overlap=min(n_overlap, max(window_data.shape[-1] - 1, 0)),
            average="mean",
            verbose="ERROR",
        )
        if freqs is None:
            freqs = np.asarray(window_freqs, dtype=float)
        elif not np.allclose(freqs, window_freqs):
            raise ValueError("Welch frequency axis changed across windows; this stage requires a fixed frequency grid.")
        psd_windows.append(np.asarray(window_psd, dtype=float))

    assert freqs is not None
    psd_tensor = np.stack(psd_windows, axis=2)
    return WindowSpectraResult(frequencies=freqs, psd=psd_tensor, method=method)
