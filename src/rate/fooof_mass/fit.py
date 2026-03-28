"""FOOOF fitting helpers for PSD tensors."""

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class FooofTensorResult:
    """FOOOF fit outputs aligned to a PSD tensor."""

    frequencies: np.ndarray
    success_mask: np.ndarray
    aperiodic_params: np.ndarray
    peak_params: np.ndarray
    error_values: np.ndarray
    r_squared_values: np.ndarray
    n_attempted: int
    n_failed: int
    max_n_peaks: int


def fit_fooof_tensor(
    frequencies: np.ndarray,
    psd_tensor: np.ndarray,
    config: Mapping[str, Any],
) -> FooofTensorResult:
    """Fit FOOOF to a PSD tensor.

    Parameters
    ----------
    frequencies
        Frequency vector with shape ``(n_freqs,)``.
    psd_tensor
        PSD tensor with shape ``(n_epochs, n_channels, n_windows, n_freqs)``.
    config
        Resolved repository config containing ``fooof.fooof`` settings.
    """
    FooofClass = _get_fooof_class()

    fooof_cfg = dict((config.get("fooof") or {}).get("fooof") or {})
    fit_range = (config.get("fooof") or {}).get("spectrum", {})
    fit_range_hz = fit_range.get("fit_range_hz", (config.get("fooof") or {}).get("freq_range_hz", [1.0, 40.0]))
    freq_range = [float(fit_range_hz[0]), float(fit_range_hz[1])]
    max_n_peaks = int(fooof_cfg.get("max_n_peaks", 6))

    psd_shape = psd_tensor.shape[:-1]
    success_mask = np.zeros(psd_shape, dtype=bool)
    aperiodic_params = np.full(psd_shape + (3,), np.nan, dtype=float)
    peak_params = np.full(psd_shape + (max_n_peaks, 3), np.nan, dtype=float)
    error_values = np.full(psd_shape, np.nan, dtype=float)
    r_squared_values = np.full(psd_shape, np.nan, dtype=float)

    n_failed = 0
    n_attempted = int(np.prod(psd_shape))
    progress_context = _build_progress_context(total=n_attempted)
    with progress_context as progress:
        task_id = progress.add_task("Sliding FOOOF fits", total=n_attempted) if progress is not None else None
        for epoch_index in range(psd_shape[0]):
            for channel_index in range(psd_shape[1]):
                for window_index in range(psd_shape[2]):
                    spectrum = np.asarray(psd_tensor[epoch_index, channel_index, window_index], dtype=float)
                    if np.any(~np.isfinite(spectrum)) or np.all(spectrum <= 0):
                        n_failed += 1
                        if progress is not None and task_id is not None:
                            progress.advance(task_id)
                        continue

                    model = FooofClass(
                        peak_width_limits=tuple(fooof_cfg.get("peak_width_limits_hz", (1.0, 8.0))),
                        max_n_peaks=max_n_peaks,
                        min_peak_height=float(fooof_cfg.get("min_peak_height", 0.0)),
                        peak_threshold=float(fooof_cfg.get("peak_threshold", 2.0)),
                        aperiodic_mode=str(fooof_cfg.get("aperiodic_mode", "fixed")),
                        verbose=False,
                    )
                    try:
                        model.fit(frequencies, spectrum, freq_range)
                    except Exception:
                        n_failed += 1
                        if progress is not None and task_id is not None:
                            progress.advance(task_id)
                        continue

                    ap_params = np.asarray(getattr(model, "aperiodic_params_", []), dtype=float)
                    if ap_params.size:
                        aperiodic_params[epoch_index, channel_index, window_index, : ap_params.size] = ap_params[:3]
                    peaks = np.asarray(getattr(model, "peak_params_", []), dtype=float)
                    if peaks.ndim == 2 and peaks.size:
                        n_copy = min(peaks.shape[0], max_n_peaks)
                        peak_params[epoch_index, channel_index, window_index, :n_copy, :] = peaks[:n_copy, :3]

                    error_value = getattr(model, "error_", np.nan)
                    r_squared = getattr(model, "r_squared_", np.nan)
                    error_values[epoch_index, channel_index, window_index] = float(error_value) if error_value is not None else np.nan
                    r_squared_values[epoch_index, channel_index, window_index] = float(r_squared) if r_squared is not None else np.nan
                    success_mask[epoch_index, channel_index, window_index] = True
                    if progress is not None and task_id is not None:
                        progress.advance(task_id)

    return FooofTensorResult(
        frequencies=np.asarray(frequencies, dtype=float),
        success_mask=success_mask,
        aperiodic_params=aperiodic_params,
        peak_params=peak_params,
        error_values=error_values,
        r_squared_values=r_squared_values,
        n_attempted=n_attempted,
        n_failed=n_failed,
        max_n_peaks=max_n_peaks,
    )


def fit_window_fooof(frequencies: np.ndarray, psd_tensor: np.ndarray, config: Mapping[str, Any]) -> FooofTensorResult:
    """Backward-compatible alias for tensor fitting."""
    return fit_fooof_tensor(frequencies, psd_tensor, config)


def _get_fooof_class():
    try:
        from fooof import FOOOF
    except ImportError as exc:  # pragma: no cover - exercised only when package missing at runtime
        raise ImportError(
            "The `fooof` package is required for the fooof-mass stage but is not installed in the current environment."
        ) from exc
    return FOOOF


def _build_progress_context(*, total: int):
    if total <= 0:
        return nullcontext(None)
    try:
        from rich.console import Console
        from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
    except ImportError:  # pragma: no cover - fallback only when rich is unavailable
        return nullcontext(None)

    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=Console(stderr=True),
        transient=False,
    )
