from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from fooof import FOOOF
from scipy.signal import welch


@dataclass(frozen=True)
class SlidingAperiodicConfig:
    window_s: float
    step_s: float
    fit_range_hz: tuple[float, float]
    edge_drop: str  # "none" or "half_window"
    aperiodic_mode: str = "fixed"  # "fixed" or "knee"
    peak_width_limits_hz: tuple[float, float] = (1.0, 12.0)
    max_n_peaks: int = 6
    min_peak_height: float = 0.0
    peak_threshold: float | None = None


def _aperiodic_psd(freqs_hz: np.ndarray, aperiodic_params: np.ndarray, mode: str) -> np.ndarray:
    f = np.asarray(freqs_hz, dtype=float)
    ap = np.asarray(aperiodic_params, dtype=float)
    out = np.full(f.shape, np.nan, dtype=float)
    valid = np.isfinite(f) & (f > 0.0)
    if not np.any(valid):
        return out

    if str(mode).strip().lower() == "knee" and ap.size >= 3:
        offset, knee, exponent = float(ap[0]), float(ap[1]), float(ap[2])
        log10_psd = offset - np.log10(np.maximum(knee + np.power(f[valid], exponent), np.finfo(float).tiny))
    elif ap.size >= 2:
        offset, exponent = float(ap[0]), float(ap[1])
        log10_psd = offset - exponent * np.log10(f[valid])
    else:
        return out

    out[valid] = np.power(10.0, log10_psd)
    return out


def _band_peak_power(peaks: np.ndarray, band_hz: tuple[float, float]) -> float:
    if peaks.ndim != 2 or peaks.shape[1] < 2:
        return np.nan
    fmin, fmax = float(band_hz[0]), float(band_hz[1])
    mask = np.isfinite(peaks[:, 0]) & np.isfinite(peaks[:, 1]) & (peaks[:, 0] >= fmin) & (peaks[:, 0] <= fmax)
    if not np.any(mask):
        return np.nan
    return float(np.nanmax(peaks[mask, 1]))


def compute_sliding_aperiodic(
    data: np.ndarray,
    sfreq: float,
    times: np.ndarray,
    cfg: SlidingAperiodicConfig,
    *,
    band_hz: tuple[float, float],
    channel_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Compute per-window/channel FOOOF decomposition statistics.

    Parameters
    ----------
    data
        EEG data with shape (n_channels, n_times).
    sfreq
        Sampling frequency in Hz.
    times
        Sample times in seconds for the input data (n_times,).
    cfg
        Sliding window and FOOOF settings.
    band_hz
        Target band used for total/aperiodic/residual band power summaries.
    channel_names
        Optional channel names with length == n_channels.
    """
    x = np.asarray(data, dtype=float)
    t = np.asarray(times, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"Expected data shape (n_channels, n_times), got {x.shape}")
    if t.ndim != 1 or t.shape[0] != x.shape[1]:
        raise ValueError(f"times must be 1D with length n_times={x.shape[1]}, got {t.shape}")
    if float(sfreq) <= 0:
        raise ValueError("sfreq must be > 0")

    win_samples = int(round(float(cfg.window_s) * float(sfreq)))
    step_samples = int(round(float(cfg.step_s) * float(sfreq)))
    if win_samples < 4:
        raise ValueError(f"window_s too short for PSD/FOOOF fitting: {cfg.window_s}")
    if step_samples <= 0:
        raise ValueError("step_s must correspond to >=1 sample")

    half_win = win_samples // 2
    if half_win <= 0 or (2 * half_win) >= x.shape[1]:
        raise ValueError(
            f"Window configuration incompatible with signal length: n_times={x.shape[1]}, window_s={cfg.window_s}"
        )

    n_channels = int(x.shape[0])
    if channel_names is None:
        channel_names = tuple(f"ch_{idx:03d}" for idx in range(n_channels))
    else:
        channel_names = tuple(str(c) for c in channel_names)
    if len(channel_names) != n_channels:
        raise ValueError(f"channel_names length {len(channel_names)} != n_channels {n_channels}")

    fmin_band, fmax_band = float(band_hz[0]), float(band_hz[1])
    if not (0.0 < fmin_band < fmax_band):
        raise ValueError(f"Invalid band_hz: {band_hz}")

    records: list[dict[str, float | str]] = []
    for center in range(half_win, x.shape[1] - half_win, step_samples):
        start = center - half_win
        stop = center + half_win
        segment = x[:, start:stop]

        for ch_idx in range(n_channels):
            segment_ch = np.asarray(segment[ch_idx, :], dtype=float)
            freqs_hz, psd = welch(
                segment_ch,
                fs=float(sfreq),
                window="hann",
                nperseg=int(win_samples),
                noverlap=0,
                detrend="constant",
                scaling="density",
                return_onesided=True,
            )
            freqs_hz = np.asarray(freqs_hz, dtype=float)
            psd = np.maximum(np.asarray(psd, dtype=float), np.finfo(float).tiny)

            fooof_kwargs: dict[str, object] = {
                "peak_width_limits": tuple(float(v) for v in cfg.peak_width_limits_hz),
                "max_n_peaks": int(cfg.max_n_peaks),
                "min_peak_height": float(cfg.min_peak_height),
                "aperiodic_mode": str(cfg.aperiodic_mode),
                "verbose": False,
            }
            if cfg.peak_threshold is not None:
                fooof_kwargs["peak_threshold"] = float(cfg.peak_threshold)
            fm = FOOOF(**fooof_kwargs)
            offset = np.nan
            exponent = np.nan
            theta_total_power = np.nan
            theta_aperiodic_power = np.nan
            theta_residual_power = np.nan
            theta_peak_power = np.nan

            try:
                fm.fit(freqs_hz, psd, tuple(float(v) for v in cfg.fit_range_hz))
                ap = np.asarray(getattr(fm, "aperiodic_params_", np.array([], dtype=float)), dtype=float)
                if ap.size >= 2:
                    offset = float(ap[0])
                    exponent = float(ap[-1])
                ap_psd = _aperiodic_psd(freqs_hz, ap, mode=str(cfg.aperiodic_mode))

                band_mask = (freqs_hz >= fmin_band) & (freqs_hz <= fmax_band)
                if np.any(band_mask):
                    theta_total_power = float(np.nanmean(psd[band_mask]))
                    theta_aperiodic_power = float(np.nanmean(ap_psd[band_mask]))
                    theta_residual_power = float(theta_total_power - theta_aperiodic_power)

                peak_params = np.asarray(getattr(fm, "peak_params_", np.empty((0, 3))), dtype=float)
                theta_peak_power = _band_peak_power(peak_params, (fmin_band, fmax_band))
            except Exception:
                pass

            records.append(
                {
                    "time": float(t[center]),
                    "channel": str(channel_names[ch_idx]),
                    "offset": float(offset),
                    "exponent": float(exponent),
                    "band_total_power": float(theta_total_power),
                    "band_aperiodic_power": float(theta_aperiodic_power),
                    "band_residual_power": float(theta_residual_power),
                    "band_peak_power": float(theta_peak_power),
                }
            )

    df = pd.DataFrame.from_records(records)
    if df.empty:
        return df

    mode = str(cfg.edge_drop).strip().lower()
    if mode == "half_window":
        tmin = float(t[0]) + (float(cfg.window_s) / 2.0)
        tmax = float(t[-1]) - (float(cfg.window_s) / 2.0)
        df = df.loc[(df["time"] >= tmin) & (df["time"] <= tmax)].reset_index(drop=True)
    elif mode not in {"none", ""}:
        raise ValueError(f"Unsupported edge_drop mode: {cfg.edge_drop}")

    return df
