from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from rate.evoked_trf.fit import _configure_mne_logging
from rate.induced_trf.aperiodic import SlidingAperiodicConfig, compute_sliding_aperiodic
from rate.induced_trf.hilbert import HilbertConfig, compute_band_envelopes


@dataclass(frozen=True)
class InducedBandRunTarget:
    y: np.ndarray
    sfreq_hz: float
    times_s: np.ndarray
    channel_names: tuple[str, ...]
    aperiodic_time_series: pd.DataFrame | None = None


def _trim_half_window(
    *,
    y: np.ndarray,
    times_s: np.ndarray,
    sfreq_hz: float,
    window_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    trim_samples = int(round((float(window_s) * float(sfreq_hz)) / 2.0))
    if trim_samples <= 0:
        return y, times_s
    if (2 * trim_samples) >= y.shape[0]:
        raise ValueError(
            f"Edge trimming removes all samples: n_samples={y.shape[0]}, trim_each_side={trim_samples}"
        )
    return y[trim_samples:-trim_samples, :], times_s[trim_samples:-trim_samples]


def load_induced_band_run_target(
    eeg_path: Path,
    *,
    band: str,
    bands_hz: dict[str, tuple[float, float]],
    downsample_hz: float,
    target_kind: str = "raw",
    fooof_sliding_cfg: SlidingAperiodicConfig | None = None,
    edge_drop: str = "none",
    edge_window_s: float | None = None,
    picks: str | list[str] = "eeg",
) -> InducedBandRunTarget:
    """
    Build induced target per run and align to the TRF target rate.

    Returns y with shape (n_samples, n_channels) to match the shared TRF engine.
    """
    import mne

    eeg_path = Path(eeg_path)
    if not eeg_path.exists():
        raise FileNotFoundError(str(eeg_path))
    if band not in bands_hz:
        raise ValueError(f"Unknown induced band '{band}'. Available: {sorted(bands_hz)}")
    if float(downsample_hz) <= 0:
        raise ValueError("downsample_hz must be > 0")
    mode_target = str(target_kind).strip().lower()
    if mode_target not in {"raw", "residual"}:
        raise ValueError(f"Unsupported target_kind '{target_kind}'. Expected one of: raw, residual")

    _configure_mne_logging()
    raw = mne.io.read_raw(eeg_path, preload=True, verbose="ERROR")
    if picks is not None:
        raw = raw.copy().pick(picks)
    channel_names = tuple(str(ch) for ch in raw.ch_names)

    aperiodic_df: pd.DataFrame | None = None
    if mode_target == "raw":
        envelopes = compute_band_envelopes(
            raw=raw,
            cfg=HilbertConfig(
                bands_hz={str(band): tuple(float(v) for v in bands_hz[band])},
                envelope="power",
            ),
        )
        data = np.asarray(envelopes[str(band)], dtype=float)  # (ch, time)
        if data.ndim != 2:
            raise ValueError(f"Expected band envelope to be 2D, got {data.shape}")
        info = raw.info.copy()
        induced_raw = mne.io.RawArray(np.asarray(data, dtype=float), info, first_samp=raw.first_samp, verbose="ERROR")
        if not np.isclose(float(induced_raw.info["sfreq"]), float(downsample_hz), atol=1e-9):
            induced_raw.resample(float(downsample_hz), npad="auto", verbose="ERROR")

        y = np.asarray(induced_raw.get_data(), dtype=float).T
        times_s = np.asarray(induced_raw.times, dtype=float)
        sfreq_hz = float(induced_raw.info["sfreq"])
    else:
        if fooof_sliding_cfg is None:
            raise ValueError("fooof_sliding_cfg is required when target_kind='residual'")
        # Residual targets are sampled directly on the sliding-window hop grid
        # at the desired TRF rate (no anti-alias resample step).
        residual_cfg = replace(fooof_sliding_cfg, step_s=(1.0 / float(downsample_hz)))
        aperiodic_df = compute_sliding_aperiodic(
            np.asarray(raw.get_data(), dtype=float),
            float(raw.info["sfreq"]),
            np.asarray(raw.times, dtype=float),
            residual_cfg,
            band_hz=tuple(float(v) for v in bands_hz[band]),
            channel_names=channel_names,
        )
        if aperiodic_df.empty:
            raise ValueError("Sliding FOOOF decomposition produced no windows.")

        n_channels = len(channel_names)
        src_times = np.sort(aperiodic_df["time"].replace([np.inf, -np.inf], np.nan).dropna().astype(float).unique())
        if src_times.size == 0:
            raise ValueError("Sliding FOOOF decomposition produced no finite residual times.")
        n_times = int(src_times.shape[0])
        data = np.full((n_channels, n_times), np.nan, dtype=float)

        for ch_idx, ch_name in enumerate(channel_names):
            ch_df = aperiodic_df.loc[aperiodic_df["channel"].astype(str) == ch_name, ["time", "band_residual_power"]].copy()
            ch_df = ch_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["time", "band_residual_power"])
            if ch_df.empty:
                raise ValueError(f"No finite residual time series values for channel '{ch_name}'")
            ch_df = ch_df.sort_values("time", kind="mergesort").drop_duplicates(subset=["time"], keep="first")
            if ch_df.shape[0] < 2:
                # Keep deterministic fallback when only one valid window exists.
                data[ch_idx, :] = float(ch_df.iloc[0]["band_residual_power"])
                continue
            x = ch_df["time"].to_numpy(dtype=float)
            y = ch_df["band_residual_power"].to_numpy(dtype=float)
            data[ch_idx, :] = np.interp(src_times, x, y, left=float(y[0]), right=float(y[-1]))

        # Fill interior NaNs per channel when interpolation encountered sparse fit failures.
        sample_idx = np.arange(data.shape[1], dtype=float)
        for ch_idx in range(data.shape[0]):
            ch = np.asarray(data[ch_idx, :], dtype=float)
            valid = np.isfinite(ch)
            if not np.any(valid):
                raise ValueError(f"Residual target contains no finite samples for channel index {ch_idx}")
            if np.all(valid):
                continue
            if np.sum(valid) < 2:
                data[ch_idx, :] = ch[valid][0]
                continue
            data[ch_idx, :] = np.interp(
                sample_idx,
                sample_idx[valid],
                ch[valid],
                left=float(ch[valid][0]),
                right=float(ch[valid][-1]),
            )

        y = np.asarray(data, dtype=float).T
        times_s = np.asarray(src_times, dtype=float)
        if times_s.size >= 2:
            hop_s = float(np.nanmedian(np.diff(times_s)))
            if not np.isfinite(hop_s) or hop_s <= 0.0:
                raise ValueError("Invalid residual hop size inferred from sliding-window times.")
            sfreq_hz = float(1.0 / hop_s)
        else:
            sfreq_hz = float(downsample_hz)

    mode = str(edge_drop).strip().lower()
    if mode == "half_window":
        if edge_window_s is None:
            raise ValueError("edge_window_s is required when edge_drop='half_window'")
        if mode_target == "raw":
            y, times_s = _trim_half_window(y=y, times_s=times_s, sfreq_hz=sfreq_hz, window_s=float(edge_window_s))
        else:
            tmin = float(times_s[0]) + (float(edge_window_s) / 2.0)
            tmax = float(times_s[-1]) - (float(edge_window_s) / 2.0)
            keep = (times_s >= tmin) & (times_s <= tmax)
            if not np.any(keep):
                raise ValueError("Residual half-window trimming removed all samples.")
            y = y[keep, :]
            times_s = times_s[keep]
    elif mode not in {"none", ""}:
        raise ValueError(f"Unsupported edge_drop mode: {edge_drop}")

    # Residual interpolation leaves boundary NaNs if edge_drop='none'; coerce finite target rows.
    if not np.all(np.isfinite(y)):
        finite_rows = np.all(np.isfinite(y), axis=1)
        if np.any(finite_rows):
            y = y[finite_rows, :]
            times_s = times_s[finite_rows]
        else:
            raise ValueError("Induced target contains no fully finite rows after construction.")

    return InducedBandRunTarget(
        y=np.asarray(y, dtype=float),
        sfreq_hz=sfreq_hz,
        times_s=np.asarray(times_s, dtype=float),
        channel_names=channel_names,
        aperiodic_time_series=aperiodic_df,
    )
