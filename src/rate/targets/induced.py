from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rate.evoked_trf.fit import _configure_mne_logging
from rate.induced_trf.hilbert import HilbertConfig, compute_band_envelopes


@dataclass(frozen=True)
class InducedBandRunTarget:
    y: np.ndarray
    sfreq_hz: float
    times_s: np.ndarray


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
    edge_drop: str = "none",
    edge_window_s: float | None = None,
    picks: str | list[str] = "eeg",
) -> InducedBandRunTarget:
    """
    Build induced (Hilbert-power) target per run and downsample to the TRF rate.

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

    _configure_mne_logging()
    raw = mne.io.read_raw(eeg_path, preload=True, verbose="ERROR")
    if picks is not None:
        raw = raw.copy().pick(picks)

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
    induced_raw = mne.io.RawArray(data, info, first_samp=raw.first_samp, verbose="ERROR")
    if not np.isclose(float(induced_raw.info["sfreq"]), float(downsample_hz), atol=1e-9):
        induced_raw.resample(float(downsample_hz), npad="auto", verbose="ERROR")

    y = np.asarray(induced_raw.get_data(), dtype=float).T
    times_s = np.asarray(induced_raw.times, dtype=float)
    sfreq_hz = float(induced_raw.info["sfreq"])

    mode = str(edge_drop).strip().lower()
    if mode == "half_window":
        if edge_window_s is None:
            raise ValueError("edge_window_s is required when edge_drop='half_window'")
        y, times_s = _trim_half_window(y=y, times_s=times_s, sfreq_hz=sfreq_hz, window_s=float(edge_window_s))
    elif mode not in {"none", ""}:
        raise ValueError(f"Unsupported edge_drop mode: {edge_drop}")

    return InducedBandRunTarget(y=np.asarray(y, dtype=float), sfreq_hz=sfreq_hz, times_s=np.asarray(times_s, dtype=float))
