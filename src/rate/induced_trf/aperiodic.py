from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from fooof import FOOOF


@dataclass(frozen=True)
class SlidingAperiodicConfig:
    window_s: float
    step_s: float
    fit_range_hz: Tuple[float, float]
    edge_drop: str  # "none" or "half_window"


def compute_sliding_aperiodic(
    data: np.ndarray,
    sfreq: float,
    times: np.ndarray,
    cfg: SlidingAperiodicConfig,
) -> pd.DataFrame:
    """
    data shape: (n_channels, n_times)
    """
    win_samples = int(cfg.window_s * sfreq)
    step_samples = int(cfg.step_s * sfreq)

    half_win = win_samples // 2

    records = []

    for center in range(half_win, data.shape[1] - half_win, step_samples):
        start = center - half_win
        stop = center + half_win

        segment = data[:, start:stop]

        psd = np.mean(np.abs(np.fft.rfft(segment, axis=1)) ** 2, axis=0)
        freqs = np.fft.rfftfreq(win_samples, 1.0 / sfreq)

        fm = FOOOF(verbose=False)
        fm.fit(freqs, psd, cfg.fit_range_hz)

        offset, exponent = fm.aperiodic_params_

        records.append(
            {
                "time": times[center],
                "offset": float(offset),
                "exponent": float(exponent),
            }
        )

    df = pd.DataFrame.from_records(records)

    if cfg.edge_drop == "half_window":
        tmin = times[0] + cfg.window_s / 2
        tmax = times[-1] - cfg.window_s / 2
        df = df[(df["time"] >= tmin) & (df["time"] <= tmax)].reset_index(drop=True)

    return df
