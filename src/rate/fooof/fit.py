from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from fooof import FOOOF
from scipy.optimize import OptimizeWarning
from scipy.signal import welch

from rate.fooof.spec import FooofSpec

PSD_EPS = np.finfo(float).tiny


def _nan_result() -> dict[str, Any]:
    return {
        "offset": np.nan,
        "exponent": np.nan,
        "knee": np.nan,
        "peaks": [],
        "r_squared": np.nan,
        "error": np.nan,
        "fit_success": False,
    }


def fit_fooof_to_window(
    window_data: np.ndarray,
    sfreq: float,
    spec: FooofSpec,
) -> dict[str, Any]:
    """Compute a PSD and fit a FOOOF model to one channel window."""
    x = np.asarray(window_data, dtype=float).reshape(-1)
    if x.size < 4 or not np.all(np.isfinite(x)):
        return _nan_result()

    nperseg = max(4, int(round(spec.welch_nperseg_s * float(sfreq))))
    nperseg = min(nperseg, x.size)
    if nperseg < 4:
        return _nan_result()
    noverlap = int(round(nperseg * spec.welch_overlap))
    noverlap = min(max(noverlap, 0), nperseg - 1)

    freqs, psd = welch(
        x,
        fs=float(sfreq),
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        scaling="density",
        return_onesided=True,
    )
    psd = np.asarray(psd, dtype=float)
    psd = np.maximum(psd, PSD_EPS)

    fm = FOOOF(
        peak_width_limits=spec.peak_width_limits_hz,
        max_n_peaks=spec.max_n_peaks,
        min_peak_height=spec.min_peak_height,
        peak_threshold=spec.peak_threshold,
        aperiodic_mode=spec.aperiodic_mode,
        verbose=False,
    )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            fm.fit(freqs, psd, spec.freq_range)
    except Exception:
        return _nan_result()

    result = _nan_result()

    try:
        ap = np.asarray(fm.aperiodic_params_, dtype=float)
        if ap.size >= 2:
            result["offset"] = float(ap[0])
            if spec.aperiodic_mode == "knee" and ap.size >= 3:
                result["knee"] = float(ap[1])
                result["exponent"] = float(ap[2])
            else:
                result["knee"] = np.nan
                result["exponent"] = float(ap[1])

        peaks_out: list[dict[str, float]] = []
        peak_params = np.asarray(getattr(fm, "peak_params_", np.empty((0, 3))), dtype=float)
        if peak_params.ndim == 2 and peak_params.shape[1] >= 3:
            for cf, power, bw in peak_params:
                peaks_out.append(
                    {
                        "center_frequency": float(cf),
                        "power": float(power),
                        "bandwidth": float(bw),
                    }
                )
        result["peaks"] = peaks_out
        result["r_squared"] = float(getattr(fm, "r_squared_", np.nan))
        result["error"] = float(getattr(fm, "error_", np.nan))
        result["fit_success"] = bool(np.isfinite(result["offset"]) and np.isfinite(result["exponent"]))
        return result
    except Exception:
        return _nan_result()
