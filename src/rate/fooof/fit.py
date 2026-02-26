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
        "freqs_hz": None,
        "psd_raw": None,
        "psd_aperiodic": None,
        "psd_periodic": None,
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
    try:
        freqs, psd = _compute_psd(x=x, sfreq=sfreq, spec=spec)
    except Exception:
        return _nan_result()
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
        _attach_spectra_to_result(result=result, fm=fm, freqs_input=freqs, psd_input=psd)
        return result
    except Exception:
        return _nan_result()


def _attach_spectra_to_result(
    *,
    result: dict[str, Any],
    fm: FOOOF,
    freqs_input: np.ndarray,
    psd_input: np.ndarray,
) -> None:
    """Attach fitted spectra components (in log10 power units) when available."""
    freqs = np.asarray(getattr(fm, "freqs", freqs_input), dtype=float).reshape(-1)
    if freqs.size == 0:
        return

    # FOOOF works on a restricted frequency range; prefer model-space arrays when present.
    raw_model = getattr(fm, "power_spectrum", None)
    raw = np.asarray(raw_model, dtype=float).reshape(-1) if raw_model is not None else np.array([], dtype=float)
    if raw.size != freqs.size:
        raw = np.log10(np.maximum(np.asarray(psd_input, dtype=float).reshape(-1), PSD_EPS))
        if raw.size != freqs.size:
            return

    ap_fit = getattr(fm, "_ap_fit", None)
    ap = np.asarray(ap_fit, dtype=float).reshape(-1) if ap_fit is not None else np.array([], dtype=float)
    if ap.size != freqs.size:
        return

    peak_fit = getattr(fm, "_peak_fit", None)
    periodic = np.asarray(peak_fit, dtype=float).reshape(-1) if peak_fit is not None else np.array([], dtype=float)
    if periodic.size != freqs.size:
        fooofed = getattr(fm, "fooofed_spectrum_", None)
        fooofed_arr = np.asarray(fooofed, dtype=float).reshape(-1) if fooofed is not None else np.array([], dtype=float)
        if fooofed_arr.size == freqs.size:
            periodic = fooofed_arr - ap
    if periodic.size != freqs.size:
        flat = getattr(fm, "_spectrum_flat", None)
        flat_arr = np.asarray(flat, dtype=float).reshape(-1) if flat is not None else np.array([], dtype=float)
        if flat_arr.size == freqs.size:
            periodic = flat_arr
    if periodic.size != freqs.size:
        return

    result["freqs_hz"] = freqs
    result["psd_raw"] = raw
    result["psd_aperiodic"] = ap
    result["psd_periodic"] = periodic


def _compute_psd(*, x: np.ndarray, sfreq: float, spec: FooofSpec) -> tuple[np.ndarray, np.ndarray]:
    method = str(spec.spectrum_method).strip().lower()
    if method == "welch":
        nperseg = max(4, int(round(spec.welch_nperseg_s * float(sfreq))))
        nperseg = min(nperseg, x.size)
        if nperseg < 4:
            raise ValueError("nperseg < 4")
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
        return np.asarray(freqs, dtype=float), np.asarray(psd, dtype=float)

    if method == "multitaper":
        import mne

        psd_mt, freqs_mt = mne.time_frequency.psd_array_multitaper(
            x[np.newaxis, :],
            sfreq=float(sfreq),
            fmin=float(spec.freq_range[0]),
            fmax=float(spec.freq_range[1]),
            bandwidth=(None if spec.multitaper_bandwidth_hz is None else float(spec.multitaper_bandwidth_hz)),
            adaptive=bool(spec.multitaper_adaptive),
            low_bias=bool(spec.multitaper_low_bias),
            normalization="full",
            verbose="ERROR",
        )
        psd_mt = np.asarray(psd_mt, dtype=float)
        if psd_mt.ndim == 2 and psd_mt.shape[0] == 1:
            psd_mt = psd_mt[0]
        return np.asarray(freqs_mt, dtype=float), np.asarray(psd_mt, dtype=float)

    raise ValueError(f"Unsupported PSD method: {spec.spectrum_method!r}")
