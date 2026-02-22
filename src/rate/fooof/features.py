from __future__ import annotations

from typing import Any

import numpy as np

from rate.fooof.spec import FooofSpec


def _band_peak_metrics(peaks: list[dict[str, Any]], band: tuple[float, float]) -> tuple[float, float]:
    fmin, fmax = band
    candidates = []
    for peak in peaks:
        cf = float(peak.get("center_frequency", np.nan))
        power = float(peak.get("power", np.nan))
        if not np.isfinite(cf) or not np.isfinite(power):
            continue
        if fmin <= cf <= fmax:
            candidates.append((cf, power))

    if not candidates:
        return (np.nan, np.nan)

    # Deterministic selection: highest power, then lowest CF on ties.
    candidates.sort(key=lambda item: (-item[1], item[0]))
    cf, power = candidates[0]
    return (float(power), float(cf))


def extract_band_features(
    fooof_result: dict,
    spec: FooofSpec,
) -> dict:
    """Convert a fitted FOOOF result into fixed band-level peak features."""
    peaks_raw = fooof_result.get("peaks", [])
    peaks = peaks_raw if isinstance(peaks_raw, list) else []

    out: dict[str, float] = {}
    for band_name in ("theta", "alpha", "beta"):
        power, cf = _band_peak_metrics(peaks, spec.bands[band_name])
        out[f"{band_name}_power"] = power
        out[f"{band_name}_cf"] = cf
    return out
