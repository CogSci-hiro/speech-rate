from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import mne
from scipy.signal import hilbert


@dataclass(frozen=True)
class HilbertConfig:
    bands_hz: Dict[str, Tuple[float, float]]
    envelope: str = "power"  # "amplitude" or "power"


def compute_band_envelopes(
    raw: mne.io.BaseRaw,
    cfg: HilbertConfig,
) -> Dict[str, np.ndarray]:
    """
    Compute Hilbert-based band envelopes.

    Returns
    -------
    dict
        band_name -> array (n_channels, n_times)
    """
    if not raw.preload:
        raise ValueError("Raw must be preloaded.")

    picks = mne.pick_types(raw.info, eeg=True, exclude="bads")

    results: Dict[str, np.ndarray] = {}

    for band, (fmin, fmax) in cfg.bands_hz.items():
        band_raw = raw.copy().pick(picks)
        band_raw.filter(l_freq=float(fmin), h_freq=float(fmax), method="fir", phase="zero")

        data = band_raw.get_data()
        analytic = hilbert(data, axis=-1)
        amplitude = np.abs(analytic)

        if cfg.envelope == "amplitude":
            results[band] = amplitude
        elif cfg.envelope == "power":
            results[band] = amplitude ** 2
        else:
            raise ValueError(f"Unknown envelope type: {cfg.envelope}")

    return results
