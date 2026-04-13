"""Helpers for induced-power lmeEEG targets."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def section_supports_targets(config: Mapping[str, Any], *, config_section: str) -> bool:
    """Return whether a config section declares target-specific analyses."""
    section_cfg = config.get(config_section) or {}
    targets = section_cfg.get("targets")
    return isinstance(targets, list) and len(targets) > 0


def resolve_target_names(
    config: Mapping[str, Any],
    *,
    config_section: str,
    target_name: str | None = None,
) -> list[str | None]:
    """Resolve configured targets for one lmeEEG section."""
    if not section_supports_targets(config, config_section=config_section):
        return [None]
    if target_name is not None:
        return [str(target_name)]
    targets = [str(value) for value in (config.get(config_section) or {}).get("targets", [])]
    if not targets:
        raise ValueError(f"No `{config_section}.targets` were configured for target-specific inference.")
    return targets


def extract_induced_power(
    epochs,
    config: Mapping[str, Any],
    *,
    config_section: str = "lmeeeg_induced",
    target_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract trialwise induced power for one configured frequency band."""
    extraction_cfg = dict((config.get(config_section) or {}).get("extraction") or {})
    method = str(extraction_cfg.get("method", "morlet")).lower()
    band_cfg = _resolve_band_config(config, config_section=config_section, target_name=target_name)
    data = epochs.get_data(copy=True)
    sfreq = float(epochs.info["sfreq"])

    if method == "morlet":
        morlet_cfg = dict(extraction_cfg.get("morlet") or {})
        freqs = _resolve_morlet_freqs(band_cfg, morlet_cfg)
        zero_mean = bool(morlet_cfg.get("zero_mean", True))
        n_cycles = _resolve_n_cycles(morlet_cfg.get("n_cycles", 5.0), freqs=freqs)
        n_cycles = _fit_morlet_cycles_to_signal(
            n_cycles,
            freqs=freqs,
            sfreq=sfreq,
            n_times=int(data.shape[-1]),
            zero_mean=zero_mean,
        )
        decim = int(morlet_cfg.get("decim", 1))

        from mne.time_frequency import tfr_array_morlet

        power = tfr_array_morlet(
            data,
            sfreq=sfreq,
            freqs=freqs,
            n_cycles=n_cycles,
            output="power",
            use_fft=bool(morlet_cfg.get("use_fft", True)),
            decim=decim,
            zero_mean=zero_mean,
            verbose="ERROR",
        )
        band_power = np.asarray(power, dtype=float).mean(axis=2)
        times = np.asarray(epochs.times, dtype=float)[::decim][: band_power.shape[-1]]
        return band_power, times

    if method == "hilbert":
        hilbert_cfg = dict(extraction_cfg.get("hilbert") or {})
        filtered_epochs = epochs.copy().filter(
            l_freq=float(band_cfg["fmin"]),
            h_freq=float(band_cfg["fmax"]),
            l_trans_bandwidth=hilbert_cfg.get("l_trans_bandwidth", "auto"),
            h_trans_bandwidth=hilbert_cfg.get("h_trans_bandwidth", "auto"),
            phase=str(hilbert_cfg.get("phase", "zero")),
            fir_design=str(hilbert_cfg.get("fir_design", "firwin")),
            verbose="ERROR",
        )
        filtered_epochs.apply_hilbert(envelope=bool(hilbert_cfg.get("envelope", True)), verbose="ERROR")
        analytic = filtered_epochs.get_data(copy=True)
        band_power = np.square(np.asarray(analytic, dtype=float))
        return band_power, np.asarray(filtered_epochs.times, dtype=float)

    raise ValueError(f"Unsupported induced extraction method `{method}`. Expected `morlet` or `hilbert`.")


def _resolve_band_config(
    config: Mapping[str, Any],
    *,
    config_section: str,
    target_name: str,
) -> dict[str, float]:
    extraction_cfg = dict((config.get(config_section) or {}).get("extraction") or {})
    bands_cfg = extraction_cfg.get("bands") or {}
    if not isinstance(bands_cfg, Mapping) or target_name not in bands_cfg:
        raise ValueError(f"Unknown induced target band `{target_name}` in `{config_section}.extraction.bands`.")
    band_cfg = dict(bands_cfg[target_name] or {})
    if "fmin" not in band_cfg or "fmax" not in band_cfg:
        raise ValueError(f"Band `{target_name}` must define `fmin` and `fmax`.")
    return {
        "fmin": float(band_cfg["fmin"]),
        "fmax": float(band_cfg["fmax"]),
    }


def _resolve_morlet_freqs(band_cfg: Mapping[str, float], morlet_cfg: Mapping[str, Any]) -> np.ndarray:
    n_freqs = int(morlet_cfg.get("n_freqs", 6))
    if n_freqs < 1:
        raise ValueError("Morlet extraction requires `n_freqs >= 1`.")
    if n_freqs == 1:
        return np.asarray([(float(band_cfg["fmin"]) + float(band_cfg["fmax"])) / 2.0], dtype=float)
    return np.linspace(float(band_cfg["fmin"]), float(band_cfg["fmax"]), num=n_freqs, dtype=float)


def _resolve_n_cycles(value: Any, *, freqs: np.ndarray) -> np.ndarray | float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list):
        if len(value) != len(freqs):
            raise ValueError("Morlet `n_cycles` list must match the number of frequencies.")
        return np.asarray(value, dtype=float)
    raise ValueError("Morlet `n_cycles` must be a number or a list of numbers.")


def _fit_morlet_cycles_to_signal(
    n_cycles: np.ndarray | float,
    *,
    freqs: np.ndarray,
    sfreq: float,
    n_times: int,
    zero_mean: bool,
) -> np.ndarray | float:
    from mne.time_frequency import morlet

    candidate = (
        np.full(len(freqs), float(n_cycles), dtype=float)
        if isinstance(n_cycles, float)
        else np.asarray(n_cycles, dtype=float).copy()
    )
    minimum_cycles = 1.0
    while True:
        wavelets = morlet(sfreq, freqs, n_cycles=candidate, zero_mean=zero_mean)
        if max(len(wavelet) for wavelet in wavelets) <= n_times:
            break
        updated = np.maximum(minimum_cycles, candidate * 0.9)
        if np.allclose(updated, candidate):
            break
        candidate = updated

    if isinstance(n_cycles, float):
        return float(candidate[0])
    return candidate
