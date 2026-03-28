"""FOOOF feature extraction helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from rate.fooof_mass.fit import FooofTensorResult


@dataclass(frozen=True)
class FooofFeatureResult:
    """Extracted feature tensors aligned to epochs, channels, and windows."""

    feature_tensors: dict[str, np.ndarray]
    valid_fraction_by_feature: dict[str, float]
    success_mask: np.ndarray


def list_fooof_epoch_feature_names(config: Mapping[str, Any]) -> list[str]:
    """Return feature names requested for feature-epochs outputs."""
    fooof_cfg = dict(config.get("fooof") or {})
    configured = fooof_cfg.get("features")
    if configured is not None:
        if not isinstance(configured, list):
            raise ValueError("`fooof.features` must be a list when provided.")
        return [str(value) for value in configured]

    bands_cfg = dict(fooof_cfg.get("bands") or {})
    return ["offset", "exponent", *bands_cfg.keys()]


def extract_fooof_features(
    fit_result: FooofTensorResult,
    config: Mapping[str, Any],
) -> FooofFeatureResult:
    """Extract named periodic and aperiodic features from FOOOF outputs."""
    bands_cfg = dict((config.get("fooof") or {}).get("bands") or {})
    feature_shape = fit_result.success_mask.shape
    feature_tensors: dict[str, np.ndarray] = {
        "offset": np.full(feature_shape, np.nan, dtype=float),
        "exponent": np.full(feature_shape, np.nan, dtype=float),
    }

    if fit_result.aperiodic_params.shape[-1] >= 1:
        feature_tensors["offset"] = np.asarray(fit_result.aperiodic_params[..., 0], dtype=float)
    if fit_result.aperiodic_params.shape[-1] >= 2:
        feature_tensors["exponent"] = np.asarray(fit_result.aperiodic_params[..., 1], dtype=float)

    for band_name, limits in bands_cfg.items():
        low_hz = float(limits[0])
        high_hz = float(limits[1])
        cf_values = np.full(feature_shape, np.nan, dtype=float)
        pw_values = np.full(feature_shape, np.nan, dtype=float)
        bw_values = np.full(feature_shape, np.nan, dtype=float)

        for epoch_index in range(feature_shape[0]):
            for channel_index in range(feature_shape[1]):
                for window_index in range(feature_shape[2]):
                    peaks = fit_result.peak_params[epoch_index, channel_index, window_index]
                    valid_peaks = peaks[np.all(np.isfinite(peaks), axis=1)]
                    if valid_peaks.size == 0:
                        continue
                    band_peaks = valid_peaks[(valid_peaks[:, 0] >= low_hz) & (valid_peaks[:, 0] <= high_hz)]
                    if band_peaks.size == 0:
                        continue
                    best_peak = band_peaks[np.argmax(band_peaks[:, 1])]
                    cf_values[epoch_index, channel_index, window_index] = float(best_peak[0])
                    pw_values[epoch_index, channel_index, window_index] = float(best_peak[1])
                    bw_values[epoch_index, channel_index, window_index] = float(best_peak[2])

        feature_tensors[f"{band_name}_cf"] = cf_values
        feature_tensors[band_name] = pw_values
        feature_tensors[f"{band_name}_pw"] = pw_values
        feature_tensors[f"{band_name}_bw"] = bw_values

    valid_fraction = {
        feature_name: float(np.isfinite(feature_values).mean()) if feature_values.size else 0.0
        for feature_name, feature_values in feature_tensors.items()
    }
    return FooofFeatureResult(
        feature_tensors=feature_tensors,
        valid_fraction_by_feature=valid_fraction,
        success_mask=np.asarray(fit_result.success_mask, dtype=bool),
    )
