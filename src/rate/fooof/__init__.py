"""FOOOF-based event-aligned spectral feature extraction."""

from importlib import import_module
from typing import Any

__all__ = [
    "FooofSpec",
    "aggregate_to_rois",
    "apply_fit_qc",
    "build_fooof_spec",
    "extract_band_features",
    "fit_fooof_to_window",
    "iter_event_windows",
    "prepare_ddc_channel_table",
    "run_fooof_pipeline",
]

_EXPORT_MAP = {
    "FooofSpec": ("rate.fooof.spec", "FooofSpec"),
    "aggregate_to_rois": ("rate.fooof.aggregate", "aggregate_to_rois"),
    "apply_fit_qc": ("rate.fooof.qc", "apply_fit_qc"),
    "build_fooof_spec": ("rate.fooof.spec", "build_fooof_spec"),
    "extract_band_features": ("rate.fooof.features", "extract_band_features"),
    "fit_fooof_to_window": ("rate.fooof.fit", "fit_fooof_to_window"),
    "iter_event_windows": ("rate.fooof.windows", "iter_event_windows"),
    "prepare_ddc_channel_table": ("rate.fooof.aggregate", "prepare_ddc_channel_table"),
    "run_fooof_pipeline": ("rate.fooof.pipeline", "run_fooof_pipeline"),
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORT_MAP:
        raise AttributeError(name)
    module_name, attr_name = _EXPORT_MAP[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
