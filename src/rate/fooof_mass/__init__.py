"""Sliding-FOOOF mass-univariate workflow package."""

from rate.fooof_mass.features import FooofFeatureResult, extract_fooof_features
from rate.fooof_mass.fit import FooofTensorResult, fit_fooof_tensor, fit_window_fooof
from rate.fooof_mass.glm import FooofMassGLMResult, fit_fooof_mass_glm, prepare_fooof_mass_design
from rate.fooof_mass.io import FooofMassRunPaths, build_fooof_mass_output_paths, load_fooof_mass_inputs, read_fooof_mass_results, save_fooof_mass_result
from rate.fooof_mass.spectra import WindowSpectraResult, compute_window_spectra
from rate.fooof_mass.windows import WindowDefinition, iter_windows, make_sliding_windows, windows_to_records

__all__ = [
    "FooofFeatureResult",
    "FooofMassGLMResult",
    "FooofMassRunPaths",
    "FooofTensorResult",
    "WindowDefinition",
    "WindowSpectraResult",
    "build_fooof_mass_output_paths",
    "compute_window_spectra",
    "extract_fooof_features",
    "fit_fooof_mass_glm",
    "fit_fooof_tensor",
    "fit_window_fooof",
    "iter_windows",
    "load_fooof_mass_inputs",
    "make_sliding_windows",
    "prepare_fooof_mass_design",
    "read_fooof_mass_results",
    "save_fooof_mass_result",
    "windows_to_records",
]
