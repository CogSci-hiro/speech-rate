"""lmeEEG workflow package."""

from rate.lmeeeg.io import LMEEEGRunPaths, build_lmeeeg_output_paths, load_lmeeeg_inputs, read_lmeeeg_outputs, save_lmeeeg_result
from rate.lmeeeg.mass_model import LMEEEGResult, fit_lmeeeg_mass_model
from rate.lmeeeg.residualise import residualise_random_effects
from rate.lmeeeg.trial_data import (
    LMEEEGTrialData,
    build_lmeeeg_trial_table,
    prepare_lmeeeg_design,
    resolve_random_effect_group,
    resolve_random_effect_structure,
)

__all__ = [
    "LMEEEGResult",
    "LMEEEGRunPaths",
    "LMEEEGTrialData",
    "build_lmeeeg_output_paths",
    "build_lmeeeg_trial_table",
    "fit_lmeeeg_mass_model",
    "load_lmeeeg_inputs",
    "prepare_lmeeeg_design",
    "read_lmeeeg_outputs",
    "residualise_random_effects",
    "resolve_random_effect_group",
    "resolve_random_effect_structure",
    "save_lmeeeg_result",
]
