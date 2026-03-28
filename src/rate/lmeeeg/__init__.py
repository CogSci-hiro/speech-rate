"""lmeEEG workflow package."""

from rate.lmeeeg.io import (
    LMEEEGModelPaths,
    LMEEEGRunPaths,
    build_lmeeeg_model_output_paths,
    build_lmeeeg_output_paths,
    load_lmeeeg_inputs,
    read_lmeeeg_outputs,
    save_lmeeeg_result,
)
from rate.lmeeeg.mass_model import LMEEEGResult, fit_lmeeeg_backend_result, fit_lmeeeg_mass_model
from rate.lmeeeg.pooling import load_pooled_lmeeeg_trial_data, resolve_dataset_index_path, select_lmeeeg_rows
from rate.lmeeeg.residualise import residualise_random_effects
from rate.lmeeeg.trial_data import (
    LMEEEGTrialData,
    build_lmeeeg_trial_data_from_arrays,
    build_lmeeeg_trial_table,
    prepare_lmeeeg_design,
    resolve_random_effect_group,
    resolve_random_effect_structure,
)

__all__ = [
    "LMEEEGResult",
    "LMEEEGModelPaths",
    "LMEEEGRunPaths",
    "LMEEEGTrialData",
    "build_lmeeeg_model_output_paths",
    "build_lmeeeg_trial_data_from_arrays",
    "build_lmeeeg_output_paths",
    "build_lmeeeg_trial_table",
    "fit_lmeeeg_backend_result",
    "fit_lmeeeg_mass_model",
    "load_pooled_lmeeeg_trial_data",
    "load_lmeeeg_inputs",
    "prepare_lmeeeg_design",
    "read_lmeeeg_outputs",
    "residualise_random_effects",
    "resolve_dataset_index_path",
    "resolve_random_effect_group",
    "resolve_random_effect_structure",
    "select_lmeeeg_rows",
    "save_lmeeeg_result",
]
