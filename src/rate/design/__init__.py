"""Shared design-matrix utilities for metadata-driven subject analyses."""

from rate.design.contrasts import (
    build_contrast_vector,
    build_contrasts,
    get_predictor_column_indices,
    get_test_predictor_contrasts,
)
from rate.design.eligibility import EligibilityResult, apply_eligibility, apply_trial_eligibility
from rate.design.formulas import (
    ModelSpec,
    ModelSpecError,
    get_derived_terms,
    get_formula_variable_names,
    get_model_formula,
    get_required_columns,
    get_standardize_columns,
    get_test_predictors,
    infer_outcome_from_formula,
    list_model_names,
    load_model_spec,
)
from rate.design.matrices import DesignMatrixResult, build_design_matrix, prepare_metadata_for_model

__all__ = [
    "DesignMatrixResult",
    "EligibilityResult",
    "ModelSpec",
    "ModelSpecError",
    "apply_eligibility",
    "apply_trial_eligibility",
    "build_contrast_vector",
    "build_contrasts",
    "build_design_matrix",
    "get_derived_terms",
    "get_formula_variable_names",
    "get_model_formula",
    "get_predictor_column_indices",
    "get_required_columns",
    "get_standardize_columns",
    "get_test_predictor_contrasts",
    "get_test_predictors",
    "infer_outcome_from_formula",
    "list_model_names",
    "load_model_spec",
    "prepare_metadata_for_model",
]
