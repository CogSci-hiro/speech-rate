"""Unit tests for shared design-matrix construction helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rate.design import (
    build_contrast_vector,
    build_design_matrix,
    get_predictor_column_indices,
    list_model_names,
    load_model_spec,
    ModelSpecError,
)
from rate.legacy.fooof_mass.glm import prepare_fooof_mass_design
from rate.legacy.glm.fit import prepare_glm_design
from rate.lmeeeg.trial_data import prepare_lmeeeg_design


def test_load_model_spec_supports_models_and_legacy_formulas() -> None:
    config = {
        "glm": {
            "models": {
                "self_rate": {
                    "formula": "self_rate ~ other_rate + latency + run",
                    "eligibility": {"not_null": ["self_rate"]},
                    "test_predictors": ["other_rate"],
                    "standardize": ["other_rate", "latency"],
                    "derived": {"latency_sq": "latency ** 2"},
                }
            }
        }
    }

    spec = load_model_spec(config, section="glm", model_name="self_rate")

    assert spec.outcome == "self_rate"
    assert spec.test_predictors == ("other_rate",)
    assert spec.standardize == ("other_rate", "latency")
    assert spec.derived == {"latency_sq": "latency ** 2"}

    legacy_config = {
        "glm": {
            "regressors": ["speech_rate"],
            "formulas": {"subject_level": "signal ~ speech_rate"},
        }
    }
    legacy_spec = load_model_spec(legacy_config, section="glm", model_name="subject_level")

    assert legacy_spec.formula == "signal ~ speech_rate"
    assert legacy_spec.test_predictors == ("speech_rate",)
    assert list_model_names(legacy_config, section="glm") == ["subject_level"]


def test_load_model_spec_rejects_string_eligibility() -> None:
    config = {
        "glm": {
            "models": {
                "self_rate": {
                    "formula": "self_rate ~ other_rate",
                    "eligibility": "self_rate",
                }
            }
        }
    }

    with pytest.raises(ModelSpecError, match="eligibility must be a mapping"):
        load_model_spec(config, section="glm", model_name="self_rate")


def test_build_design_matrix_reports_row_drops_and_standardizes_after_eligibility() -> None:
    metadata = pd.DataFrame(
        {
            "self_rate": [1.0, np.nan, 3.0, 4.0],
            "other_rate": [1.0, 2.0, np.nan, 4.0],
            "latency": [10.0, 20.0, 30.0, 40.0],
            "run": ["A", "A", "B", "B"],
        },
        index=pd.Index([10, 11, 12, 13], name="epoch_id"),
    )
    config = {
        "glm": {
            "models": {
                "self_rate": {
                    "formula": "self_rate ~ other_rate + latency + run",
                    "eligibility": {"not_null": ["self_rate"]},
                    "test_predictors": ["other_rate", "run"],
                    "standardize": ["other_rate", "latency"],
                }
            }
        }
    }

    result = build_design_matrix(metadata, config=config, section="glm", model_name="self_rate")

    assert list(result.row_index) == [10, 13]
    assert result.n_rows_input == 4
    assert result.n_rows_after_eligibility == 3
    assert result.n_rows_after_missing == 2
    assert result.n_rows_final == 2
    assert result.drop_report == {
        "excluded_by_eligibility": 1,
        "excluded_by_required_columns": 1,
        "excluded_by_formula_construction": 0,
    }
    assert result.row_mask.tolist() == [True, False, False, True]
    assert np.isclose(result.filtered_metadata["other_rate"].mean(), 0.0)
    assert np.isclose(result.filtered_metadata["latency"].mean(), 0.0)
    assert "Intercept" in result.column_names
    assert result.test_predictors == ["other_rate", "run"]


def test_build_design_matrix_standardizes_integer_columns_without_dtype_errors() -> None:
    metadata = pd.DataFrame(
        {
            "self_rate": [1.0, 2.0, 3.0, 4.0],
            "latency": [10, 20, 30, 40],
        }
    )
    config = {
        "glm": {
            "models": {
                "int_standardize": {
                    "formula": "self_rate ~ latency",
                    "eligibility": {"not_null": ["self_rate", "latency"]},
                    "test_predictors": ["latency"],
                    "standardize": ["latency"],
                }
            }
        }
    }

    result = build_design_matrix(metadata, config=config, section="glm", model_name="int_standardize")

    assert np.isclose(result.filtered_metadata["latency"].mean(), 0.0)
    assert np.isclose(result.filtered_metadata["latency"].std(ddof=0), 1.0)
    assert pd.api.types.is_float_dtype(result.filtered_metadata["latency"])


def test_build_design_matrix_supports_threshold_and_expression_eligibility_filters() -> None:
    metadata = pd.DataFrame(
        {
            "self_rate": [1.0, 2.0, 3.0, 4.0],
            "other_rate": [1.0, 2.0, 3.0, 4.0],
            "self_syllable_count": [5, 2, 5, 6],
            "other_syllable_count": [4, 4, 1, 7],
            "self_ipu_duration": [0.50, 0.50, 0.50, 0.10],
            "other_ipu_duration": [0.60, 0.10, 0.60, 0.60],
            "latency": [0.1, 0.2, 0.3, 0.4],
        },
        index=pd.Index([100, 101, 102, 103], name="epoch_id"),
    )
    config = {
        "glm": {
            "models": {
                "speech_rate_qc": {
                    "formula": "self_rate ~ other_rate + latency",
                    "eligibility": {
                        "not_null": ["self_rate", "other_rate"],
                        "min_values": {
                            "self_syllable_count": 3,
                            "other_syllable_count": 3,
                            "self_ipu_duration": 0.2,
                            "other_ipu_duration": 0.2,
                        },
                        "expressions": [
                            "self_ipu_duration >= 0.2",
                            "other_ipu_duration >= 0.2",
                        ],
                    },
                    "test_predictors": ["other_rate"],
                }
            }
        }
    }

    result = build_design_matrix(metadata, config=config, section="glm", model_name="speech_rate_qc")

    assert list(result.row_index) == [100]
    assert result.n_rows_input == 4
    assert result.n_rows_after_eligibility == 1
    assert result.n_rows_after_missing == 1
    assert result.n_rows_final == 1
    assert result.drop_report == {
        "excluded_by_eligibility": 3,
        "excluded_by_required_columns": 0,
        "excluded_by_formula_construction": 0,
    }


def test_contrasts_use_term_slices_for_formula_expansions() -> None:
    metadata = pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0],
            "run": ["A", "B", "C", "A"],
        }
    )
    config = {
        "glm": {
            "models": {
                "categorical": {
                    "formula": "y ~ run",
                    "test_predictors": ["run"],
                }
            }
        }
    }

    result = build_design_matrix(metadata, config=config, section="glm", model_name="categorical")
    indices = get_predictor_column_indices(
        "run",
        column_names=result.column_names,
        term_name_slices=result.term_name_slices,
    )
    contrast = build_contrast_vector(
        "run",
        column_names=result.column_names,
        term_name_slices=result.term_name_slices,
    )

    assert indices == [1, 2]
    assert contrast.tolist() == [0.0, 1.0, 1.0]


def test_downstream_adapters_use_shared_design_layer() -> None:
    metadata = pd.DataFrame(
        {
            "signal": [1.0, 2.0, 3.0],
            "speech_rate": [0.2, 0.5, 0.8],
        }
    )
    config = {
        "glm": {"formulas": {"subject_level": "signal ~ speech_rate"}, "regressors": ["speech_rate"]},
        "fooof": {
            "models": {
                "theta_cf_other_rate": {
                    "formula": "theta_cf ~ other_rate",
                    "test_predictors": ["other_rate"],
                }
            }
        },
        "lmeeeg": {"fixed_effects": ["speech_rate"]},
    }

    glm_result = prepare_glm_design(metadata, config, model_name="subject_level")
    assert glm_result.column_names == ["Intercept", "speech_rate"]

    fooof_metadata = pd.DataFrame({"theta_cf": [5.0, 6.0], "other_rate": [1.0, 2.0]})
    fooof_result = prepare_fooof_mass_design(fooof_metadata, config, model_name="theta_cf_other_rate")
    assert fooof_result.column_names == ["Intercept", "other_rate"]

    lmeeeg_result = prepare_lmeeeg_design(metadata, config)
    assert lmeeeg_result.column_names == ["Intercept", "speech_rate"]
    assert lmeeeg_result.response_dataframe is None
