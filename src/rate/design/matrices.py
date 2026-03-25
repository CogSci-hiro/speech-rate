"""Design-matrix construction utilities shared across downstream analyses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import pandas as pd
import patsy

from rate.design.eligibility import EligibilityResult, apply_eligibility
from rate.design.formulas import (
    ModelSpec,
    get_formula_variable_names,
    get_test_predictors,
    load_model_spec,
)


@dataclass(frozen=True)
class DesignMatrixResult:
    """Container holding aligned metadata and matrix outputs for one model."""

    model_spec: ModelSpec
    filtered_metadata: pd.DataFrame
    design_dataframe: pd.DataFrame
    design_matrix: np.ndarray
    response_dataframe: pd.DataFrame | None
    response_matrix: np.ndarray | None
    column_names: list[str]
    row_index: pd.Index
    row_mask: np.ndarray
    test_predictors: list[str]
    term_name_slices: dict[str, slice] = field(default_factory=dict)
    design_info_column_names: list[str] = field(default_factory=list)
    n_rows_input: int = 0
    n_rows_after_eligibility: int = 0
    n_rows_after_missing: int = 0
    n_rows_final: int = 0
    drop_report: dict[str, int] = field(default_factory=dict)


def prepare_metadata_for_model(
    metadata_df: pd.DataFrame,
    model_spec: ModelSpec,
) -> tuple[pd.DataFrame, EligibilityResult, np.ndarray, tuple[str, ...]]:
    """Apply derived columns, eligibility, required-column filtering, and standardization."""
    prepared = metadata_df.copy()
    _apply_derived_columns(prepared, model_spec)

    eligibility_result = apply_eligibility(prepared, model_spec)
    eligible_metadata = prepared.loc[eligibility_result.row_mask].copy()
    required_columns = tuple(_resolve_required_columns(eligible_metadata, model_spec))
    required_mask = np.ones((len(eligible_metadata),), dtype=bool)
    if required_columns:
        required_mask &= eligible_metadata.loc[:, list(required_columns)].notna().all(axis=1).to_numpy(dtype=bool)
    eligible_metadata = eligible_metadata.loc[required_mask].copy()
    _standardize_columns(eligible_metadata, model_spec)

    return eligible_metadata, eligibility_result, required_mask, required_columns


def build_design_matrix(
    metadata_df: pd.DataFrame,
    model_spec: ModelSpec | None = None,
    *,
    config: Mapping[str, Any] | None = None,
    section: str | None = None,
    model_name: str | None = None,
) -> DesignMatrixResult:
    """Build aligned design-matrix outputs from metadata and config-defined models.

    Parameters
    ----------
    metadata_df
        Metadata table containing one row per candidate observation.
    model_spec
        Optional preloaded model specification.
    config, section, model_name
        Alternative config-based specification lookup.

    Returns
    -------
    DesignMatrixResult
        Filtered metadata, design matrix, response values, and row-accounting
        diagnostics suitable for downstream subject-level analyses.
    """
    resolved_spec = _resolve_model_spec(model_spec, config=config, section=section, model_name=model_name)
    eligible_metadata, eligibility_result, required_mask, _required_columns = prepare_metadata_for_model(
        metadata_df,
        resolved_spec,
    )

    design_df, response_df = _build_patsy_matrices(eligible_metadata, resolved_spec)
    final_index = design_df.index
    final_metadata = eligible_metadata.loc[final_index].copy()
    row_mask = np.asarray(metadata_df.index.isin(final_index), dtype=bool)

    _validate_test_predictors(
        get_test_predictors(resolved_spec),
        column_names=list(design_df.columns),
        term_slices=dict(design_df.attrs.get("term_name_slices", {})),
    )

    drop_report = {
        "excluded_by_eligibility": int(len(metadata_df) - eligibility_result.n_rows_selected),
        "excluded_by_required_columns": int(len(required_mask) - int(required_mask.sum())),
        "excluded_by_formula_construction": int(len(eligible_metadata) - len(final_metadata)),
    }

    return DesignMatrixResult(
        model_spec=resolved_spec,
        filtered_metadata=final_metadata,
        design_dataframe=design_df,
        design_matrix=design_df.to_numpy(dtype=float, copy=True),
        response_dataframe=response_df,
        response_matrix=None if response_df is None else response_df.to_numpy(copy=True),
        column_names=list(design_df.columns),
        row_index=final_index,
        row_mask=row_mask,
        test_predictors=get_test_predictors(resolved_spec),
        term_name_slices=dict(design_df.attrs.get("term_name_slices", {})),
        design_info_column_names=list(design_df.columns),
        n_rows_input=len(metadata_df),
        n_rows_after_eligibility=eligibility_result.n_rows_selected,
        n_rows_after_missing=len(eligible_metadata),
        n_rows_final=len(final_metadata),
        drop_report=drop_report,
    )


def _resolve_model_spec(
    model_spec: ModelSpec | None,
    *,
    config: Mapping[str, Any] | None,
    section: str | None,
    model_name: str | None,
) -> ModelSpec:
    if model_spec is not None:
        return model_spec
    if config is None or section is None:
        raise ValueError("Provide either `model_spec` or both `config` and `section`.")
    return load_model_spec(config, section=section, model_name=model_name)


def _resolve_required_columns(metadata_df: pd.DataFrame, model_spec: ModelSpec) -> list[str]:
    candidate_columns = set(get_formula_variable_names(model_spec))
    candidate_columns.update(model_spec.required_columns)
    candidate_columns.update(model_spec.standardize)
    if model_spec.outcome:
        candidate_columns.add(model_spec.outcome)
    return sorted(column for column in candidate_columns if column in metadata_df.columns)


def _apply_derived_columns(metadata_df: pd.DataFrame, model_spec: ModelSpec) -> None:
    for column_name, expression in model_spec.derived.items():
        try:
            metadata_df.loc[:, column_name] = metadata_df.eval(expression, engine="python")
        except Exception as exc:  # pragma: no cover - defensive branch
            raise ValueError(
                f"Unable to compute derived column `{column_name}` using expression `{expression}`."
            ) from exc


def _standardize_columns(metadata_df: pd.DataFrame, model_spec: ModelSpec) -> None:
    for column_name in model_spec.standardize:
        if column_name not in metadata_df.columns:
            raise ValueError(f"Requested standardization column `{column_name}` is missing from metadata.")
        series = metadata_df.loc[:, column_name]
        if not pd.api.types.is_numeric_dtype(series):
            raise ValueError(f"Requested standardization column `{column_name}` must be numeric.")
        mean = float(series.mean())
        std = float(series.std(ddof=0))
        if np.isclose(std, 0.0):
            metadata_df.loc[:, column_name] = 0.0
        else:
            metadata_df.loc[:, column_name] = (series - mean) / std


def _build_patsy_matrices(
    metadata_df: pd.DataFrame,
    model_spec: ModelSpec,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    na_action = patsy.NAAction(on_NA="drop")
    has_outcome = bool(model_spec.outcome) and "~" in model_spec.formula
    if has_outcome:
        response_df, design_df = patsy.dmatrices(
            model_spec.formula,
            metadata_df,
            return_type="dataframe",
            NA_action=na_action,
        )
    else:
        design_df = patsy.dmatrix(
            model_spec.formula,
            metadata_df,
            return_type="dataframe",
            NA_action=na_action,
        )
        response_df = None

    design_info = design_df.design_info
    design_df.attrs["term_name_slices"] = dict(design_info.term_name_slices)
    return design_df, response_df


def _validate_test_predictors(
    test_predictors: list[str],
    *,
    column_names: list[str],
    term_slices: Mapping[str, slice],
) -> None:
    missing_predictors = [
        predictor
        for predictor in test_predictors
        if predictor not in term_slices and predictor not in column_names
    ]
    if missing_predictors:
        missing_text = ", ".join(sorted(missing_predictors))
        raise ValueError(f"Requested test predictors are absent from the final design matrix: {missing_text}")
