"""Shared eligibility helpers used by design construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from rate.design.formulas import ModelSpec


@dataclass(frozen=True)
class EligibilityResult:
    """Boolean inclusion mask plus lightweight filtering diagnostics."""

    row_mask: np.ndarray
    applied_columns: tuple[str, ...]
    rule_name: str | None
    n_rows_input: int
    n_rows_selected: int


@dataclass(frozen=True)
class ResolvedEligibility:
    """Normalized eligibility configuration."""

    not_null: tuple[str, ...]
    min_values: Mapping[str, float]
    max_values: Mapping[str, float]
    equals: Mapping[str, Any]
    expressions: tuple[str, ...]


def apply_eligibility(
    metadata_df: pd.DataFrame,
    model_spec: ModelSpec | None = None,
    *,
    eligibility: Mapping[str, Any] | None = None,
    required_columns: Iterable[str] | None = None,
) -> EligibilityResult:
    """Apply explicit row-inclusion rules to metadata.

    Parameters
    ----------
    metadata_df
        Metadata table aligned to candidate observations.
    model_spec
        Optional model specification providing eligibility and additional
        required columns.
    eligibility
        Optional explicit eligibility rule name.
    required_columns
        Additional columns that must be non-missing.

    Returns
    -------
    EligibilityResult
        Boolean mask aligned to ``metadata_df`` and summary counts.
    """
    if metadata_df.empty:
        empty_mask = np.zeros((0,), dtype=bool)
        return EligibilityResult(
            row_mask=empty_mask,
            applied_columns=(),
            rule_name=None,
            n_rows_input=0,
            n_rows_selected=0,
        )

    resolved = _resolve_eligibility(model_spec=model_spec, eligibility=eligibility)
    columns_to_check = tuple(dict.fromkeys([*resolved.not_null, *(required_columns or [])]))
    mask = np.ones((len(metadata_df),), dtype=bool)
    if columns_to_check:
        missing_columns = [column for column in columns_to_check if column not in metadata_df.columns]
        if missing_columns:
            missing_text = ", ".join(sorted(missing_columns))
            raise ValueError(f"Eligibility filtering requested missing metadata columns: {missing_text}")
        mask &= metadata_df.loc[:, list(columns_to_check)].notna().all(axis=1).to_numpy(dtype=bool)

    mask &= _apply_numeric_thresholds(
        metadata_df,
        min_values=resolved.min_values,
        max_values=resolved.max_values,
    )
    mask &= _apply_equals_filters(metadata_df, resolved.equals)
    mask &= _apply_expressions(metadata_df, resolved.expressions)

    return EligibilityResult(
        row_mask=mask,
        applied_columns=columns_to_check,
        rule_name=None,
        n_rows_input=len(metadata_df),
        n_rows_selected=int(mask.sum()),
    )


def apply_trial_eligibility(*args, **kwargs) -> EligibilityResult:
    """Backward-compatible alias for the shared design eligibility helper."""
    return apply_eligibility(*args, **kwargs)


def _resolve_eligibility(
    *,
    model_spec: ModelSpec | None,
    eligibility: Mapping[str, Any] | None,
) -> ResolvedEligibility:
    raw_value = eligibility if eligibility is not None else getattr(model_spec, "eligibility", None)
    if raw_value is None:
        return ResolvedEligibility(
            not_null=(),
            min_values={},
            max_values={},
            equals={},
            expressions=(),
        )
    if not isinstance(raw_value, Mapping):
        raise ValueError("Eligibility must be a mapping.")

    return ResolvedEligibility(
        not_null=tuple(_coerce_string_list(raw_value.get("not_null"), field_name="eligibility.not_null")),
        min_values=_coerce_numeric_mapping(raw_value.get("min_values"), field_name="eligibility.min_values"),
        max_values=_coerce_numeric_mapping(raw_value.get("max_values"), field_name="eligibility.max_values"),
        equals=dict(raw_value.get("equals") or {}),
        expressions=tuple(_coerce_string_list(raw_value.get("expressions"), field_name="eligibility.expressions")),
    )


def _apply_numeric_thresholds(
    metadata_df: pd.DataFrame,
    *,
    min_values: Mapping[str, float],
    max_values: Mapping[str, float],
) -> np.ndarray:
    mask = np.ones((len(metadata_df),), dtype=bool)
    for field_name, threshold in min_values.items():
        series = _coerce_numeric_series(metadata_df, field_name, filter_name="minimum threshold")
        mask &= series.ge(threshold).to_numpy(dtype=bool)
    for field_name, threshold in max_values.items():
        series = _coerce_numeric_series(metadata_df, field_name, filter_name="maximum threshold")
        mask &= series.le(threshold).to_numpy(dtype=bool)
    return mask


def _apply_equals_filters(metadata_df: pd.DataFrame, equals: Mapping[str, Any]) -> np.ndarray:
    mask = np.ones((len(metadata_df),), dtype=bool)
    for field_name, expected_value in equals.items():
        if field_name not in metadata_df.columns:
            raise ValueError(f"Eligibility filtering requested missing metadata column: {field_name}")
        mask &= metadata_df.loc[:, field_name].eq(expected_value).to_numpy(dtype=bool)
    return mask


def _apply_expressions(metadata_df: pd.DataFrame, expressions: Iterable[str]) -> np.ndarray:
    mask = np.ones((len(metadata_df),), dtype=bool)
    for expression in expressions:
        try:
            result = metadata_df.eval(expression, engine="python")
        except Exception as exc:
            raise ValueError(f"Unable to evaluate eligibility expression `{expression}`.") from exc
        if not isinstance(result, pd.Series):
            raise ValueError(f"Eligibility expression `{expression}` did not produce a row-wise result.")
        mask &= result.fillna(False).astype(bool).to_numpy(dtype=bool)
    return mask


def _coerce_string_list(value: Any, *, field_name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field_name} must be a list of strings when provided.")
    return list(value)


def _coerce_numeric_mapping(value: Any, *, field_name: str) -> dict[str, float]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping when provided.")
    coerced: dict[str, float] = {}
    for key, raw_number in value.items():
        if not isinstance(key, str):
            raise ValueError(f"{field_name} keys must be strings.")
        try:
            coerced[key] = float(raw_number)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name}.{key} must be numeric.") from exc
    return coerced


def _coerce_numeric_series(metadata_df: pd.DataFrame, field_name: str, *, filter_name: str) -> pd.Series:
    if field_name not in metadata_df.columns:
        raise ValueError(f"Eligibility {filter_name} requested missing metadata column: {field_name}")
    series = metadata_df.loc[:, field_name]
    if not pd.api.types.is_numeric_dtype(series):
        raise ValueError(f"Eligibility {filter_name} requires numeric metadata column: {field_name}")
    return series
