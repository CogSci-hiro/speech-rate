"""Shared eligibility helpers used by design construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

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


_ELIGIBILITY_COLUMN_GROUPS: dict[str, tuple[tuple[str, ...], ...]] = {
    "self_rate": (("self_rate", "speech_rate"),),
    "alignment": (("self_rate", "speech_rate"), ("other_rate",),),
    "convergence": (
        ("convergence",),
        ("baseline_self_rate",),
        ("baseline_other_rate",),
    ),
}


def apply_eligibility(
    metadata_df: pd.DataFrame,
    model_spec: ModelSpec | None = None,
    *,
    eligibility: str | None = None,
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
            rule_name=eligibility or getattr(model_spec, "eligibility", None),
            n_rows_input=0,
            n_rows_selected=0,
        )

    resolved_rule = eligibility or getattr(model_spec, "eligibility", None)
    rule_columns = _resolve_rule_columns(metadata_df, resolved_rule)
    columns_to_check = tuple(dict.fromkeys([*rule_columns, *(required_columns or [])]))
    mask = np.ones((len(metadata_df),), dtype=bool)
    if columns_to_check:
        missing_columns = [column for column in columns_to_check if column not in metadata_df.columns]
        if missing_columns:
            missing_text = ", ".join(sorted(missing_columns))
            raise ValueError(f"Eligibility filtering requested missing metadata columns: {missing_text}")
        mask &= metadata_df.loc[:, list(columns_to_check)].notna().all(axis=1).to_numpy(dtype=bool)

    return EligibilityResult(
        row_mask=mask,
        applied_columns=columns_to_check,
        rule_name=resolved_rule,
        n_rows_input=len(metadata_df),
        n_rows_selected=int(mask.sum()),
    )


def apply_trial_eligibility(*args, **kwargs) -> EligibilityResult:
    """Backward-compatible alias for the shared design eligibility helper."""
    return apply_eligibility(*args, **kwargs)


def _resolve_rule_columns(metadata_df: pd.DataFrame, rule_name: str | None) -> tuple[str, ...]:
    if not rule_name:
        return ()
    if rule_name not in _ELIGIBILITY_COLUMN_GROUPS:
        raise ValueError(f"Unsupported eligibility rule `{rule_name}`.")

    resolved_columns: list[str] = []
    for alternatives in _ELIGIBILITY_COLUMN_GROUPS[rule_name]:
        matched = next((column for column in alternatives if column in metadata_df.columns), None)
        if matched is not None:
            resolved_columns.append(matched)
    return tuple(resolved_columns)
