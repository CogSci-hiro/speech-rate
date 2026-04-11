"""Trial-data preparation helpers for lmeEEG."""

from __future__ import annotations

from dataclasses import dataclass, replace
import logging
from typing import Any, Mapping

import numpy as np
import pandas as pd

from rate.design import DesignMatrixResult, build_design_matrix, load_model_spec

LOGGER = logging.getLogger(__name__)
DEFAULT_LMEEEG_SECTION = "lmeeeg"

_LEGACY_METADATA_ALIASES: dict[str, str] = {
    "proximity": "alignment",
    "proximity_sq": "alignment_sq",
}


@dataclass(frozen=True)
class LMEEEGTrialData:
    """Trial-level metadata and aligned EEG rows for the lmeEEG branch."""

    design_result: DesignMatrixResult
    trial_metadata: pd.DataFrame
    row_index: np.ndarray
    group_column: str
    eeg_data: np.ndarray
    channel_names: list[str]
    times: np.ndarray


def prepare_lmeeeg_design(
    metadata_df: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    config_section: str = DEFAULT_LMEEEG_SECTION,
    model_name: str | None = None,
) -> DesignMatrixResult:
    """Build the fixed-effect design matrix for the lmeEEG branch."""
    metadata_df = _apply_legacy_metadata_aliases(metadata_df)
    model_spec = load_model_spec(config, section=config_section, model_name=model_name)
    design_spec = _make_rhs_only_model_spec(model_spec)
    return build_design_matrix(metadata_df, model_spec=design_spec)


def build_lmeeeg_trial_table(
    epochs,
    metadata_df: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    config_section: str = DEFAULT_LMEEEG_SECTION,
    model_name: str | None = None,
) -> LMEEEGTrialData:
    """Prepare the filtered trial-level table aligned to EEG epochs.

    Parameters
    ----------
    epochs
        MNE epochs object with trialwise EEG data.
    metadata_df
        Epoch metadata table aligned to ``epochs`` rows.
    config
        Resolved repository config.
    model_name
        lmeEEG model name to resolve from config.
    """
    return build_lmeeeg_trial_data_from_arrays(
        eeg_data=epochs.get_data(copy=True),
        metadata_df=metadata_df,
        channel_names=list(epochs.ch_names),
        times=np.asarray(epochs.times, dtype=float),
        config=config,
        config_section=config_section,
        model_name=model_name,
    )


def build_lmeeeg_trial_data_from_arrays(
    *,
    eeg_data: np.ndarray,
    metadata_df: pd.DataFrame,
    channel_names: list[str],
    times: np.ndarray,
    config: Mapping[str, Any],
    config_section: str = DEFAULT_LMEEEG_SECTION,
    model_name: str | None = None,
) -> LMEEEGTrialData:
    """Prepare filtered trial data from pooled EEG arrays and metadata."""
    if int(np.asarray(eeg_data).shape[0]) != len(metadata_df):
        raise ValueError("EEG rows and metadata rows must match before lmeEEG design filtering.")

    prepared_metadata_df = _apply_legacy_metadata_aliases(metadata_df)
    design_result = prepare_lmeeeg_design(prepared_metadata_df, config, config_section=config_section, model_name=model_name)
    group_column = resolve_random_effect_group(config, config_section=config_section, model_name=model_name)
    if group_column not in prepared_metadata_df.columns:
        raise ValueError(f"Configured lmeEEG random-effect group column `{group_column}` is missing from metadata.")

    row_index = design_result.row_index.to_numpy(dtype=int, copy=True)
    trial_metadata = prepared_metadata_df.iloc[row_index].copy().reset_index(drop=True)
    if trial_metadata[group_column].isna().any():
        raise ValueError(f"Configured lmeEEG random-effect group column `{group_column}` contains missing values.")

    eeg_array = np.asarray(eeg_data, dtype=float)
    filtered_eeg_data = eeg_array[row_index, :, :]
    if filtered_eeg_data.shape[0] != design_result.design_matrix.shape[0]:
        raise ValueError("Filtered EEG rows and lmeEEG design rows do not align.")

    LOGGER.info(
        "Prepared lmeEEG trials for model=%s: input=%d, after_eligibility=%d, after_missing=%d, final=%d",
        model_name or "<default>",
        int(design_result.n_rows_input),
        int(design_result.n_rows_after_eligibility),
        int(design_result.n_rows_after_missing),
        int(design_result.n_rows_final),
    )

    return LMEEEGTrialData(
        design_result=design_result,
        trial_metadata=trial_metadata,
        row_index=row_index,
        group_column=group_column,
        eeg_data=filtered_eeg_data,
        channel_names=list(channel_names),
        times=np.asarray(times, dtype=float),
    )


def resolve_random_effect_group(
    config: Mapping[str, Any],
    *,
    config_section: str = DEFAULT_LMEEEG_SECTION,
    model_name: str | None = None,
) -> str:
    """Resolve the grouping column for the random-intercept stage."""
    lmeeeg_cfg = dict(config.get(config_section) or {})
    model_cfg = {}
    if model_name is not None:
        models_cfg = dict(lmeeeg_cfg.get("models") or {})
        model_cfg = dict(models_cfg.get(model_name) or {})

    if model_cfg.get("random_effect_group"):
        return str(model_cfg["random_effect_group"])

    random_effects_cfg = lmeeeg_cfg.get("random_effects") or {}
    if isinstance(random_effects_cfg, Mapping) and random_effects_cfg.get("group"):
        return str(random_effects_cfg["group"])

    return str(lmeeeg_cfg.get("group_column", "subject_id"))


def resolve_random_effect_structure(
    config: Mapping[str, Any],
    *,
    config_section: str = DEFAULT_LMEEEG_SECTION,
    model_name: str | None = None,
) -> str:
    """Resolve and validate the supported random-effects structure."""
    lmeeeg_cfg = dict(config.get(config_section) or {})
    model_cfg = {}
    if model_name is not None:
        models_cfg = dict(lmeeeg_cfg.get("models") or {})
        model_cfg = dict(models_cfg.get(model_name) or {})

    structure = str(
        model_cfg.get(
            "random_effect_structure",
            (lmeeeg_cfg.get("random_effects") or {}).get("structure", "random_intercept")
            if isinstance(lmeeeg_cfg.get("random_effects"), Mapping)
            else "random_intercept",
        )
    )
    if structure != "random_intercept":
        raise ValueError(f"Unsupported lmeEEG random-effect structure `{structure}`. Only `random_intercept` is implemented.")
    return structure


def _make_rhs_only_model_spec(model_spec):
    if "~" not in model_spec.formula:
        return model_spec
    lhs, rhs = model_spec.formula.split("~", maxsplit=1)
    if not lhs.strip():
        return model_spec
    return replace(model_spec, formula=f"~ {rhs.strip()}", outcome=None)


def _apply_legacy_metadata_aliases(metadata_df: pd.DataFrame) -> pd.DataFrame:
    """Backfill renamed metadata columns so older cached epoch tables remain readable."""
    aliased = metadata_df.copy()
    for target_column, source_column in _LEGACY_METADATA_ALIASES.items():
        if target_column in aliased.columns or source_column not in aliased.columns:
            continue
        aliased.loc[:, target_column] = aliased[source_column]
    return aliased
