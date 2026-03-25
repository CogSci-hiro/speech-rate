"""Epoch metadata helpers."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import pandas as pd


def prepare_epoch_event_rows(event_rows: pd.DataFrame) -> pd.DataFrame:
    """Add epoch-facing timing aliases derived from the canonical events table."""
    prepared = event_rows.copy()
    if "t_onset" in prepared.columns:
        prepared["self_ipu_onset"] = pd.to_numeric(prepared["t_onset"], errors="coerce")
        if "latency" in prepared.columns:
            prepared["other_ipu_offset"] = pd.to_numeric(prepared["t_onset"], errors="coerce") - pd.to_numeric(
                prepared["latency"],
                errors="coerce",
            )
    return prepared


def build_epoch_metadata(
    event_rows: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    epoch_samples: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Build the metadata table attached to epochs.

    Parameters
    ----------
    event_rows
        Event-table rows that survived selection and bounds checks.
    config
        Resolved repository config.
    epoch_samples
        Optional absolute sample indices used for the final epochs array.

    Returns
    -------
    pandas.DataFrame
        Metadata rows aligned 1:1 with the final epochs order.
    """
    prepared_rows = prepare_epoch_event_rows(event_rows)
    epochs_cfg = dict(config.get("epochs") or {})
    selected_columns = epochs_cfg.get("metadata_columns")

    if selected_columns:
        keep_columns = [column for column in selected_columns if column in prepared_rows.columns]
        metadata_df = prepared_rows.loc[:, keep_columns].copy()
    else:
        metadata_df = prepared_rows.copy()

    metadata_df = metadata_df.drop(columns=["t_onset", "t_offset"], errors="ignore")

    if epoch_samples is not None:
        metadata_df["epoch_sample"] = list(epoch_samples)
    return metadata_df.reset_index(drop=True)
