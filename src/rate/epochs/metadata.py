"""Epoch metadata helpers."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import pandas as pd


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
    epochs_cfg = dict(config.get("epochs") or {})
    selected_columns = epochs_cfg.get("metadata_columns")

    if selected_columns:
        keep_columns = [column for column in selected_columns if column in event_rows.columns]
        metadata_df = event_rows.loc[:, keep_columns].copy()
    else:
        metadata_df = event_rows.copy()

    if epoch_samples is not None:
        metadata_df["epoch_sample"] = list(epoch_samples)
    return metadata_df.reset_index(drop=True)
