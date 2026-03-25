"""Eligibility helpers for trial/epoch filtering."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def select_eligible_epochs(metadata_df: pd.DataFrame, *, required_columns: Iterable[str] | None = None) -> np.ndarray:
    """Return a minimal boolean eligibility mask for epoch metadata.

    Parameters
    ----------
    metadata_df
        Epoch metadata table.
    required_columns
        Columns that must be non-missing to count as eligible.

    Returns
    -------
    numpy.ndarray
        Boolean mask aligned to ``metadata_df`` rows.
    """
    if metadata_df.empty:
        return np.zeros((0,), dtype=bool)

    required = list(required_columns or [])
    if not required:
        return np.ones((len(metadata_df),), dtype=bool)

    existing_columns = [column for column in required if column in metadata_df.columns]
    if not existing_columns:
        return np.ones((len(metadata_df),), dtype=bool)

    return metadata_df.loc[:, existing_columns].notna().all(axis=1).to_numpy(dtype=bool)
