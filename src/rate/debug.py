from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

import pandas as pd


def get_debug_max_files(config: Mapping[str, Any]) -> int | None:
    """Return configured debug file limit, or ``None`` when disabled."""
    debug_cfg = config.get("debug", {})
    if debug_cfg is None or not isinstance(debug_cfg, Mapping):
        return None

    enabled = bool(debug_cfg.get("enabled", False))
    raw_limit = debug_cfg.get("max_files")
    if raw_limit is None:
        return None

    limit = int(raw_limit)
    if limit < 1:
        return None
    if not enabled:
        return None
    return limit


def limit_dataframe_by_debug_max_files(
    df: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    key_columns: Sequence[str] = ("dyad_id", "subject_id", "run"),
    logger: logging.Logger | None = None,
    label: str = "rows",
) -> pd.DataFrame:
    """Limit a dataframe to the first N unique file-like units in debug mode."""
    limit = get_debug_max_files(config)
    if limit is None or df.empty:
        return df

    keys = [col for col in key_columns if col in df.columns]
    if not keys:
        limited = df.head(limit).copy()
        if logger is not None and len(limited) != len(df):
            logger.info("Debug mode active: limiting %s from %d to %d rows.", label, len(df), len(limited))
        return limited

    # Prefer processable units when a status column exists so debug mode
    # does not accidentally select only excluded/missing rows.
    unique_source = df
    if "status" in df.columns:
        status_norm = df["status"].astype(str).str.lower().str.strip()
        ok_df = df.loc[status_norm == "ok"].copy()
        if not ok_df.empty:
            unique_source = ok_df

    unique_units = unique_source.loc[:, keys].drop_duplicates(keep="first")
    if len(unique_units) <= limit:
        return df

    keep_keys = {
        tuple(row)
        for row in unique_units.head(limit).itertuples(index=False, name=None)
    }
    mask = [
        tuple(row) in keep_keys
        for row in df.loc[:, keys].itertuples(index=False, name=None)
    ]
    limited = df.loc[mask].copy()

    if logger is not None:
        logger.info(
            "Debug mode active: limiting %s to first %d file(s) (%d -> %d rows).",
            label,
            limit,
            len(df),
            len(limited),
        )
    return limited


def filter_dataframe_to_unit_keys(
    df: pd.DataFrame,
    units_df: pd.DataFrame,
    *,
    key_columns: Sequence[str] = ("dyad_id", "subject_id", "run"),
) -> pd.DataFrame:
    """Filter ``df`` to rows matching file-like keys present in ``units_df``."""
    if df.empty or units_df.empty:
        return df.iloc[0:0].copy()

    keys = [col for col in key_columns if col in df.columns and col in units_df.columns]
    if not keys:
        return df.copy()

    keep_keys = {
        tuple(row)
        for row in units_df.loc[:, keys].drop_duplicates(keep="first").itertuples(index=False, name=None)
    }
    mask = [
        tuple(row) in keep_keys
        for row in df.loc[:, keys].itertuples(index=False, name=None)
    ]
    return df.loc[mask].copy()
