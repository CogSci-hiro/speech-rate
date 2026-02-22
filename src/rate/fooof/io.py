from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rate.fooof.spec import FooofSpec
from rate.util import write_parquet_dataframe


def _fooof_dir(out_dir: Path) -> Path:
    return Path(out_dir) / "fooof"


def _trial_level_dir(out_dir: Path) -> Path:
    return _fooof_dir(out_dir) / "trial_level"


def _aggregated_dir(out_dir: Path) -> Path:
    return _fooof_dir(out_dir) / "aggregated"


def _meta_dir(out_dir: Path) -> Path:
    return _fooof_dir(out_dir) / "meta"


def write_trial_channels(df: pd.DataFrame, out_dir: Path) -> None:
    path = _trial_level_dir(out_dir) / "fooof_trial_channels.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_parquet_dataframe(df, path)


def write_trial_rois(df: pd.DataFrame, out_dir: Path) -> None:
    path = _trial_level_dir(out_dir) / "fooof_trial_rois.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def write_qc_summary(df: pd.DataFrame, out_dir: Path) -> None:
    path = _meta_dir(out_dir) / "qc_summary.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = [{"metric": "n_rows", "value": int(len(df))}]
    if not df.empty:
        bool_cols: list[str] = []
        for col in df.columns:
            series = df[col]
            if pd.api.types.is_bool_dtype(series):
                bool_cols.append(col)
                continue
            nonnull = series.dropna()
            if nonnull.empty:
                continue
            if nonnull.map(lambda x: isinstance(x, (bool, np.bool_))).all():
                bool_cols.append(col)
        for col in sorted(bool_cols):
            rows.append({"metric": f"{col}_count", "value": int(df[col].fillna(False).sum())})
            rows.append({"metric": f"{col}_rate", "value": float(df[col].fillna(False).mean())})

        for col in ("offset", "exponent", "r_squared", "error"):
            if col in df.columns:
                s = pd.to_numeric(df[col], errors="coerce")
                rows.append({"metric": f"{col}_nonnull", "value": int(s.notna().sum())})
                rows.append({"metric": f"{col}_mean", "value": float(s.mean()) if s.notna().any() else float("nan")})

    summary_df = pd.DataFrame.from_records(rows)
    write_parquet_dataframe(summary_df, path)


def write_settings(spec: FooofSpec, out_dir: Path) -> None:
    path = _meta_dir(out_dir) / "fooof_settings.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(asdict(spec), f, indent=2, sort_keys=True)


def write_roi_lmm_table(df: pd.DataFrame, out_dir: Path) -> None:
    path = _aggregated_dir(out_dir) / "roi_level_for_lmm.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_parquet_dataframe(df, path)


def write_channel_ddc_table(df: pd.DataFrame, out_dir: Path) -> None:
    path = _aggregated_dir(out_dir) / "channel_level_for_ddc.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_parquet_dataframe(df, path)
