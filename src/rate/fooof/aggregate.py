from __future__ import annotations

import numpy as np
import pandas as pd

from rate.fooof.spec import FooofSpec

TRIAL_KEY_COLS = ["dyad_id", "subject_id", "run", "event_id"]
WINDOW_COLS = ["window_start", "window_end", "window_length"]


def _mean_or_nan(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() == 0:
        return np.nan
    return float(numeric.mean())


def _cf_mean_with_coverage(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce")
    if len(numeric) == 0:
        return np.nan
    coverage = float(numeric.notna().mean())
    if coverage < 0.5:
        return np.nan
    return _mean_or_nan(numeric)


def aggregate_to_rois(
    trial_channels_df: pd.DataFrame,
    spec: FooofSpec,
) -> pd.DataFrame:
    """Aggregate channel-level trial features to configured ROIs."""
    if trial_channels_df.empty:
        return pd.DataFrame(
            columns=[
                *TRIAL_KEY_COLS,
                "roi",
                *WINDOW_COLS,
                "n_channels_total",
                "n_channels_observed",
                "n_fit_success",
                "fit_success_rate",
                "offset",
                "exponent",
                "knee",
                "theta_power",
                "alpha_power",
                "beta_power",
                "theta_cf",
                "alpha_cf",
                "beta_cf",
                "r_squared",
                "error",
            ]
        )

    records: list[dict[str, object]] = []
    roi_members = {roi: set(chs) for roi, chs in spec.rois.items()}

    for (dyad_id, subject_id, run, event_id), trial_df in trial_channels_df.groupby(TRIAL_KEY_COLS, sort=False, dropna=False):
        for roi_name, members in roi_members.items():
            roi_df = trial_df.loc[trial_df["channel"].astype(str).isin(members)].copy()

            record: dict[str, object] = {
                "dyad_id": dyad_id,
                "subject_id": subject_id,
                "run": run,
                "event_id": event_id,
                "roi": roi_name,
                "n_channels_total": len(members),
                "n_channels_observed": int(len(roi_df)),
                "n_fit_success": 0,
            }
            if not roi_df.empty and "fit_success" in roi_df.columns:
                fit_success_s = roi_df["fit_success"].astype(bool)
                record["n_fit_success"] = int(fit_success_s.sum())

            for col in WINDOW_COLS:
                record[col] = trial_df.iloc[0][col] if col in trial_df.columns and not trial_df.empty else np.nan

            denom = float(record["n_channels_observed"]) if float(record["n_channels_observed"]) > 0 else np.nan
            record["fit_success_rate"] = (
                float(record["n_fit_success"]) / denom if np.isfinite(denom) else np.nan
            )

            for col in ("offset", "exponent", "knee", "theta_power", "alpha_power", "beta_power", "r_squared", "error"):
                record[col] = _mean_or_nan(roi_df[col]) if (not roi_df.empty and col in roi_df.columns) else np.nan

            for col in ("theta_cf", "alpha_cf", "beta_cf"):
                record[col] = _cf_mean_with_coverage(roi_df[col]) if (not roi_df.empty and col in roi_df.columns) else np.nan

            records.append(record)

    out = pd.DataFrame.from_records(records)
    if out.empty:
        return out

    out = out.sort_values(by=[*TRIAL_KEY_COLS, "roi"], kind="mergesort").reset_index(drop=True)
    return out


def prepare_ddc_channel_table(
    trial_channels_df: pd.DataFrame,
) -> pd.DataFrame:
    """Reorder/select the channel-level FOOOF table for downstream DDC modules."""
    preferred_cols = [
        "dyad_id",
        "subject_id",
        "run",
        "event_id",
        "channel",
        "window_start",
        "window_end",
        "window_length",
        "offset",
        "exponent",
        "knee",
        "theta_power",
        "alpha_power",
        "beta_power",
        "theta_cf",
        "alpha_cf",
        "beta_cf",
        "r_squared",
        "error",
        "fit_success",
    ]
    qc_cols = [
        "nonfinite_aperiodic",
        "nonfinite_diagnostics",
        "extreme_exponent",
        "extreme_offset",
        "low_r_squared",
        "high_error",
        "qc_any_flag",
    ]
    cols = [col for col in [*preferred_cols, *qc_cols] if col in trial_channels_df.columns]
    out = trial_channels_df.loc[:, cols].copy()
    sort_cols = [col for col in ("dyad_id", "subject_id", "run", "event_id", "channel") if col in out.columns]
    if sort_cols:
        out = out.sort_values(by=sort_cols, kind="mergesort").reset_index(drop=True)
    return out
