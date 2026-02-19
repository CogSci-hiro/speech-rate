"""Feature engineering for canonical event rows."""

import numpy as np
import pandas as pd

SUPPORTED_TIME_METHODS = {"run_duration", "minmax_0_1"}


def compute_other_rate(events_df: pd.DataFrame, window_s: float) -> tuple[pd.Series, pd.Series]:
    """Compute nearest-other-speaker rate and listener label per event."""
    other_rates: list[float] = []
    listeners: list[str | None] = []

    for row in events_df.itertuples(index=False):
        candidate_df = events_df[events_df["speaker"] != row.speaker].copy()
        if candidate_df.empty:
            other_rates.append(np.nan)
            listeners.append(None)
            continue

        candidate_df["delta_onset"] = (candidate_df["t_onset"] - float(row.t_onset)).abs()
        within_window = candidate_df[candidate_df["delta_onset"] <= window_s]
        if within_window.empty:
            # Fallback to nearest partner event so other_rate is defined when both streams exist.
            winner = candidate_df.sort_values(["delta_onset", "t_onset", "speaker"], kind="mergesort").iloc[0]
        else:
            winner = within_window.sort_values(["delta_onset", "t_onset", "speaker"], kind="mergesort").iloc[0]
        other_rates.append(float(winner["self_rate"]))
        listeners.append(str(winner["speaker"]))

    return pd.Series(other_rates, dtype="float64"), pd.Series(listeners, dtype="object")


def compute_time_within_run(group_df: pd.DataFrame, normalize: bool, method: str) -> pd.Series:
    """Compute time within run using configured normalization method."""
    if not normalize:
        return group_df["t_onset"].astype(float)

    if method not in SUPPORTED_TIME_METHODS:
        raise ValueError(f"Unsupported time normalization method: {method}")

    t_run = float(group_df["t_offset"].max())
    if t_run <= 0:
        return pd.Series(np.zeros(len(group_df), dtype=float), index=group_df.index)
    return (group_df["t_onset"].astype(float) / t_run).clip(lower=0.0, upper=1.0)


def add_event_rate_features(unit_df: pd.DataFrame, absdiff_window_s: float, normalize: bool, method: str) -> pd.DataFrame:
    """Add self_rate/other_rate/abs_diff/time_within_run features."""
    enriched = unit_df.copy()
    enriched["self_rate"] = enriched["syllables"] / enriched["duration_s"]
    enriched["other_rate"], enriched["listener"] = compute_other_rate(
        enriched,
        window_s=absdiff_window_s,
    )
    enriched["abs_diff"] = (enriched["self_rate"] - enriched["other_rate"]).abs()
    enriched["time_within_run"] = compute_time_within_run(
        enriched,
        normalize=normalize,
        method=method,
    )
    return enriched
