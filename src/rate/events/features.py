"""Feature engineering for canonical event rows."""

from __future__ import annotations

import numpy as np
import pandas as pd

SUPPORTED_TIME_METHODS = {"run_duration", "minmax_0_1"}
SUPPORTED_MATCHING_TIME_LOCKS = {"onset", "offset"}


def compute_other_rate(
    events_df: pd.DataFrame,
    window_s: float,
    time_lock: str = "onset",
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Compute nearest-other-speaker features per event."""
    if time_lock not in SUPPORTED_MATCHING_TIME_LOCKS:
        raise ValueError(f"Unsupported matching time_lock: {time_lock!r}")

    origin_col = "t_onset" if time_lock == "onset" else "t_offset"
    target_col = "t_offset" if time_lock == "onset" else "t_onset"

    other_rates: list[float] = []
    listeners: list[str | None] = []
    latencies: list[float] = []
    other_syllables: list[float] = []
    other_durations: list[float] = []

    for row in events_df.itertuples(index=False):
        candidate_df = events_df[events_df["speaker"] != row.speaker].copy()
        if candidate_df.empty:
            other_rates.append(np.nan)
            listeners.append(None)
            latencies.append(np.nan)
            other_syllables.append(np.nan)
            other_durations.append(np.nan)
            continue

        origin = float(getattr(row, origin_col))
        low = origin - window_s
        high = origin + window_s
        within_window = candidate_df[(candidate_df[target_col] >= low) & (candidate_df[target_col] < high)]
        if within_window.empty:
            other_rates.append(np.nan)
            listeners.append(None)
            latencies.append(np.nan)
            other_syllables.append(np.nan)
            other_durations.append(np.nan)
            continue

        deltas = (within_window[target_col] - origin).abs().to_numpy(dtype=float)
        winner = within_window.iloc[int(np.argmin(deltas))]
        other_rates.append(float(winner["self_rate"]))
        listeners.append(str(winner["speaker"]))
        latencies.append(origin - float(winner[target_col]))
        other_syllables.append(float(winner["syllables"]))
        other_durations.append(float(winner["duration_s"]))

    return (
        pd.Series(other_rates, dtype="float64"),
        pd.Series(listeners, dtype="object"),
        pd.Series(latencies, dtype="float64"),
        pd.Series(other_syllables, dtype="float64"),
        pd.Series(other_durations, dtype="float64"),
    )


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


def compute_mean_rate(self_rate: pd.Series, other_rate: pd.Series) -> pd.Series:
    """Compute dyadic mean speaking rate per matched event."""
    return (self_rate.astype(float) + other_rate.astype(float)) / 2.0


def add_event_rate_features(
    unit_df: pd.DataFrame,
    absdiff_window_s: float,
    normalize: bool,
    method: str,
    matching_time_lock: str = "onset",
) -> pd.DataFrame:
    """Add rate, derived behavioral, squared-rate, and time-within-run features."""
    enriched = unit_df.copy()
    enriched["self_n_syllables"] = enriched["syllables"].astype(float)
    enriched["self_ipu_duration"] = enriched["duration_s"].astype(float)
    enriched["self_rate"] = enriched["syllables"] / enriched["duration_s"]
    (
        enriched["other_rate"],
        enriched["listener"],
        enriched["latency"],
        enriched["other_n_syllables"],
        enriched["other_ipu_duration"],
    ) = compute_other_rate(
        enriched,
        window_s=absdiff_window_s,
        time_lock=matching_time_lock,
    )
    enriched["mean_rate"] = compute_mean_rate(enriched["self_rate"], enriched["other_rate"])
    enriched["self_rate_sq"] = enriched["self_rate"] ** 2
    enriched["other_rate_sq"] = enriched["other_rate"] ** 2
    enriched["time_within_run"] = compute_time_within_run(
        enriched,
        normalize=normalize,
        method=method,
    )
    return enriched
