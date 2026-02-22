from __future__ import annotations

from collections.abc import Iterator

import mne
import numpy as np
import pandas as pd

from rate.fooof.spec import FooofSpec


def _run_token(value: object) -> str:
    text = str(value).strip()
    try:
        return str(int(float(text)))
    except ValueError:
        return text


def _event_sample_bounds(event_onset_s: float, sfreq: float, spec: FooofSpec) -> tuple[int, int]:
    start_s = event_onset_s + spec.window_tmin_s
    end_s = event_onset_s + spec.window_tmax_s
    start = int(np.round(start_s * sfreq))
    stop = int(np.round(end_s * sfreq))
    return start, stop


def iter_event_windows(
    raw: mne.io.BaseRaw,
    events_df: pd.DataFrame,
    spec: FooofSpec,
    subject_id: str,
    run: int,
) -> Iterator[tuple[str, int, str, str, np.ndarray]]:
    """
    Yield channel-level pre-speech windows for one subject/run.

    Yields
    ------
    tuple
        (subject_id, run, event_id, channel_name, window_data_array)
    """
    if events_df.empty:
        return

    required = {"subject_id", "run", "event_id", "t_onset"}
    missing = required - set(events_df.columns)
    if missing:
        raise ValueError(f"events_df missing required columns for window extraction: {sorted(missing)}")

    run_key = _run_token(run)
    event_rows = events_df.loc[
        (events_df["subject_id"].astype(str) == str(subject_id))
        & (events_df["run"].map(_run_token) == run_key)
    ].copy()
    if event_rows.empty:
        return

    event_rows = event_rows.sort_values(
        by=["t_onset", "event_id"],
        kind="mergesort",
    ).reset_index(drop=True)

    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    if len(picks) == 0:
        raise ValueError("Raw object contains no EEG channels for FOOOF window extraction.")

    sfreq = float(raw.info["sfreq"])
    channel_names = [raw.ch_names[idx] for idx in picks]

    for row in event_rows.itertuples(index=False):
        event_onset_s = float(getattr(row, "t_onset"))
        start, stop = _event_sample_bounds(event_onset_s, sfreq, spec)

        if start < 0 or stop > raw.n_times or stop <= start:
            continue

        window_data = raw.get_data(picks=picks, start=start, stop=stop)
        for channel_name, channel_data in zip(channel_names, window_data, strict=True):
            yield (str(subject_id), int(run), str(getattr(row, "event_id")), channel_name, np.asarray(channel_data))
