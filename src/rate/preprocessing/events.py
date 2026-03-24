"""Event extraction helpers for preprocessing."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd


def extract_events_table(raw, config: Mapping[str, Any]) -> tuple[pd.DataFrame, dict[str, int]]:
    """Extract events from a raw object and return a tabular representation.

    Parameters
    ----------
    raw
        MNE raw object.
    config
        Resolved configuration mapping.

    Returns
    -------
    tuple[pandas.DataFrame, dict[str, int]]
        Event table and optional event-id mapping.
    """
    import mne

    preprocessing_cfg = dict(config.get("preprocessing") or {})
    events_cfg = dict(preprocessing_cfg.get("events") or {})
    stim_channel = events_cfg.get("stim_channel")
    shortest_event = int(events_cfg.get("shortest_event", 1))
    consecutive = bool(events_cfg.get("consecutive", True))

    try:
        events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
    except ValueError:
        events = np.empty((0, 3), dtype=int)
        event_id = {}

    if events.size == 0:
        try:
            kwargs: dict[str, Any] = {"shortest_event": shortest_event, "consecutive": consecutive, "verbose": "ERROR"}
            if stim_channel:
                kwargs["stim_channel"] = stim_channel
            events = mne.find_events(raw, **kwargs)
        except ValueError:
            events = np.empty((0, 3), dtype=int)
        event_id = {}

    if events.size == 0:
        table = pd.DataFrame(columns=["sample", "time_s", "previous", "event_id"])
        return table, event_id

    sfreq = float(raw.info["sfreq"])
    table = pd.DataFrame(
        {
            "sample": events[:, 0].astype(int),
            "time_s": events[:, 0].astype(float) / sfreq,
            "previous": events[:, 1].astype(int),
            "event_id": events[:, 2].astype(int),
        }
    )
    return table, event_id
