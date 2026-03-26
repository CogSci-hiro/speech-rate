"""Bad-channel loading and interpolation utilities."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def load_bad_channels_from_bids_tsv(channels_tsv_path: Path | str, *, raw) -> list[str]:
    """Load bad-channel annotations from a BIDS ``channels.tsv`` file.

    Parameters
    ----------
    channels_tsv_path
        Path to the BIDS channels table.
    raw
        Raw object whose channel names should be matched.

    Returns
    -------
    list[str]
        Matched bad EEG channel names present in ``raw``.
    """
    path = Path(channels_tsv_path)
    if not path.exists():
        raise FileNotFoundError(f"BIDS channels.tsv file does not exist: {path}")

    table = pd.read_csv(path, sep="\t")
    if table.empty:
        raise ValueError(f"{path} is empty; expected BIDS channel rows with bad-channel annotations.")

    name_column = "name" if "name" in table.columns else "channel_name" if "channel_name" in table.columns else None
    if name_column is None:
        raise ValueError(f"{path} is missing a channel-name column ('name' or 'channel_name').")
    if "status" not in table.columns:
        raise ValueError(f"{path} is missing the required BIDS 'status' column.")

    bad_rows = table.loc[table["status"].astype(str).str.strip().str.lower() == "bad"].copy()
    if bad_rows.empty:
        LOGGER.info("No bad channels declared in BIDS channels.tsv: %s", path)
        raw.info["bads"] = []
        return []

    raw_lookup = {name: name for name in raw.ch_names}
    raw_lookup_lower = {name.lower(): name for name in raw.ch_names}

    matched_names: list[str] = []
    missing_names: list[str] = []
    for channel_name in bad_rows[name_column].astype(str):
        stripped = channel_name.strip()
        if not stripped:
            continue
        if stripped in raw_lookup:
            matched_names.append(raw_lookup[stripped])
            continue
        lowered = stripped.lower()
        if lowered in raw_lookup_lower:
            matched_names.append(raw_lookup_lower[lowered])
            continue
        missing_names.append(stripped)

    if missing_names:
        message = f"Bad channels in {path} were not found in raw data: {sorted(missing_names)}"
        raise ValueError(message)

    matched_unique = list(dict.fromkeys(matched_names))
    raw.info["bads"] = matched_unique
    return matched_unique


def interpolate_bad_channels(raw) -> None:
    """Interpolate channels currently listed in ``raw.info['bads']``."""
    if not raw.info["bads"]:
        LOGGER.info("No bad channels are marked on raw.info['bads']; skipping interpolation.")
        return
    invalid_position_names: list[str] = []
    bad_name_set = set(raw.info["bads"])
    channel_types = raw.get_channel_types()
    for channel_name, channel_type, channel_info in zip(raw.ch_names, channel_types, raw.info["chs"], strict=False):
        if channel_type != "eeg":
            continue
        position = np.asarray(channel_info["loc"][:3], dtype=float)
        if not np.isfinite(position).all():
            invalid_position_names.append(channel_name)

    invalid_position_names = list(dict.fromkeys(invalid_position_names))
    invalid_bad_position_names = [name for name in invalid_position_names if name in bad_name_set]

    if invalid_bad_position_names:
        message = (
            "Cannot interpolate bad EEG channels without finite sensor positions: "
            f"{invalid_bad_position_names}"
        )
        raise ValueError(message)

    raw.interpolate_bads(
        reset_bads=True,
        exclude=invalid_position_names,
        verbose="ERROR",
    )
