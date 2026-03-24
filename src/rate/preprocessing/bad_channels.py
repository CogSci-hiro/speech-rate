"""Bad-channel loading and interpolation utilities."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

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
        return []

    name_column = "name" if "name" in table.columns else "channel_name" if "channel_name" in table.columns else None
    if name_column is None:
        raise ValueError(f"{path} is missing a channel-name column ('name' or 'channel_name').")
    if "status" not in table.columns:
        raise ValueError(f"{path} is missing the required BIDS 'status' column.")

    bad_rows = table.loc[table["status"].astype(str).str.strip().str.lower() == "bad"].copy()
    if bad_rows.empty:
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
        if not matched_names:
            raise ValueError(message)
        warnings.warn(message, RuntimeWarning)
        LOGGER.warning(message)

    matched_unique = list(dict.fromkeys(matched_names))
    raw.info["bads"] = matched_unique
    return matched_unique


def interpolate_bad_channels(raw) -> None:
    """Interpolate channels currently listed in ``raw.info['bads']``."""
    if not raw.info["bads"]:
        return
    raw.interpolate_bads(reset_bads=False, verbose="ERROR")
