"""Channel-selection helpers for preprocessing."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ChannelSplit:
    """Description of separated channel groups."""

    eeg_names: list[str]
    emg_names: list[str]
    dropped_names: list[str]


def _as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _compile_patterns(patterns: Sequence[str]) -> list[re.Pattern[str]]:
    return [re.compile(pattern, flags=re.IGNORECASE) for pattern in patterns]


def _match_any(name: str, patterns: Sequence[re.Pattern[str]]) -> bool:
    return any(pattern.search(name) for pattern in patterns)


def split_eeg_and_emg_channels(raw, config: Mapping[str, Any]) -> ChannelSplit:
    """Split a raw object into EEG and EMG channel groups.

    Parameters
    ----------
    raw
        MNE raw object.
    config
        Resolved configuration mapping.

    Returns
    -------
    ChannelSplit
        Channel names for EEG, EMG, and dropped channels.
    """
    preprocessing_cfg = dict(config.get("preprocessing") or {})
    channel_cfg = dict(preprocessing_cfg.get("channels") or {})

    emg_exact_names = {name.strip() for name in _as_string_list(channel_cfg.get("emg_names")) if name.strip()}
    emg_patterns = _compile_patterns(_as_string_list(channel_cfg.get("emg_patterns")))
    eeg_exact_names = {name.strip() for name in _as_string_list(channel_cfg.get("eeg_names")) if name.strip()}
    eeg_patterns = _compile_patterns(_as_string_list(channel_cfg.get("eeg_name_patterns")))
    drop_exact_names = {name.strip() for name in _as_string_list(channel_cfg.get("drop_names")) if name.strip()}

    channel_type_map: dict[str, str] = {}
    emg_names: list[str] = []

    for channel_name in raw.ch_names:
        if channel_name in emg_exact_names or _match_any(channel_name, emg_patterns):
            channel_type_map[channel_name] = "emg"
            emg_names.append(channel_name)
    if channel_type_map:
        raw.set_channel_types(channel_type_map, on_unit_change="ignore")

    eeg_names = [name for name in raw.copy().pick("eeg").ch_names]
    if not eeg_names:
        for channel_name in raw.ch_names:
            if channel_name in emg_names or channel_name in drop_exact_names:
                continue
            if channel_name in eeg_exact_names or _match_any(channel_name, eeg_patterns):
                eeg_names.append(channel_name)

    if not eeg_names:
        raise ValueError("No EEG channels could be identified for preprocessing.")

    keep_names = set(eeg_names) | set(emg_names)
    dropped_names = [name for name in raw.ch_names if name not in keep_names]
    return ChannelSplit(
        eeg_names=sorted(eeg_names, key=raw.ch_names.index),
        emg_names=sorted(set(emg_names), key=raw.ch_names.index),
        dropped_names=dropped_names,
    )


def apply_montage(raw, config: Mapping[str, Any]) -> None:
    """Attach a configured montage to the EEG raw object."""
    import mne

    preprocessing_cfg = dict(config.get("preprocessing") or {})
    montage_cfg = preprocessing_cfg.get("montage", preprocessing_cfg.get("montage_name"))

    if isinstance(montage_cfg, Mapping):
        montage_name = str(montage_cfg.get("name", "standard_1020"))
        on_missing = str(montage_cfg.get("on_missing", "ignore"))
        match_case = bool(montage_cfg.get("match_case", False))
    else:
        montage_name = str(montage_cfg or "standard_1020")
        on_missing = "ignore"
        match_case = False

    montage = mne.channels.make_standard_montage(montage_name)
    raw.set_montage(montage, on_missing=on_missing, match_case=match_case)
