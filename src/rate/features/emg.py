import numpy as np
import mne
import re
from pathlib import Path
from typing import Sequence, Tuple


def _normalize_channel_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def _resolve_emg_channels(
    ch_names: Sequence[str],
    requested: Sequence[str],
    patterns: Sequence[str],
) -> list[str]:
    available = [str(ch) for ch in ch_names]
    wanted = [str(ch) for ch in requested if str(ch).strip()]
    if not available:
        return []

    # 1) Exact
    exact = [ch for ch in wanted if ch in available]
    if exact:
        return exact

    # 2) Normalized exact (e.g., "EMG 1" vs "EMG1")
    norm_map = {_normalize_channel_name(ch): ch for ch in available}
    norm_match: list[str] = []
    for ch in wanted:
        key = _normalize_channel_name(ch)
        if key in norm_map:
            norm_match.append(norm_map[key])
    if norm_match:
        seen = set()
        return [ch for ch in norm_match if not (ch in seen or seen.add(ch))]

    # 3) Pattern fallback
    compiled = [re.compile(pat, flags=re.IGNORECASE) for pat in patterns if str(pat).strip()]
    if not compiled:
        return []
    out = [ch for ch in available if any(p.search(ch) for p in compiled)]
    return out


def compute_emg_envelope(
    eeg_path: Path,
    target_sfreq: float,
    emg_channels: Sequence[str] = ("EMG1", "EMG2"),
    emg_channel_patterns: Sequence[str] = (r"\bEMG\b", r"\bEXG\b", r"masseter", r"jaw"),
    missing_policy: str = "error",
    l_freq: float = 30.0,
    h_freq: float = 150.0,
    log_transform: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute broadband EMG envelope and resample to TRF feature sampling rate.

    Parameters
    ----------
    eeg_path : Path
        Path to EEG file (EDF/FIF/BDF/VHDR supported by MNE).
    target_sfreq : float
        Sampling rate to match TRF feature space.
    emg_channels : Sequence[str]
        EMG channel names to combine with RMS.
    emg_channel_patterns : Sequence[str]
        Regex patterns used when exact channels are unavailable.
    missing_policy : str
        What to do when no EMG-like channels are found: "error" or "zeros".
    l_freq : float
        High-pass cutoff (Hz).
    h_freq : float
        Low-pass cutoff (Hz).
    log_transform : bool
        Apply log1p transform to envelope.

    Returns
    -------
    times : ndarray
        Time vector (seconds) at target_sfreq.
    emg_env : ndarray
        EMG envelope at target_sfreq.

    Usage example
    -------------
        times, emg = compute_emg_envelope(
            Path("sub-014_task-conversation_run-2_eeg.edf"),
            target_sfreq=32.0,
        )
    """
    raw = mne.io.read_raw(eeg_path, preload=True, verbose="ERROR")

    wanted = [str(ch) for ch in emg_channels]
    if not wanted:
        raise ValueError("emg_channels must be non-empty.")
    selected = _resolve_emg_channels(raw.ch_names, requested=wanted, patterns=emg_channel_patterns)
    if not selected:
        if str(missing_policy).lower() == "zeros":
            info = mne.create_info(
                ch_names=["EMG_RMS"],
                sfreq=raw.info["sfreq"],
                ch_types=["misc"],
            )
            emg_raw = mne.io.RawArray(np.zeros((1, raw.n_times), dtype=float), info)
            emg_raw.resample(target_sfreq, npad="auto")
            return emg_raw.times, emg_raw.get_data()[0]
        raise ValueError(f"EMG channels not found. Requested={wanted}, patterns={list(emg_channel_patterns)}")

    raw.pick(selected)

    # RMS combine channels
    data = raw.get_data()
    emg_rms = np.sqrt(np.mean(data ** 2, axis=0))

    info = mne.create_info(
        ch_names=["EMG_RMS"],
        sfreq=raw.info["sfreq"],
        ch_types=["misc"],
    )

    emg_raw = mne.io.RawArray(emg_rms[np.newaxis, :], info)

    # Bandpass in native sampling space
    emg_raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)

    # Hilbert envelope
    emg_raw.apply_hilbert(envelope=True)

    emg_env = emg_raw.get_data()[0]

    if log_transform:
        emg_env = np.log1p(emg_env)

    # Now resample to TRF feature rate
    emg_raw_resampled = mne.io.RawArray(
        emg_env[np.newaxis, :],
        mne.create_info(ch_names=["EMG_RMS"], sfreq=raw.info["sfreq"], ch_types=["misc"]),
    )
    emg_raw_resampled.resample(target_sfreq, npad="auto")

    emg_env_resampled = emg_raw_resampled.get_data()[0]
    times_resampled = emg_raw_resampled.times

    return times_resampled, emg_env_resampled
