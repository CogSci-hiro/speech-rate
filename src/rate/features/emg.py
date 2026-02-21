import numpy as np
import mne
import pandas as pd
from pathlib import Path
from typing import Tuple


def compute_emg_envelope(
    edf_path: Path,
    target_sfreq: float,
    l_freq: float = 30.0,
    h_freq: float = 150.0,
    log_transform: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute broadband EMG envelope and resample to TRF feature sampling rate.

    Parameters
    ----------
    edf_path : Path
        Path to EDF file.
    target_sfreq : float
        Sampling rate to match TRF feature space.
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
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

    if not {"EMG1", "EMG2"}.issubset(set(raw.ch_names)):
        raise ValueError("EMG1 and EMG2 channels not found.")

    raw.pick(["EMG1", "EMG2"])

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
        info,
    )
    emg_raw_resampled.resample(target_sfreq, npad="auto")

    emg_env_resampled = emg_raw_resampled.get_data()[0]
    times_resampled = emg_raw_resampled.times

    return times_resampled, emg_env_resampled
