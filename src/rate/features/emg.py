import numpy as np
import mne
import pandas as pd
from pathlib import Path
from typing import Tuple


def compute_emg_envelope(
    edf_path: Path,
    l_freq: float = 30.0,
    h_freq: float = 150.0,
    log_transform: bool = True,
    target_sfreq: float | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute broadband EMG envelope from two EMG channels in EDF file.

    Parameters
    ----------
    edf_path : Path
        Path to BIDS EDF file.
    l_freq : float
        High-pass cutoff for EMG band (Hz).
    h_freq : float
        Low-pass cutoff for EMG band (Hz).
    log_transform : bool
        Whether to apply log transform to envelope.
    target_sfreq : float | None
        If provided, resample to this sampling rate.

    Returns
    -------
    times : ndarray, shape (n_samples,)
        Time vector in seconds.
    emg_env : ndarray, shape (n_samples,)
        EMG envelope time series.

    Usage example
    -------------
        times, emg = compute_emg_envelope(
            Path("sub-014_task-conversation_run-2_eeg.edf")
        )
    """
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

    if not {"EMG1", "EMG2"}.issubset(set(raw.ch_names)):
        raise ValueError("EMG1 and EMG2 channels not found in EDF.")

    raw.pick(["EMG1", "EMG2"])

    # RMS combine channels to avoid polarity cancellation
    data = raw.get_data()
    emg_rms = np.sqrt(np.mean(data ** 2, axis=0))

    info = mne.create_info(
        ch_names=["EMG_RMS"],
        sfreq=raw.info["sfreq"],
        ch_types=["misc"],
    )

    emg_raw = mne.io.RawArray(emg_rms[np.newaxis, :], info)

    # Bandpass filter
    emg_raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)

    # Hilbert transform for analytic amplitude
    emg_raw.apply_hilbert(envelope=True)

    emg_env = emg_raw.get_data()[0]

    if log_transform:
        emg_env = np.log1p(emg_env)

    if target_sfreq is not None:
        emg_raw_resampled = emg_raw.copy().resample(target_sfreq)
        emg_env = emg_raw_resampled.get_data()[0]
        times = emg_raw_resampled.times
    else:
        times = emg_raw.times

    return times, emg_env