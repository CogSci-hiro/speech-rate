from __future__ import annotations

from pathlib import Path


def read_raw_eeg(path: Path | str, *, preload: bool = False, verbose: str | bool = "ERROR"):
    """Read a supported EEG raw file with an MNE loader chosen by suffix."""
    import mne

    eeg_path = Path(path)
    suffixes = tuple(s.lower() for s in eeg_path.suffixes)

    if suffixes[-2:] == (".fif", ".gz") or suffixes[-1:] == (".fif",):
        return mne.io.read_raw_fif(eeg_path, preload=preload, verbose=verbose)
    if suffixes[-1:] == (".edf",):
        return mne.io.read_raw_edf(eeg_path, preload=preload, verbose=verbose)
    if suffixes[-1:] == (".bdf",):
        return mne.io.read_raw_bdf(eeg_path, preload=preload, verbose=verbose)
    if suffixes[-1:] == (".vhdr",):
        return mne.io.read_raw_brainvision(eeg_path, preload=preload, verbose=verbose)

    # Fallback to MNE's generic reader for formats it can infer directly.
    return mne.io.read_raw(eeg_path, preload=preload, verbose=verbose)
