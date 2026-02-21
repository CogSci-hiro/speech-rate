from typing import List, Tuple

import mne
import numpy as np
from mne.stats import combine_adjacency
from scipy import sparse


def make_eeg_channel_adjacency(
    ch_names: List[str],
    montage_name: str,
) -> sparse.csr_matrix:
    """
    Build EEG sensor adjacency (channels-only) using an MNE montage.

    Parameters
    ----------
    ch_names
        EEG channel names in the same order as the kernel arrays.
    montage_name
        MNE montage name (e.g., "biosemi64").

    Returns
    -------
    scipy.sparse.csr_matrix
        Channel adjacency matrix.

    Usage example
    -------------
        ch_adj = make_eeg_channel_adjacency(ch_names, montage_name="biosemi64")
    """
    info = mne.create_info(ch_names=ch_names, sfreq=100.0, ch_types="eeg")
    montage = mne.channels.make_standard_montage(montage_name)
    info.set_montage(montage, match_case=False)

    ch_adj, _ = mne.channels.find_ch_adjacency(info, ch_type="eeg")
    return ch_adj


def make_spatiotemporal_adjacency(
    ch_names: List[str],
    n_lags: int,
    montage_name: str,
) -> sparse.csr_matrix:
    """
    Build spatiotemporal adjacency over (channels × lags).

    Parameters
    ----------
    ch_names
        EEG channel names.
    n_lags
        Number of lag samples.
    montage_name
        Montage used for channel adjacency.

    Returns
    -------
    scipy.sparse.csr_matrix
        Spatiotemporal adjacency matrix compatible with MNE cluster functions.

    Usage example
    -------------
        adj = make_spatiotemporal_adjacency(ch_names, n_lags=321, montage_name="biosemi64")
    """
    ch_adj = make_eeg_channel_adjacency(ch_names=ch_names, montage_name=montage_name)
    # combine_adjacency: first dimension is time/lag, second is space
    return combine_adjacency(n_lags, ch_adj)