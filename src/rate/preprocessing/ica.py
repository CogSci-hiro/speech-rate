"""Loading and applying precomputed ICA solutions."""

from __future__ import annotations

import logging
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def apply_precomputed_ica(raw, ica_path: Path | str | None) -> None:
    """Load a precomputed ICA file and apply it in-place.

    Parameters
    ----------
    raw
        EEG raw object.
    ica_path
        Path to a precomputed MNE ICA file.
    """
    import mne

    if ica_path is None:
        raise FileNotFoundError("No precomputed ICA path was resolved for this run.")

    path = Path(ica_path)
    if not path.exists():
        raise FileNotFoundError(f"Precomputed ICA file does not exist: {path}")

    ica = mne.preprocessing.read_ica(path, verbose="ERROR")
    ica.apply(raw, verbose="ERROR")
