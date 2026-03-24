"""Average rereferencing utilities."""

from __future__ import annotations


def apply_average_reference(raw) -> None:
    """Apply an average EEG reference in-place."""
    raw.set_eeg_reference(ref_channels="average", projection=False, verbose="ERROR")
