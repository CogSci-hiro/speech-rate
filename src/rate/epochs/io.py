"""IO helpers for epochs artifacts."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from rate.paths import ensure_parent_dir, resolve_out_dir
from rate.preprocessing.io import resolve_preprocessed_eeg_path

DEFAULT_EPOCHS_SECTION = "epochs"


@dataclass(frozen=True)
class EpochRunPaths:
    """Resolved input and output paths for one epoching run."""

    raw_eeg_path: Path
    events_table_path: Path
    epochs_output_path: Path
    metadata_output_path: Path
    events_array_output_path: Path
    summary_output_path: Path


def _sanitize_lock_name(value: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip()).strip("_").lower()
    return token or "events"


def _render_filename(pattern: str, *, lock_name: str) -> str:
    return pattern.format(lock_name=_sanitize_lock_name(lock_name))


def get_epochs_config(config: Mapping[str, Any], *, section: str = DEFAULT_EPOCHS_SECTION) -> dict[str, Any]:
    """Return one epoching config section as a plain dictionary."""
    epochs_cfg = config.get(section) or {}
    if not isinstance(epochs_cfg, Mapping):
        raise TypeError(f"`{section}` must be a mapping.")
    return dict(epochs_cfg)


def resolve_events_table_path(config: Mapping[str, Any], *, override_path: Path | None = None) -> Path:
    """Resolve the project-level events table path."""
    if override_path is not None:
        return override_path
    return resolve_out_dir(config) / "events" / "events.csv"


def build_epoch_output_paths(
    config: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    epochs_section: str = DEFAULT_EPOCHS_SECTION,
    raw_eeg_path: Path | None = None,
    events_table_path: Path | None = None,
) -> EpochRunPaths:
    """Build canonical input and output paths for one epoching run."""
    out_dir = resolve_out_dir(config)
    epochs_cfg = get_epochs_config(config, section=epochs_section)
    output_cfg = dict(epochs_cfg.get("output") or {})
    lock_name = str(epochs_cfg.get("event_column", "time_s"))

    epochs_filename = _render_filename(str(output_cfg.get("epochs_filename", "epochs-{lock_name}.fif")), lock_name=lock_name)
    metadata_filename = _render_filename(str(output_cfg.get("metadata_filename", "metadata-{lock_name}.csv")), lock_name=lock_name)
    events_array_filename = _render_filename(str(output_cfg.get("events_array_filename", "events-{lock_name}.npy")), lock_name=lock_name)
    summary_filename = _render_filename(str(output_cfg.get("summary_filename", "epoching_summary-{lock_name}.json")), lock_name=lock_name)

    base_dir = out_dir / "epochs" / str(row["subject_id"])
    return EpochRunPaths(
        raw_eeg_path=raw_eeg_path or resolve_preprocessed_eeg_path(config, row),
        events_table_path=resolve_events_table_path(config, override_path=events_table_path),
        epochs_output_path=base_dir / epochs_filename,
        metadata_output_path=base_dir / metadata_filename,
        events_array_output_path=base_dir / events_array_filename,
        summary_output_path=base_dir / summary_filename,
    )


def read_events_table(path: Path) -> pd.DataFrame:
    """Read the project-level events table."""
    return pd.read_csv(path)


def _configure_mne_runtime() -> None:
    """Set conservative runtime defaults before importing MNE.

    This mirrors the test environment and avoids Numba cache issues in some
    local Python/MNE installations.
    """
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    os.environ.setdefault("MNE_DONTWRITE_HOME", "true")


def read_epochs(path: Path, *, preload: bool = False):
    """Read an MNE epochs file."""
    _configure_mne_runtime()
    import mne

    return mne.read_epochs(path, preload=preload, verbose="ERROR")


def write_epochs(epochs, output_path: Path) -> Path:
    """Write epochs to FIF."""
    _configure_mne_runtime()
    ensure_parent_dir(output_path)
    epochs.save(output_path, overwrite=True, verbose="ERROR")
    return output_path


def write_epoch_metadata(metadata_df: pd.DataFrame, output_path: Path) -> Path:
    """Write epoch metadata as CSV."""
    ensure_parent_dir(output_path)
    metadata_df.to_csv(output_path, index=False)
    return output_path


def write_epoch_events_array(events_array: np.ndarray, output_path: Path) -> Path:
    """Write the integer events array used for epoching."""
    ensure_parent_dir(output_path)
    np.save(output_path, events_array.astype(int, copy=False))
    return output_path


def write_epoch_summary(summary: Mapping[str, Any], output_path: Path) -> Path:
    """Write a JSON epoching summary."""
    ensure_parent_dir(output_path)
    output_path.write_text(json.dumps(dict(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path
