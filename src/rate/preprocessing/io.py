"""IO helpers for preprocessing inputs and outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from rate.paths import ensure_parent_dir, resolve_out_dir


@dataclass(frozen=True)
class PreprocessingRunPaths:
    """Resolved input and output paths for one preprocessing run."""

    raw_eeg_path: Path
    channels_tsv_path: Path
    ica_path: Path
    eeg_output_path: Path
    emg_output_path: Path
    events_output_path: Path
    summary_output_path: Path


def _format_context(row: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, str]:
    paths_cfg = dict(config.get("paths") or {})
    return {
        "dyad_id": str(row["dyad_id"]),
        "subject_id": str(row["subject_id"]),
        "task": str(row["task"]),
        "run": str(row["run"]),
        "run_padded": f"{int(row['run']):02d}",
        "out_dir": str(resolve_out_dir(config)),
        "bids_dir": str(paths_cfg.get("bids_dir", "")),
    }


def _resolve_from_pattern(pattern: str, *, root_dir: Path | None, context: Mapping[str, str]) -> Path:
    rendered = pattern.format(**context)
    path = Path(rendered)
    if path.is_absolute() or root_dir is None:
        return path
    return root_dir / path


def _find_matching_bids_channels_tsv(*, bids_dir: Path, row: Mapping[str, Any]) -> Path:
    subject_id = str(row["subject_id"]).lower()
    task = str(row["task"]).lower()
    run = str(row["run"])
    run_tokens = {
        f"run-{run}".lower(),
        f"run_{run}".lower(),
        f"run-{int(run):02d}".lower(),
        f"run_{int(run):02d}".lower(),
    }
    task_tokens = {f"task-{task}".lower(), f"task_{task}".lower(), task}

    candidates: list[Path] = []
    for path in sorted(bids_dir.rglob("*channels.tsv")):
        path_text = path.as_posix().lower()
        if subject_id not in path_text:
            continue
        if not any(token in path_text for token in task_tokens):
            continue
        if not any(token in path_text for token in run_tokens):
            continue
        candidates.append(path)

    if not candidates:
        raise FileNotFoundError(
            "Could not locate a BIDS channels.tsv file for "
            f"subject_id={row['subject_id']}, task={row['task']}, run={row['run']} under {bids_dir}"
        )
    return candidates[0]


def resolve_channels_tsv_path(
    config: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    override_path: Path | None = None,
) -> Path:
    """Resolve the BIDS ``channels.tsv`` file for one run."""
    if override_path is not None:
        return override_path

    preprocessing_cfg = dict(config.get("preprocessing") or {})
    bad_cfg = dict(preprocessing_cfg.get("bad_channels") or {})
    bids_dir = Path(str((config.get("paths") or {})["bids_dir"])).expanduser()
    context = _format_context(row, config)

    pattern = bad_cfg.get("path_pattern")
    if pattern:
        return _resolve_from_pattern(str(pattern), root_dir=bids_dir, context=context)

    return _find_matching_bids_channels_tsv(bids_dir=bids_dir, row=row)


def resolve_precomputed_ica_path(
    config: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    override_path: Path | None = None,
) -> Path:
    """Resolve the precomputed ICA path for one run."""
    if override_path is not None:
        return override_path

    preprocessing_cfg = dict(config.get("preprocessing") or {})
    ica_cfg = dict(preprocessing_cfg.get("ica") or {})
    context = _format_context(row, config)

    pattern = str(ica_cfg.get("path_pattern", "{subject_id}/task-{task}/run-{run_padded}_ica.fif"))
    root_dir_value = ica_cfg.get("root_dir")
    root_dir = Path(str(root_dir_value)).expanduser() if root_dir_value else resolve_out_dir(config) / "ica"
    return _resolve_from_pattern(pattern, root_dir=root_dir, context=context)


def build_preprocessing_output_paths(config: Mapping[str, Any], row: Mapping[str, Any]) -> PreprocessingRunPaths:
    """Build canonical output paths for one preprocessing run."""
    out_dir = resolve_out_dir(config)
    preprocessing_cfg = dict(config.get("preprocessing") or {})
    output_cfg = dict(preprocessing_cfg.get("output") or {})
    filenames = {
        "eeg": str(output_cfg.get("eeg_filename", "eeg_preprocessed_raw.fif")),
        "emg": str(output_cfg.get("emg_filename", "emg.npz")),
        "events": str(output_cfg.get("events_filename", "events.tsv")),
        "summary": str(output_cfg.get("summary_filename", "preprocessing_summary.json")),
    }

    base_dir = (
        out_dir
        / "preprocessing"
        / str(row["subject_id"])
        / f"task-{row['task']}"
        / f"run-{row['run']}"
    )
    raw_eeg_path = Path(str(row["eeg_path"])).expanduser()
    channels_tsv_path = resolve_channels_tsv_path(config, row)
    ica_path = resolve_precomputed_ica_path(config, row)
    return PreprocessingRunPaths(
        raw_eeg_path=raw_eeg_path,
        channels_tsv_path=channels_tsv_path,
        ica_path=ica_path,
        eeg_output_path=base_dir / filenames["eeg"],
        emg_output_path=base_dir / filenames["emg"],
        events_output_path=base_dir / filenames["events"],
        summary_output_path=base_dir / filenames["summary"],
    )


def write_events_table(output_path: Path, events_df: pd.DataFrame) -> Path:
    """Write a tab-separated event artifact."""
    ensure_parent_dir(output_path)
    events_df.to_csv(output_path, sep="\t", index=False)
    return output_path


def write_emg_artifact(output_path: Path, emg_raw) -> Path:
    """Write an EMG-only artifact as a compressed NumPy bundle."""
    ensure_parent_dir(output_path)
    if emg_raw is None:
        data = np.empty((0, 0), dtype=float)
        channel_names: list[str] = []
        sfreq = 0.0
        times = np.empty((0,), dtype=float)
    else:
        data = emg_raw.get_data()
        channel_names = list(emg_raw.ch_names)
        sfreq = float(emg_raw.info["sfreq"])
        times = emg_raw.times

    np.savez_compressed(
        output_path,
        data=data,
        channel_names=np.asarray(channel_names, dtype=str),
        sfreq=np.asarray([sfreq], dtype=float),
        times=times,
    )
    return output_path


def write_preprocessing_summary(output_path: Path, payload: Mapping[str, Any]) -> Path:
    """Write a JSON preprocessing summary."""
    ensure_parent_dir(output_path)
    output_path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path
