"""Source-space preparation for the experimental lmeEEG branch."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from rate.epochs.io import build_epoch_output_paths, read_epochs
from rate.lmeeeg_source.config import (
    DEFAULT_LMEEEG_SOURCE_SECTION,
    LMEEEGSourceSettings,
    load_lmeeeg_source_settings,
    resolve_freesurfer_subject,
)
from rate.paths import ensure_parent_dir, resolve_out_dir

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SourcePreparedPaths:
    """Resolved input/output paths for one prepared source-space subject array."""

    epochs_input_path: Path
    metadata_input_path: Path
    source_data_output_path: Path
    metadata_output_path: Path
    times_output_path: Path
    channel_names_output_path: Path
    vertices_output_path: Path
    summary_output_path: Path


def build_source_prepared_paths(
    config: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    config_section: str = DEFAULT_LMEEEG_SOURCE_SECTION,
) -> SourcePreparedPaths:
    """Build canonical source-preparation paths for one subject-level epochs artifact."""
    settings = load_lmeeeg_source_settings(config, config_section=config_section)
    epoch_paths = build_epoch_output_paths(config, row, epochs_section=settings.epochs.section)
    root_dir = resolve_out_dir(config) / settings.outputs.namespace / settings.outputs.prepared_dirname / str(row["subject_id"])
    return SourcePreparedPaths(
        epochs_input_path=epoch_paths.epochs_output_path,
        metadata_input_path=epoch_paths.metadata_output_path,
        source_data_output_path=root_dir / "source_data.npy",
        metadata_output_path=root_dir / "metadata.csv",
        times_output_path=root_dir / "times.npy",
        channel_names_output_path=root_dir / "channel_names.json",
        vertices_output_path=root_dir / "vertices.json",
        summary_output_path=root_dir / "summary.json",
    )


def prepare_source_subject(
    config: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    config_section: str = DEFAULT_LMEEEG_SOURCE_SECTION,
) -> dict[str, Any]:
    """Project one subject-level epochs file into a tiny fsaverage source space."""
    settings = load_lmeeeg_source_settings(config, config_section=config_section)
    if not settings.enabled:
        raise ValueError(f"`{config_section}.enabled` is false. Refusing to run the experimental source-space pipeline.")

    paths = build_source_prepared_paths(config, row, config_section=config_section)
    ensure_parent_dir(paths.summary_output_path)

    _configure_mne_runtime()
    import mne
    from mne.minimum_norm import apply_inverse_epochs, make_inverse_operator

    subject_id = str(row["subject_id"])
    fs_subject = resolve_freesurfer_subject(subject_id, settings)
    subjects_dir = settings.source_space.subjects_dir
    fsaverage_subject = settings.source_space.fsaverage_subject

    LOGGER.info(
        "Preparing source-space lmeEEG subject=%s with spacing=%s from %s",
        subject_id,
        settings.source_space.spacing,
        paths.epochs_input_path,
    )
    epochs = read_epochs(paths.epochs_input_path, preload=True)
    metadata_df = pd.read_csv(paths.metadata_input_path).reset_index(drop=True)
    if len(epochs) != len(metadata_df):
        raise ValueError(
            "Epochs length and metadata length must match before source projection: "
            f"{len(epochs)} != {len(metadata_df)}"
        )
    if _needs_inverse_reference_normalization(epochs):
        LOGGER.info("Adding average-reference projector for inverse modeling on %s", subject_id)
        epochs.set_eeg_reference(projection=True, verbose="ERROR")

    trans_path = _resolve_required_path(
        settings.forward.trans_pattern,
        row=row,
        subject_id=subject_id,
        fs_subject=fs_subject,
        subjects_dir=subjects_dir,
        label="transformation",
    )
    bem_solution_path = _resolve_required_path(
        settings.forward.bem_solution_pattern,
        row=row,
        subject_id=subject_id,
        fs_subject=fs_subject,
        subjects_dir=subjects_dir,
        label="BEM solution",
    )
    subject_src = _build_source_space(
        mne,
        subject=fs_subject,
        settings=settings,
    )
    fsaverage_src = _build_source_space(
        mne,
        subject=fsaverage_subject,
        settings=settings,
    )
    noise_cov = _compute_noise_covariance(mne, epochs, settings=settings)
    forward = mne.make_forward_solution(
        epochs.info,
        trans=str(trans_path),
        src=subject_src,
        bem=str(bem_solution_path),
        meg=settings.forward.meg,
        eeg=settings.forward.eeg,
        mindist=settings.forward.mindist,
        n_jobs=settings.fitting.n_jobs,
        verbose="ERROR",
    )
    inverse_operator = make_inverse_operator(
        epochs.info,
        forward,
        noise_cov,
        loose=settings.inverse.loose,
        depth=settings.inverse.depth,
        verbose="ERROR",
    )
    morph = mne.compute_source_morph(
        src=inverse_operator["src"],
        subject_from=fs_subject,
        subject_to=fsaverage_subject,
        src_to=fsaverage_src,
        subjects_dir=str(subjects_dir),
        verbose="ERROR",
    )

    stc_iterator = apply_inverse_epochs(
        epochs,
        inverse_operator,
        lambda2=settings.inverse.lambda2,
        method=settings.inverse.method,
        pick_ori=settings.inverse.pick_ori,
        return_generator=True,
        verbose="ERROR",
    )
    morphed_epochs: list[np.ndarray] = []
    morphed_vertices: list[np.ndarray] | None = None
    times: np.ndarray | None = None
    for stc in tqdm(
        stc_iterator,
        total=len(epochs),
        disable=not settings.fitting.progress,
        desc=f"source:{subject_id}",
    ):
        morphed = morph.apply(stc)
        morphed_epochs.append(np.asarray(morphed.data, dtype=float))
        if morphed_vertices is None:
            morphed_vertices = [np.asarray(hemi_vertices, dtype=int) for hemi_vertices in morphed.vertices]
            times = np.asarray(morphed.times, dtype=float)

    if not morphed_epochs or morphed_vertices is None or times is None:
        raise ValueError(f"No source estimates were produced for subject `{subject_id}`.")

    source_data = np.stack(morphed_epochs, axis=0)
    channel_names = build_source_channel_names(morphed_vertices)

    np.save(paths.source_data_output_path, source_data)
    np.save(paths.times_output_path, times)
    paths.metadata_output_path.write_text(metadata_df.to_csv(index=False), encoding="utf-8")
    paths.channel_names_output_path.write_text(json.dumps(channel_names, indent=2) + "\n", encoding="utf-8")
    paths.vertices_output_path.write_text(
        json.dumps(
            {
                "subject_id": subject_id,
                "freesurfer_subject": fs_subject,
                "fsaverage_subject": fsaverage_subject,
                "spacing": settings.source_space.spacing,
                "surface": settings.source_space.surface,
                "vertices": {
                    "lh": morphed_vertices[0].tolist(),
                    "rh": morphed_vertices[1].tolist(),
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    summary = {
        "status": "ok",
        "subject_id": subject_id,
        "freesurfer_subject": fs_subject,
        "epochs_input": str(paths.epochs_input_path),
        "metadata_input": str(paths.metadata_input_path),
        "transformation_path": str(trans_path),
        "bem_solution_path": str(bem_solution_path),
        "source_space_spacing": settings.source_space.spacing,
        "n_trials": int(source_data.shape[0]),
        "n_vertices": int(source_data.shape[1]),
        "n_times": int(source_data.shape[2]),
        "outputs": {
            "source_data": str(paths.source_data_output_path),
            "metadata": str(paths.metadata_output_path),
            "times": str(paths.times_output_path),
            "channel_names": str(paths.channel_names_output_path),
            "vertices": str(paths.vertices_output_path),
        },
    }
    paths.summary_output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    LOGGER.info("Wrote source-space preparation outputs for %s -> %s", subject_id, paths.summary_output_path)
    return summary


def build_source_channel_names(vertices: list[np.ndarray]) -> list[str]:
    """Label fsaverage vertices so the pooled lmeEEG code can treat them like channels.

    This is a computational adapter only. Vertices are not conceptually sensors.
    """
    channel_names = [f"lh_{int(vertex):06d}" for vertex in vertices[0]]
    channel_names.extend(f"rh_{int(vertex):06d}" for vertex in vertices[1])
    return channel_names


def _resolve_required_path(
    pattern: str,
    *,
    row: Mapping[str, Any],
    subject_id: str,
    fs_subject: str,
    subjects_dir: Path,
    label: str,
) -> Path:
    format_values = {str(key): value for key, value in row.items()}
    format_values.update(
        {
            "subject_id": subject_id,
            "fs_subject": fs_subject,
            "subjects_dir": str(subjects_dir),
        }
    )
    rendered = pattern.format(**format_values)
    path = Path(rendered).expanduser()
    if not path.exists():
        raise FileNotFoundError(
            f"Required {label} file is missing for subject `{subject_id}` (FreeSurfer subject `{fs_subject}`): {path}"
        )
    return path


def _build_source_space(mne_module, *, subject: str, settings: LMEEEGSourceSettings):
    subject_dir = settings.source_space.subjects_dir / subject
    if not subject_dir.exists():
        raise FileNotFoundError(f"FreeSurfer subject directory is missing: {subject_dir}")
    return mne_module.setup_source_space(
        subject=subject,
        spacing=settings.source_space.spacing,
        surface=settings.source_space.surface,
        subjects_dir=str(settings.source_space.subjects_dir),
        add_dist=settings.source_space.add_dist,
        verbose="ERROR",
    )


def _compute_noise_covariance(mne_module, epochs, *, settings: LMEEEGSourceSettings):
    try:
        return mne_module.compute_covariance(
            epochs,
            tmin=settings.covariance.tmin,
            tmax=settings.covariance.tmax,
            method=settings.covariance.method,
            rank=None,
            verbose="ERROR",
        )
    except Exception as exc:
        if settings.covariance.tmin is None and settings.covariance.tmax == 0.0 and float(epochs.times.min()) > 0.0:
            LOGGER.warning(
                "Source covariance baseline tmax=0.0 is unavailable because epochs start at %.3fs; retrying across the full epoch. "
                "This is acceptable for feasibility testing but should be reviewed before interpreting results.",
                float(epochs.times.min()),
            )
            return mne_module.compute_covariance(
                epochs,
                tmin=None,
                tmax=None,
                method=settings.covariance.method,
                rank=None,
                verbose="ERROR",
            )
        raise ValueError(f"Unable to compute source-space noise covariance: {exc}") from exc


def _needs_inverse_reference_normalization(epochs) -> bool:
    if "eeg" not in epochs.get_channel_types(unique=True):
        return False
    if epochs.info.get("custom_ref_applied", False):
        return True
    return not any(proj.get("desc") == "Average EEG reference" for proj in epochs.info.get("projs", []))


def _configure_mne_runtime() -> None:
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
