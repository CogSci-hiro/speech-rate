"""Visualization helpers for GLM beta-map outputs."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")

from matplotlib import pyplot as plt


def _infer_sfreq(times: np.ndarray) -> float:
    """Infer a sampling frequency from a 1D time vector."""
    if times.ndim != 1:
        raise ValueError("times must be a 1D array")
    if times.size < 2:
        return 1.0
    deltas = np.diff(times.astype(float))
    finite = deltas[np.isfinite(deltas)]
    if finite.size == 0:
        return 1.0
    delta = float(np.median(finite))
    if delta <= 0:
        return 1.0
    return 1.0 / delta


def _sanitize_token(value: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value).strip()).strip("._-")
    return token or "item"


def _extract_overlapping_channels(message: str) -> list[str]:
    lines = [line.strip() for line in str(message).splitlines() if line.strip()]
    if not lines:
        return []
    channel_line = lines[-1]
    return [name.strip() for name in channel_line.split(",") if name.strip()]


def _same_axis_metadata(
    left_times: np.ndarray,
    left_channels: Sequence[str],
    right_times: np.ndarray,
    right_channels: Sequence[str],
) -> bool:
    return tuple(left_channels) == tuple(right_channels) and np.array_equal(
        np.asarray(left_times, dtype=float),
        np.asarray(right_times, dtype=float),
        equal_nan=True,
    )


def _nanmean_stack(beta_maps: Sequence[np.ndarray]) -> np.ndarray:
    stacked = np.stack([np.asarray(beta_map, dtype=float) for beta_map in beta_maps], axis=0)
    with np.errstate(invalid="ignore"):
        return np.nanmean(stacked, axis=0)


def plot_joint_model_weights(
    beta_map: np.ndarray,
    *,
    times: np.ndarray,
    channel_names: Sequence[str],
    output_stem: Path,
    title: str | None = None,
    formats: Sequence[str] = ("png", "pdf"),
    dpi: int = 300,
    line_width: float = 2.5,
    joint_times: str | Sequence[float] = "peaks",
) -> list[Path]:
    """Plot one GLM beta map with an MNE joint plot.

    This uses ``nave=1`` so the figure reflects a single estimated beta map
    rather than implying averaging across observations.
    """
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
    import mne

    beta_array = np.asarray(beta_map, dtype=float)
    time_array = np.asarray(times, dtype=float)
    if beta_array.ndim != 2:
        raise ValueError("beta_map must have shape (n_channels, n_times)")
    if beta_array.shape[0] != len(channel_names):
        raise ValueError("beta_map channel dimension must match channel_names")
    if beta_array.shape[1] != time_array.shape[0]:
        raise ValueError("beta_map time dimension must match times")

    info = mne.create_info(list(channel_names), sfreq=_infer_sfreq(time_array), ch_types=["eeg"] * len(channel_names))
    try:
        info.set_montage("standard_1020", on_missing="ignore")
    except Exception:
        pass

    evoked = mne.EvokedArray(beta_array, info, tmin=float(time_array[0]), nave=1, comment=title or "beta")
    try:
        figure = evoked.plot_joint(times=joint_times, title=title, show=False)
    except ValueError as exc:
        if "overlapping positions" not in str(exc):
            raise
        overlapping_channels = _extract_overlapping_channels(str(exc))
        keep_channels = [name for name in evoked.ch_names if name not in overlapping_channels]
        if not keep_channels:
            raise
        figure = evoked.copy().pick(keep_channels).plot_joint(times=joint_times, title=title, show=False)
    for axis in figure.axes:
        for line in axis.lines:
            line.set_linewidth(float(line_width))

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    written_paths: list[Path] = []
    for fmt in formats:
        output_path = output_stem.with_suffix(f".{fmt}")
        figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
        written_paths.append(output_path)
    plt.close(figure)
    return written_paths


def build_model_weights_manifest(
    *,
    out_dir: Path,
    glm_root: Path,
    manifest_path: Path,
    formats: Sequence[str] = ("png", "pdf"),
    dpi: int = 300,
    line_width: float = 2.5,
) -> dict:
    """Render joint plots for all saved GLM predictor betas and return a manifest."""
    subject_plots: list[dict[str, object]] = []
    group_inputs: dict[tuple[str, str], dict[str, object]] = {}
    for summary_path in sorted(glm_root.glob("*/" "*/summary.json")):
        model_dir = summary_path.parent
        subject_id = model_dir.parent.name
        model_name = model_dir.name
        predictor_betas_path = model_dir / "predictor_betas.npz"
        times_path = model_dir / "times.npy"
        channel_names_path = model_dir / "channel_names.json"

        if not predictor_betas_path.exists() or not times_path.exists() or not channel_names_path.exists():
            continue

        predictor_betas = np.load(predictor_betas_path)
        times = np.load(times_path)
        channel_names = json.loads(channel_names_path.read_text(encoding="utf-8"))
        for predictor_name in predictor_betas.files:
            beta_map = np.asarray(predictor_betas[predictor_name], dtype=float)
            if beta_map.ndim != 2 or not np.isfinite(beta_map).any():
                continue
            output_stem = out_dir / "figures" / "glm" / "model_weights" / subject_id / model_name / _sanitize_token(predictor_name)
            written_paths = plot_joint_model_weights(
                beta_map,
                times=times,
                channel_names=channel_names,
                output_stem=output_stem,
                title=f"{subject_id} | {model_name} | {predictor_name}",
                formats=formats,
                dpi=dpi,
                line_width=line_width,
            )
            subject_plots.append(
                {
                    "level": "subject",
                    "subject_id": subject_id,
                    "model_name": model_name,
                    "predictor": predictor_name,
                    "files": [str(path) for path in written_paths],
                }
            )
            group_key = (model_name, predictor_name)
            group_entry = group_inputs.setdefault(
                group_key,
                {
                    "times": np.asarray(times, dtype=float),
                    "channel_names": list(channel_names),
                    "beta_maps": [],
                    "subject_ids": [],
                },
            )
            if not _same_axis_metadata(
                group_entry["times"],
                group_entry["channel_names"],
                times,
                channel_names,
            ):
                continue
            group_entry["beta_maps"].append(beta_map)
            group_entry["subject_ids"].append(subject_id)

    group_plots: list[dict[str, object]] = []
    for (model_name, predictor_name), group_entry in sorted(group_inputs.items()):
        beta_maps = list(group_entry["beta_maps"])
        if not beta_maps:
            continue
        mean_beta_map = _nanmean_stack(beta_maps)
        if not np.isfinite(mean_beta_map).any():
            continue
        output_stem = out_dir / "figures" / "glm" / "model_weights" / "group" / model_name / f"{_sanitize_token(predictor_name)}_mean"
        written_paths = plot_joint_model_weights(
            mean_beta_map,
            times=np.asarray(group_entry["times"], dtype=float),
            channel_names=list(group_entry["channel_names"]),
            output_stem=output_stem,
            title=f"group mean | {model_name} | {predictor_name}",
            formats=formats,
            dpi=dpi,
            line_width=line_width,
        )
        group_plots.append(
            {
                "level": "group",
                "model_name": model_name,
                "predictor": predictor_name,
                "n_subjects": len(group_entry["subject_ids"]),
                "subject_ids": list(group_entry["subject_ids"]),
                "files": [str(path) for path in written_paths],
            }
        )

    manifest = {
        "status": "ok",
        "plot_count": len(subject_plots) + len(group_plots),
        "subject_plot_count": len(subject_plots),
        "group_plot_count": len(group_plots),
        "plots": [*subject_plots, *group_plots],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def plot_glm_summary(*_args, **_kwargs):
    """Compatibility placeholder for the older scaffold path."""
    raise NotImplementedError("GLM summary plotting is not implemented in the scaffold.")
