"""Shared joint-plot helpers for topographic weight maps."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.collections import Collection, LineCollection, PathCollection, QuadMesh
from matplotlib.contour import QuadContourSet


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


def sanitize_token(value: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value).strip()).strip("._-")
    return token or "item"


def _extract_overlapping_channels(message: str) -> list[str]:
    lines = [line.strip() for line in str(message).splitlines() if line.strip()]
    if not lines:
        return []
    channel_line = lines[-1]
    return [name.strip() for name in channel_line.split(",") if name.strip()]


def _dash_pattern_has_no_positive_lengths(style: object) -> bool:
    """Return whether a Matplotlib linestyle payload resolves to an invalid dash list."""
    if isinstance(style, tuple) and len(style) == 2:
        _, dash_pattern = style
        if dash_pattern is None:
            return False
        try:
            dash_values = tuple(float(value) for value in dash_pattern)
        except TypeError:
            return False
        return bool(dash_values) and not any(value > 0.0 for value in dash_values)
    if isinstance(style, (list, tuple)):
        return any(_dash_pattern_has_no_positive_lengths(item) for item in style)
    return False


def _normalise_invalid_collection_linestyles(figure) -> None:
    """Coerce zero-length contour dash patterns to solid before vector export."""
    for axis in figure.axes:
        for collection in axis.collections:
            if not isinstance(collection, Collection):
                continue
            get_linestyle = getattr(collection, "get_linestyle", None)
            set_linestyle = getattr(collection, "set_linestyle", None)
            if get_linestyle is None or set_linestyle is None:
                continue
            try:
                linestyle = get_linestyle()
            except Exception:
                continue
            if _dash_pattern_has_no_positive_lengths(linestyle):
                set_linestyle("solid")


def _ensure_visible_topomap_contours(figure, *, line_width: float = 0.75) -> None:
    """Restore contour strokes when MNE creates them with zero-width lines."""
    for axis in figure.axes:
        for collection in axis.collections:
            if isinstance(collection, QuadContourSet):
                collection.set_linewidths(float(line_width))
                collection.set_linestyle("solid")
                collection.set_edgecolor("k")


def _style_butterfly_axis_lines(figure, *, n_times: int, line_width: float) -> None:
    """Only thicken the main time-series traces, not topomap outlines or mask joins."""
    for axis in figure.axes:
        if not axis.lines:
            continue
        if not any(len(np.asarray(line.get_xdata())) == int(n_times) for line in axis.lines):
            continue
        for line in axis.lines:
            line.set_linewidth(float(line_width))


def _remove_spatial_colors_inset_axis(figure, *, n_times: int) -> None:
    """Remove MNE's small sensor-color inset while preserving colored butterfly traces."""
    for axis in tuple(figure.axes):
        has_time_series = any(len(np.asarray(line.get_xdata())) == int(n_times) for line in axis.lines)
        if has_time_series:
            continue
        if axis.images:
            continue
        if any(isinstance(collection, (LineCollection, QuadMesh)) for collection in axis.collections):
            continue
        if any(isinstance(collection, PathCollection) for collection in axis.collections):
            figure.delaxes(axis)


def _pick_peak_indices(score: np.ndarray, *, n_peaks: int, min_separation: int) -> list[int]:
    ranking = np.argsort(np.asarray(score, dtype=float))[::-1]
    peaks: list[int] = []
    for index in ranking:
        if not np.isfinite(score[index]):
            continue
        if any(abs(int(index) - existing) < min_separation for existing in peaks):
            continue
        peaks.append(int(index))
        if len(peaks) >= n_peaks:
            break
    return sorted(peaks)


def _resolve_joint_times(
    beta_map: np.ndarray,
    time_array: np.ndarray,
    *,
    joint_times: str | Sequence[float],
    significance_mask: np.ndarray | None = None,
) -> np.ndarray:
    if isinstance(joint_times, str) and joint_times != "peaks":
        raise ValueError(f"Unsupported joint_times mode: {joint_times}")
    if not isinstance(joint_times, str):
        return np.asarray(joint_times, dtype=float)

    score = np.nanmax(np.abs(np.asarray(beta_map, dtype=float)), axis=0)
    if significance_mask is not None:
        significant_times = np.any(np.asarray(significance_mask, dtype=bool), axis=0)
        if np.any(significant_times):
            score = np.where(significant_times, score, -np.inf)
    min_separation = max(1, int(round(time_array.shape[0] / 10.0)))
    peak_indices = _pick_peak_indices(score, n_peaks=3, min_separation=min_separation)
    if not peak_indices:
        fallback = np.linspace(0, max(0, time_array.shape[0] - 1), num=min(3, time_array.shape[0]), dtype=int)
        peak_indices = sorted({int(index) for index in fallback.tolist()})
    return time_array[np.asarray(peak_indices, dtype=int)]


def _contiguous_true_spans(mask: np.ndarray, time_array: np.ndarray) -> list[tuple[float, float]]:
    boolean_mask = np.asarray(mask, dtype=bool)
    if boolean_mask.ndim != 1 or boolean_mask.shape[0] != time_array.shape[0]:
        raise ValueError("mask must be a 1D boolean array aligned to time_array")
    if not np.any(boolean_mask):
        return []

    if time_array.shape[0] > 1:
        finite_deltas = np.diff(np.asarray(time_array, dtype=float))
        finite_deltas = finite_deltas[np.isfinite(finite_deltas) & (finite_deltas > 0)]
        half_step = float(np.median(finite_deltas)) / 2.0 if finite_deltas.size else 0.0
    else:
        half_step = 0.0

    spans: list[tuple[float, float]] = []
    start_index: int | None = None
    for index, flag in enumerate(boolean_mask):
        if flag and start_index is None:
            start_index = index
        if not flag and start_index is not None:
            spans.append((float(time_array[start_index]) - half_step, float(time_array[index - 1]) + half_step))
            start_index = None
    if start_index is not None:
        spans.append((float(time_array[start_index]) - half_step, float(time_array[-1]) + half_step))
    return spans


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
    significance_mask: np.ndarray | None = None,
) -> list[Path]:
    """Plot one weight map with an MNE joint plot."""
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
    significance_array = None
    if significance_mask is not None:
        significance_array = np.asarray(significance_mask, dtype=bool)
        if significance_array.shape != beta_array.shape:
            raise ValueError("significance_mask must match beta_map shape")

    info = mne.create_info(list(channel_names), sfreq=_infer_sfreq(time_array), ch_types=["eeg"] * len(channel_names))
    try:
        info.set_montage("standard_1020", on_missing="ignore")
    except Exception:
        pass

    evoked = mne.EvokedArray(beta_array, info, tmin=float(time_array[0]), nave=1, comment=title or "beta")
    selected_joint_times = _resolve_joint_times(
        beta_array,
        time_array,
        joint_times=joint_times,
        significance_mask=significance_array,
    )
    topomap_args: dict[str, object] = {"contours": 6}
    if significance_array is not None:
        topomap_args["mask"] = significance_array
        topomap_args["mask_params"] = {
            "marker": "o",
            "markerfacecolor": "0.6",
            "markeredgecolor": "0.6",
            "markeredgewidth": 0.0,
            "markersize": 8.0,
        }
    try:
        figure = evoked.plot_joint(
            times=selected_joint_times,
            title=title,
            show=False,
            ts_args={"spatial_colors": True},
            topomap_args=topomap_args,
        )
    except ValueError as exc:
        if "overlapping positions" not in str(exc):
            raise
        overlapping_channels = _extract_overlapping_channels(str(exc))
        keep_channels = [name for name in evoked.ch_names if name not in overlapping_channels]
        if not keep_channels:
            raise
        keep_indices = [evoked.ch_names.index(name) for name in keep_channels]
        reduced_topomap_args = dict(topomap_args)
        if significance_array is not None:
            reduced_topomap_args["mask"] = significance_array[np.asarray(keep_indices, dtype=int)]
        figure = evoked.copy().pick(keep_channels).plot_joint(
            times=selected_joint_times,
            title=title,
            show=False,
            ts_args={"spatial_colors": True},
            topomap_args=reduced_topomap_args,
        )
    _ensure_visible_topomap_contours(figure)
    _remove_spatial_colors_inset_axis(figure, n_times=time_array.shape[0])
    _style_butterfly_axis_lines(figure, n_times=time_array.shape[0], line_width=line_width)
    if significance_array is not None and np.any(significance_array):
        significant_time_mask = np.any(significance_array, axis=0)
        line_axes = [axis for axis in figure.axes if axis.lines]
        if line_axes:
            main_axis = max(line_axes, key=lambda axis: len(axis.lines))
            for start_time, end_time in _contiguous_true_spans(significant_time_mask, time_array):
                main_axis.axvspan(start_time, end_time, color="0.85", alpha=0.7, zorder=0)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    written_paths: list[Path] = []
    for fmt in formats:
        output_path = output_stem.with_suffix(f".{fmt}")
        _normalise_invalid_collection_linestyles(figure)
        figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
        written_paths.append(output_path)
    plt.close(figure)
    return written_paths
