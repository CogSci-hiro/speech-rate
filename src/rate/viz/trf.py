"""Unified TRF visualization utilities.

This module is rendering-only and intentionally contains no model fitting,
statistics, file loading, or cluster computation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import matplotlib as mpl
import mne
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

from .base import add_time_cluster_shading, save_figure


def plot_joint_kernel(
    *,
    evoked: mne.Evoked,
    out_dir: Path,
    family: str,
    name: str,
    times: Optional[Sequence[float]] = None,
    time_clusters: Optional[Iterable[Tuple[float, float]]] = None,
    title: Optional[str] = None,
) -> None:
    """Plot a TRF kernel with MNE's joint layout and save it.

    Parameters
    ----------
    evoked
        TRF kernel as an ``mne.Evoked`` object.
    out_dir
        Root output directory.
    family
        Figure family subdirectory under ``final/figures``.
    name
        Figure base filename.
    times
        Optional topomap times (seconds). If ``None``, MNE selects defaults.
    time_clusters
        Optional iterable of ``(start_s, end_s)`` intervals for shading.
    title
        Optional figure title.
    """
    joint_times = _normalize_joint_times(evoked=evoked, times=times)
    fig = evoked.plot_joint(times=joint_times, show=False, title=title)
    main_ax = _find_time_axis(fig)

    if main_ax is not None and time_clusters is not None:
        add_time_cluster_shading(main_ax, evoked.times, time_clusters)

    save_figure(fig, out_dir, family, name)


def _normalize_joint_times(
    *,
    evoked: mne.Evoked,
    times: Optional[Sequence[float]],
) -> np.ndarray:
    """Return a 1D times array suitable for ``Evoked.plot_joint``."""
    if times is not None:
        arr = np.asarray(times, dtype=float)
        arr = np.atleast_1d(arr).astype(float)
        if arr.size < 1:
            raise ValueError("times must contain at least one time point")
        return arr

    t = np.asarray(evoked.times, dtype=float)
    if t.ndim != 1 or t.size < 1:
        raise ValueError("evoked.times must be a non-empty 1D array")
    if t.size == 1:
        return t.copy()

    # 3 representative topomap anchors across the available time span.
    idx = np.array([0, t.size // 2, t.size - 1], dtype=int)
    return np.unique(t[idx])


def plot_electrode_timecourse_panel(
    *,
    condition_evokeds: Dict[str, mne.Evoked],
    electrodes: Sequence[str],
    colors: Dict[str, str],
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    out_dir: Path,
    family: str,
    name: str,
    xlabel: str = "Time (s)",
    ylabel: str = "Amplitude",
) -> None:
    """Plot multi-condition TRF time courses for selected electrodes.

    Parameters
    ----------
    condition_evokeds
        Mapping from condition name to already-aggregated ``mne.Evoked``.
    electrodes
        Electrode names to plot (one row per electrode).
    colors
        Mapping from condition name to line color.
    xmin, xmax
        X-axis limits in seconds.
    ymin, ymax
        Y-axis limits.
    out_dir
        Root output directory.
    family
        Figure family subdirectory under ``final/figures``.
    name
        Figure base filename.
    xlabel, ylabel
        Axis labels for the panel.
    """
    if not electrodes:
        raise ValueError("electrodes must contain at least one channel name")
    if not condition_evokeds:
        raise ValueError("condition_evokeds must not be empty")

    n_rows = len(electrodes)
    fig, axes = plt.subplots(n_rows, 1, figsize=(6.0, 2.5 * n_rows), sharex=True)

    if n_rows == 1:
        axes = [axes]

    for ax, electrode in zip(axes, electrodes):
        for condition, evoked in condition_evokeds.items():
            picked = evoked.copy().pick([electrode])
            ax.plot(
                picked.times,
                picked.data.squeeze(),
                label=condition,
                color=colors.get(condition),
                linewidth=2.0,
            )

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.axvline(0.0, linestyle="--", color="black", linewidth=1.0)
        ax.set_ylabel(ylabel)
        ax.set_title(electrode)

    axes[-1].set_xlabel(xlabel)
    axes[0].legend(frameon=False)
    fig.tight_layout()

    save_figure(fig, out_dir, family, name)


def plot_stat_topomap_sequence(
    *,
    stat: np.ndarray,
    info: mne.Info,
    data_tmin: float,
    tmin: float,
    tmax: float,
    step_ms: float,
    title: str,
    out_dir: Path,
    family: str,
    name: str,
    mask: Optional[np.ndarray] = None,
    mask_params: Optional[dict] = None,
    lim_val: float | None = None,
    max_cols: int = 10,
    cmap: str = "RdBu_r",
    cbar_label: str = "Value",
) -> None:
    """Plot a time-sequenced topomap grid from channel-by-time statistics.

    Parameters
    ----------
    stat
        Array with shape ``(n_times, n_channels)``.
    info
        MNE sensor info for topomap plotting.
    data_tmin
        Start time in seconds of the first row in ``stat``.
    tmin, tmax
        Requested plotting window in seconds.
    step_ms
        Temporal spacing between maps in milliseconds.
    title
        Figure title.
    out_dir
        Root output directory.
    family
        Figure family subdirectory under ``final/figures``.
    name
        Figure base filename.
    mask
        Optional boolean mask with same shape as ``stat``.
    lim_val
        Optional symmetric color limit. If ``None``, uses max abs value.
    max_cols
        Maximum number of map columns.
    cmap
        Colormap name.
    cbar_label
        Colorbar label.
    """
    _validate_topomap_inputs(stat=stat, mask=mask, step_ms=step_ms, max_cols=max_cols)

    evoked = mne.EvokedArray(stat.T, info, tmin=float(data_tmin))
    plot_times, labels_ms = _make_step_times_and_labels(
        evoked=evoked,
        tmin_s=tmin,
        tmax_s=tmax,
        step_ms=step_ms,
    )

    fig, axes, cbar_ax = _make_topomap_grid(n_maps=int(plot_times.size), max_cols=max_cols)

    topomap_mask_params = None
    if mask is not None:
        topomap_mask_params = {
            "marker": "o",
            "markerfacecolor": "0.75",
            "markeredgecolor": "black",
            "markersize": 10,
        }
        if mask_params is not None:
            topomap_mask_params.update(mask_params)

    if lim_val is None:
        plot_idx = np.asarray(evoked.time_as_index(plot_times), dtype=int)
        plot_idx = np.clip(plot_idx, 0, evoked.data.shape[1] - 1)
        plotted_values = evoked.data[:, plot_idx]
        finite_values = plotted_values[np.isfinite(plotted_values)]
        if finite_values.size == 0:
            limit = 1.0
        else:
            limit = float(np.max(np.abs(finite_values)))
            if limit == 0.0:
                limit = 1.0
    else:
        limit = _resolve_lim_val(stat=stat, lim_val=lim_val)

    evoked.plot_topomap(
        times=plot_times,
        axes=axes,
        show=False,
        mask=None if mask is None else mask.T,
        mask_params=topomap_mask_params,
        vlim=(-limit, limit),
        scalings=1.0,
        cmap=cmap,
        colorbar=False,
    )

    _annotate_topomap_axes(axes, labels_ms)
    fig.suptitle(title)
    _add_colorbar(fig=fig, cbar_ax=cbar_ax, lim_val=limit, cmap=cmap, label=cbar_label)

    save_figure(fig, out_dir, family, name)


def _find_time_axis(fig: plt.Figure) -> Optional[plt.Axes]:
    """Return the most likely time-course axis from an MNE joint figure."""
    for ax in fig.axes:
        xlabel = ax.get_xlabel().lower()
        if "time" in xlabel:
            return ax

    if fig.axes:
        return fig.axes[0]
    return None


def _validate_topomap_inputs(
    *,
    stat: np.ndarray,
    mask: Optional[np.ndarray],
    step_ms: float,
    max_cols: int,
) -> None:
    if stat.ndim != 2:
        raise ValueError("stat must be 2D with shape (n_times, n_channels)")
    if stat.shape[0] == 0 or stat.shape[1] == 0:
        raise ValueError("stat must have at least one time and one channel")
    if mask is not None and mask.shape != stat.shape:
        raise ValueError("mask must match stat shape")
    if step_ms <= 0:
        raise ValueError("step_ms must be > 0")
    if max_cols < 1:
        raise ValueError("max_cols must be >= 1")


def _resolve_lim_val(*, stat: np.ndarray, lim_val: float | None) -> float:
    if lim_val is not None:
        resolved = abs(float(lim_val))
        return resolved if np.isfinite(resolved) and resolved > 0.0 else 1.0

    finite_stat = np.asarray(stat, dtype=float)
    finite_stat = finite_stat[np.isfinite(finite_stat)]
    if finite_stat.size == 0:
        return 1.0

    data_limit = float(np.max(np.abs(finite_stat)))
    if not np.isfinite(data_limit) or data_limit == 0.0:
        return 1.0
    return data_limit


def _make_step_times_and_labels(
    *,
    evoked: mne.Evoked,
    tmin_s: float,
    tmax_s: float,
    step_ms: float,
) -> tuple[np.ndarray, np.ndarray]:
    start, end = _clip_time_range(evoked=evoked, tmin_s=tmin_s, tmax_s=tmax_s)

    requested_ms = np.arange(start * 1e3, end * 1e3 + step_ms, step_ms)
    sample_idx = _snap_ms_to_sample_indices(evoked=evoked, times_ms=requested_ms)

    keep = np.concatenate(([True], sample_idx[1:] != sample_idx[:-1]))
    sample_idx = sample_idx[keep]
    requested_ms = requested_ms[keep]

    snapped_times = evoked.times[sample_idx]
    labels_ms = np.round(requested_ms).astype(int)
    return snapped_times, labels_ms


def _clip_time_range(*, evoked: mne.Evoked, tmin_s: float, tmax_s: float) -> tuple[float, float]:
    data_start = float(evoked.times[0])
    data_end = float(evoked.times[-1])

    start = max(float(tmin_s), data_start)
    end = min(float(tmax_s), data_end)

    if end < start:
        raise ValueError("Requested time window does not overlap data range")
    return start, end


def _snap_ms_to_sample_indices(*, evoked: mne.Evoked, times_ms: np.ndarray) -> np.ndarray:
    t0 = float(evoked.times[0])
    sfreq = float(evoked.info["sfreq"])
    idx = np.round((times_ms / 1e3 - t0) * sfreq).astype(int)
    return np.clip(idx, 0, len(evoked.times) - 1)


def _make_topomap_grid(n_maps: int, max_cols: int) -> tuple[plt.Figure, list[plt.Axes], plt.Axes]:
    n_cols = min(max_cols, n_maps)
    n_rows = int(np.ceil(n_maps / n_cols))

    fig = plt.figure(figsize=(1.35 * n_cols + 1.0, 1.35 * n_rows + 1.0))
    gs = GridSpec(
        nrows=n_rows,
        ncols=n_cols + 1,
        figure=fig,
        width_ratios=[1.0] * n_cols + [0.08],
        wspace=0.05,
        hspace=0.10,
    )

    axes = [fig.add_subplot(gs[row, col]) for row in range(n_rows) for col in range(n_cols)]
    for ax in axes[n_maps:]:
        ax.set_visible(False)

    cbar_ax = fig.add_subplot(gs[:, -1])
    return fig, axes[:n_maps], cbar_ax


def _annotate_topomap_axes(axes: Sequence[plt.Axes], labels_ms: np.ndarray) -> None:
    for ax, label in zip(axes, labels_ms):
        ax.set_title(f"{int(label)} ms")


def _add_colorbar(
    *,
    fig: plt.Figure,
    cbar_ax: plt.Axes,
    lim_val: float,
    cmap: str,
    label: str,
) -> None:
    norm = mpl.colors.Normalize(vmin=-lim_val, vmax=lim_val)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(label)
