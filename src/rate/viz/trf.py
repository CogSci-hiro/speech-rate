"""
TRF-related figures (evoked and induced share rendering grammar).
"""

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib as mpl
import mne
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec


from .base import save_figure


# =============================================================================
# Public API
# =============================================================================

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
    lim_val: float | None = None,
    max_cols: int = 10,
    cmap: str = "RdBu_r",
    cbar_label: str = "Value",
) -> None:
    """
    Plot statistical topomap grid (generic TRF).

    stat shape: (n_times, n_channels)
    mask shape: (n_times, n_channels) or None
    """

    _validate_topomap_inputs(stat=stat, mask=mask, step_ms=step_ms)
    lim_val = _resolve_lim_val(stat=stat, lim_val=lim_val)

    evoked = mne.EvokedArray(stat.T, info, tmin=float(data_tmin))

    times_s, labels_ms = _make_step_times_s_and_labels_ms(
        evoked,
        tmin_s=tmin,
        tmax_s=tmax,
        step_ms=step_ms,
    )

    fig, axes_flat, cbar_ax = _make_topomap_grid(
        n_maps=int(times_s.size),
        max_cols=max_cols,
    )

    draw_axes = axes_flat[: int(times_s.size)]

    evoked.plot_topomap(
        axes=draw_axes,
        times=times_s,
        show=False,
        mask=None if mask is None else mask.T,
        vlim=(-lim_val, lim_val),
        cmap=cmap,
        colorbar=False,
    )

    _annotate_topomap_times(draw_axes, labels_ms=labels_ms)

    fig.suptitle(title)
    _add_topomap_colorbar(
        fig,
        cbar_ax=cbar_ax,
        lim_val=lim_val,
        cmap=cmap,
        cbar_label=cbar_label,
    )

    save_figure(fig, out_dir, family, name)


def _validate_topomap_inputs(*, stat: np.ndarray, mask: Optional[np.ndarray], step_ms: float) -> None:
    if stat.ndim != 2:
        raise ValueError("stat must be 2D (n_times, n_channels)")
    if mask is not None and mask.shape != stat.shape:
        raise ValueError("mask must match stat shape")
    if step_ms <= 0:
        raise ValueError("step_ms must be > 0")


def _resolve_lim_val(*, stat: np.ndarray, lim_val: float | None) -> float:
    if lim_val is None:
        return float(np.max(np.abs(stat)))
    return float(lim_val)


def _make_topomap_grid(n_maps: int, max_cols: int):
    n_cols = int(min(max_cols, n_maps))
    n_rows = int(np.ceil(n_maps / n_cols))

    fig = plt.figure(figsize=(1.35 * n_cols + 1, 1.35 * n_rows + 1))
    gs = GridSpec(
        n_rows,
        n_cols + 1,
        figure=fig,
        width_ratios=[1.0] * n_cols + [0.08],
        wspace=0.05,
        hspace=0.1,
    )

    axes_flat = [
        fig.add_subplot(gs[r, c])
        for r in range(n_rows)
        for c in range(n_cols)
    ]

    cbar_ax = fig.add_subplot(gs[:, -1])

    for ax in axes_flat[n_maps:]:
        ax.set_visible(False)

    return fig, axes_flat, cbar_ax


def _make_step_times_s_and_labels_ms(
    evoked: mne.Evoked,
    *,
    tmin_s: float,
    tmax_s: float,
    step_ms: float,
):
    tmin_s, tmax_s = _clip_time_range_to_evoked(evoked, tmin_s=tmin_s, tmax_s=tmax_s)

    grid_ms = np.arange(tmin_s * 1e3, tmax_s * 1e3 + step_ms, step_ms)

    t0 = float(evoked.times[0])
    sfreq = float(evoked.info["sfreq"])
    idx = np.round((grid_ms / 1e3 - t0) * sfreq).astype(int)
    idx = np.clip(idx, 0, len(evoked.times) - 1)

    keep = np.concatenate(([True], idx[1:] != idx[:-1]))
    idx = idx[keep]
    grid_ms = grid_ms[keep]

    times_s = evoked.times[idx]
    labels_ms = np.round(grid_ms / step_ms) * step_ms

    return times_s, labels_ms.astype(int)


def _add_topomap_colorbar(
    fig,
    *,
    cbar_ax,
    lim_val,
    cmap,
    cbar_label,
):
    norm = mpl.colors.Normalize(vmin=-lim_val, vmax=lim_val)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(cbar_label)


def plot_electrode_timecourse_panel(
    *,
    condition_evokeds: Dict[str, List[mne.Evoked]],
    electrodes: Sequence[str],
    colors: Dict[str, str],
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    out_dir: Path,
    family: str,
    name: str,
    ylabel: str = "Amplitude",
    xlabel: str = "Time (s)",
) -> None:
    """
    Plot multi-condition TRF electrode timecourse panel.

    Parameters
    ----------
    condition_evokeds : dict
        Mapping condition_name -> list of subject-level Evoked objects.
    electrodes : list of str
        Electrodes to plot (rows).
    colors : dict
        Mapping condition_name -> color.
    xmin, xmax : float
        Time limits (seconds).
    ymin, ymax : float
        Amplitude limits.
    """

    n_rows = len(electrodes)
    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(6, 2.5 * n_rows),
        sharex=True,
    )

    if n_rows == 1:
        axes = [axes]

    # Combine per condition
    combined = {
        cond: mne.combine_evoked(ev_list, weights="equal")
        for cond, ev_list in condition_evokeds.items()
    }

    for ax, electrode in zip(axes, electrodes):

        for cond, evoked in combined.items():

            ev = evoked.copy().pick([electrode])

            ax.plot(
                ev.times,
                ev.data.squeeze(),
                label=cond,
                color=colors.get(cond, None),
                linewidth=2,
            )

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.axvline(0, linestyle="--", color="black", linewidth=1)
        ax.set_ylabel(electrode)

    axes[-1].set_xlabel(xlabel)
    axes[0].legend(frameon=False)

    fig.tight_layout()

    save_figure(fig, out_dir, family, name)


# =============================================================================
# Joint plot
# =============================================================================

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
    """
    Plot MNE joint TRF kernel.

    Parameters
    ----------
    evoked : mne.Evoked
        Group-level TRF kernel.
    out_dir : Path
        Output directory.
    family : str
        "evoked_trf" or "induced_trf".
    name : str
        Figure name.
    times : sequence of float | None
        Specific times for topomaps. If None, MNE chooses automatically.
    time_clusters : iterable of (t_start, t_end)
        Significant time windows (seconds).
    title : str | None
        Figure title.
    """

    fig = evoked.plot_joint(
        times=times,
        show=False,
        title=title,
    )

    # Main timecourse axis is first axis
    main_ax = fig.axes[0]

    if time_clusters is not None:
        add_time_cluster_shading(
            main_ax,
            evoked.times,
            time_clusters,
        )

    save_figure(fig, out_dir, family, name)


# =============================================================================
# Electrode panel (multi-condition)
# =============================================================================

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
    """
    Plot multi-condition TRF timecourses for selected electrodes.

    Parameters
    ----------
    condition_evokeds : dict
        Mapping condition name -> group-level Evoked.
    electrodes : list of str
        Electrodes to plot (one row per electrode).
    colors : dict
        Mapping condition name -> color.
    """

    n_rows = len(electrodes)
    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(6, 2.5 * n_rows),
        sharex=True,
    )

    if n_rows == 1:
        axes = [axes]

    for ax, electrode in zip(axes, electrodes):

        for cond, evoked in condition_evokeds.items():

            ev = evoked.copy().pick([electrode])

            ax.plot(
                ev.times,
                ev.data.squeeze(),
                label=cond,
                color=colors.get(cond, None),
                linewidth=2,
            )

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.axvline(0, linestyle="--", color="black", linewidth=1)
        ax.set_ylabel(electrode)

    axes[-1].set_xlabel(xlabel)
    axes[0].legend(frameon=False)

    fig.tight_layout()

    save_figure(fig, out_dir, family, name)


# =============================================================================
# Topomap sequence
# =============================================================================

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
    lim_val: float | None = None,
    max_cols: int = 10,
    cmap: str = "RdBu_r",
    cbar_label: str = "Value",
) -> None:
    """
    Plot statistical TRF topomap grid.

    stat shape: (n_times, n_channels)
    mask shape: same shape or None
    """

    if stat.ndim != 2:
        raise ValueError("stat must be (n_times, n_channels)")

    if mask is not None and mask.shape != stat.shape:
        raise ValueError("mask must match stat shape")

    if lim_val is None:
        lim_val = float(np.max(np.abs(stat)))

    evoked = mne.EvokedArray(stat.T, info, tmin=float(data_tmin))

    times_s, labels_ms = _make_step_times_s_and_labels_ms(
        evoked,
        tmin_s=tmin,
        tmax_s=tmax,
        step_ms=step_ms,
    )

    fig = evoked.plot_topomap(
        times=times_s,
        show=False,
        mask=None if mask is None else mask.T,
        vlim=(-lim_val, lim_val),
        cmap=cmap,
        colorbar=True,
    )

    fig.suptitle(title)

    save_figure(fig, out_dir, family, name)


# =============================================================================
# Utilities
# =============================================================================

def _make_step_times_s_and_labels_ms(
    evoked: mne.Evoked,
    *,
    tmin_s: float,
    tmax_s: float,
    step_ms: float,
):
    if step_ms <= 0:
        raise ValueError("step_ms must be > 0")

    tmin_s = max(tmin_s, float(evoked.times[0]))
    tmax_s = min(tmax_s, float(evoked.times[-1]))

    grid_ms = np.arange(tmin_s * 1e3, tmax_s * 1e3 + step_ms, step_ms)

    t0 = float(evoked.times[0])
    sfreq = float(evoked.info["sfreq"])

    idx = np.round((grid_ms / 1e3 - t0) * sfreq).astype(int)
    idx = np.clip(idx, 0, len(evoked.times) - 1)

    keep = np.concatenate(([True], idx[1:] != idx[:-1]))
    idx = idx[keep]
    grid_ms = grid_ms[keep]

    times_s = evoked.times[idx]
    labels_ms = np.round(grid_ms / step_ms) * step_ms

    return times_s, labels_ms.astype(int)
