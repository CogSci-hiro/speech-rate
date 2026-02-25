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
    mask: Optional[np.ndarray] = None,
    mask_params: Optional[dict] = None,
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
    mask
        Optional boolean significance mask with shape ``(n_times, n_channels)``.
        Markers are drawn on joint-plot topomaps at significant sensors.
    mask_params
        Optional marker styling overrides passed to MNE topomap plotting.
    title
        Optional figure title.
    """
    joint_times = _normalize_joint_times(evoked=evoked, times=times)
    topomap_args: dict[str, object] = {}
    if mask is not None:
        topomap_mask_params = {
            "marker": "o",
            "markerfacecolor": "0.75",
            "markeredgecolor": "none",
            "linestyle": "None",
            "markersize": 10,
        }
        if mask_params is not None:
            topomap_mask_params.update(mask_params)
        topomap_args["mask"] = mask.T
        topomap_args["mask_params"] = topomap_mask_params

    fig = evoked.plot_joint(
        times=joint_times,
        show=False,
        title=title,
        topomap_args=topomap_args if topomap_args else None,
    )
    _style_joint_figure(fig)
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
    condition_se: Optional[Dict[str, np.ndarray]] = None,
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
    condition_se
        Optional mapping from condition name to standard-error arrays with shape
        matching ``evoked.data`` (``n_channels, n_times``).
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
    info_for_insets = next(iter(condition_evokeds.values())).info if condition_evokeds else None

    if n_rows == 1:
        axes = [axes]

    for ax, electrode in zip(axes, electrodes):
        for condition, evoked in condition_evokeds.items():
            picked = evoked.copy().pick([electrode])
            line_color = colors.get(condition)
            if condition_se is not None and condition in condition_se:
                se = np.asarray(condition_se[condition], dtype=float)
                if se.shape == evoked.data.shape:
                    ch_idx = evoked.ch_names.index(electrode)
                    se_trace = se[ch_idx, :]
                    mean_trace = picked.data.squeeze()
                    ax.fill_between(
                        picked.times,
                        mean_trace - se_trace,
                        mean_trace + se_trace,
                        color=line_color,
                        alpha=0.18,
                        linewidth=0,
                    )
            ax.plot(
                picked.times,
                picked.data.squeeze(),
                label=condition,
                color=line_color,
                linewidth=2.0,
            )

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.axvline(0.0, linestyle="--", color="black", linewidth=1.0)
        ax.set_ylabel(ylabel)
        ax.set_title(electrode)
        _add_electrode_inset(ax=ax, info=info_for_insets, electrode=electrode)

    axes[-1].set_xlabel(xlabel)
    axes[0].legend(frameon=False)
    fig.tight_layout()

    save_figure(fig, out_dir, family, name)


def plot_reconstruction_erp_grid(
    *,
    panel_data: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    colors: Dict[str, str],
    out_dir: Path,
    family: str,
    name: str,
    title: str,
    electrodes: Sequence[str] = ("Fz", "Pz"),
    signal_order: Sequence[str] = ("original", "reconstruction"),
    xlabel: str = "Time (s)",
    ylabel: str = "Amplitude",
    info: mne.Info | None = None,
) -> None:
    """Plot a 2x2 ERP panel (rows=electrodes, cols=original/reconstruction).

    ``panel_data`` is expected as:
    ``panel_data[electrode][signal][condition] -> array(n_time, 3)``, where columns are
    ``time_s, mean, se``.
    """
    if len(electrodes) != 2:
        raise ValueError("electrodes must contain exactly two entries for the 2x2 layout")
    if len(signal_order) != 2:
        raise ValueError("signal_order must contain exactly two entries for the 2x2 layout")

    all_times: list[np.ndarray] = []
    ymins: list[float] = []
    ymaxs: list[float] = []
    for electrode in electrodes:
        if electrode not in panel_data:
            raise ValueError(f"Missing panel data for electrode: {electrode}")
        for signal in signal_order:
            if signal not in panel_data[electrode]:
                raise ValueError(f"Missing signal='{signal}' for electrode='{electrode}'")
            for condition, arr in panel_data[electrode][signal].items():
                vals = np.asarray(arr, dtype=float)
                if vals.ndim != 2 or vals.shape[1] != 3:
                    raise ValueError(f"Expected shape (n_time, 3) for {electrode}/{signal}/{condition}")
                t = vals[:, 0]
                m = vals[:, 1]
                se = vals[:, 2]
                all_times.append(t)
                finite_lo = (m - se)[np.isfinite(m - se)]
                finite_hi = (m + se)[np.isfinite(m + se)]
                if finite_lo.size:
                    ymins.append(float(np.min(finite_lo)))
                if finite_hi.size:
                    ymaxs.append(float(np.max(finite_hi)))

    if not all_times or not ymins or not ymaxs:
        raise ValueError("panel_data does not contain plottable values")

    xmin = min(float(np.min(t)) for t in all_times if t.size)
    xmax = max(float(np.max(t)) for t in all_times if t.size)
    ymin = min(ymins)
    ymax = max(ymaxs)
    if not np.isfinite(ymin) or not np.isfinite(ymax):
        ymin, ymax = -1.0, 1.0
    if np.isclose(ymin, ymax):
        pad = 1.0 if np.isclose(ymax, 0.0) else abs(ymax) * 0.1
        ymin -= pad
        ymax += pad

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.0), sharex=True, sharey=True)
    signal_titles = {"original": "Original", "reconstruction": "Reconstruction"}

    for row, electrode in enumerate(electrodes):
        for col, signal in enumerate(signal_order):
            ax = axes[row, col]
            for condition, arr in panel_data[electrode][signal].items():
                vals = np.asarray(arr, dtype=float)
                t = vals[:, 0]
                m = vals[:, 1]
                se = vals[:, 2]
                color = colors.get(condition)
                ax.fill_between(t, m - se, m + se, color=color, alpha=0.18, linewidth=0)
                ax.plot(t, m, color=color, linewidth=2.2, label=condition)
            ax.axvline(0.0, linestyle="--", color="black", linewidth=1.0)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            if row == 0:
                ax.set_title(signal_titles.get(signal, str(signal).title()))
            if col == 0:
                ax.set_ylabel(f"{electrode}\n{ylabel}")
            if row == 1:
                ax.set_xlabel(xlabel)
            _add_electrode_inset(ax=ax, info=info, electrode=electrode)

    axes[0, 0].legend(frameon=False, loc="best")
    fig.suptitle(title)
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


def plot_spatial_stat_topomap(
    *,
    stat: np.ndarray,
    info: mne.Info,
    title: str,
    out_dir: Path,
    family: str,
    name: str,
    mask: Optional[np.ndarray] = None,
    mask_params: Optional[dict] = None,
    lim_val: float | None = None,
    cmap: str = "RdBu_r",
    cbar_label: str = "Value",
) -> None:
    """Plot a single-channel-space topomap (no time axis) with optional significance markers."""
    values = np.asarray(stat, dtype=float).reshape(-1)
    if values.size != len(info["ch_names"]):
        raise ValueError("stat length must match number of channels in info")

    sig_mask = None
    if mask is not None:
        sig_mask = np.asarray(mask, dtype=bool).reshape(-1)
        if sig_mask.shape != values.shape:
            raise ValueError("mask must match stat shape")

    if lim_val is None:
        finite = values[np.isfinite(values)]
        limit = float(np.max(np.abs(finite))) if finite.size else 1.0
        if not np.isfinite(limit) or limit <= 0.0:
            limit = 1.0
    else:
        limit = _resolve_lim_val(stat=values, lim_val=lim_val)

    topomap_mask_params = {
        "marker": "o",
        "markerfacecolor": "0.75",
        "markeredgecolor": "black",
        "markersize": 10,
    }
    if mask_params is not None:
        topomap_mask_params.update(mask_params)

    fig = plt.figure(figsize=(5.8, 4.8))
    gs = GridSpec(1, 2, width_ratios=[20, 1], figure=fig, wspace=0.15)
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])

    topo_out = mne.viz.plot_topomap(
        values,
        info,
        axes=ax,
        show=False,
        cmap=cmap,
        vlim=(-limit, limit),
        mask=sig_mask,
        mask_params=topomap_mask_params if sig_mask is not None else None,
    )
    im = topo_out[0] if isinstance(topo_out, tuple) else topo_out
    ax.set_title(title)

    norm = mpl.colors.Normalize(vmin=-limit, vmax=limit)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, cax=cax, label=cbar_label)
    # Suppress "im" unused complaints in some linters while keeping MNE call explicit.
    _ = im

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


def _style_joint_figure(fig: plt.Figure, line_width: float = 2.0) -> None:
    """Apply consistent styling to MNE joint plots."""
    if not fig.axes:
        return

    # Convert topomap titles to ms when MNE leaves them in seconds.
    for ax in fig.axes:
        title = (ax.get_title() or "").strip()
        if title.endswith(" s"):
            try:
                sec = float(title.replace(" s", "").strip())
            except ValueError:
                continue
            ax.set_title(f"{int(round(sec * 1000.0))} ms")

    for ax in fig.axes:
        for line in ax.lines:
            line.set_linewidth(line_width)

    # MNE joint plots typically append a colorbar axis at the end.
    last_ax = fig.axes[-1]
    if not last_ax.lines and not last_ax.collections and not last_ax.images:
        # empty/placeholder axis: keep it
        pass
    else:
        bbox = last_ax.get_position()
        if bbox.width < 0.06 and bbox.height > 0.2:
            fig.delaxes(last_ax)

    main_ax = _find_time_axis(fig)
    if main_ax is not None:
        main_ax.set_ylabel("A.U.")
        main_ax.set_yticks([])
        for text in list(main_ax.texts):
            if "N$_" in text.get_text():
                text.remove()


def _add_electrode_inset(
    *,
    ax: plt.Axes,
    info: mne.Info | None,
    electrode: str,
    bounds: tuple[float, float, float, float] = (0.80, 0.54, 0.18, 0.34),
) -> None:
    """Add a small head inset highlighting one electrode location."""
    if info is None:
        return
    ch_names = list(info.get("ch_names", []))
    if electrode not in ch_names:
        return

    idx = ch_names.index(electrode)
    values = np.zeros(len(ch_names), dtype=float)
    mask = np.zeros(len(ch_names), dtype=bool)
    mask[idx] = True

    inset_ax = ax.inset_axes(bounds)
    try:
        topo = mne.viz.plot_topomap(
            values,
            info,
            axes=inset_ax,
            show=False,
            contours=0,
            sensors=True,
            cmap="Greys",
            vlim=(-1.0, 1.0),
            mask=mask,
            mask_params={
                "marker": "o",
                "markerfacecolor": "white",
                "markeredgecolor": "black",
                "markersize": 6,
            },
        )
        image = topo[0] if isinstance(topo, tuple) else topo
        if image is not None and hasattr(image, "set_alpha"):
            image.set_alpha(0.0)
    except Exception:
        inset_ax.remove()
        return

    inset_ax.set_xticks([])
    inset_ax.set_yticks([])
    inset_ax.set_title("")
    inset_ax.set_facecolor("none")
    for spine in inset_ax.spines.values():
        spine.set_visible(False)


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
