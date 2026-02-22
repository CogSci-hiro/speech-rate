
from pathlib import Path
from typing import Iterable, Optional, Tuple

import mne
import numpy as np
import matplotlib.pyplot as plt

from .base import save_figure, add_time_cluster_shading


def plot_joint_kernel(
    evoked: mne.Evoked,
    out_dir: Path,
    name: str,
    family: str = "evoked_trf",
    time_clusters: Optional[Iterable[Tuple[float, float]]] = None,
) -> None:
    """
    Plot MNE joint kernel with optional time cluster shading.

    Parameters
    ----------
    evoked : mne.Evoked
        Group-level kernel.
    out_dir : Path
        Output directory.
    name : str
        Figure name.
    family : str
        Analysis family.
    time_clusters : list of (t_start, t_end)
        Significant time intervals in seconds.
    """
    fig = evoked.plot_joint(show=False, ts_args=dict(time_unit="ms"))

    # Convert topomap titles to ms
    for ax in fig.axes[0:]:  # skip main time-series axis
        title = ax.get_title()
        if "s" in title:
            try:
                # Extract numeric seconds
                sec = float(title.replace(" s", "").strip())
                ax.set_title(f"{int(sec * 1000)} ms")
            except ValueError:
                pass

    for ax in fig.axes:
        for line in ax.lines:
            line.set_linewidth(1)

    # Remove last axis (colorbar is typically last)
    fig.delaxes(fig.axes[-1])

    # Add shading to main time axis
    main_ax = fig.axes[0]
    times = evoked.times

    # Set y axis
    main_ax = fig.axes[-2]  # the main figure
    main_ax.set_ylabel("A.U.")
    main_ax.set_yticks([])

    # Remove N_ave from title if present
    main_ax = fig.axes[-2]

    for text in main_ax.texts:
        if "N$_" in text.get_text():
            text.remove()

    # Add shading to main time axis
    main_ax = fig.axes[0]
    times = evoked.times
    add_time_cluster_shading(main_ax, times, time_clusters)

    save_figure(fig, out_dir, family, name)


def plot_timecourse(
    evoked: mne.Evoked,
    picks: list[str],
    out_dir: Path,
    name: str,
    family: str = "evoked_trf",
    time_clusters: Optional[Iterable[Tuple[float, float]]] = None,
) -> None:
    """
    Plot time course for selected electrodes.

    Parameters
    ----------
    evoked : mne.Evoked
        Group-level kernel.
    picks : list of str
        Electrode names.
    out_dir : Path
        Output directory.
    name : str
        Figure name.
    """
    fig, ax = plt.subplots()

    evoked.copy().pick(picks).plot(
        axes=ax,
        show=False,
        spatial_colors=True,
    )

    add_time_cluster_shading(ax, evoked.times, time_clusters)

    ax.set_title("Time Course")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")

    save_figure(fig, out_dir, family, name)


def plot_topomap_sequence(
    evoked: mne.Evoked,
    times: np.ndarray,
    out_dir: Path,
    name: str,
    family: str = "evoked_trf",
) -> None:
    """
    Plot sequence of topomaps at specified times.

    Parameters
    ----------
    evoked : mne.Evoked
        Group-level kernel.
    times : ndarray
        Time points in seconds.
    out_dir : Path
        Output directory.
    name : str
        Figure name.
    """
    fig = evoked.plot_topomap(
        times=times,
        show=False,
        colorbar=True,
        contours=0,
    )

    save_figure(fig, out_dir, family, name)
