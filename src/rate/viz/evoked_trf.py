from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import mne
import numpy as np
import matplotlib.pyplot as plt

from .trf import (
    plot_joint_kernel as _plot_joint_kernel_shared,
)
from .base import add_time_cluster_shading, save_figure


def plot_joint_kernel(
    evoked: mne.Evoked,
    out_dir: Path,
    name: str,
    family: str = "evoked_trf",
    time_clusters: Optional[Iterable[Tuple[float, float]]] = None,
    times: Optional[Sequence[float]] = None,
    mask: Optional[np.ndarray] = None,
    mask_params: Optional[dict] = None,
    title: Optional[str] = None,
) -> None:
    """Evoked TRF joint plot wrapper using the shared custom-styled implementation."""
    _plot_joint_kernel_shared(
        evoked=evoked,
        out_dir=out_dir,
        family=family,
        name=name,
        times=times,
        time_clusters=time_clusters,
        mask=mask,
        mask_params=mask_params,
        title=title,
    )


def plot_timecourse(
    evoked: mne.Evoked,
    picks: list[str],
    out_dir: Path,
    name: str,
    family: str = "evoked_trf",
    time_clusters: Optional[Iterable[Tuple[float, float]]] = None,
) -> None:
    """Plot time course for selected electrodes."""
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
    """Plot sequence of topomaps at specified times."""
    fig = evoked.plot_topomap(
        times=times,
        show=False,
        colorbar=True,
        contours=0,
    )

    save_figure(fig, out_dir, family, name)
