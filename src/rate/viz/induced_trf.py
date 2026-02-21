
from pathlib import Path
import numpy as np
import mne

from .evoked_trf import (
    plot_joint_kernel,
    plot_timecourse,
    plot_topomap_sequence,
)


def plot_joint_induced(
    evoked: mne.Evoked,
    out_dir: Path,
    name: str,
    time_clusters=None,
):
    plot_joint_kernel(
        evoked=evoked,
        out_dir=out_dir,
        name=name,
        family="induced_trf",
        time_clusters=time_clusters,
    )


def plot_timecourse_induced(
    evoked: mne.Evoked,
    picks: list[str],
    out_dir: Path,
    name: str,
    time_clusters=None,
):
    plot_timecourse(
        evoked=evoked,
        picks=picks,
        out_dir=out_dir,
        name=name,
        family="induced_trf",
        time_clusters=time_clusters,
    )


def plot_topomap_sequence_induced(
    evoked: mne.Evoked,
    times: np.ndarray,
    out_dir: Path,
    name: str,
):
    plot_topomap_sequence(
        evoked=evoked,
        times=times,
        out_dir=out_dir,
        name=name,
        family="induced_trf",
    )