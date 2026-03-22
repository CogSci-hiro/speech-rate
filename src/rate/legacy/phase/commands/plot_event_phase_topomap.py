"""CLI command for event-phase locking topomap grid."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any

import mne
import numpy as np

from cli.commands._io import save_figure_all_formats
from viz.phase_topomap import plot_event_phase_topomaps

REQUIRES_CONFIG = False


def add_subparser(subparsers: Any) -> ArgumentParser:
    """Register the `plot-event-phase-topomap` command."""
    parser = subparsers.add_parser(
        "plot-event-phase-topomap",
        help="Plot 4-panel event phase-locking EEG topomap grid.",
    )
    parser.add_argument("--t-values", required=True, help="Path to t_values.npy")
    parser.add_argument("--clusters", required=True, help="Path to cluster_masks.npy")
    parser.add_argument("--info", required=True, help="Path to MNE info FIF file.")
    parser.add_argument(
        "--output",
        required=True,
        help="Output stem path (without extension), e.g. figures/event_phase_topomap",
    )
    parser.set_defaults(command_func=run)
    return parser


def register_subparser(subparsers: Any) -> ArgumentParser:
    """Backward-compatible alias for CLI registration."""
    return add_subparser(subparsers)


def _load_dict_from_npy(path: str, *, label: str) -> dict[str, np.ndarray]:
    """Load dictionary payload from `.npy` or `.npz`."""
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".npz":
        with np.load(source, allow_pickle=False) as archive:
            return {str(key): np.asarray(archive[key]) for key in archive.files}
    payload = np.load(source, allow_pickle=True)
    if isinstance(payload, np.ndarray) and payload.dtype == object and payload.shape == ():
        item = payload.item()
    else:
        item = payload
    if not isinstance(item, dict):
        raise ValueError(f"{label} must contain a dictionary payload: {source}")
    return {str(key): np.asarray(value) for key, value in item.items()}


def run(args: Namespace, config: Any | None = None) -> int:
    """Execute event phase topomap plotting."""
    del config
    t_values = _load_dict_from_npy(args.t_values, label="--t-values")
    cluster_masks = _load_dict_from_npy(args.clusters, label="--clusters")
    info = mne.io.read_info(args.info)

    figure = plot_event_phase_topomaps(
        t_values=t_values,
        cluster_masks=cluster_masks,
        info=info,
        vlim=None,
    )

    output_stem = Path(args.output)
    output_dir = output_stem.parent if str(output_stem.parent) != "" else Path(".")
    save_figure_all_formats(
        figure=figure,
        output_dir=output_dir,
        base_name=output_stem.name,
    )
    return 0
