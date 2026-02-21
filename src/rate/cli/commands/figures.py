"""`figures` CLI subcommand for TRF visualization rendering."""

from __future__ import annotations

import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)
REQUIRES_CONFIG = False

_DEFAULT_ELECTRODES = ["Fz", "Pz"]
_DEFAULT_TOPO_STEP_MS = 50.0
_CONDITION_LABEL = "kernel"
_CLUSTER_TSTART_COL = "t_start"
_CLUSTER_TEND_COL = "t_end"


def add_subparser(subparsers) -> ArgumentParser:
    """Register the `figures` subcommand."""
    parser = subparsers.add_parser(
        "figures",
        help="Render TRF figures from frozen outputs.",
    )
    parser.add_argument("--out-dir", required=True, help="Root output directory.")
    parser.add_argument("--family", required=True, help="Analysis family (e.g. evoked_trf).")
    parser.add_argument("--kernel", required=True, help="Kernel evoked filename inside out_dir/family.")
    parser.add_argument(
        "--clusters",
        default=None,
        help="Optional cluster parquet filename inside out_dir/family.",
    )
    parser.add_argument(
        "--electrodes",
        nargs="+",
        default=list(_DEFAULT_ELECTRODES),
        help="Electrodes for the timecourse panel (default: Fz Pz).",
    )
    parser.set_defaults(command_func=run)
    return parser


def register_subparser(subparsers) -> ArgumentParser:
    """Backward-compatible alias for CLI registration."""
    return add_subparser(subparsers)


def run(args: Namespace, config=None) -> int:
    """Run TRF figure generation from CLI arguments."""
    del config  # Unused; command does not require project config.

    try:
        run_trf_figures(
            out_dir=Path(args.out_dir),
            family=str(args.family),
            kernel_filename=str(args.kernel),
            cluster_filename=str(args.clusters) if args.clusters else None,
            electrodes=list(args.electrodes) if args.electrodes else None,
        )
    except Exception as exc:
        LOGGER.error("figures command failed: %s", exc)
        raise

    LOGGER.info("TRF figures generated under %s", Path(args.out_dir) / "final" / "figures" / str(args.family))
    return 0


def run_trf_figures(
    *,
    out_dir: Path,
    family: str,
    kernel_filename: str,
    cluster_filename: str | None = None,
    electrodes: list[str] | None = None,
) -> None:
    """Render TRF figures from frozen outputs.

    Parameters
    ----------
    out_dir
        Root output directory.
    family
        Analysis family subdirectory containing frozen inputs.
    kernel_filename
        Evoked file name under ``{out_dir}/{family}``.
    cluster_filename
        Optional parquet file with ``t_start`` and ``t_end`` columns.
    electrodes
        Optional electrode list for timecourse panel.
    """
    import mne
    from rate.viz.trf import (
        plot_electrode_timecourse_panel,
        plot_joint_kernel,
        plot_stat_topomap_sequence,
    )

    family_dir = out_dir / family
    kernel_path = family_dir / kernel_filename

    if not kernel_path.exists():
        raise FileNotFoundError(f"Kernel file not found: {kernel_path}")

    evokeds = mne.read_evokeds(kernel_path)
    if not evokeds:
        raise ValueError(f"No evoked entries found in kernel file: {kernel_path}")
    evoked = evokeds[0]

    cluster_intervals = _load_cluster_intervals(
        cluster_path=(family_dir / cluster_filename) if cluster_filename else None
    )

    panel_electrodes = electrodes if electrodes is not None else list(_DEFAULT_ELECTRODES)

    plot_joint_kernel(
        evoked=evoked,
        out_dir=out_dir,
        family=family,
        name="joint_kernel",
        time_clusters=cluster_intervals,
        title=f"{family} kernel",
    )

    plot_electrode_timecourse_panel(
        condition_evokeds={_CONDITION_LABEL: evoked},
        electrodes=panel_electrodes,
        colors={_CONDITION_LABEL: "tab:blue"},
        xmin=float(evoked.times[0]),
        xmax=float(evoked.times[-1]),
        ymin=float(np.nanmin(evoked.data)),
        ymax=float(np.nanmax(evoked.data)),
        out_dir=out_dir,
        family=family,
        name="timecourse_panel",
        ylabel="Amplitude",
    )

    plot_stat_topomap_sequence(
        stat=evoked.data.T,
        info=evoked.info,
        data_tmin=float(evoked.times[0]),
        tmin=float(evoked.times[0]),
        tmax=float(evoked.times[-1]),
        step_ms=_DEFAULT_TOPO_STEP_MS,
        title=f"{family} topomap sequence",
        out_dir=out_dir,
        family=family,
        name="topomap_sequence",
    )


def _load_cluster_intervals(
    cluster_path: Optional[Path],
) -> Optional[list[tuple[float, float]]]:
    """Load optional time-cluster intervals from parquet."""
    if cluster_path is None or not cluster_path.exists():
        return None

    table = pd.read_parquet(cluster_path)

    missing = {
        _CLUSTER_TSTART_COL,
        _CLUSTER_TEND_COL,
    }.difference(table.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Cluster file missing required columns: {missing_list}")

    intervals: list[tuple[float, float]] = []
    for start, end in table[[_CLUSTER_TSTART_COL, _CLUSTER_TEND_COL]].itertuples(index=False):
        intervals.append((float(start), float(end)))

    return intervals
