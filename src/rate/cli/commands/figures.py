"""`figures` CLI subcommand for deterministic TRF figure rendering."""

from __future__ import annotations

import logging
from argparse import ArgumentParser, Namespace
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)
REQUIRES_CONFIG = True

FIGURE_JOINT = "joint"
FIGURE_TOPO = "topo"
FIGURE_TIMECOURSE = "timecourse"
FIGURE_CHOICES = (FIGURE_JOINT, FIGURE_TOPO, FIGURE_TIMECOURSE)

DEFAULT_ELECTRODES = ["Fz", "Pz"]
DEFAULT_TOPO_STEP_MS = 50.0
CLUSTER_TSTART_COL = "t_start"
CLUSTER_TEND_COL = "t_end"

KERNEL_FILE_TEMPLATE = "group_kernel_{regressor}.fif"
CLUSTER_FILE_TEMPLATE = "clusters_{regressor}.parquet"


def add_subparser(subparsers) -> ArgumentParser:
    """Register the `figures` subcommand."""
    parser = subparsers.add_parser(
        "figures",
        help="Render TRF figures from frozen outputs.",
    )
    parser.add_argument("--out-dir", required=True, help="Root output directory.")
    parser.add_argument("--family", required=True, help="Analysis family (e.g. evoked_trf).")
    parser.add_argument(
        "--figure",
        required=True,
        choices=FIGURE_CHOICES,
        help="Figure type to render (joint, topo, timecourse).",
    )
    parser.add_argument(
        "--regressor",
        default=None,
        help="Regressor name. Required for joint and topo figures.",
    )
    parser.add_argument(
        "--electrodes",
        nargs="+",
        default=None,
        help="Optional override for timecourse electrodes.",
    )
    parser.add_argument("--config", default="config/config.yaml", help="Path to YAML config file.")
    parser.add_argument("--schema", default="config/schema.json", help="Path to JSON schema file.")
    parser.set_defaults(command_func=run)
    return parser


def register_subparser(subparsers) -> ArgumentParser:
    """Backward-compatible alias for CLI registration."""
    return add_subparser(subparsers)


def run(args: Namespace, config: Mapping[str, Any] | None = None) -> int:
    """Run TRF figure generation from CLI arguments."""
    if config is None:
        raise ValueError("figures command requires validated config.")

    try:
        run_trf_figures(
            out_dir=Path(args.out_dir),
            family=str(args.family),
            figure=str(args.figure),
            regressor=str(args.regressor) if args.regressor else None,
            config=config,
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
    figure: str,
    regressor: str | None,
    config: Mapping[str, Any],
    electrodes: list[str] | None = None,
) -> None:
    """Render selected TRF figure from frozen outputs.

    Parameters
    ----------
    out_dir
        Root output directory.
    family
        Analysis family subdirectory containing frozen inputs.
    figure
        Figure mode: ``joint``, ``topo``, or ``timecourse``.
    regressor
        Regressor name for per-regressor figures.
    config
        Validated project configuration mapping.
    electrodes
        Optional electrode override for the timecourse panel.
    """
    import mne
    from rate.viz.trf import (
        plot_electrode_timecourse_panel,
        plot_joint_kernel,
        plot_stat_topomap_sequence,
    )

    regressors = _get_trf_regressors(config)

    if figure in (FIGURE_JOINT, FIGURE_TOPO):
        reg_name = _require_regressor(regressor=regressor, figure=figure)
        evoked = _load_group_evoked(mne=mne, out_dir=out_dir, family=family, regressor=reg_name)
        cluster_intervals = _load_cluster_intervals(
            cluster_path=_cluster_path(out_dir=out_dir, family=family, regressor=reg_name)
        )

        if figure == FIGURE_JOINT:
            plot_joint_kernel(
                evoked=evoked,
                out_dir=out_dir,
                family=family,
                name=f"joint_{reg_name}",
                time_clusters=cluster_intervals,
                title=f"{family} kernel ({reg_name})",
            )
            return

        alpha = float(config.get("reporting", {}).get("alpha", 0.05))
        stat_map = _load_topomap_stat_map(
            out_dir=out_dir,
            family=family,
            regressor=reg_name,
            fallback=evoked.data.T,
        )
        sig_mask = _load_topomap_significance_mask(
            out_dir=out_dir,
            family=family,
            regressor=reg_name,
            alpha=alpha,
            expected_shape=stat_map.shape,
        )
        plot_stat_topomap_sequence(
            stat=stat_map,
            info=evoked.info,
            data_tmin=float(evoked.times[0]),
            tmin=float(evoked.times[0]),
            tmax=float(evoked.times[-1]),
            step_ms=DEFAULT_TOPO_STEP_MS,
            title=f"{family} topomap ({reg_name})",
            out_dir=out_dir,
            family=family,
            name=f"topomap_{reg_name}",
            mask=sig_mask,
            mask_params={
                "marker": "o",
                "markerfacecolor": "0.5",
                "markeredgecolor": "0.5",
                "markersize": 10,
            },
        )
        return

    if figure == FIGURE_TIMECOURSE:
        panel_electrodes = electrodes if electrodes is not None else _get_trf_electrodes(config)
        condition_evokeds = {
            reg: _load_group_evoked(mne=mne, out_dir=out_dir, family=family, regressor=reg)
            for reg in regressors
        }
        xmin, xmax, ymin, ymax = _global_axis_bounds(condition_evokeds.values())
        colors = _make_condition_colors(regressors)

        plot_electrode_timecourse_panel(
            condition_evokeds=condition_evokeds,
            electrodes=panel_electrodes,
            colors=colors,
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
            out_dir=out_dir,
            family=family,
            name="timecourse_panel",
            ylabel="Amplitude",
        )
        return

    raise ValueError(f"Unsupported figure mode: {figure}")


def _get_trf_regressors(config: Mapping[str, Any]) -> list[str]:
    """Return configured TRF regressors from ``reporting.trf.regressors``."""
    reporting = config.get("reporting", {})
    trf_block = reporting.get("trf", {}) if isinstance(reporting, Mapping) else {}
    regressors = trf_block.get("regressors", []) if isinstance(trf_block, Mapping) else []

    if not isinstance(regressors, Sequence) or isinstance(regressors, (str, bytes)):
        raise ValueError("config.reporting.trf.regressors must be a non-empty list of strings")

    values = [str(item) for item in regressors if str(item)]
    if not values:
        raise ValueError("config.reporting.trf.regressors must contain at least one regressor")
    return values


def _get_trf_electrodes(config: Mapping[str, Any]) -> list[str]:
    """Return configured TRF electrodes with a stable default fallback."""
    reporting = config.get("reporting", {})
    trf_block = reporting.get("trf", {}) if isinstance(reporting, Mapping) else {}
    electrodes = trf_block.get("electrodes") if isinstance(trf_block, Mapping) else None

    if electrodes is None:
        return list(DEFAULT_ELECTRODES)

    if not isinstance(electrodes, Sequence) or isinstance(electrodes, (str, bytes)):
        raise ValueError("config.reporting.trf.electrodes must be a list of channel names")

    values = [str(item) for item in electrodes if str(item)]
    if not values:
        raise ValueError("config.reporting.trf.electrodes must contain at least one channel")
    return values


def _require_regressor(*, regressor: str | None, figure: str) -> str:
    """Validate that per-regressor figure modes receive a regressor."""
    if regressor is None or not regressor.strip():
        raise ValueError(f"--regressor is required when --figure is '{figure}'")
    return regressor.strip()


def _kernel_path(*, out_dir: Path, family: str, regressor: str) -> Path:
    """Build path to group-level kernel file for a regressor."""
    legacy = out_dir / family / KERNEL_FILE_TEMPLATE.format(regressor=regressor)
    model = "baseline" if str(regressor) == "emg" else str(regressor)
    stats_path = out_dir / "stats" / family / model / str(regressor) / "grand_average_trfs-ave.fif"
    if stats_path.exists():
        return stats_path
    return legacy


def _cluster_path(*, out_dir: Path, family: str, regressor: str) -> Path:
    """Build path to optional cluster table for a regressor."""
    legacy = out_dir / family / CLUSTER_FILE_TEMPLATE.format(regressor=regressor)
    model = "baseline" if str(regressor) == "emg" else str(regressor)
    stats_csv = out_dir / "stats" / family / model / str(regressor) / "clusters.csv"
    stats_parquet = out_dir / "stats" / family / model / str(regressor) / "clusters.parquet"
    if stats_csv.exists():
        return stats_csv
    if stats_parquet.exists():
        return stats_parquet
    return legacy


def _stats_feature_dir(*, out_dir: Path, family: str, regressor: str) -> Path:
    model = "baseline" if str(regressor) == "emg" else str(regressor)
    return out_dir / "stats" / family / model / str(regressor)


def _load_group_evoked(*, mne, out_dir: Path, family: str, regressor: str):
    """Load first evoked entry for a frozen group kernel file."""
    kernel_path = _kernel_path(out_dir=out_dir, family=family, regressor=regressor)
    if not kernel_path.exists():
        raise FileNotFoundError(f"Kernel file not found: {kernel_path}")

    evokeds = mne.read_evokeds(kernel_path)
    if not evokeds:
        raise ValueError(f"No evoked entries found in kernel file: {kernel_path}")
    return evokeds[0]


def _global_axis_bounds(evokeds: Iterable[Any]) -> tuple[float, float, float, float]:
    """Compute global axis bounds across a sequence of evoked objects."""
    evoked_list = list(evokeds)
    xmins = [float(evoked.times[0]) for evoked in evoked_list]
    xmaxs = [float(evoked.times[-1]) for evoked in evoked_list]
    ymins = [float(np.nanmin(evoked.data)) for evoked in evoked_list]
    ymaxs = [float(np.nanmax(evoked.data)) for evoked in evoked_list]
    return min(xmins), max(xmaxs), min(ymins), max(ymaxs)


def _make_condition_colors(regressors: Sequence[str]) -> dict[str, str]:
    """Build deterministic color mapping for regressor overlays."""
    palette = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:brown", "tab:pink"]
    return {name: palette[idx % len(palette)] for idx, name in enumerate(regressors)}


def _load_cluster_intervals(cluster_path: Path) -> list[tuple[float, float]] | None:
    """Load optional time-cluster intervals from cluster table if present."""
    if not cluster_path.exists():
        return None

    if cluster_path.suffix.lower() == ".csv":
        table = pd.read_csv(cluster_path)
    else:
        table = pd.read_parquet(cluster_path)
    missing = {CLUSTER_TSTART_COL, CLUSTER_TEND_COL}.difference(table.columns)
    if missing:
        # Newer stats outputs may only have cluster summary columns.
        return None

    intervals: list[tuple[float, float]] = []
    for start, end in table[[CLUSTER_TSTART_COL, CLUSTER_TEND_COL]].itertuples(index=False):
        intervals.append((float(start), float(end)))
    return intervals


def _load_topomap_stat_map(
    *,
    out_dir: Path,
    family: str,
    regressor: str,
    fallback: np.ndarray,
) -> np.ndarray:
    stats_dir = _stats_feature_dir(out_dir=out_dir, family=family, regressor=regressor)
    t_obs_map_path = stats_dir / "t_obs_map.npy"
    if t_obs_map_path.exists():
        arr = np.asarray(np.load(t_obs_map_path), dtype=float)
        if arr.ndim == 2 and arr.size > 0:
            return arr
    return np.asarray(fallback, dtype=float)


def _load_topomap_significance_mask(
    *,
    out_dir: Path,
    family: str,
    regressor: str,
    alpha: float,
    expected_shape: tuple[int, int],
) -> np.ndarray | None:
    if not (0.0 < float(alpha) < 1.0):
        return None

    stats_dir = _stats_feature_dir(out_dir=out_dir, family=family, regressor=regressor)
    labels_path = stats_dir / "cluster_labels.npy"
    cluster_path = _cluster_path(out_dir=out_dir, family=family, regressor=regressor)
    if not labels_path.exists() or not cluster_path.exists():
        return None

    labels = np.asarray(np.load(labels_path), dtype=np.int32)
    if labels.shape != expected_shape:
        return None

    if cluster_path.suffix.lower() == ".csv":
        table = pd.read_csv(cluster_path)
    else:
        table = pd.read_parquet(cluster_path)
    if "cluster_id" not in table.columns or "p_value" not in table.columns:
        return None

    sig_ids = (
        table.loc[pd.to_numeric(table["p_value"], errors="coerce") <= float(alpha), "cluster_id"]
        .astype(int)
        .to_numpy()
        + 1
    )
    if sig_ids.size == 0:
        return np.zeros_like(labels, dtype=bool)
    return np.isin(labels, sig_ids)
