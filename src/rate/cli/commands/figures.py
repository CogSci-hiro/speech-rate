"""`figures` CLI subcommand for deterministic TRF figure rendering."""

from __future__ import annotations

import logging
from argparse import ArgumentParser, Namespace
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rate.config import load_yaml_config

LOGGER = logging.getLogger(__name__)
REQUIRES_CONFIG = False

FIGURE_JOINT = "joint"
FIGURE_TOPO = "topo"
FIGURE_TIMECOURSE = "timecourse"
FIGURE_FOOOF_PSD_PANEL = "fooof_psd_panel"
FIGURE_FOOOF_PSD_RAW_ROW = "fooof_psd_raw_row"
FIGURE_RECON_ERP_LINEAR = "reconstruction_erp_linear"
FIGURE_RECON_ERP_QUADRATIC = "reconstruction_erp_quadratic"
FIGURE_RECON_ERP_R2_TOPO = "reconstruction_erp_r2_topo"
FIGURE_CHOICES = (
    FIGURE_JOINT,
    FIGURE_TOPO,
    FIGURE_TIMECOURSE,
    FIGURE_FOOOF_PSD_PANEL,
    FIGURE_FOOOF_PSD_RAW_ROW,
    FIGURE_RECON_ERP_LINEAR,
    FIGURE_RECON_ERP_QUADRATIC,
    FIGURE_RECON_ERP_R2_TOPO,
)

DEFAULT_ELECTRODES = ["Fz", "Pz"]
DEFAULT_TOPO_STEP_MS = 50.0
CLUSTER_TSTART_COL = "t_start"
CLUSTER_TEND_COL = "t_end"
SEMANTIC_COLOR_MAP = {
    "self_rate": "firebrick",
    "other_rate": "royalblue",
    "abs_diff": "seagreen",
    "self_rate_sq": "firebrick",
    "other_rate_sq": "royalblue",
    "abs_diff_sq": "seagreen",
    "self": "firebrick",
    "other": "royalblue",
}

KERNEL_FILE_TEMPLATE = "group_kernel_{regressor}.fif"
CLUSTER_FILE_TEMPLATE = "clusters_{regressor}.parquet"
BASELINE_CONTROL_REGRESSORS = {
    "emg",
    "self_onset_impulse",
    "other_offset_impulse",
    "self_ipu_onset",
    "other_ipu_offset",
}
TARGET_BEHAVIOR_REGRESSORS = {"self_rate", "other_rate", "abs_diff"}


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
    try:
        if config is None:
            config = load_yaml_config(Path(args.config))
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
    from rate.viz import evoked_trf as evoked_trf_viz
    from rate.viz.fooof import plot_psd_quantile_grid, plot_psd_quantile_raw_row
    from rate.viz.trf import (
        plot_electrode_timecourse_panel,
        plot_joint_kernel,
        plot_reconstruction_erp_grid,
        plot_spatial_stat_topomap,
        plot_stat_topomap_sequence,
    )

    regressors = _get_trf_regressors(config)

    if figure in (FIGURE_JOINT, FIGURE_TOPO):
        reg_name = _require_regressor(regressor=regressor, figure=figure)
        is_target_regressor = reg_name in TARGET_BEHAVIOR_REGRESSORS
        evoked = _load_group_evoked(mne=mne, out_dir=out_dir, family=family, regressor=reg_name)
        cluster_intervals = (
            _load_cluster_intervals(
                cluster_path=_cluster_path(out_dir=out_dir, family=family, regressor=reg_name)
            )
            if is_target_regressor
            else None
        )

        if figure == FIGURE_JOINT:
            alpha = _get_viz_significance_p_threshold(config)
            stat_map = _load_topomap_stat_map(
                out_dir=out_dir,
                family=family,
                regressor=reg_name,
                fallback=evoked.data.T,
            )
            sig_mask = (
                _load_topomap_significance_mask(
                    out_dir=out_dir,
                    family=family,
                    regressor=reg_name,
                    alpha=alpha,
                    expected_shape=stat_map.shape,
                )
                if is_target_regressor
                else None
            )
            joint_time_clusters = cluster_intervals
            if sig_mask is not None:
                collapsed_intervals = _collapse_topomap_mask_to_time_intervals(
                    mask=sig_mask,
                    times=np.asarray(evoked.times, dtype=float),
                )
                if collapsed_intervals is not None:
                    joint_time_clusters = collapsed_intervals
            joint_plot_fn = evoked_trf_viz.plot_joint_kernel if family == "evoked_trf" else plot_joint_kernel
            joint_plot_fn(
                evoked=evoked,
                out_dir=out_dir,
                family=family,
                name=f"joint_{reg_name}",
                times=_get_trf_joint_times(config, regressor=reg_name),
                time_clusters=joint_time_clusters,
                mask=sig_mask,
                mask_params={
                    "marker": "o",
                    "markerfacecolor": "0.75",
                    "markeredgecolor": "none",
                    "linestyle": "None",
                    "markersize": 10,
                },
                title=f"{family} kernel ({reg_name})",
            )
            return

        alpha = _get_viz_significance_p_threshold(config)
        stat_map = _load_topomap_stat_map(
            out_dir=out_dir,
            family=family,
            regressor=reg_name,
            fallback=evoked.data.T,
        )
        sig_mask = (
            _load_topomap_significance_mask(
                out_dir=out_dir,
                family=family,
                regressor=reg_name,
                alpha=alpha,
                expected_shape=stat_map.shape,
            )
            if is_target_regressor
            else None
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
                "markerfacecolor": "0.75",
                "markeredgecolor": "black",
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
        condition_se = {
            reg: se
            for reg in regressors
            if (se := _load_condition_standard_error(
                mne=mne,
                out_dir=out_dir,
                family=family,
                regressor=reg,
                template_evoked=condition_evokeds[reg],
                config=config,
            )) is not None
        }
        for reg in regressors:
            se = condition_se.get(reg)
            if se is None:
                LOGGER.warning("No SEM available for timecourse condition '%s' (%s)", reg, family)
                continue
            LOGGER.info(
                "Loaded SEM for '%s' (%s): shape=%s, finite_range=[%.6g, %.6g]",
                reg,
                family,
                tuple(se.shape),
                float(np.nanmin(se)),
                float(np.nanmax(se)),
            )
        if not condition_se:
            LOGGER.warning(
                "No subject-level SEM traces found for %s timecourse panel; rendering lines without SE shading.",
                family,
            )
        xmin, xmax, ymin, ymax = _global_axis_bounds(
            condition_evokeds.values(),
            se_by_condition=condition_se if condition_se else None,
            condition_evokeds=condition_evokeds,
        )
        colors = _make_condition_colors(regressors)

        plot_electrode_timecourse_panel(
            condition_evokeds=condition_evokeds,
            condition_se=condition_se if condition_se else None,
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

    if figure == FIGURE_FOOOF_PSD_PANEL:
        try:
            df = _load_trial_psd_spectra_table(out_dir=out_dir)
            plot_psd_quantile_grid(
                subject_spectra=df,
                q=_get_fooof_psd_quantiles(config),
                out_dir=out_dir,
                family="fooof",
                name="psd_quantile_grid",
                subject_col="subject_key" if "subject_key" in df.columns else "subject_id",
                trial_col="trial_key" if "trial_key" in df.columns else "event_id",
                title="FOOOF PSD Quantile Panel",
                log_y=_get_fooof_psd_log_y(config),
                xlim_hz=_get_fooof_psd_xlim_hz(config),
            )
        except Exception as exc:
            LOGGER.warning("Falling back to placeholder FOOOF PSD panel: %s", exc)
            _write_placeholder_figure(
                out_dir=out_dir,
                family="fooof",
                name="psd_quantile_grid",
                title="FOOOF PSD Quantile Panel",
                message=(
                    "Subject-level PSD spectra table not available yet "
                    f"({exc.__class__.__name__})."
                ),
            )
        return

    if figure == FIGURE_FOOOF_PSD_RAW_ROW:
        try:
            df = _load_trial_psd_spectra_table(out_dir=out_dir)
            plot_psd_quantile_raw_row(
                subject_spectra=df,
                q=_get_fooof_psd_quantiles(config),
                out_dir=out_dir,
                family="fooof",
                name="psd_quantile_raw_row",
                subject_col="subject_key" if "subject_key" in df.columns else "subject_id",
                trial_col="trial_key" if "trial_key" in df.columns else "event_id",
                title="FOOOF Raw PSD Quantile Row",
                log_y=_get_fooof_psd_log_y(config),
                xlim_hz=_get_fooof_psd_xlim_hz(config),
            )
        except Exception as exc:
            LOGGER.warning("Falling back to placeholder FOOOF raw PSD row: %s", exc)
            _write_placeholder_figure(
                out_dir=out_dir,
                family="fooof",
                name="psd_quantile_raw_row",
                title="FOOOF Raw PSD Quantile Row",
                message=(
                    "Trial-level PSD spectra table not available yet "
                    f"({exc.__class__.__name__})."
                ),
            )
        return

    if figure in (FIGURE_RECON_ERP_LINEAR, FIGURE_RECON_ERP_QUADRATIC):
        if family != "evoked_trf":
            raise ValueError(f"{figure} is only supported for family='evoked_trf'")
        panel_kind = "linear" if figure == FIGURE_RECON_ERP_LINEAR else "quadratic"
        condition_to_model = (
            {"self": "self_rate", "other": "other_rate", "abs_diff": "abs_diff"}
            if panel_kind == "linear"
            else {"self": "self_rate_sq", "other": "other_rate_sq", "abs_diff": "abs_diff_sq"}
        )
        try:
            info_for_insets = _load_group_evoked(
                mne=mne,
                out_dir=out_dir,
                family=family,
                regressor=next(iter(condition_to_model.values())),
            ).info
            panel_data = _build_reconstruction_erp_panel_data(
                mne=mne,
                out_dir=out_dir,
                condition_to_model=condition_to_model,
                electrodes=("Fz", "Pz"),
            )
            plot_reconstruction_erp_grid(
                panel_data=panel_data,
                colors={
                    "self": SEMANTIC_COLOR_MAP["self"],
                    "other": SEMANTIC_COLOR_MAP["other"],
                    "abs_diff": SEMANTIC_COLOR_MAP["abs_diff"],
                },
                out_dir=out_dir,
                family=family,
                name=f"reconstruction_erp_{panel_kind}",
                title=f"Evoked TRF Reconstruction ERP ({panel_kind})",
                electrodes=("Fz", "Pz"),
                info=info_for_insets,
            )
        except Exception as exc:
            LOGGER.warning(
                "Falling back to placeholder reconstruction ERP figure (%s): %s",
                panel_kind,
                exc,
            )
            _write_placeholder_figure(
                out_dir=out_dir,
                family=family,
                name=f"reconstruction_erp_{panel_kind}",
                title=f"Evoked TRF Reconstruction ERP ({panel_kind})",
                message=f"Unavailable in current run ({exc.__class__.__name__}).",
            )
        return

    if figure == FIGURE_RECON_ERP_R2_TOPO:
        if family != "evoked_trf":
            raise ValueError(f"{figure} is only supported for family='evoked_trf'")
        model = _require_regressor(regressor=regressor, figure=figure)
        if model == "baseline":
            raise ValueError(f"{figure} expects a non-baseline model in --regressor")
        stats_dir = _reconstruction_erp_r2_stats_dir(out_dir=out_dir, family=family, model=model)
        mean_delta = np.asarray(np.load(stats_dir / "mean_delta_r2.npy"), dtype=float).reshape(-1)
        ch_names = _load_reconstruction_erp_r2_channel_names(stats_dir=stats_dir)
        info = _make_reconstruction_topomap_info(config=config, mne=mne, ch_names=ch_names)
        alpha = _get_viz_significance_p_threshold(config)
        sig_mask = _load_reconstruction_erp_r2_significance_mask(stats_dir=stats_dir, alpha=alpha, n_channels=mean_delta.size)
        plot_spatial_stat_topomap(
            stat=mean_delta,
            info=info,
            title=f"{family} reconstruction ERP ΔR² ({model} - baseline)",
            out_dir=out_dir,
            family=family,
            name=f"reconstruction_erp_r2_topomap_{model}_vs_baseline",
            mask=sig_mask,
            mask_params={
                "marker": "o",
                "markerfacecolor": "0.75",
                "markeredgecolor": "black",
                "markersize": 10,
            },
            cbar_label="Mean ΔR²",
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


def _get_trf_joint_times(
    config: Mapping[str, Any],
    *,
    regressor: str | None = None,
) -> list[float] | None:
    """Return optional joint topomap times from ``config.reporting.trf.joint_times_s``.

    Supports either:
    - a global list of times (seconds), or
    - a mapping like ``{"default": [...], "self_rate": [...], ...}``
    """
    reporting = config.get("reporting", {})
    trf_block = reporting.get("trf", {}) if isinstance(reporting, Mapping) else {}
    times = trf_block.get("joint_times_s") if isinstance(trf_block, Mapping) else None

    if times is None:
        return None

    if isinstance(times, Mapping):
        selected = None
        if regressor is not None and regressor in times:
            selected = times.get(regressor)
        elif "default" in times:
            selected = times.get("default")
        else:
            return None
        times = selected

    if not isinstance(times, Sequence) or isinstance(times, (str, bytes)):
        raise ValueError(
            "config.reporting.trf.joint_times_s must be a list of times or a mapping "
            "of regressor -> list of times (seconds)"
        )

    values = [float(t) for t in times]
    if not values:
        raise ValueError("config.reporting.trf.joint_times_s must contain at least one time value")
    return values


def _get_fooof_psd_quantiles(config: Mapping[str, Any]) -> int:
    reporting = config.get("reporting", {})
    fooof_block = reporting.get("fooof", {}) if isinstance(reporting, Mapping) else {}
    if not isinstance(fooof_block, Mapping):
        return 4
    value = fooof_block.get("psd_quantiles", fooof_block.get("q", 4))
    q = int(value)
    if q < 2:
        raise ValueError("config.reporting.fooof.psd_quantiles must be >= 2")
    return q


def _get_viz_significance_p_threshold(config: Mapping[str, Any]) -> float:
    """Return p-value cutoff for figure significance masks."""
    viz_block = config.get("viz", {})
    if isinstance(viz_block, Mapping):
        sig_block = viz_block.get("significance", {})
        if isinstance(sig_block, Mapping) and "p_value_threshold" in sig_block:
            value = float(sig_block["p_value_threshold"])
            if not (0.0 < value < 1.0):
                raise ValueError("config.viz.significance.p_value_threshold must be in (0, 1)")
            return value

    # Backward-compatible fallback for existing viz configs.
    reporting = config.get("reporting", {})
    if isinstance(reporting, Mapping):
        value = float(reporting.get("alpha", 0.05))
        if not (0.0 < value < 1.0):
            raise ValueError("config.reporting.alpha must be in (0, 1)")
        return value
    return 0.05


def _get_fooof_psd_log_y(config: Mapping[str, Any]) -> bool:
    reporting = config.get("reporting", {})
    fooof_block = reporting.get("fooof", {}) if isinstance(reporting, Mapping) else {}
    if not isinstance(fooof_block, Mapping):
        return False
    return bool(fooof_block.get("psd_log_y", False))


def _get_fooof_psd_xlim_hz(config: Mapping[str, Any]) -> tuple[float, float] | None:
    reporting = config.get("reporting", {})
    fooof_block = reporting.get("fooof", {}) if isinstance(reporting, Mapping) else {}
    if not isinstance(fooof_block, Mapping):
        return None
    vals = fooof_block.get("psd_xlim_hz")
    if vals is None:
        return None
    if not isinstance(vals, Sequence) or isinstance(vals, (str, bytes)) or len(vals) != 2:
        raise ValueError("config.reporting.fooof.psd_xlim_hz must be [fmin, fmax]")
    fmin = float(vals[0])
    fmax = float(vals[1])
    if not np.isfinite(fmin) or not np.isfinite(fmax) or fmin >= fmax:
        raise ValueError("config.reporting.fooof.psd_xlim_hz must satisfy finite fmin < fmax")
    return (fmin, fmax)


def _require_regressor(*, regressor: str | None, figure: str) -> str:
    """Validate that per-regressor figure modes receive a regressor."""
    if regressor is None or not regressor.strip():
        raise ValueError(f"--regressor is required when --figure is '{figure}'")
    return regressor.strip()


def _write_placeholder_figure(
    *,
    out_dir: Path,
    family: str,
    name: str,
    title: str,
    message: str,
) -> None:
    import matplotlib.pyplot as plt

    fig_dir = out_dir / "final" / "figures" / family
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / f"{name}.pdf"

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axis("off")
    ax.text(0.5, 0.68, title, ha="center", va="center", fontsize=14, weight="bold")
    ax.text(0.5, 0.45, message, ha="center", va="center", fontsize=11)
    ax.text(0.5, 0.25, "Placeholder figure generated to keep debug pipeline runnable.", ha="center", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)


def _load_trial_psd_spectra_table(*, out_dir: Path) -> pd.DataFrame:
    """Load a trial-level PSD spectra table for within-subject quantile FOOOF plotting."""
    candidates = [
        out_dir / "fooof" / "aggregated" / "trial_psd_spectra.parquet",
        out_dir / "fooof" / "aggregated" / "trial_psd_spectra.csv",
        out_dir / "fooof" / "aggregated" / "subject_psd_spectra.parquet",
        out_dir / "fooof" / "aggregated" / "subject_psd_spectra.csv",
        out_dir / "fooof" / "aggregated" / "subject_level_psd_spectra.parquet",
        out_dir / "fooof" / "aggregated" / "subject_level_psd_spectra.csv",
        out_dir / "fooof" / "aggregated" / "fooof_subject_psd_spectra.parquet",
        out_dir / "fooof" / "aggregated" / "fooof_subject_psd_spectra.csv",
    ]
    for path in candidates:
        if not path.exists():
            continue
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        else:
            df = pd.read_parquet(path)
        if df.empty:
            raise ValueError(f"PSD spectra table is empty: {path}")
        return df

    raise FileNotFoundError(
        "Expected a trial-level PSD spectra table under fooof/aggregated "
        "(e.g., trial_psd_spectra.parquet) with columns "
        "subject_key,trial_key,freq_hz,psd_raw,psd_aperiodic,psd_periodic,"
        "exponent,offset,theta_power,theta_cf"
    )


def _kernel_path(*, out_dir: Path, family: str, regressor: str) -> Path:
    """Build path to group-level kernel file for a regressor."""
    legacy = out_dir / family / KERNEL_FILE_TEMPLATE.format(regressor=regressor)
    model = _trf_model_for_regressor(regressor)
    stats_path = out_dir / "stats" / family / model / str(regressor) / "grand_average_trfs-ave.fif"
    if stats_path.exists():
        return stats_path
    return legacy


def _cluster_path(*, out_dir: Path, family: str, regressor: str) -> Path:
    """Build path to optional cluster table for a regressor."""
    legacy = out_dir / family / CLUSTER_FILE_TEMPLATE.format(regressor=regressor)
    model = _trf_model_for_regressor(regressor)
    stats_csv = out_dir / "stats" / family / model / str(regressor) / "clusters.csv"
    stats_parquet = out_dir / "stats" / family / model / str(regressor) / "clusters.parquet"
    if stats_csv.exists():
        return stats_csv
    if stats_parquet.exists():
        return stats_parquet
    return legacy


def _stats_feature_dir(*, out_dir: Path, family: str, regressor: str) -> Path:
    model = _trf_model_for_regressor(regressor)
    return out_dir / "stats" / family / model / str(regressor)


def _reconstruction_erp_r2_stats_dir(*, out_dir: Path, family: str, model: str) -> Path:
    return out_dir / "stats" / family / str(model) / "reconstruction_erp_r2_vs_baseline"


def _load_group_evoked(*, mne, out_dir: Path, family: str, regressor: str):
    """Load first evoked entry for a frozen group kernel file."""
    kernel_path = _kernel_path(out_dir=out_dir, family=family, regressor=regressor)
    if not kernel_path.exists():
        # Fallback for runs where stats-level aggregates are absent for a regressor
        # (e.g., baseline controls): build a grand average from subject FIFs.
        subject_evokeds = _load_subject_evokeds(
            mne=mne,
            out_dir=out_dir,
            family=family,
            regressor=regressor,
        )
        if subject_evokeds:
            return mne.grand_average(subject_evokeds, interpolate_bads=False)
        artifact_evoked = _build_group_evoked_from_trf_artifacts(
            mne=mne,
            out_dir=out_dir,
            family=family,
            regressor=regressor,
        )
        if artifact_evoked is not None:
            return artifact_evoked
        raise FileNotFoundError(f"Kernel file not found: {kernel_path}")

    evokeds = mne.read_evokeds(kernel_path)
    if not evokeds:
        raise ValueError(f"No evoked entries found in kernel file: {kernel_path}")
    return evokeds[0]


def _load_reconstruction_erp_r2_channel_names(*, stats_dir: Path) -> list[str]:
    path = stats_dir / "channel_names.npy"
    if not path.exists():
        raise FileNotFoundError(f"Channel names not found: {path}")
    arr = np.load(path)
    vals = [str(x) for x in np.asarray(arr).reshape(-1)]
    if not vals:
        raise ValueError(f"No channel names found in {path}")
    return vals


def _make_reconstruction_topomap_info(*, config: Mapping[str, Any], mne, ch_names: Sequence[str]):
    stats_root = config.get("stats", {})
    stats_cfg = stats_root.get("kernel_cluster") or stats_root.get("kernel") or {}
    adjacency_cfg = stats_cfg.get("adjacency", {}) if isinstance(stats_cfg, Mapping) else {}
    montage_name = str(adjacency_cfg.get("montage_name", "biosemi64"))
    info = mne.create_info(ch_names=list(ch_names), sfreq=1.0, ch_types="eeg")
    montage = mne.channels.make_standard_montage(montage_name)
    info.set_montage(montage, on_missing="ignore")
    return info


def _load_reconstruction_erp_r2_significance_mask(
    *,
    stats_dir: Path,
    alpha: float,
    n_channels: int,
) -> np.ndarray | None:
    if not (0.0 < float(alpha) < 1.0):
        return None

    labels_path = stats_dir / "cluster_labels.npy"
    if not labels_path.exists():
        return None
    labels = np.asarray(np.load(labels_path), dtype=np.int32).reshape(-1)
    if labels.size != int(n_channels):
        return None

    cluster_csv = stats_dir / "clusters.csv"
    cluster_parquet = stats_dir / "clusters.parquet"
    if cluster_csv.exists():
        table = pd.read_csv(cluster_csv)
    elif cluster_parquet.exists():
        table = pd.read_parquet(cluster_parquet)
    else:
        return None
    if "cluster_id" not in table.columns or "p_value" not in table.columns:
        return None
    sig_ids = (
        table.loc[pd.to_numeric(table["p_value"], errors="coerce") <= float(alpha), "cluster_id"]
        .astype(int)
        .to_numpy()
        + 1
    )
    if sig_ids.size == 0:
        return np.zeros(int(n_channels), dtype=bool)
    return np.isin(labels, sig_ids)


def _global_axis_bounds(
    evokeds: Iterable[Any],
    *,
    se_by_condition: Mapping[str, np.ndarray] | None = None,
    condition_evokeds: Mapping[str, Any] | None = None,
) -> tuple[float, float, float, float]:
    """Compute global axis bounds across a sequence of evoked objects."""
    evoked_list = list(evokeds)
    xmins = [float(evoked.times[0]) for evoked in evoked_list]
    xmaxs = [float(evoked.times[-1]) for evoked in evoked_list]
    ymins = [float(np.nanmin(evoked.data)) for evoked in evoked_list]
    ymaxs = [float(np.nanmax(evoked.data)) for evoked in evoked_list]

    if se_by_condition is not None and condition_evokeds is not None:
        for cond, se in se_by_condition.items():
            evoked = condition_evokeds.get(cond)
            if evoked is None:
                continue
            se_arr = np.asarray(se, dtype=float)
            if se_arr.shape != evoked.data.shape:
                continue
            ymins.append(float(np.nanmin(evoked.data - se_arr)))
            ymaxs.append(float(np.nanmax(evoked.data + se_arr)))
    return min(xmins), max(xmaxs), min(ymins), max(ymaxs)


def _load_condition_standard_error(
    *,
    mne,
    out_dir: Path,
    family: str,
    regressor: str,
    template_evoked,
    config: Mapping[str, Any] | None = None,
) -> np.ndarray | None:
    """Load subject-level kernels and return SEM array matching ``template_evoked.data``."""
    # Preferred source: subject-level z maps written by stats pipeline. This matches
    # the quantity plotted in `grand_average_trfs-ave.fif` for figure timecourses.
    zmap_se = _load_condition_standard_error_from_subject_zmaps(
        out_dir=out_dir,
        family=family,
        regressor=regressor,
        template_evoked=template_evoked,
    )
    if zmap_se is not None:
        return zmap_se

    subject_evokeds = _load_subject_evokeds(mne=mne, out_dir=out_dir, family=family, regressor=regressor)
    ref_shape = tuple(template_evoked.data.shape)
    ref_times = np.asarray(template_evoked.times, dtype=float)
    matched = []

    for evoked in subject_evokeds:
        if tuple(evoked.data.shape) != ref_shape:
            continue
        if not np.allclose(np.asarray(evoked.times, dtype=float), ref_times):
            continue
        matched.append(np.asarray(evoked.data, dtype=float))

    if len(matched) < 2:
        # Fallback for analyses that do not persist subject-level per-regressor FIFs
        # (e.g., induced TRF): derive one subject kernel per subject by averaging
        # outer-fold kernels in `kernels_outer.npy`.
        for arr in _load_subject_kernel_arrays_from_trf_artifacts(
            out_dir=out_dir,
            family=family,
            regressor=regressor,
            template_evoked=template_evoked,
        ):
            matched.append(arr)

    if len(matched) < 2:
        return None

    stack = np.stack(matched, axis=0)
    return np.nanstd(stack, axis=0, ddof=1) / np.sqrt(float(stack.shape[0]))


def _load_condition_standard_error_from_subject_zmaps(
    *,
    out_dir: Path,
    family: str,
    regressor: str,
    template_evoked,
) -> np.ndarray | None:
    """Return SEM from `subject_z_maps.npy` when available (subjects x channels x lags)."""
    stats_dir = _stats_feature_dir(out_dir=out_dir, family=family, regressor=regressor)
    path = stats_dir / "subject_z_maps.npy"
    if not path.exists():
        return None
    try:
        subj = np.asarray(np.load(path), dtype=float)
    except Exception:
        return None
    if subj.ndim != 3 or subj.shape[0] < 2:
        return None
    ref_shape = tuple(np.asarray(template_evoked.data).shape)
    if tuple(subj.shape[1:]) != ref_shape:
        return None
    sem = np.nanstd(subj, axis=0, ddof=1) / np.sqrt(float(subj.shape[0]))
    return np.asarray(sem, dtype=float)


def _load_subject_evokeds(*, mne, out_dir: Path, family: str, regressor: str) -> list[Any]:
    """Load subject-level TRF kernels for one regressor when available."""
    subjects_root = out_dir / str(family) / "subjects"
    if not subjects_root.exists():
        return []

    model = _trf_model_for_regressor(regressor)
    fname = "self_onset-ave.fif" if regressor == "self_onset_impulse" else (
        "other_offset-ave.fif" if regressor == "other_offset_impulse" else f"{regressor}-ave.fif"
    )

    evokeds: list[Any] = []
    candidates = sorted(subjects_root.rglob(f"models/{model}/{fname}"))
    # Some pipelines may store a regressor kernel under a model directory that differs
    # from the regressor name; fall back to a broader search when the canonical path misses.
    if not candidates:
        candidates = sorted(subjects_root.rglob(f"models/*/{fname}"))
    for path in candidates:
        try:
            loaded = mne.read_evokeds(path)
        except Exception:
            continue
        if loaded:
            evokeds.append(loaded[0])
    return evokeds


def _load_subject_kernel_arrays_from_trf_artifacts(
    *,
    out_dir: Path,
    family: str,
    regressor: str,
    template_evoked,
) -> list[np.ndarray]:
    """Return subject-level kernel arrays from `kernels_outer.npy`, averaged across folds."""
    subjects_root = out_dir / str(family) / "subjects"
    if not subjects_root.exists():
        return []

    model = _trf_model_for_regressor(regressor)
    lags_ref = np.asarray(template_evoked.times, dtype=float)
    shape_ref = tuple(np.asarray(template_evoked.data).shape)
    LOGGER.info("SEM fallback for %s/%s: using per-subject kernels_outer after fold averaging", family, regressor)

    arrays: list[np.ndarray] = []
    for kernels_path in sorted(subjects_root.rglob(f"models/{model}/kernels_outer.npy")):
        lags_path = kernels_path.with_name("lags.npy")
        if not lags_path.exists():
            continue
        try:
            kernels = np.asarray(np.load(kernels_path), dtype=float)
            lags = np.asarray(np.load(lags_path), dtype=float).reshape(-1)
        except Exception:
            continue
        if kernels.size == 0 or lags.size == 0:
            continue
        if lags.shape != lags_ref.shape:
            continue
        step = float(np.nanmedian(np.abs(np.diff(lags_ref)))) if lags_ref.size > 1 else 0.0
        atol = max(1e-8, step * 0.25 if np.isfinite(step) and step > 0 else 1e-8)
        if not np.allclose(lags, lags_ref, atol=atol, rtol=1e-6):
            continue

        feature_index = _feature_index_for_regressor(regressor=regressor, n_features=int(kernels.shape[2]))
        if feature_index is None:
            continue
        subj_kernel = _subject_kernel_mean_from_outer_folds(
            kernels=kernels,
            n_channels=int(shape_ref[0]),
            n_lags=int(shape_ref[1]),
            feature_index=int(feature_index),
        )
        if subj_kernel is None:
            continue
        if tuple(subj_kernel.shape) != shape_ref:
            continue
        arrays.append(subj_kernel)
    return arrays


def _subject_kernel_mean_from_outer_folds(
    *,
    kernels: np.ndarray,
    n_channels: int,
    n_lags: int,
    feature_index: int,
) -> np.ndarray | None:
    """Collapse outer-fold TRF kernels to one subject-level (channels x lags) array."""
    arr = np.asarray(kernels, dtype=float)
    if arr.ndim != 4:
        return None
    _, arr_n_channels, n_features, arr_n_lags = arr.shape
    if int(arr_n_channels) != int(n_channels) or int(arr_n_lags) != int(n_lags):
        return None
    if int(feature_index) < 0 or int(feature_index) >= int(n_features):
        return None
    # Subject estimate = mean across outer folds, then select the requested feature.
    subj_mean = np.nanmean(arr, axis=0)          # (n_channels, n_features, n_lags)
    return np.asarray(subj_mean[:, int(feature_index), :], dtype=float)


def _feature_index_for_regressor(*, regressor: str, n_features: int) -> int | None:
    token = str(regressor).strip()
    unified = {
        "self_onset_impulse": 0,
        "other_offset_impulse": 1,
        "emg": 2,
        "self_rate": 3,
        "other_rate": 4,
        "abs_diff": 5,
    }
    idx = unified.get(token)
    if idx is not None and idx < int(n_features):
        return int(idx)
    # Backward-compatible fallback for legacy single-target model artifacts.
    if token in {"self_rate", "other_rate", "abs_diff"} and int(n_features) > 0:
        return int(n_features - 1)
    return None


def _build_group_evoked_from_trf_artifacts(
    *,
    mne,
    out_dir: Path,
    family: str,
    regressor: str,
):
    subjects_root = out_dir / str(family) / "subjects"
    if not subjects_root.exists():
        return None

    model = _trf_model_for_regressor(regressor)
    subject_arrays: list[np.ndarray] = []
    lags_ref: np.ndarray | None = None
    ch_names_ref: list[str] | None = None

    for kernels_path in sorted(subjects_root.rglob(f"models/{model}/kernels_outer.npy")):
        lags_path = kernels_path.with_name("lags.npy")
        if not lags_path.exists():
            continue
        try:
            kernels = np.asarray(np.load(kernels_path), dtype=float)
            lags = np.asarray(np.load(lags_path), dtype=float).reshape(-1)
        except Exception:
            continue
        if kernels.ndim != 4 or lags.ndim != 1 or lags.size == 0:
            continue
        feature_index = _feature_index_for_regressor(regressor=regressor, n_features=int(kernels.shape[2]))
        if feature_index is None:
            continue
        subj_kernel = _subject_kernel_mean_from_outer_folds(
            kernels=kernels,
            n_channels=int(kernels.shape[1]),
            n_lags=int(kernels.shape[3]),
            feature_index=int(feature_index),
        )
        if subj_kernel is None:
            continue

        if lags_ref is None:
            lags_ref = lags
        else:
            if lags.shape != lags_ref.shape or not np.allclose(lags, lags_ref, atol=1e-8, rtol=1e-6):
                continue

        ch_names_path = kernels_path.with_name("channel_names.npy")
        if ch_names_path.exists():
            try:
                ch_names = [str(v) for v in np.asarray(np.load(ch_names_path), dtype=str).reshape(-1)]
            except Exception:
                ch_names = []
        else:
            ch_names = []
        if not ch_names:
            ch_names = [f"EEG{idx+1:03d}" for idx in range(int(subj_kernel.shape[0]))]

        if ch_names_ref is None:
            ch_names_ref = ch_names
        elif ch_names != ch_names_ref:
            continue

        subject_arrays.append(np.asarray(subj_kernel, dtype=float))

    if not subject_arrays or lags_ref is None:
        return None

    mean_data = np.nanmean(np.stack(subject_arrays, axis=0), axis=0)  # (ch, lag)
    sfreq = 1.0
    if lags_ref.size > 1:
        dt = float(np.nanmedian(np.diff(lags_ref)))
        if np.isfinite(dt) and dt > 0:
            sfreq = 1.0 / dt
    ch_names_final = ch_names_ref if ch_names_ref is not None else [f"EEG{idx+1:03d}" for idx in range(mean_data.shape[0])]
    info = mne.create_info(ch_names=ch_names_final, sfreq=float(sfreq), ch_types="eeg")
    _attach_default_eeg_montage(info=info, mne=mne)
    return mne.EvokedArray(
        data=np.asarray(mean_data, dtype=float),
        info=info,
        tmin=float(lags_ref[0]),
        comment=f"{family}:{regressor}:from_subject_kernels_outer",
        nave=int(len(subject_arrays)),
        verbose=False,
    )


def _attach_default_eeg_montage(*, info, mne) -> None:
    """Attach a standard EEG montage when channel names look mappable."""
    for montage_name in ("biosemi64", "standard_1020"):
        try:
            montage = mne.channels.make_standard_montage(montage_name)
            info.set_montage(montage, on_missing="ignore")
            return
        except Exception:
            continue


def _make_condition_colors(regressors: Sequence[str]) -> dict[str, str]:
    """Build deterministic color mapping for regressor overlays."""
    palette = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:brown", "tab:pink"]
    colors: dict[str, str] = {}
    fallback_idx = 0
    for name in regressors:
        if name in SEMANTIC_COLOR_MAP:
            colors[name] = SEMANTIC_COLOR_MAP[name]
            continue
        colors[name] = palette[fallback_idx % len(palette)]
        fallback_idx += 1
    return colors


def _trf_model_for_regressor(regressor: str) -> str:
    """Map regressor to model directory used in stats outputs."""
    value = str(regressor).strip()
    if value in TARGET_BEHAVIOR_REGRESSORS:
        return "unified"
    if value in BASELINE_CONTROL_REGRESSORS:
        return "unified"
    return value


def _build_reconstruction_erp_panel_data(
    *,
    mne,
    out_dir: Path,
    condition_to_model: Mapping[str, str],
    electrodes: tuple[str, str],
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """Aggregate subject-level reconstruction ERP traces into plotting arrays."""
    per_condition: dict[str, pd.DataFrame] = {}
    for condition, model in condition_to_model.items():
        per_condition[condition] = _load_reconstruction_waveform_subject_means(
            mne=mne,
            out_dir=out_dir,
            model=model,
            electrodes=electrodes,
        )

    panel_data: dict[str, dict[str, dict[str, np.ndarray]]] = {
        elec: {"original": {}, "reconstruction": {}} for elec in electrodes
    }
    for condition, df in per_condition.items():
        for electrode in electrodes:
            dfe = df.loc[df["electrode"] == electrode].copy()
            if dfe.empty:
                raise ValueError(f"No reconstruction ERP rows for model='{condition_to_model[condition]}' electrode='{electrode}'")
            grp = (
                dfe.groupby("time_s", as_index=False)
                .agg(
                    orig_mean=("orig_amp", "mean"),
                    orig_se=("orig_amp", _se_series),
                    rec_mean=("rec_amp", "mean"),
                    rec_se=("rec_amp", _se_series),
                )
                .sort_values("time_s", kind="mergesort")
                .reset_index(drop=True)
            )
            panel_data[electrode]["original"][condition] = np.column_stack(
                [
                    grp["time_s"].to_numpy(dtype=float),
                    grp["orig_mean"].to_numpy(dtype=float),
                    grp["orig_se"].to_numpy(dtype=float),
                ]
            )
            panel_data[electrode]["reconstruction"][condition] = np.column_stack(
                [
                    grp["time_s"].to_numpy(dtype=float),
                    grp["rec_mean"].to_numpy(dtype=float),
                    grp["rec_se"].to_numpy(dtype=float),
                ]
            )
    return panel_data


def _load_reconstruction_waveform_subject_means(
    *,
    mne,
    out_dir: Path,
    model: str,
    electrodes: tuple[str, str],
) -> pd.DataFrame:
    """Return subject-averaged (across folds) reconstruction ERP waveforms for one model."""
    subjects_root = out_dir / "evoked_trf" / "subjects"
    if not subjects_root.exists():
        raise FileNotFoundError(f"Subjects root not found: {subjects_root}")

    wanted = set(str(e) for e in electrodes)
    rows: list[pd.DataFrame] = []
    for metrics_path in sorted(subjects_root.rglob(f"models/{model}/reconstruction_metrics.parquet")):
        df = pd.read_parquet(metrics_path)
        required = {"channel", "time_s", "orig_amp", "rec_amp"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(
                f"{metrics_path} is missing columns {sorted(missing)}. "
                "Re-run evoked_trf subject jobs with the updated code to regenerate reconstruction waveforms."
            )

        dyad_id = str(df["dyad_id"].iloc[0]) if "dyad_id" in df.columns and not df.empty else str(metrics_path.parents[3].name)
        subject = str(df["subject"].iloc[0]) if "subject" in df.columns and not df.empty else str(metrics_path.parents[2].name)

        kernel_path = metrics_path.parent / f"{model}-ave.fif"
        if not kernel_path.exists():
            raise FileNotFoundError(f"Subject kernel file not found for channel mapping: {kernel_path}")
        evokeds = mne.read_evokeds(kernel_path)
        if not evokeds:
            raise ValueError(f"No evoked entries found in {kernel_path}")
        ch_names = list(evokeds[0].ch_names)
        idx_to_name = {int(i): str(name) for i, name in enumerate(ch_names)}

        dfx = df.copy()
        dfx["channel"] = pd.to_numeric(dfx["channel"], errors="coerce").astype("Int64")
        dfx = dfx.dropna(subset=["channel", "time_s", "orig_amp", "rec_amp"]).copy()
        dfx["channel"] = dfx["channel"].astype(int)
        dfx["electrode"] = dfx["channel"].map(idx_to_name)
        dfx = dfx.loc[dfx["electrode"].isin(wanted)].copy()
        if dfx.empty:
            continue

        # Subject-level waveform = average across outer folds before group averaging.
        dfx["dyad_id"] = dyad_id
        dfx["subject"] = subject
        rows.append(
            dfx.groupby(["dyad_id", "subject", "electrode", "time_s"], as_index=False)
            .agg(orig_amp=("orig_amp", "mean"), rec_amp=("rec_amp", "mean"))
            .sort_values(["dyad_id", "subject", "electrode", "time_s"], kind="mergesort")
            .reset_index(drop=True)
        )

    if not rows:
        raise FileNotFoundError(f"No reconstruction metrics found for model='{model}' under {subjects_root}")
    return pd.concat(rows, ignore_index=True)


def _se_series(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return 0.0
    return float(np.std(arr, ddof=1) / np.sqrt(float(arr.size)))


def _load_cluster_intervals(cluster_path: Path) -> list[tuple[float, float]] | None:
    """Load optional time-cluster intervals from cluster table if present."""
    if not cluster_path.exists():
        return None
    if cluster_path.suffix.lower() not in {".csv", ".parquet"}:
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
    if cluster_path.suffix.lower() not in {".csv", ".parquet"}:
        return None
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


def _collapse_topomap_mask_to_time_intervals(
    *,
    mask: np.ndarray,
    times: np.ndarray,
) -> list[tuple[float, float]] | None:
    """Collapse a (time, channel) significance mask into contiguous time intervals."""
    arr = np.asarray(mask, dtype=bool)
    t = np.asarray(times, dtype=float)

    if arr.ndim != 2 or t.ndim != 1 or arr.shape[0] != t.size or t.size == 0:
        return None

    active = np.any(arr, axis=1)
    if not np.any(active):
        return []

    # Boundaries of contiguous True runs in the collapsed time mask.
    padded = np.pad(active.astype(np.int8), (1, 1), constant_values=0)
    transitions = np.diff(padded)
    starts = np.flatnonzero(transitions == 1)
    stops = np.flatnonzero(transitions == -1) - 1

    intervals: list[tuple[float, float]] = []
    for start_idx, stop_idx in zip(starts, stops):
        intervals.append((float(t[start_idx]), float(t[stop_idx])))
    return intervals
