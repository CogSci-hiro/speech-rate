"""Topographic main-result figures for pooled lmeEEG inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import mne
import numpy as np
from matplotlib import gridspec
from matplotlib.patches import ConnectionPatch

from rate.lmeeeg.io import build_lmeeeg_model_output_paths
from rate.viz.theme import apply_theme, get_role_palette, get_theme_color


def build_lmeeeg_main_results_manifest(
    *,
    config: Mapping[str, Any],
    out_dir: Path,
    manifest_path: Path,
    figure_config: Mapping[str, Any] | None = None,
    theme_config: Mapping[str, Any] | None = None,
    config_section: str = "lmeeeg",
    figure_subdir: str = "main_figures",
    formats: Sequence[str] = ("png", "pdf"),
    dpi: int = 300,
) -> dict[str, object]:
    """Render the main lmeEEG results figure from two configured effects."""
    figure_config = dict(figure_config or {})
    similarity_figure_config = dict(figure_config.pop("similarity_summary", {}) or {})
    behavioural_figure_config = dict(figure_config.pop("behavioural", {}) or {})
    variance_partition_figure_config = dict(figure_config.pop("variance_partition", {}) or {})
    theme = apply_theme(theme_config)
    model_specs = dict(figure_config.get("models") or {})
    self_spec = dict(model_specs.get("self") or {})
    other_spec = dict(model_specs.get("other") or {})
    if not self_spec or not other_spec:
        raise ValueError("`viz.figures.lmeeeg_main_results.models.self` and `.other` must both be configured.")

    self_payload = _load_effect_payload(
        config,
        out_dir=out_dir,
        config_section=config_section,
        effect_config=self_spec,
    )
    other_payload = _load_effect_payload(
        config,
        out_dir=out_dir,
        config_section=config_section,
        effect_config=other_spec,
    )
    _validate_payload_compatibility(self_payload, other_payload)

    times = np.asarray(self_payload["times"], dtype=float)
    plotting_tmin_s = figure_config.get("plotting_tmin_s")
    plotting_tmax_s = figure_config.get("plotting_tmax_s")
    time_mask = _resolve_time_mask(times, plotting_tmin_s=plotting_tmin_s, plotting_tmax_s=plotting_tmax_s)
    alpha_threshold = float(figure_config.get("significance_threshold", 0.05))
    robust_percentile = float(figure_config.get("robust_percentile", 97.5))
    manual_vlim = figure_config.get("manual_vlim")
    font_scale = float(figure_config.get("font_scale", 3.0))
    shared_vmin, shared_vmax = _compute_shared_vlim(
        np.asarray(self_payload["observed_map"], dtype=float),
        np.asarray(other_payload["observed_map"], dtype=float),
        requested_vlim=float(manual_vlim) if manual_vlim is not None else None,
        percentile=robust_percentile,
    )
    info = _build_topomap_info(list(self_payload["channel_names"]), config=config)
    palette = get_role_palette(theme)
    line_self_color = str(self_spec.get("line_color", palette["self"]))
    line_other_color = str(other_spec.get("line_color", palette["other"]))
    shared_label_color = get_theme_color("neutral", theme)
    reference_color = get_theme_color("reference", theme)

    self_pvalues = np.asarray(self_payload["corrected_p_values"], dtype=float)
    other_pvalues = np.asarray(other_payload["corrected_p_values"], dtype=float)
    sig_self_matrix = self_pvalues < alpha_threshold
    sig_other_matrix = other_pvalues < alpha_threshold
    self_significant_counts = sig_self_matrix.sum(axis=0)
    other_significant_counts = sig_other_matrix.sum(axis=0)

    self_topomap_times_s = _resolve_requested_topomap_times(
        times,
        requested_times=self_spec.get("topomap_times_s") or [],
        time_mask=time_mask,
        label=str(self_spec.get("label", "Self")),
    )
    other_topomap_times_s = _resolve_requested_topomap_times(
        times,
        requested_times=other_spec.get("topomap_times_s") or [],
        time_mask=time_mask,
        label=str(other_spec.get("label", "Other")),
    )

    n_self_maps = len(self_topomap_times_s)
    n_other_maps = len(other_topomap_times_s)
    n_cols = max(n_self_maps, n_other_maps, 1)
    figure = plt.figure(
        figsize=(
            float(figure_config.get("figure_width", 14.0)),
            float(figure_config.get("figure_height", 10.5)),
        )
    )
    outer_grid = gridspec.GridSpec(
        nrows=4,
        ncols=2,
        figure=figure,
        width_ratios=[1.0, 0.04],
        height_ratios=[1.1, 1.5, float(figure_config.get("bottom_row_spacer_height", 0.38)), 1.1],
        hspace=float(figure_config.get("hspace", 0.20)),
        wspace=float(figure_config.get("wspace", 0.12)),
    )
    top_grid = gridspec.GridSpecFromSubplotSpec(1, n_cols, subplot_spec=outer_grid[0, 0], wspace=0.35)
    line_axis = figure.add_subplot(outer_grid[1, 0])
    bottom_grid = gridspec.GridSpecFromSubplotSpec(1, n_cols, subplot_spec=outer_grid[3, 0], wspace=0.35)
    colorbar_host_axis = figure.add_subplot(outer_grid[:, 1])
    colorbar_host_axis.axis("off")
    colorbar_axis = colorbar_host_axis.inset_axes([0.25, 0.25, 0.5, 0.5])
    self_topomap_axes = [figure.add_subplot(top_grid[0, index]) for index in range(n_cols)]
    other_topomap_axes = [figure.add_subplot(bottom_grid[0, index]) for index in range(n_cols)]
    _align_bottom_right_topomap_to_main_panel(
        other_topomap_axes,
        plotted_count=n_other_maps,
        right_edge=line_axis.get_position().x1,
        extra_offset=float(figure_config.get("bottom_right_topomap_extra_offset", 0.0)),
    )

    times_window_s = times[time_mask]
    times_window_ms = times_window_s * 1000.0
    self_counts_window = self_significant_counts[time_mask]
    other_counts_window = other_significant_counts[time_mask]
    line_axis.plot(times_window_ms, self_counts_window, color=line_self_color, linewidth=2.5, label=str(self_spec.get("label", "Unique self")))
    line_axis.plot(times_window_ms, other_counts_window, color=line_other_color, linewidth=2.5, label=str(other_spec.get("label", "Unique other")))
    line_axis.axvline(0.0, color=reference_color, linewidth=1.0)
    line_axis.set_ylabel("Number of significant electrodes", fontsize=_scaled_fontsize(11, font_scale))
    line_axis.set_xlabel("Time (ms)", fontsize=_scaled_fontsize(11, font_scale))
    line_axis.spines["top"].set_visible(False)
    line_axis.spines["right"].set_visible(False)
    line_axis.tick_params(axis="both", labelsize=_scaled_fontsize(9, font_scale))
    line_axis.legend(frameon=False, loc="upper right", fontsize=_scaled_fontsize(9, font_scale))
    figure.suptitle(
        str(figure_config.get("line_title", "Spatial extent of significant effects over time")),
        fontsize=_scaled_fontsize(13, font_scale),
        y=float(figure_config.get("title_y", 1.01)),
    )

    marker_size = float(figure_config.get("significant_marker_size", 40.0))
    marker_facecolor = str(figure_config.get("significant_marker_facecolor", shared_label_color))
    marker_edgecolor = str(figure_config.get("significant_marker_edgecolor", get_theme_color("background", theme)))
    marker_linewidth = float(figure_config.get("significant_marker_linewidth", 1.2))
    topomap_cmap = str(figure_config.get("topomap_cmap", "RdBu_r"))
    top_title_pad = float(figure_config.get("topomap_title_pad", 2.0))
    bottom_title_pad = float(figure_config.get("bottom_topomap_title_pad", -18.0))
    topomap_title_fontsize = _scaled_fontsize(11, font_scale)
    lower_panel_link_target_y = float(figure_config.get("lower_panel_link_target_y", 0.88))
    last_image = None

    for index, selected_time_s in enumerate(self_topomap_times_s):
        time_index = _get_time_index(times, selected_time_s)
        selected_time_ms = selected_time_s * 1000.0
        last_image = _draw_topomap_with_significant_markers(
            axis=self_topomap_axes[index],
            mne_info=info,
            values=np.asarray(self_payload["observed_map"], dtype=float)[:, time_index],
            significance_mask=sig_self_matrix[:, time_index],
            title=f"{str(self_spec.get('label', 'Self'))}\n{selected_time_ms:.0f} ms",
            vmin=shared_vmin,
            vmax=shared_vmax,
            cmap=topomap_cmap,
            marker_size=marker_size,
            marker_facecolor=marker_facecolor,
            marker_edgecolor=marker_edgecolor,
            marker_linewidth=marker_linewidth,
            title_pad=top_title_pad,
            title_fontsize=topomap_title_fontsize,
        )
        _add_time_link(
            figure=figure,
            line_axis=line_axis,
            topomap_axis=self_topomap_axes[index],
            selected_time_ms=selected_time_ms,
            y_value=float(np.interp(selected_time_ms, times_window_ms, self_counts_window)),
            color=line_self_color,
            connect_to_y=0.0,
        )
    for index in range(n_self_maps, n_cols):
        self_topomap_axes[index].axis("off")

    for index, selected_time_s in enumerate(other_topomap_times_s):
        time_index = _get_time_index(times, selected_time_s)
        selected_time_ms = selected_time_s * 1000.0
        last_image = _draw_topomap_with_significant_markers(
            axis=other_topomap_axes[index],
            mne_info=info,
            values=np.asarray(other_payload["observed_map"], dtype=float)[:, time_index],
            significance_mask=sig_other_matrix[:, time_index],
            title=f"{str(other_spec.get('label', 'Other'))}\n{selected_time_ms:.0f} ms",
            vmin=shared_vmin,
            vmax=shared_vmax,
            cmap=topomap_cmap,
            marker_size=marker_size,
            marker_facecolor=marker_facecolor,
            marker_edgecolor=marker_edgecolor,
            marker_linewidth=marker_linewidth,
            title_pad=bottom_title_pad,
            title_fontsize=topomap_title_fontsize,
        )
        _add_time_link(
            figure=figure,
            line_axis=line_axis,
            topomap_axis=other_topomap_axes[index],
            selected_time_ms=selected_time_ms,
            y_value=float(np.interp(selected_time_ms, times_window_ms, other_counts_window)),
            color=line_other_color,
            connect_to_y=lower_panel_link_target_y,
        )
    for index in range(n_other_maps, n_cols):
        other_topomap_axes[index].axis("off")

    if last_image is None:
        raise ValueError("No topomap snapshots were drawn. Configure at least one topomap time for self or other.")
    colorbar = figure.colorbar(last_image, cax=colorbar_axis)
    colorbar.set_label(str(figure_config.get("colorbar_label", "Observed t value")), rotation=90, fontsize=_scaled_fontsize(11, font_scale))
    colorbar.ax.tick_params(labelsize=_scaled_fontsize(9, font_scale))
    colorbar.ax.yaxis.get_offset_text().set_size(_scaled_fontsize(9, font_scale))

    output_stem = out_dir / "figures" / figure_subdir / "main_results"
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    written_paths: list[str] = []
    for fmt in formats:
        output_path = output_stem.with_suffix(f".{fmt}")
        figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
        written_paths.append(str(output_path))
    plt.close(figure)

    plots = [
        {
            "plot_name": "lmeeeg_main_results",
            "files": written_paths,
            "config_section": config_section,
            "significance_threshold": alpha_threshold,
            "plotting_tmin_s": float(times_window_s.min()),
            "plotting_tmax_s": float(times_window_s.max()),
            "models": [
                _manifest_model_entry(self_spec, self_payload, self_topomap_times_s),
                _manifest_model_entry(other_spec, other_payload, other_topomap_times_s),
            ],
        }
    ]

    if similarity_figure_config and bool(similarity_figure_config.get("enabled", True)):
        from rate.viz.lmeeeg_similarity import build_similarity_summary_figure_manifest

        similarity_manifest = build_similarity_summary_figure_manifest(
            config=config,
            out_dir=out_dir,
            manifest_path=manifest_path,
            output_stem=out_dir / "figures" / figure_subdir / "similarity_summary",
            figure_config=similarity_figure_config,
            theme_config=theme_config,
            formats=formats,
            dpi=dpi,
            write_manifest=False,
        )
        plots.extend(similarity_manifest["plots"])

    if behavioural_figure_config:
        from rate.viz.behavioural import build_behavioural_manifest

        behavioural_manifest = build_behavioural_manifest(
            out_dir=out_dir,
            residuals_path=out_dir / "stats" / "behavioural_residuals" / "trialwise_residuals.csv",
            random_effects_path=out_dir / "stats" / "behavioural_residuals" / "random_effects.csv",
            subject_slopes_path=None,
            lmm_summary_path=out_dir / "stats" / "r_analysis" / "analysis_model_summaries.txt",
            manifest_path=manifest_path,
            output_stem=out_dir / "figures" / figure_subdir / "behavioural_summary",
            figure_config=behavioural_figure_config,
            theme_config=theme_config,
            figure_subdir=figure_subdir,
            formats=formats,
            dpi=dpi,
            write_manifest=False,
        )
        plots.extend(behavioural_manifest["plots"])

    if variance_partition_figure_config:
        from rate.lmeeeg.variance_partition import build_variance_partition_group_output_paths
        from rate.viz.variance_partition import build_variance_partition_manifest

        variance_partition_paths = build_variance_partition_group_output_paths(config)
        variance_partition_table_path = _resolve_variance_partition_table_path(
            default_path=variance_partition_paths.table_output_path,
            out_dir=out_dir,
        )
        variance_partition_manifest = build_variance_partition_manifest(
            out_dir=out_dir,
            table_path=variance_partition_table_path,
            manifest_path=manifest_path,
            output_stem=out_dir / "figures" / figure_subdir / "variance_partition_summary",
            figure_config=variance_partition_figure_config,
            theme_config=theme_config,
            figure_subdir=figure_subdir,
            formats=formats,
            dpi=dpi,
            write_manifest=False,
        )
        plots.extend(variance_partition_manifest["plots"])

    manifest = {
        "status": "ok",
        "plot_count": len(plots),
        "plots": plots,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _manifest_model_entry(effect_config: Mapping[str, Any], payload: Mapping[str, object], topomap_times_s: Sequence[float]) -> dict[str, object]:
    return {
        "label": str(effect_config.get("label", "")),
        "model_name": str(effect_config["model_name"]),
        "effect_name": str(effect_config["effect_name"]),
        "observed_statistic": str(payload["observed_path"]),
        "observed_statistic_kind": str(payload.get("observed_statistic_kind", "observed_statistic")),
        "corrected_p_values": str(payload["corrected_p_path"]),
        "topomap_times_s": [float(value) for value in topomap_times_s],
    }


def _resolve_variance_partition_table_path(*, default_path: Path, out_dir: Path) -> Path:
    if default_path.exists():
        return default_path
    fallback_path = out_dir / "_previous" / "last" / "lmeeeg_variance_partition" / "group" / "descriptive_maps.parquet"
    if fallback_path.exists():
        return fallback_path
    return default_path


def _load_effect_payload(
    config: Mapping[str, Any],
    *,
    out_dir: Path,
    config_section: str,
    effect_config: Mapping[str, Any],
) -> dict[str, object]:
    model_name = str(effect_config["model_name"])
    effect_name = str(effect_config["effect_name"])
    stats_dir = out_dir / "stats" / config_section / model_name / effect_name
    summary_path = stats_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"lmeEEG main-results summary not found: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    observed_path = Path(str(summary["observed_statistic"]))
    corrected_p_path = Path(str(summary["corrected_p_values"]))
    fit_paths = build_lmeeeg_model_output_paths(config, config_section=config_section, model_name=model_name)
    if not all(path.exists() for path in (observed_path, corrected_p_path, fit_paths.times_output_path, fit_paths.channel_names_output_path)):
        raise FileNotFoundError(
            "Missing lmeEEG main-results inputs for "
            f"model={model_name}, effect={effect_name}."
        )
    return {
        "model_name": model_name,
        "effect_name": effect_name,
        "observed_path": observed_path,
        "corrected_p_path": corrected_p_path,
        "observed_statistic_kind": str(summary.get("observed_statistic_kind", "observed_statistic")),
        "observed_map": np.asarray(np.load(observed_path), dtype=float),
        "corrected_p_values": np.asarray(np.load(corrected_p_path), dtype=float),
        "times": np.asarray(np.load(fit_paths.times_output_path), dtype=float),
        "channel_names": json.loads(fit_paths.channel_names_output_path.read_text(encoding="utf-8")),
    }


def _validate_payload_compatibility(self_payload: Mapping[str, object], other_payload: Mapping[str, object]) -> None:
    self_times = np.asarray(self_payload["times"], dtype=float)
    other_times = np.asarray(other_payload["times"], dtype=float)
    self_channels = list(self_payload["channel_names"])
    other_channels = list(other_payload["channel_names"])
    self_observed = np.asarray(self_payload["observed_map"], dtype=float)
    other_observed = np.asarray(other_payload["observed_map"], dtype=float)
    self_p = np.asarray(self_payload["corrected_p_values"], dtype=float)
    other_p = np.asarray(other_payload["corrected_p_values"], dtype=float)
    if self_channels != other_channels:
        raise ValueError("Configured self/other lmeEEG main-results effects must share the same channel order.")
    if not np.array_equal(self_times, other_times):
        raise ValueError("Configured self/other lmeEEG main-results effects must share the same time axis.")
    if self_observed.shape != other_observed.shape or self_observed.shape != self_p.shape or other_observed.shape != other_p.shape:
        raise ValueError("Observed-statistic and corrected-p arrays must align across configured self/other effects.")


def _resolve_time_mask(times: np.ndarray, *, plotting_tmin_s: float | None, plotting_tmax_s: float | None) -> np.ndarray:
    lower = float(times.min()) if plotting_tmin_s is None else float(plotting_tmin_s)
    upper = float(times.max()) if plotting_tmax_s is None else float(plotting_tmax_s)
    mask = (times >= lower) & (times <= upper)
    if not np.any(mask):
        raise ValueError(f"No lmeEEG time samples fall within the configured plotting window [{lower}, {upper}].")
    return mask


def _resolve_requested_topomap_times(
    available_times: np.ndarray,
    *,
    requested_times: Sequence[float],
    time_mask: np.ndarray,
    label: str,
) -> list[float]:
    if not requested_times:
        return []
    window_times = available_times[time_mask]
    resolved: list[float] = []
    for requested_time in requested_times:
        nearest_time = _get_nearest_time_value(available_times, float(requested_time))
        if nearest_time < float(window_times.min()) or nearest_time > float(window_times.max()):
            raise ValueError(
                f"Requested topomap time {requested_time} s for `{label}` falls outside the configured plotting window."
            )
        resolved.append(nearest_time)
    return resolved


def _get_time_index(available_time_values: np.ndarray, target_time: float) -> int:
    return int(np.argmin(np.abs(available_time_values - target_time)))


def _get_nearest_time_value(available_time_values: np.ndarray, target_time: float) -> float:
    return float(available_time_values[_get_time_index(available_time_values, target_time)])


def _compute_shared_vlim(
    self_values_array: np.ndarray,
    other_values_array: np.ndarray,
    *,
    requested_vlim: float | None,
    percentile: float,
) -> tuple[float, float]:
    if requested_vlim is not None:
        return -float(requested_vlim), float(requested_vlim)
    all_abs_values = np.abs(np.concatenate([self_values_array.ravel(), other_values_array.ravel()]))
    vmax = float(np.nanpercentile(all_abs_values, percentile))
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 1.0
    return -vmax, vmax


def _build_topomap_info(channel_names: list[str], *, config: Mapping[str, Any]) -> mne.Info:
    info = mne.create_info(channel_names, sfreq=1.0, ch_types=["eeg"] * len(channel_names))
    try:
        montage = mne.channels.make_standard_montage(_resolve_preprocessing_montage_name(config))
        info.set_montage(montage, on_missing="ignore")
    except Exception:
        pass
    return info


def _resolve_preprocessing_montage_name(config: Mapping[str, Any]) -> str:
    preprocessing_cfg = dict(config.get("preprocessing") or {})
    montage_cfg = preprocessing_cfg.get("montage", preprocessing_cfg.get("montage_name"))
    if isinstance(montage_cfg, Mapping):
        return str(montage_cfg.get("name", "standard_1020"))
    return str(montage_cfg or "standard_1020")


def _get_topomap_xy(mne_info: mne.Info, picks: np.ndarray | None = None) -> np.ndarray:
    from mne.channels.layout import _find_topomap_coords

    if picks is None:
        picks = np.arange(len(mne_info.ch_names))
    return _find_topomap_coords(mne_info, picks=picks)


def _draw_topomap_with_significant_markers(
    *,
    axis: plt.Axes,
    mne_info: mne.Info,
    values: np.ndarray,
    significance_mask: np.ndarray,
    title: str,
    vmin: float,
    vmax: float,
    cmap: str,
    marker_size: float,
    marker_facecolor: str,
    marker_edgecolor: str,
    marker_linewidth: float,
    title_pad: float,
    title_fontsize: float,
):
    image, _ = mne.viz.plot_topomap(
        values,
        mne_info,
        axes=axis,
        show=False,
        cmap=cmap,
        vlim=(vmin, vmax),
        contours=3,
        sensors=False,
        outlines="head",
        mask=None,
    )
    xy = _get_topomap_xy(mne_info)
    sig_xy = xy[np.asarray(significance_mask, dtype=bool)]
    if sig_xy.size > 0:
        axis.scatter(
            sig_xy[:, 0],
            sig_xy[:, 1],
            s=marker_size,
            facecolors=marker_facecolor,
            edgecolors=marker_edgecolor,
            linewidths=marker_linewidth,
            zorder=10,
        )
    axis.set_title(title, fontsize=title_fontsize, pad=title_pad)
    return image


def _add_time_link(
    *,
    figure: plt.Figure,
    line_axis: plt.Axes,
    topomap_axis: plt.Axes,
    selected_time_ms: float,
    y_value: float,
    color: str,
    connect_to_y: float,
) -> None:
    line_axis.axvline(selected_time_ms, color="0.65", linestyle="--", linewidth=1.0, alpha=0.7)
    line_axis.scatter([selected_time_ms], [y_value], color=color, s=35, zorder=5)
    figure.add_artist(
        ConnectionPatch(
            xyA=(selected_time_ms, y_value),
            coordsA=line_axis.transData,
            xyB=(0.5, connect_to_y),
            coordsB=topomap_axis.transAxes,
            arrowstyle="->",
            shrinkA=4,
            shrinkB=4,
            mutation_scale=12,
            color=color,
            linewidth=1.5,
        )
    )


def _align_bottom_right_topomap_to_main_panel(
    axes: Sequence[plt.Axes],
    *,
    plotted_count: int,
    right_edge: float,
    extra_offset: float,
) -> None:
    if len(axes) < 1 or plotted_count < 1:
        return
    target_index = min(plotted_count, len(axes)) - 1
    if target_index < 0:
        return
    target_axis = axes[target_index]
    position = target_axis.get_position()
    new_x0 = right_edge - position.width + extra_offset
    target_axis.set_position([new_x0, position.y0, position.width, position.height])


def _scaled_fontsize(base_size: float, scale: float) -> float:
    return float(base_size) * float(scale)
