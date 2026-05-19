"""Paper-style similarity summary figure for pooled lmeEEG map comparisons."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import mne
import numpy as np
from matplotlib import colors as mcolors
from matplotlib import gridspec
from matplotlib.collections import QuadMesh
from matplotlib.colors import ListedColormap
from matplotlib.patches import ConnectionPatch

from rate.lmeeeg.similarity import build_similarity_output_paths, load_predictor_map, resolve_similarity_settings
from rate.viz.lmeeeg_results import _build_topomap_info, _resolve_time_mask
from rate.viz.theme import apply_theme, get_role_palette, get_theme_color


@dataclass(frozen=True)
class SimilarityFigurePayload:
    """Loaded similarity outputs required for the publication summary figure."""

    analysis_name: str
    predictor_a: str
    predictor_b: str
    predictor_a_map: np.ndarray
    predictor_b_map: np.ndarray
    difference_map: np.ndarray
    times: np.ndarray
    channel_names: list[str]
    observed_curve: np.ndarray
    corrected_p_values: np.ndarray | None
    null_curves: np.ndarray | None


def build_similarity_summary_figure_manifest(
    *,
    config: Mapping[str, Any],
    out_dir: Path,
    manifest_path: Path,
    output_stem: Path | None = None,
    figure_config: Mapping[str, Any] | None = None,
    theme_config: Mapping[str, Any] | None = None,
    formats: Sequence[str] = ("png", "pdf"),
    dpi: int = 300,
    write_manifest: bool = True,
) -> dict[str, object]:
    """Render the similarity-analysis analogue of the main lmeEEG summary figure."""
    resolved_figure_config = dict(figure_config or {})
    if not bool(resolved_figure_config.get("enabled", True)):
        manifest = {"status": "skipped", "plot_count": 0, "plots": []}
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return manifest

    payload = _load_similarity_payload(config, out_dir=out_dir)
    figure, plot_metadata = plot_similarity_summary_figure(
        payload=payload,
        config=config,
        figure_config=resolved_figure_config,
        theme_config=theme_config,
    )

    selected_formats = tuple(resolved_figure_config.get("save_formats", formats))
    resolved_output_stem = output_stem
    if resolved_output_stem is None:
        output_paths = build_similarity_output_paths(config)
        resolved_output_stem = output_paths.figures_dir / "similarity_summary"
    resolved_output_stem.parent.mkdir(parents=True, exist_ok=True)
    written_paths: list[str] = []
    for fmt in selected_formats:
        output_path = resolved_output_stem.with_suffix(f".{fmt}")
        figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
        written_paths.append(str(output_path))
    plt.close(figure)

    manifest = {
        "status": "ok",
        "plot_count": 1,
        "plots": [
            {
                "plot_name": "similarity_summary",
                "files": written_paths,
                "analysis_name": payload.analysis_name,
                "predictor_a": payload.predictor_a,
                "predictor_b": payload.predictor_b,
                **plot_metadata,
            }
        ],
    }
    if write_manifest:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def plot_similarity_summary_figure(
    *,
    payload: SimilarityFigurePayload,
    config: Mapping[str, Any],
    figure_config: Mapping[str, Any] | None = None,
    theme_config: Mapping[str, Any] | None = None,
) -> tuple[plt.Figure, dict[str, object]]:
    """Build the figure canvas without writing it to disk."""
    figure_config = dict(figure_config or {})
    theme = apply_theme(theme_config)
    palette = get_role_palette(theme)
    line_color = str(figure_config.get("line_color", palette["shared"]))
    envelope_color = str(figure_config.get("null_envelope_color", get_theme_color("neutral", theme)))
    reference_color = str(figure_config.get("reference_color", get_theme_color("reference", theme)))
    topomap_cmap = str(figure_config.get("topomap_cmap", "RdBu_r"))
    topomap_title_pad = float(figure_config.get("topomap_title_pad", 6.0))
    colorbar_label = str(figure_config.get("colorbar_label", "Beta coefficient"))
    significance_threshold = float(figure_config.get("significance_threshold", 0.05))
    window_radius_ms = float(figure_config.get("window_radius_ms", 50.0))
    font_scale = float(figure_config.get("font_scale", 3.0))
    topomap_row_label_xpad = float(figure_config.get("topomap_row_label_xpad", 0.015))
    legend_headroom_fraction = float(figure_config.get("legend_headroom_fraction", 0.35))
    lower_padding_fraction = float(figure_config.get("lower_padding_fraction", 0.18))
    legend_anchor_x = float(figure_config.get("legend_anchor_x", 0.0))
    legend_anchor_y = float(figure_config.get("legend_anchor_y", 1.0))
    significance_inset_bottom = float(figure_config.get("significance_inset_bottom", 0.08))
    significance_inset_height = float(figure_config.get("significance_inset_height", 0.10))
    hide_topomaps = bool(figure_config.get("hide_topomaps", False))
    requested_topomaps = 0 if hide_topomaps else max(1, int(figure_config.get("n_topomaps", 4)))
    selection_mode = str(figure_config.get("topomap_selection", "automatic")).lower()
    if selection_mode != "automatic":
        raise ValueError(f"Unsupported similarity topomap selection mode `{selection_mode}`.")
    lower_percentile, upper_percentile = _resolve_envelope_percentiles(figure_config.get("null_envelope_percentiles"))

    times = np.asarray(payload.times, dtype=float)
    plotting_tmin_s = figure_config.get("plotting_tmin_s")
    plotting_tmax_s = figure_config.get("plotting_tmax_s")
    time_mask = _resolve_time_mask(times, plotting_tmin_s=plotting_tmin_s, plotting_tmax_s=plotting_tmax_s)
    observed_curve = np.asarray(payload.observed_curve, dtype=float)
    corrected_p_values = None if payload.corrected_p_values is None else np.asarray(payload.corrected_p_values, dtype=float)
    significant_mask = np.zeros(times.shape, dtype=bool) if corrected_p_values is None else corrected_p_values < significance_threshold

    selected_indices: list[int] = []
    if requested_topomaps > 0:
        n_topomaps = min(requested_topomaps, int(np.count_nonzero(time_mask)))
        selected_indices = _select_topomap_indices(
            times=times,
            observed_curve=observed_curve,
            significant_mask=significant_mask,
            time_mask=time_mask,
            n_topomaps=n_topomaps,
            window_radius_ms=window_radius_ms,
        )
    predictor_a_maps: list[np.ndarray] = []
    predictor_b_maps: list[np.ndarray] = []
    topomap_time_labels: list[str] = []
    topomap_windows_ms: list[list[float]] = []
    for selected_index in selected_indices:
        predictor_a_map, window_mask = _average_map_window(
            values_map=np.asarray(payload.predictor_a_map, dtype=float),
            times=times,
            center_index=selected_index,
            window_radius_ms=window_radius_ms,
        )
        predictor_b_map, _ = _average_map_window(
            values_map=np.asarray(payload.predictor_b_map, dtype=float),
            times=times,
            center_index=selected_index,
            window_radius_ms=window_radius_ms,
        )
        predictor_a_maps.append(predictor_a_map)
        predictor_b_maps.append(predictor_b_map)
        window_start_ms = float(times[window_mask][0] * 1000.0)
        window_end_ms = float(times[window_mask][-1] * 1000.0)
        topomap_windows_ms.append([window_start_ms, window_end_ms])
        topomap_time_labels.append(_format_topomap_time_label(float(times[selected_index] * 1000.0)))

    vmin = vmax = None
    info = None
    if selected_indices:
        vmin, vmax = _compute_symmetric_topomap_limits(
            [*predictor_a_maps, *predictor_b_maps],
            requested_vlim=float(figure_config["manual_vlim"]) if figure_config.get("manual_vlim") is not None else None,
        )
        info = _build_topomap_info(list(payload.channel_names), config=config)
    predictor_a_label = str(figure_config.get("predictor_a_label", "Production"))
    predictor_b_label = str(figure_config.get("predictor_b_label", "Perception"))

    figure = plt.figure(
        figsize=(
            float(figure_config.get("figure_width", 13.5)),
            float(figure_config.get("figure_height", 9.0)),
        )
    )
    if selected_indices:
        outer_grid = gridspec.GridSpec(
            nrows=2,
            ncols=2,
            figure=figure,
            width_ratios=[1.0, 0.045],
            height_ratios=[1.5, 1.6],
            hspace=float(figure_config.get("hspace", 0.28)),
            wspace=float(figure_config.get("wspace", 0.12)),
        )
        top_grid = gridspec.GridSpecFromSubplotSpec(
            2,
            len(selected_indices),
            subplot_spec=outer_grid[0, 0],
            hspace=0.18,
            wspace=0.35,
        )
        topomap_axes_a = [figure.add_subplot(top_grid[0, index]) for index in range(len(selected_indices))]
        topomap_axes_b = [figure.add_subplot(top_grid[1, index]) for index in range(len(selected_indices))]
        line_axis = figure.add_subplot(outer_grid[1, 0])
        colorbar_host_axis = figure.add_subplot(outer_grid[:, 1])
        colorbar_host_axis.axis("off")
        colorbar_axis = colorbar_host_axis.inset_axes([0.25, 0.25, 0.5, 0.5])
    else:
        topomap_axes_a = []
        topomap_axes_b = []
        line_axis = figure.add_subplot(111)
        colorbar_axis = None

    times_window_ms = times[time_mask] * 1000.0
    observed_window = observed_curve[time_mask]
    line_axis.axhline(0.0, color=reference_color, linewidth=1.0, alpha=0.8, zorder=1)
    line_axis.axvline(0.0, color=reference_color, linewidth=1.0, alpha=0.8, zorder=1)

    envelope_lower = None
    envelope_upper = None
    if payload.null_curves is not None:
        envelope_lower, envelope_upper = _compute_null_envelope(
            null_curves=np.asarray(payload.null_curves, dtype=float),
            lower_percentile=lower_percentile,
            upper_percentile=upper_percentile,
        )
        line_axis.fill_between(
            times_window_ms,
            envelope_lower[time_mask],
            envelope_upper[time_mask],
            color=envelope_color,
            alpha=0.18,
            linewidth=0.0,
            zorder=2,
            label=f"Permutation {upper_percentile - lower_percentile:.0f}% envelope",
        )

    line_axis.plot(
        times_window_ms,
        observed_window,
        color=line_color,
        linewidth=2.8,
        zorder=3,
        label="Observed similarity",
    )
    line_axis.set_ylabel(str(figure_config.get("y_label", "Topographic similarity (r)")), fontsize=_scaled_fontsize(11, font_scale))
    line_axis.spines["top"].set_visible(False)
    line_axis.spines["right"].set_visible(False)
    line_axis.tick_params(axis="both", labelsize=_scaled_fontsize(9, font_scale))

    ymin, ymax = _resolve_similarity_ylim(observed_window, envelope_lower, envelope_upper, time_mask=time_mask)
    ymin -= (ymax - ymin) * lower_padding_fraction
    ymax += (ymax - ymin) * legend_headroom_fraction
    line_axis.set_ylim(ymin, ymax)
    line_axis.set_xlim(float(times_window_ms[0]), float(times_window_ms[-1]))

    if payload.null_curves is not None:
        line_axis.legend(
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(legend_anchor_x, legend_anchor_y),
            borderaxespad=0.0,
            fontsize=_scaled_fontsize(9, font_scale),
        )

    sig_label = str(figure_config.get("significance_label", "TFCE-corrected p < 0.05"))
    if selected_indices:
        last_image = None
        for top_axis, bottom_axis, predictor_a_map, predictor_b_map, label, selected_index in zip(
            topomap_axes_a,
            topomap_axes_b,
            predictor_a_maps,
            predictor_b_maps,
            topomap_time_labels,
            selected_indices,
            strict=True,
        ):
            last_image = _draw_similarity_topomap(
                axis=top_axis,
                mne_info=info,
                values=predictor_a_map,
                title=label,
                vmin=vmin,
                vmax=vmax,
                cmap=topomap_cmap,
                title_pad=topomap_title_pad,
                title_fontsize=_scaled_fontsize(11, font_scale),
            )
            _draw_similarity_topomap(
                axis=bottom_axis,
                mne_info=info,
                values=predictor_b_map,
                title="",
                vmin=vmin,
                vmax=vmax,
                cmap=topomap_cmap,
                title_pad=topomap_title_pad,
                title_fontsize=_scaled_fontsize(11, font_scale),
            )
            _add_similarity_time_link(
                figure=figure,
                line_axis=line_axis,
                topomap_axis=bottom_axis,
                center_time_ms=float(times[selected_index] * 1000.0),
                y_value=float(observed_curve[selected_index]),
                color=get_theme_color("neutral", theme),
            )
        if last_image is None:
            raise ValueError("No similarity topomaps were drawn.")
        _add_topomap_row_labels(
            figure=figure,
            top_axes=topomap_axes_a,
            bottom_axes=topomap_axes_b,
            top_label=predictor_a_label,
            bottom_label=predictor_b_label,
            xpad=topomap_row_label_xpad,
            fontsize=_scaled_fontsize(11, font_scale),
        )
        colorbar = figure.colorbar(last_image, cax=colorbar_axis)
        colorbar.set_label(colorbar_label, rotation=90, fontsize=_scaled_fontsize(11, font_scale))
        colorbar.ax.tick_params(labelsize=_scaled_fontsize(9, font_scale))
        colorbar.ax.yaxis.get_offset_text().set_size(_scaled_fontsize(9, font_scale))
    sig_mesh = None
    if corrected_p_values is not None:
        significance_axis = line_axis.inset_axes(
            [0.0, significance_inset_bottom, 1.0, significance_inset_height],
            sharex=line_axis,
        )
        sig_mesh = _plot_significance_bar(
            significance_axis,
            times_ms=times_window_ms,
            significant_mask=significant_mask[time_mask],
            color=line_color,
            label=sig_label,
            fontsize=_scaled_fontsize(9, font_scale),
        )
    line_axis.set_xlabel("Time (ms)", fontsize=_scaled_fontsize(11, font_scale))

    plot_metadata = {
        "n_topomaps": len(selected_indices),
        "hide_topomaps": hide_topomaps,
        "window_radius_ms": window_radius_ms,
        "topomap_windows_ms": topomap_windows_ms,
        "topomap_centers_s": [float(times[index]) for index in selected_indices],
        "plotting_tmin_s": float(times[time_mask][0]),
        "plotting_tmax_s": float(times[time_mask][-1]),
        "significance_threshold": significance_threshold,
        "null_envelope_percentiles": [lower_percentile, upper_percentile] if payload.null_curves is not None else None,
        "has_corrected_p_values": corrected_p_values is not None,
        "has_null_envelope": payload.null_curves is not None,
        "topomap_color_limits": ([vmin, vmax] if vmin is not None and vmax is not None else None),
        "significance_timepoints": (int(np.count_nonzero(significant_mask[time_mask])) if corrected_p_values is not None else 0),
        "significance_artist": sig_mesh is not None,
        "topomap_predictors": [predictor_a_label, predictor_b_label],
    }
    return figure, plot_metadata


def _load_similarity_payload(config: Mapping[str, Any], *, out_dir: Path) -> SimilarityFigurePayload:
    output_paths = build_similarity_output_paths(config)
    summary_path = output_paths.summary_output_path
    if not summary_path.exists():
        raise FileNotFoundError(f"Similarity summary not found: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    required_paths = {
        "difference_map": output_paths.difference_map_output_path,
        "times": output_paths.times_output_path,
        "channel_names": output_paths.channel_names_output_path,
        "time_resolved_curve": output_paths.time_resolved_curve_output_path,
    }
    missing = [name for name, path in required_paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing similarity figure inputs: "
            + ", ".join(f"{name}={required_paths[name]}" for name in missing)
        )

    difference_map = np.asarray(np.load(output_paths.difference_map_output_path), dtype=float)
    times = np.asarray(np.load(output_paths.times_output_path), dtype=float)
    channel_names = json.loads(output_paths.channel_names_output_path.read_text(encoding="utf-8"))
    observed_curve = np.asarray(np.load(output_paths.time_resolved_curve_output_path), dtype=float)

    time_resolved_summary = dict(summary.get("time_resolved_topography") or {})
    corrected_p_values = _load_optional_array(time_resolved_summary.get("corrected_p_values_path"))
    null_curves = _load_optional_array(time_resolved_summary.get("null_curves_path"))

    _validate_similarity_payload(
        difference_map=difference_map,
        times=times,
        channel_names=channel_names,
        observed_curve=observed_curve,
        corrected_p_values=corrected_p_values,
        null_curves=null_curves,
    )

    input_summary = dict(summary.get("input") or {})
    settings = resolve_similarity_settings(config)
    predictor_a_name = str(input_summary.get("predictor_a", settings.input_settings.predictor_a))
    predictor_b_name = str(input_summary.get("predictor_b", settings.input_settings.predictor_b))
    predictor_a_map = load_predictor_map(config, predictor_name=predictor_a_name)
    predictor_b_map = load_predictor_map(config, predictor_name=predictor_b_name)
    _validate_predictor_maps_for_figure(
        predictor_a_map=np.asarray(predictor_a_map.values, dtype=float),
        predictor_b_map=np.asarray(predictor_b_map.values, dtype=float),
        reference_difference_map=difference_map,
        reference_times=times,
        reference_channel_names=channel_names,
    )
    return SimilarityFigurePayload(
        analysis_name=str(summary.get("analysis_name", settings.analysis_name)),
        predictor_a=predictor_a_name,
        predictor_b=predictor_b_name,
        predictor_a_map=np.asarray(predictor_a_map.values, dtype=float),
        predictor_b_map=np.asarray(predictor_b_map.values, dtype=float),
        difference_map=difference_map,
        times=times,
        channel_names=[str(name) for name in channel_names],
        observed_curve=observed_curve,
        corrected_p_values=corrected_p_values,
        null_curves=null_curves,
    )


def _load_optional_array(path_value: object) -> np.ndarray | None:
    if not path_value:
        return None
    path = Path(str(path_value))
    if not path.exists():
        raise FileNotFoundError(f"Configured similarity figure input not found: {path}")
    return np.asarray(np.load(path), dtype=float)


def _validate_similarity_payload(
    *,
    difference_map: np.ndarray,
    times: np.ndarray,
    channel_names: Sequence[str],
    observed_curve: np.ndarray,
    corrected_p_values: np.ndarray | None,
    null_curves: np.ndarray | None,
) -> None:
    if difference_map.ndim != 2:
        raise ValueError(f"Expected 2D difference map (channels x time), got shape {difference_map.shape}.")
    n_channels, n_times = difference_map.shape
    if times.ndim != 1:
        raise ValueError(f"Expected 1D time axis, got shape {times.shape}.")
    if observed_curve.ndim != 1:
        raise ValueError(f"Expected 1D observed similarity curve, got shape {observed_curve.shape}.")
    if len(channel_names) != n_channels:
        raise ValueError(
            f"Difference map channel dimension ({n_channels}) does not match channel metadata ({len(channel_names)})."
        )
    if times.shape[0] != n_times:
        raise ValueError(f"Difference map time dimension ({n_times}) does not match time axis length ({times.shape[0]}).")
    if observed_curve.shape[0] != n_times:
        raise ValueError(
            f"Observed similarity curve length ({observed_curve.shape[0]}) does not match time axis length ({n_times})."
        )
    if corrected_p_values is not None and corrected_p_values.shape != observed_curve.shape:
        raise ValueError(
            f"Corrected p-values shape {corrected_p_values.shape} does not align with time axis length {observed_curve.shape}."
        )
    if null_curves is not None:
        if null_curves.ndim != 2:
            raise ValueError(f"Expected 2D null curves array, got shape {null_curves.shape}.")
        if null_curves.shape[1] != n_times:
            raise ValueError(
                f"Null curves time dimension ({null_curves.shape[1]}) does not match time axis length ({n_times})."
            )


def _validate_predictor_maps_for_figure(
    *,
    predictor_a_map: np.ndarray,
    predictor_b_map: np.ndarray,
    reference_difference_map: np.ndarray,
    reference_times: np.ndarray,
    reference_channel_names: Sequence[str],
) -> None:
    if predictor_a_map.shape != reference_difference_map.shape:
        raise ValueError(
            f"Predictor A map shape {predictor_a_map.shape} does not match similarity map shape {reference_difference_map.shape}."
        )
    if predictor_b_map.shape != reference_difference_map.shape:
        raise ValueError(
            f"Predictor B map shape {predictor_b_map.shape} does not match similarity map shape {reference_difference_map.shape}."
        )
    if predictor_a_map.shape[1] != reference_times.shape[0]:
        raise ValueError(
            f"Predictor A time dimension ({predictor_a_map.shape[1]}) does not match similarity times ({reference_times.shape[0]})."
        )
    if predictor_b_map.shape[1] != reference_times.shape[0]:
        raise ValueError(
            f"Predictor B time dimension ({predictor_b_map.shape[1]}) does not match similarity times ({reference_times.shape[0]})."
        )
    if predictor_a_map.shape[0] != len(reference_channel_names):
        raise ValueError(
            f"Predictor A channel dimension ({predictor_a_map.shape[0]}) does not match similarity channels ({len(reference_channel_names)})."
        )
    if predictor_b_map.shape[0] != len(reference_channel_names):
        raise ValueError(
            f"Predictor B channel dimension ({predictor_b_map.shape[0]}) does not match similarity channels ({len(reference_channel_names)})."
        )


def _resolve_envelope_percentiles(config_value: object) -> tuple[float, float]:
    if config_value is None:
        return 2.5, 97.5
    values = list(config_value)
    if len(values) != 2:
        raise ValueError("`null_envelope_percentiles` must contain exactly two values.")
    lower = float(values[0])
    upper = float(values[1])
    if not 0.0 <= lower < upper <= 100.0:
        raise ValueError("`null_envelope_percentiles` must satisfy 0 <= lower < upper <= 100.")
    return lower, upper


def _find_contiguous_true_runs(mask: np.ndarray) -> list[np.ndarray]:
    true_indices = np.flatnonzero(np.asarray(mask, dtype=bool))
    if true_indices.size == 0:
        return []
    split_points = np.where(np.diff(true_indices) > 1)[0] + 1
    return [chunk for chunk in np.split(true_indices, split_points) if chunk.size > 0]


def _select_topomap_indices(
    *,
    times: np.ndarray,
    observed_curve: np.ndarray,
    significant_mask: np.ndarray,
    time_mask: np.ndarray,
    n_topomaps: int,
    window_radius_ms: float,
) -> list[int]:
    """Choose deterministic, representative topomap centers for the similarity curve."""
    available_indices = np.flatnonzero(time_mask)
    if available_indices.size == 0:
        raise ValueError("No time points remain after applying the similarity plotting window.")
    if n_topomaps < 1:
        raise ValueError("`n_topomaps` must be at least 1.")

    significant_clusters = _find_contiguous_true_runs(significant_mask & time_mask)
    if significant_clusters:
        return [_pick_cluster_center(cluster_indices) for cluster_indices in significant_clusters]

    min_separation_ms = max(float(window_radius_ms), _estimate_time_step_ms(times))
    selected: list[int] = []
    candidate_order: list[int] = []
    candidate_order.append(_pick_peak_index(observed_curve, available_indices))
    candidate_order.append(_pick_near_zero_index(observed_curve, available_indices))

    candidate_order.extend(_pick_quantile_landmarks(available_indices, n_topomaps=n_topomaps))
    candidate_order.extend(int(index) for index in available_indices)

    for candidate in candidate_order:
        if candidate not in available_indices:
            continue
        if _is_sufficiently_separated(candidate, selected, times=times, min_separation_ms=min_separation_ms):
            selected.append(int(candidate))
        if len(selected) >= min(n_topomaps, available_indices.size):
            break

    if not selected:
        raise ValueError("Unable to select any similarity topomap windows.")
    return sorted(selected)


def _pick_cluster_center(cluster_indices: np.ndarray) -> int:
    center_offset = int((cluster_indices.size - 1) // 2)
    return int(cluster_indices[center_offset])


def _estimate_time_step_ms(times: np.ndarray) -> float:
    if times.size < 2:
        return 0.0
    diffs_ms = np.diff(times) * 1000.0
    finite_diffs = diffs_ms[np.isfinite(diffs_ms)]
    if finite_diffs.size == 0:
        return 0.0
    return float(np.median(np.abs(finite_diffs)))


def _pick_peak_index(observed_curve: np.ndarray, candidate_indices: np.ndarray) -> int:
    curve_values = np.abs(observed_curve[candidate_indices])
    return int(candidate_indices[int(np.nanargmax(curve_values))])


def _pick_near_zero_index(observed_curve: np.ndarray, candidate_indices: np.ndarray) -> int:
    curve_values = np.abs(observed_curve[candidate_indices])
    rank_order = np.argsort(curve_values, kind="stable")
    shortlist_size = min(candidate_indices.size, max(2, int(np.ceil(candidate_indices.size / 3.0))))
    shortlisted = candidate_indices[rank_order[:shortlist_size]]
    center_position = float(np.median(candidate_indices.astype(float)))
    center_distances = np.abs(shortlisted.astype(float) - center_position)
    return int(shortlisted[int(np.argmin(center_distances))])


def _pick_quantile_landmarks(available_indices: np.ndarray, *, n_topomaps: int) -> list[int]:
    if available_indices.size == 1:
        return [int(available_indices[0])]
    landmark_positions = np.linspace(0, available_indices.size - 1, num=n_topomaps + 2)[1:-1]
    landmarks = [int(available_indices[int(round(position))]) for position in landmark_positions]
    return list(dict.fromkeys(landmarks))


def _is_sufficiently_separated(candidate: int, selected: Sequence[int], *, times: np.ndarray, min_separation_ms: float) -> bool:
    candidate_ms = float(times[candidate] * 1000.0)
    for existing in selected:
        existing_ms = float(times[existing] * 1000.0)
        if abs(candidate_ms - existing_ms) < min_separation_ms:
            return False
    return True


def _average_map_window(
    *,
    values_map: np.ndarray,
    times: np.ndarray,
    center_index: int,
    window_radius_ms: float,
) -> tuple[np.ndarray, np.ndarray]:
    center_time_s = float(times[center_index])
    radius_s = float(window_radius_ms) / 1000.0
    window_mask = np.abs(times - center_time_s) <= (radius_s + 1.0e-12)
    if not np.any(window_mask):
        raise ValueError(
            f"Selected similarity topomap window around {center_time_s * 1000.0:.1f} ms contains no samples."
        )
    averaged_map = np.nanmean(np.asarray(values_map, dtype=float)[:, window_mask], axis=1)
    return averaged_map, window_mask


def _format_topomap_time_label(center_time_ms: float) -> str:
    return f"{center_time_ms:.0f} ms"


def _compute_symmetric_topomap_limits(
    averaged_maps: Sequence[np.ndarray],
    *,
    requested_vlim: float | None,
) -> tuple[float, float]:
    if requested_vlim is not None:
        return -float(requested_vlim), float(requested_vlim)
    if not averaged_maps:
        raise ValueError("At least one averaged topomap is required to determine a shared color scale.")
    vmax = float(max(np.nanmax(np.abs(np.asarray(averaged_map, dtype=float))) for averaged_map in averaged_maps))
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 1.0
    return -vmax, vmax


def _draw_similarity_topomap(
    *,
    axis: plt.Axes,
    mne_info: mne.Info,
    values: np.ndarray,
    title: str,
    vmin: float,
    vmax: float,
    cmap: str,
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
    axis.set_title(title, fontsize=title_fontsize, pad=title_pad)
    return image


def _compute_null_envelope(
    *,
    null_curves: np.ndarray,
    lower_percentile: float,
    upper_percentile: float,
) -> tuple[np.ndarray, np.ndarray]:
    if null_curves.ndim != 2:
        raise ValueError(f"Expected 2D null curves array, got shape {null_curves.shape}.")
    lower = np.nanpercentile(null_curves, lower_percentile, axis=0)
    upper = np.nanpercentile(null_curves, upper_percentile, axis=0)
    return np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)


def _resolve_similarity_ylim(
    observed_window: np.ndarray,
    envelope_lower: np.ndarray | None,
    envelope_upper: np.ndarray | None,
    *,
    time_mask: np.ndarray,
) -> tuple[float, float]:
    values = [observed_window]
    if envelope_lower is not None and envelope_upper is not None:
        values.extend([envelope_lower[time_mask], envelope_upper[time_mask]])
    stacked = np.concatenate([np.asarray(value, dtype=float).ravel() for value in values])
    finite_values = stacked[np.isfinite(stacked)]
    if finite_values.size == 0:
        return -1.0, 1.0
    span = float(finite_values.max() - finite_values.min())
    padding = 0.08 * (span if span > 0.0 else 1.0)
    return float(finite_values.min() - padding), float(finite_values.max() + padding)


def _time_bin_edges(times_ms: np.ndarray) -> np.ndarray:
    if times_ms.ndim != 1:
        raise ValueError(f"Expected 1D time values, got shape {times_ms.shape}.")
    if times_ms.size == 0:
        raise ValueError("Cannot compute time-bin edges for an empty time axis.")
    if times_ms.size == 1:
        return np.asarray([times_ms[0] - 0.5, times_ms[0] + 0.5], dtype=float)
    edges = np.empty(times_ms.size + 1, dtype=float)
    edges[1:-1] = (times_ms[:-1] + times_ms[1:]) / 2.0
    edges[0] = times_ms[0] - (edges[1] - times_ms[0])
    edges[-1] = times_ms[-1] + (times_ms[-1] - edges[-2])
    return edges


def _plot_significance_bar(
    axis: plt.Axes,
    *,
    times_ms: np.ndarray,
    significant_mask: np.ndarray,
    color: str,
    label: str,
    fontsize: float,
) -> QuadMesh:
    if times_ms.shape != significant_mask.shape:
        raise ValueError(
            f"Significance mask shape {significant_mask.shape} does not align with plotted time axis {times_ms.shape}."
        )
    rgba = mcolors.to_rgba(color, alpha=0.75)
    cmap = ListedColormap([(1.0, 1.0, 1.0, 0.0), rgba])
    mesh = axis.pcolormesh(
        _time_bin_edges(np.asarray(times_ms, dtype=float)),
        np.asarray([0.0, 1.0], dtype=float),
        np.asarray(significant_mask, dtype=float)[np.newaxis, :],
        shading="flat",
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
    )
    axis.set_ylim(0.0, 1.0)
    axis.set_yticks([])
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_visible(False)
    axis.tick_params(axis="x", which="both", length=0, labelbottom=False)
    axis.text(0.0, 0.5, label, transform=axis.transAxes, ha="left", va="center", fontsize=fontsize, color=color)
    return mesh


def _add_topomap_row_labels(
    *,
    figure: plt.Figure,
    top_axes: Sequence[plt.Axes],
    bottom_axes: Sequence[plt.Axes],
    top_label: str,
    bottom_label: str,
    xpad: float,
    fontsize: float,
) -> None:
    if not top_axes or not bottom_axes:
        return
    top_bbox = top_axes[0].get_position()
    bottom_bbox = bottom_axes[0].get_position()
    x_position = top_bbox.x0 - xpad
    figure.text(x_position, (top_bbox.y0 + top_bbox.y1) / 2.0, top_label, ha="right", va="center", fontsize=fontsize)
    figure.text(x_position, (bottom_bbox.y0 + bottom_bbox.y1) / 2.0, bottom_label, ha="right", va="center", fontsize=fontsize)


def _scaled_fontsize(base_size: float, scale: float) -> float:
    return float(base_size) * float(scale)


def _add_similarity_time_link(
    *,
    figure: plt.Figure,
    line_axis: plt.Axes,
    topomap_axis: plt.Axes,
    center_time_ms: float,
    y_value: float,
    color: str,
) -> None:
    line_axis.axvline(center_time_ms, color=color, linestyle="--", linewidth=0.9, alpha=0.45, zorder=1)
    figure.add_artist(
        ConnectionPatch(
            xyA=(center_time_ms, y_value),
            coordsA=line_axis.transData,
            xyB=(0.5, 0.0),
            coordsB=topomap_axis.transAxes,
            arrowstyle="-",
            linewidth=0.9,
            color=color,
            alpha=0.55,
        )
    )
