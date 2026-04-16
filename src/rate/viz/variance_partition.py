"""Variance-partition figure helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from matplotlib import gridspec
from matplotlib.patches import Circle

from rate.util import read_parquet_dataframe
from rate.viz.theme import apply_theme, get_role_palette, get_theme_color

try:
    from matplotlib_venn import venn2
except ImportError:  # pragma: no cover - exercised implicitly via fallback path
    venn2 = None


DEFAULT_ANTERIOR_CHANNELS = (
    "Fp1",
    "Fp2",
    "AF3",
    "AF4",
    "AF7",
    "AF8",
    "Fz",
    "F1",
    "F2",
    "F3",
    "F4",
    "F5",
    "F6",
    "F7",
    "F8",
    "FCz",
    "FC1",
    "FC2",
    "FC3",
    "FC4",
    "FC5",
    "FC6",
)
DEFAULT_POSTERIOR_CHANNELS = (
    "CPz",
    "CP1",
    "CP2",
    "CP3",
    "CP4",
    "CP5",
    "CP6",
    "Pz",
    "P1",
    "P2",
    "P3",
    "P4",
    "P5",
    "P6",
    "P7",
    "P8",
    "POz",
    "PO3",
    "PO4",
    "PO7",
    "PO8",
    "Oz",
    "O1",
    "O2",
)
_COMPONENT_COLUMNS = ("unique_self", "shared", "unique_other")
_DEFAULT_MONTAGE_NAME = "standard_1020"


def build_variance_partition_manifest(
    *,
    out_dir: Path,
    table_path: Path,
    manifest_path: Path,
    figure_config: Mapping[str, Any] | None = None,
    theme_config: Mapping[str, Any] | None = None,
    figure_subdir: str = "lmeeeg_variance_partition",
    formats: Sequence[str] = ("png", "pdf"),
    dpi: int = 300,
) -> dict[str, object]:
    """Render variance-partition summary figures and write a manifest."""
    figure_config = dict(figure_config or {})
    theme = apply_theme(theme_config)
    dataframe = _prepare_component_values(read_parquet_dataframe(table_path))
    roi_config = dict(figure_config.get("rois") or {})
    use_proportions = bool(figure_config.get("use_proportions", False))
    epsilon = float(figure_config.get("epsilon", 1e-12))
    palette = get_role_palette(theme)

    anterior_channels = tuple(roi_config.get("anterior", DEFAULT_ANTERIOR_CHANNELS))
    posterior_channels = tuple(roi_config.get("posterior", DEFAULT_POSTERIOR_CHANNELS))
    global_summary = _summarise_global_components(dataframe)
    anterior_timecourse = _summarise_roi_over_time(
        dataframe,
        roi_channels=anterior_channels,
        use_proportions_flag=use_proportions,
        epsilon=epsilon,
    )
    posterior_timecourse = _summarise_roi_over_time(
        dataframe,
        roi_channels=posterior_channels,
        use_proportions_flag=use_proportions,
        epsilon=epsilon,
    )

    figure_width = float(figure_config.get("figure_width", 14.0))
    figure_height = float(figure_config.get("figure_height", 7.0))
    figure = plt.figure(figsize=(figure_width, figure_height), constrained_layout=True)
    layout_engine = figure.get_layout_engine()
    if layout_engine is not None:
        layout_engine.set(
            w_pad=float(figure_config.get("constrained_w_pad", 0.02)),
            h_pad=float(figure_config.get("constrained_h_pad", 0.02)),
            wspace=float(figure_config.get("wspace", 0.04)),
            hspace=float(figure_config.get("hspace", 0.06)),
        )
    grid_spec = gridspec.GridSpec(
        nrows=2,
        ncols=2,
        figure=figure,
        width_ratios=[1.0, 1.8],
        height_ratios=[1.0, 1.0],
        wspace=float(figure_config.get("wspace", 0.04)),
        hspace=float(figure_config.get("hspace", 0.06)),
    )

    venn_axis = figure.add_subplot(grid_spec[:, 0])
    anterior_axis = figure.add_subplot(grid_spec[0, 1])
    posterior_axis = figure.add_subplot(grid_spec[1, 1], sharex=anterior_axis)

    _draw_weighted_venn(
        venn_axis,
        unique_self_value=float(global_summary["unique_self"]),
        shared_value=float(global_summary["shared"]),
        unique_other_value=float(global_summary["unique_other"]),
        title=str(figure_config.get("global_title", "Global variance partitioning")),
        colors=palette,
        theme_config=theme,
    )
    y_axis_label = "Proportion of explained variance" if use_proportions else "Explained variance"
    _plot_stacked_area(
        anterior_axis,
        anterior_timecourse,
        title=str(figure_config.get("anterior_title", f"Anterior ROI (n={len(anterior_channels)} channels)")),
        ylabel=y_axis_label,
        colors=palette,
        theme_config=theme,
    )
    _plot_stacked_area(
        posterior_axis,
        posterior_timecourse,
        title=str(figure_config.get("posterior_title", f"Posterior ROI (n={len(posterior_channels)} channels)")),
        ylabel=y_axis_label,
        colors=palette,
        theme_config=theme,
    )
    _add_roi_inset(
        anterior_axis,
        roi_channels=anterior_channels,
        label=str(figure_config.get("anterior_inset_label", "Anterior ROI")),
        highlight_color=palette["self"],
        theme_config=theme,
        anchor=tuple(figure_config.get("anterior_inset_anchor", (0.01, 0.50, 0.18, 0.42))),
    )
    _add_roi_inset(
        posterior_axis,
        roi_channels=posterior_channels,
        label=str(figure_config.get("posterior_inset_label", "Posterior ROI")),
        highlight_color=palette["other"],
        theme_config=theme,
        anchor=tuple(figure_config.get("posterior_inset_anchor", (0.01, 0.50, 0.18, 0.42))),
    )
    posterior_axis.set_xlabel("Time (s)")

    if use_proportions:
        anterior_axis.set_ylim(0.0, 1.0)
        posterior_axis.set_ylim(0.0, 1.0)
    else:
        common_y_max = max(
            float(anterior_timecourse["total"].max()),
            float(posterior_timecourse["total"].max()),
        ) * 1.05
        anterior_axis.set_ylim(0.0, common_y_max)
        posterior_axis.set_ylim(0.0, common_y_max)

    handles, labels = anterior_axis.get_legend_handles_labels()
    if handles:
        anterior_axis.legend(handles, labels, loc="upper right", frameon=False)

    output_stem = out_dir / "figures" / figure_subdir / "variance_partition_summary"
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    written_paths: list[str] = []
    for fmt in formats:
        output_path = output_stem.with_suffix(f".{fmt}")
        figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
        written_paths.append(str(output_path))
    plt.close(figure)

    manifest = {
        "status": "ok",
        "plot_count": 1,
        "plots": [
            {
                "plot_name": "variance_partition_summary",
                "files": written_paths,
                "source_paths": [str(table_path)],
                "use_proportions": use_proportions,
                "global_summary": global_summary,
                "rois": {
                    "anterior": list(anterior_channels),
                    "posterior": list(posterior_channels),
                },
            }
        ],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _prepare_component_values(data_frame: pd.DataFrame) -> pd.DataFrame:
    prepared_df = data_frame.copy()
    required_columns = {"channel", "time", "unique_self", "unique_other", "shared"}
    missing = required_columns.difference(prepared_df.columns)
    if missing:
        missing_text = ", ".join(sorted(str(column) for column in missing))
        raise ValueError(f"Variance-partition figure input is missing required columns: {missing_text}")

    prepared_df["plot_unique_self"] = prepared_df["unique_self"].clip(lower=0.0)
    prepared_df["plot_unique_other"] = prepared_df["unique_other"].clip(lower=0.0)
    prepared_df["plot_shared"] = prepared_df["shared"].clip(lower=0.0)
    prepared_df["plot_total"] = (
        prepared_df["plot_unique_self"]
        + prepared_df["plot_shared"]
        + prepared_df["plot_unique_other"]
    )
    return prepared_df


def _summarise_global_components(data_frame: pd.DataFrame) -> dict[str, float]:
    unique_self_value = float(data_frame["plot_unique_self"].mean())
    shared_value = float(data_frame["plot_shared"].mean())
    unique_other_value = float(data_frame["plot_unique_other"].mean())
    total_value = unique_self_value + shared_value + unique_other_value
    return {
        "unique_self": unique_self_value,
        "shared": shared_value,
        "unique_other": unique_other_value,
        "total": total_value,
    }


def _summarise_roi_over_time(
    data_frame: pd.DataFrame,
    *,
    roi_channels: Sequence[str],
    use_proportions_flag: bool,
    epsilon: float,
) -> pd.DataFrame:
    roi_df = data_frame[data_frame["channel"].isin(roi_channels)].copy()
    if roi_df.empty:
        roi_text = ", ".join(str(channel) for channel in roi_channels[:5])
        raise ValueError(f"ROI selection returned no rows. Check channel names. Example channels: {roi_text}")

    summary_df = (
        roi_df.groupby("time", as_index=False)[["plot_unique_self", "plot_shared", "plot_unique_other"]]
        .mean()
        .rename(
            columns={
                "plot_unique_self": "unique_self",
                "plot_shared": "shared",
                "plot_unique_other": "unique_other",
            }
        )
        .sort_values("time")
        .reset_index(drop=True)
    )
    summary_df["total"] = summary_df["unique_self"] + summary_df["shared"] + summary_df["unique_other"]

    if use_proportions_flag:
        denominator = summary_df["total"].to_numpy(dtype=float) + float(epsilon)
        for component_name in _COMPONENT_COLUMNS:
            summary_df[component_name] = summary_df[component_name].to_numpy(dtype=float) / denominator
        summary_df["total"] = 1.0

    return summary_df


def _plot_stacked_area(
    axis: plt.Axes,
    timecourse_df: pd.DataFrame,
    *,
    title: str,
    ylabel: str,
    colors: Mapping[str, str],
    theme_config: Mapping[str, Any] | None = None,
) -> None:
    time_values = timecourse_df["time"].to_numpy(dtype=float)
    axis.stackplot(
        time_values,
        timecourse_df["unique_self"].to_numpy(dtype=float),
        timecourse_df["shared"].to_numpy(dtype=float),
        timecourse_df["unique_other"].to_numpy(dtype=float),
        labels=["Unique self", "Shared", "Unique other"],
        colors=[colors["self"], colors["shared"], colors["other"]],
        alpha=0.9,
    )
    axis.axvline(0.0, linewidth=1.0, color=get_theme_color("reference", theme_config))
    axis.set_title(title, fontsize=13)
    axis.set_ylabel(ylabel)
    axis.set_xlim(float(time_values.min()), float(time_values.max()))
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def _add_roi_inset(
    axis: plt.Axes,
    *,
    roi_channels: Sequence[str],
    label: str,
    highlight_color: str,
    theme_config: Mapping[str, Any] | None = None,
    anchor: Sequence[float] = (0.01, 0.50, 0.18, 0.42),
) -> None:
    positions = _get_roi_channel_positions(roi_channels)
    if positions is None:
        return

    inset_axis = axis.inset_axes(anchor)
    head_color = get_theme_color("spine", theme_config)
    neutral_color = get_theme_color("grid", theme_config)
    background_color = get_theme_color("background", theme_config)

    inset_axis.set_facecolor(background_color)
    inset_axis.add_patch(Circle((0.0, 0.0), radius=1.0, fill=False, edgecolor=head_color, linewidth=1.0))
    inset_axis.plot([-0.18, 0.0, 0.18], [1.0, 1.15, 1.0], color=head_color, linewidth=1.0)
    inset_axis.plot([-1.0, -1.08, -1.0], [0.12, 0.0, -0.12], color=head_color, linewidth=1.0)
    inset_axis.plot([1.0, 1.08, 1.0], [0.12, 0.0, -0.12], color=head_color, linewidth=1.0)
    inset_axis.scatter(
        positions[:, 0],
        positions[:, 1],
        s=2,
        facecolors=get_theme_color("neutral", theme_config),
        edgecolors=background_color,
        linewidths=0.15,
        zorder=3,
    )
    inset_axis.scatter(
        positions[:, 0],
        positions[:, 1],
        s=5,
        facecolors="none",
        edgecolors="black",
        linewidths=0.3,
        zorder=2,
    )
    inset_axis.text(0.0, -1.22, label, ha="center", va="top", fontsize=8.5, fontweight="semibold")
    inset_axis.set_xlim(-1.15, 1.15)
    inset_axis.set_ylim(-1.28, 1.2)
    inset_axis.set_aspect("equal")
    inset_axis.axis("off")


def _get_roi_channel_positions(roi_channels: Sequence[str]) -> np.ndarray | None:
    channel_names = [str(channel) for channel in roi_channels]
    if not channel_names:
        return None

    try:
        info = mne.create_info(channel_names, sfreq=1.0, ch_types=["eeg"] * len(channel_names))
        montage = mne.channels.make_standard_montage(_DEFAULT_MONTAGE_NAME)
        info.set_montage(montage, on_missing="ignore")
        positions = _find_topomap_coords(info)
    except Exception:
        return None

    if positions.size == 0:
        return None

    radii = np.sqrt((positions**2).sum(axis=1))
    finite_mask = np.isfinite(radii)
    if not finite_mask.any():
        return None

    positions = positions[finite_mask]
    max_radius = float(np.max(np.sqrt((positions**2).sum(axis=1))))
    if not np.isfinite(max_radius) or max_radius <= 0.0:
        return None
    return positions / max_radius


def _draw_weighted_venn(
    axis: plt.Axes,
    *,
    unique_self_value: float,
    shared_value: float,
    unique_other_value: float,
    title: str,
    colors: Mapping[str, str],
    theme_config: Mapping[str, Any] | None = None,
) -> None:
    if venn2 is not None:
        _draw_weighted_venn_with_matplotlib_venn(
            axis,
            unique_self_value=unique_self_value,
            shared_value=shared_value,
            unique_other_value=unique_other_value,
            title=title,
            colors=colors,
            theme_config=theme_config,
        )
        return
    _draw_weighted_venn_fallback(
        axis,
        unique_self_value=unique_self_value,
        shared_value=shared_value,
        unique_other_value=unique_other_value,
        title=title,
        colors=colors,
        theme_config=theme_config,
    )


def _draw_weighted_venn_with_matplotlib_venn(
    axis: plt.Axes,
    *,
    unique_self_value: float,
    shared_value: float,
    unique_other_value: float,
    title: str,
    colors: Mapping[str, str],
    theme_config: Mapping[str, Any] | None = None,
) -> None:
    venn_object = venn2(
        subsets=(unique_self_value, unique_other_value, shared_value),
        set_labels=("Self", "Other"),
        ax=axis,
    )
    patch_styles = {"10": colors["self"], "01": colors["other"], "11": colors["shared"]}
    for region_id, region_color in patch_styles.items():
        patch = venn_object.get_patch_by_id(region_id)
        if patch is not None:
            patch.set_color(region_color)
            patch.set_alpha(0.78)
            patch.set_edgecolor(get_theme_color("spine", theme_config))
            patch.set_linewidth(1.25)

    label_map = {
        "10": f"Unique self\n{_format_scientific_mathtext(unique_self_value, precision=1)}",
        "01": f"Unique other\n{_format_scientific_mathtext(unique_other_value, precision=1)}",
        "11": f"Shared\n{_format_scientific_mathtext(shared_value, precision=1)}",
    }
    for region_id, region_text in label_map.items():
        label = venn_object.get_label_by_id(region_id)
        if label is not None:
            label.set_text(region_text)
            label.set_fontsize(11)
            label.set_fontweight("semibold")
            label.set_color("white")

    shared_label = venn_object.get_label_by_id("11")
    if shared_label is not None:
        x_position, y_position = shared_label.get_position()
        shared_label.set_position((x_position, y_position - 0.06))

    for set_id in ("A", "B"):
        set_label = venn_object.get_label_by_id(set_id)
        if set_label is not None:
            set_label.set_fontsize(12)
            set_label.set_fontweight("semibold")
            set_label.set_color("white")
    axis.set_title(title, fontsize=14, pad=12)


def _draw_weighted_venn_fallback(
    axis: plt.Axes,
    *,
    unique_self_value: float,
    shared_value: float,
    unique_other_value: float,
    title: str,
    colors: Mapping[str, str],
    theme_config: Mapping[str, Any] | None = None,
) -> None:
    total_self = max(unique_self_value + shared_value, 0.0)
    total_other = max(unique_other_value + shared_value, 0.0)
    max_total = max(total_self, total_other, 1e-9)
    self_radius = 0.75 * np.sqrt(total_self / max_total) if total_self > 0.0 else 0.2
    other_radius = 0.75 * np.sqrt(total_other / max_total) if total_other > 0.0 else 0.2
    edge_color = get_theme_color("spine", theme_config)

    left_center = (-0.35, 0.0)
    right_center = (0.35, 0.0)
    axis.add_patch(Circle(left_center, self_radius, facecolor=colors["self"], edgecolor=edge_color, linewidth=1.25, alpha=0.35))
    axis.add_patch(Circle(right_center, other_radius, facecolor=colors["other"], edgecolor=edge_color, linewidth=1.25, alpha=0.35))
    axis.text(left_center[0] - 0.12, 0.0, f"Unique self\n{_format_scientific_mathtext(unique_self_value, 1)}", ha="center", va="center", fontsize=11, fontweight="semibold", color="white")
    axis.text(right_center[0] + 0.12, 0.0, f"Unique other\n{_format_scientific_mathtext(unique_other_value, 1)}", ha="center", va="center", fontsize=11, fontweight="semibold", color="white")
    axis.text(0.0, -0.02, f"Shared\n{_format_scientific_mathtext(shared_value, 1)}", ha="center", va="center", fontsize=11, fontweight="semibold", color=colors["shared"])
    axis.text(left_center[0], self_radius + 0.16, "Self", ha="center", va="bottom", fontsize=12, fontweight="semibold", color="white")
    axis.text(right_center[0], other_radius + 0.16, "Other", ha="center", va="bottom", fontsize=12, fontweight="semibold", color="white")
    axis.set_title(title, fontsize=14, pad=12)
    axis.set_xlim(-1.35, 1.35)
    axis.set_ylim(-1.0, 1.1)
    axis.set_aspect("equal")
    axis.axis("off")


def _format_scientific_mathtext(value: float, precision: int = 1) -> str:
    if value == 0:
        return "$0$"
    if 1e-3 <= abs(value) < 1e3:
        return f"{value:.{precision + 2}f}"
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / (10**exponent)
    return rf"${mantissa:.{precision}f}\times10^{{{exponent}}}$"


def _find_topomap_coords(mne_info: mne.Info) -> np.ndarray:
    from mne.channels.layout import _find_topomap_coords as mne_find_topomap_coords

    return mne_find_topomap_coords(mne_info, picks=np.arange(len(mne_info.ch_names)))
