"""Behavioural figure helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rate.viz.theme import apply_theme, get_role_palette, make_dyad_palette


def build_behavioural_manifest(
    *,
    out_dir: Path,
    residuals_path: Path,
    subject_slopes_path: Path | None,
    manifest_path: Path,
    figure_config: Mapping[str, Any] | None = None,
    theme_config: Mapping[str, Any] | None = None,
    figure_subdir: str = "behavioural",
    formats: Sequence[str] = ("png", "pdf"),
    dpi: int = 300,
) -> dict[str, object]:
    """Render one combined behavioural figure and write a manifest."""
    figure_config = dict(figure_config or {})
    speech_rate_cfg = dict(figure_config.get("speech_rate") or {})
    subject_slopes_cfg = dict(figure_config.get("subject_slopes") or {})
    if subject_slopes_path is None:
        raise ValueError("subject_slopes_path is required for the combined behavioural figure.")

    plot = _build_combined_behavioural_plot(
        out_dir=out_dir,
        residuals_path=residuals_path,
        subject_slopes_path=subject_slopes_path,
        figure_subdir=figure_subdir,
        speech_rate_cfg=speech_rate_cfg,
        subject_slopes_cfg=subject_slopes_cfg,
        theme_config=theme_config,
        formats=tuple(speech_rate_cfg.get("formats", formats)),
        dpi=int(speech_rate_cfg.get("dpi", dpi)),
    )

    manifest = {
        "status": "ok",
        "plot_count": 1,
        "plots": [plot],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _build_combined_behavioural_plot(
    *,
    out_dir: Path,
    residuals_path: Path,
    subject_slopes_path: Path,
    figure_subdir: str,
    speech_rate_cfg: Mapping[str, Any],
    subject_slopes_cfg: Mapping[str, Any],
    theme_config: Mapping[str, Any] | None,
    formats: Sequence[str],
    dpi: int,
) -> dict[str, object]:
    theme = apply_theme(theme_config)
    palette = get_role_palette(theme)
    subject_col = str(speech_rate_cfg.get("subject_col", "subject_id"))
    dyad_col = str(speech_rate_cfg.get("dyad_col", "dyad_id"))
    rate_col = str(speech_rate_cfg.get("rate_col", "self_rate"))
    min_trials_per_subject = int(speech_rate_cfg.get("min_trials_per_subject", 5))
    histogram_bins = int(speech_rate_cfg.get("histogram_bins", 30))
    point_size = float(speech_rate_cfg.get("point_size", 70))
    dot_center_y = float(speech_rate_cfg.get("dot_center_y", 0.0))
    dot_spread_y = float(speech_rate_cfg.get("dot_spread_y", 0.10))
    jitter_sd = float(speech_rate_cfg.get("jitter_sd", 0.012))
    connector_alpha = float(speech_rate_cfg.get("connector_alpha", 0.75))
    dot_alpha = float(speech_rate_cfg.get("dot_alpha", 0.95))
    rng_seed = int(speech_rate_cfg.get("random_seed", 42))
    rng = np.random.default_rng(rng_seed)

    data = pd.read_csv(residuals_path)
    required_columns = [subject_col, dyad_col, rate_col]
    _require_columns(data, required_columns, source_path=residuals_path)

    data = data.dropna(subset=required_columns).copy()
    subject_counts = data.groupby(subject_col).size()
    valid_subjects = subject_counts[subject_counts >= min_trials_per_subject].index
    data = data[data[subject_col].isin(valid_subjects)].copy()

    subject_summary = (
        data.groupby(subject_col, as_index=False)
        .agg(
            median_rate=(rate_col, "median"),
            mean_rate=(rate_col, "mean"),
            n_trials=(rate_col, "size"),
            dyad_id=(dyad_col, "first"),
        )
        .sort_values(["dyad_id", subject_col])
        .reset_index(drop=True)
    )

    valid_dyads = [str(dyad) for dyad in subject_summary["dyad_id"].dropna().astype(str).unique().tolist()]
    dyad_to_color = _build_dyad_palette(valid_dyads, theme_config=theme)
    slope_dataframe, ordered_slope_df, dyad_slope_summary = _prepare_subject_slopes(
        subject_slopes_path=subject_slopes_path,
        plot_config=subject_slopes_cfg,
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    ax_hist, ax_subject, ax_slope = axes

    ax_hist.hist(
        data[rate_col].to_numpy(dtype=float),
        bins=histogram_bins,
        color=str(speech_rate_cfg.get("histogram_color", palette["self"])),
        edgecolor="none",
        linewidth=0.0,
    )
    ax_hist.set_title("A. Distribution of self speech rate")
    ax_hist.set_xlabel("Self speech rate")
    ax_hist.set_ylabel("Count")

    for dyad_id, dyad_df in subject_summary.groupby("dyad_id", dropna=True):
        dyad_df = dyad_df.sort_values("median_rate").reset_index(drop=True)
        dyad_key = str(dyad_id)
        color = dyad_to_color[dyad_key]

        if len(dyad_df) == 1:
            base_y_positions = np.array([dot_center_y], dtype=float)
        else:
            base_y_positions = np.linspace(
                dot_center_y - (dot_spread_y / 2.0),
                dot_center_y + (dot_spread_y / 2.0),
                len(dyad_df),
            )

        y_positions = base_y_positions + rng.normal(loc=0.0, scale=jitter_sd, size=len(dyad_df))
        median_rates = dyad_df["median_rate"].to_numpy(dtype=float)

        if len(dyad_df) >= 2:
            order = np.argsort(median_rates)
            ax_subject.plot(
                median_rates[order],
                y_positions[order],
                color=color,
                linewidth=2.0,
                alpha=connector_alpha,
                zorder=2,
            )

        ax_subject.scatter(
            median_rates,
            y_positions,
            s=point_size,
            color=[color],
            edgecolor="white",
            linewidth=0.9,
            alpha=dot_alpha,
            zorder=3,
        )

    ax_subject.set_title("B. Subject median speech rates by dyad")
    ax_subject.set_xlabel("Median self speech rate")
    ax_subject.set_ylabel("")
    ax_subject.set_yticks([])
    ax_subject.set_ylim(dot_center_y - dot_spread_y - 0.08, dot_center_y + dot_spread_y + 0.08)

    if valid_dyads:
        legend_handles = [
            plt.Line2D(
                [0],
                [0],
                color=dyad_to_color[dyad],
                marker="o",
                linestyle="-",
                linewidth=2.0,
                markersize=7,
                label=dyad,
            )
            for dyad in valid_dyads
        ]
        ax_subject.legend(
            handles=legend_handles,
            title="Dyad ID",
            bbox_to_anchor=(1.02, 1.0),
            loc="upper left",
            frameon=False,
        )

    plot_grouped_subject_slopes(
        ax=ax_slope,
        slope_dataframe=slope_dataframe,
        subject_column=str(subject_slopes_cfg.get("subject_col", "subject_id")),
        dyad_column=str(subject_slopes_cfg.get("dyad_col", "dyad_id")),
        title="C. Subject-specific other-rate slopes by dyad",
        xlabel="Subject-specific slope estimate",
        point_color=str(subject_slopes_cfg.get("point_color", palette["other"])),
        line_color=str(subject_slopes_cfg.get("line_color", palette["other"])),
        within_dyad_spacing=float(subject_slopes_cfg.get("within_dyad_spacing", 0.8)),
        between_dyad_spacing=float(subject_slopes_cfg.get("between_dyad_spacing", 1.8)),
    )

    fig.tight_layout()

    output_stem = out_dir / "figures" / figure_subdir / "behavioural_summary"
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    written_paths: list[str] = []
    for fmt in formats:
        output_path = output_stem.with_suffix(f".{fmt}")
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        written_paths.append(str(output_path))
    plt.close(fig)

    return {
        "plot_name": "behavioural_summary",
        "source_paths": [str(residuals_path), str(subject_slopes_path)],
        "files": written_paths,
        "n_rows": int(len(data)),
        "n_subjects": int(data[subject_col].nunique()),
        "n_dyads": int(data[dyad_col].nunique()),
        "min_trials_per_subject": min_trials_per_subject,
        "ordered_subject_ids": ordered_slope_df[str(subject_slopes_cfg.get("subject_col", "subject_id"))].astype(str).tolist(),
        "ordered_dyads": dyad_slope_summary[str(subject_slopes_cfg.get("dyad_col", "dyad_id"))].astype(str).tolist(),
    }


def _require_columns(data: pd.DataFrame, required_columns: Sequence[str], *, source_path: Path) -> None:
    missing = [column for column in required_columns if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required behavioural figure columns in {source_path}: {missing}")


def _build_dyad_palette(
    valid_dyads: Sequence[str],
    *,
    theme_config: Mapping[str, Any] | None = None,
) -> dict[str, tuple[float, float, float]]:
    return make_dyad_palette(tuple(valid_dyads), theme_config=theme_config)


def _prepare_subject_slopes(
    *,
    subject_slopes_path: Path,
    plot_config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    subject_col = str(plot_config.get("subject_col", "subject_id"))
    dyad_col = str(plot_config.get("dyad_col", "dyad_id"))
    slope_col = str(plot_config.get("slope_col", "slope"))
    se_col = str(plot_config.get("se_col", "se"))
    slope_dataframe = pd.read_csv(subject_slopes_path)
    _require_columns(slope_dataframe, [subject_col, dyad_col, slope_col, se_col], source_path=subject_slopes_path)
    slope_dataframe = slope_dataframe.dropna(subset=[subject_col, dyad_col, slope_col, se_col]).copy()

    ordered_plot_df, dyad_summary = _order_grouped_subject_slopes(
        slope_dataframe=slope_dataframe,
        dyad_column=dyad_col,
    )
    return slope_dataframe, ordered_plot_df, dyad_summary


def plot_grouped_subject_slopes(
    ax: plt.Axes,
    slope_dataframe: pd.DataFrame,
    subject_column: str,
    dyad_column: str,
    title: str,
    xlabel: str,
    point_color: str = "0.25",
    line_color: str = "0.35",
    within_dyad_spacing: float = 0.8,
    between_dyad_spacing: float = 1.8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Plot grouped subject-wise slopes with 95% confidence intervals."""
    ordered_plot_df, dyad_summary = _order_grouped_subject_slopes(
        slope_dataframe=slope_dataframe,
        dyad_column=dyad_column,
    )
    ordered_plot_df = ordered_plot_df.copy()
    current_y = 0.0
    y_positions: list[float] = []
    for dyad_id in dyad_summary[dyad_column].tolist():
        dyad_size = int((ordered_plot_df[dyad_column] == dyad_id).sum())
        for _ in range(dyad_size):
            y_positions.append(current_y)
            current_y += within_dyad_spacing
        current_y += between_dyad_spacing
    ordered_plot_df["y_position"] = y_positions

    for _, row in ordered_plot_df.iterrows():
        ax.hlines(
            y=float(row["y_position"]),
            xmin=float(row["slope"]) - (1.96 * float(row["se"])),
            xmax=float(row["slope"]) + (1.96 * float(row["se"])),
            linewidth=1.5,
            color=line_color,
            alpha=0.9,
        )
        ax.scatter(
            float(row["slope"]),
            float(row["y_position"]),
            s=35,
            color=point_color,
            alpha=0.95,
            zorder=3,
        )

    current_y = 0.0
    for dyad_id in dyad_summary[dyad_column].tolist()[:-1]:
        dyad_size = int((ordered_plot_df[dyad_column] == dyad_id).sum())
        current_y += dyad_size * within_dyad_spacing
        separator_y = current_y - (0.5 * within_dyad_spacing) + (0.5 * between_dyad_spacing)
        ax.axhline(separator_y, color="0.9", linewidth=1, zorder=0)
        current_y += between_dyad_spacing

    ax.axvline(0.0, linewidth=1, color="0.4")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Subjects grouped by dyad")
    ax.set_yticks([])

    return ordered_plot_df, dyad_summary


def _order_grouped_subject_slopes(
    *,
    slope_dataframe: pd.DataFrame,
    dyad_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    plot_df = slope_dataframe.copy()
    dyad_summary = (
        plot_df.groupby(dyad_column, as_index=False)
        .agg(dyad_mean_slope=("slope", "mean"))
        .sort_values("dyad_mean_slope")
        .reset_index(drop=True)
    )
    ordered_dyads = dyad_summary[dyad_column].tolist()

    ordered_rows: list[pd.DataFrame] = []
    for dyad_id in ordered_dyads:
        dyad_df = (
            plot_df[plot_df[dyad_column] == dyad_id]
            .sort_values("slope")
            .reset_index(drop=True)
        )
        ordered_rows.append(dyad_df)

    ordered_plot_df = pd.concat(ordered_rows, ignore_index=True) if ordered_rows else pd.DataFrame(columns=plot_df.columns)
    if ordered_plot_df.empty:
        raise ValueError("No subject slope rows available after filtering missing values.")
    return ordered_plot_df, dyad_summary


def _plot_regression_scatter(
    *,
    ax: plt.Axes,
    dataframe: pd.DataFrame,
    x_column: str,
    y_column: str,
    color: str,
    label: str,
) -> None:
    plot_df = dataframe.dropna(subset=[x_column, y_column]).copy()
    x_values = plot_df[x_column].to_numpy(dtype=float)
    y_values = plot_df[y_column].to_numpy(dtype=float)
    ax.scatter(
        x_values,
        y_values,
        s=60,
        alpha=0.9,
        color=color,
        label=label,
    )
    if len(plot_df) >= 2 and np.unique(x_values).size >= 2:
        slope, intercept = np.polyfit(x_values, y_values, 1)
        x_line = np.linspace(float(np.min(x_values)), float(np.max(x_values)), 100)
        y_line = (slope * x_line) + intercept
        if len(plot_df) >= 3:
            predicted = (slope * x_values) + intercept
            residuals = y_values - predicted
            degrees_of_freedom = len(x_values) - 2
            if degrees_of_freedom > 0:
                residual_sd = np.sqrt(np.sum(residuals**2) / degrees_of_freedom)
                x_mean = float(np.mean(x_values))
                sxx = float(np.sum((x_values - x_mean) ** 2))
                if sxx > 0:
                    standard_error = residual_sd * np.sqrt((1.0 / len(x_values)) + (((x_line - x_mean) ** 2) / sxx))
                    ax.fill_between(
                        x_line,
                        y_line - (1.96 * standard_error),
                        y_line + (1.96 * standard_error),
                        color=color,
                        alpha=0.15,
                        linewidth=0,
                    )
        ax.plot(x_line, y_line, linewidth=2, color=color)
