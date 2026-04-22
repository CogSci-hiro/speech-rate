"""Behavioural figure helpers."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from rate.viz.theme import apply_theme, get_role_palette, make_dyad_palette


def build_behavioural_manifest(
    *,
    out_dir: Path,
    residuals_path: Path,
    random_effects_path: Path | None,
    subject_slopes_path: Path | None,
    lmm_summary_path: Path | None,
    manifest_path: Path,
    output_stem: Path | None = None,
    figure_config: Mapping[str, Any] | None = None,
    theme_config: Mapping[str, Any] | None = None,
    figure_subdir: str = "behavioural",
    formats: Sequence[str] = ("png", "pdf"),
    dpi: int = 300,
    write_manifest: bool = True,
) -> dict[str, object]:
    """Render one combined behavioural figure and write a manifest."""
    figure_config = dict(figure_config or {})
    speech_rate_cfg = dict(figure_config.get("speech_rate") or {})
    if lmm_summary_path is None:
        raise ValueError("lmm_summary_path is required for the combined behavioural figure.")

    plot = _build_combined_behavioural_plot(
        out_dir=out_dir,
        residuals_path=residuals_path,
        random_effects_path=random_effects_path,
        lmm_summary_path=lmm_summary_path,
        output_stem=output_stem,
        figure_subdir=figure_subdir,
        speech_rate_cfg=speech_rate_cfg,
        theme_config=theme_config,
        formats=tuple(speech_rate_cfg.get("formats", formats)),
        dpi=int(speech_rate_cfg.get("dpi", dpi)),
    )

    manifest = {
        "status": "ok",
        "plot_count": 1,
        "plots": [plot],
    }
    if write_manifest:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _build_combined_behavioural_plot(
    *,
    out_dir: Path,
    residuals_path: Path,
    random_effects_path: Path | None,
    lmm_summary_path: Path,
    output_stem: Path | None,
    figure_subdir: str,
    speech_rate_cfg: Mapping[str, Any],
    theme_config: Mapping[str, Any] | None,
    formats: Sequence[str],
    dpi: int,
) -> dict[str, object]:
    theme = apply_theme(theme_config)
    palette = get_role_palette(theme)
    subject_col = str(speech_rate_cfg.get("subject_col", "subject_id"))
    dyad_col = str(speech_rate_cfg.get("dyad_col", "dyad_id"))
    rate_col = str(speech_rate_cfg.get("rate_col", "self_rate"))
    regression_x_col = str(speech_rate_cfg.get("regression_x_col", "other_rate"))
    regression_y_col = str(speech_rate_cfg.get("regression_y_col", rate_col))
    lmm_analysis_id = str(speech_rate_cfg.get("lmm_analysis_id", "self_rate_by_other_rate"))
    min_trials_per_subject = int(speech_rate_cfg.get("min_trials_per_subject", 5))
    histogram_bins = int(speech_rate_cfg.get("histogram_bins", 30))
    point_size = float(speech_rate_cfg.get("point_size", 70))
    dot_center_y = float(speech_rate_cfg.get("dot_center_y", 0.0))
    dot_spread_y = float(speech_rate_cfg.get("dot_spread_y", 0.10))
    jitter_sd = float(speech_rate_cfg.get("jitter_sd", 0.012))
    connector_alpha = float(speech_rate_cfg.get("connector_alpha", 0.75))
    dot_alpha = float(speech_rate_cfg.get("dot_alpha", 0.95))
    font_scale = float(speech_rate_cfg.get("font_scale", 3.0))
    rng_seed = int(speech_rate_cfg.get("random_seed", 42))
    regression_line_width = float(speech_rate_cfg.get("regression_line_width", 2.4))
    show_marginalised_scatter = bool(speech_rate_cfg.get("show_marginalised_scatter", False))
    marginalised_scatter_mode = str(speech_rate_cfg.get("marginalised_scatter_mode", "partial_residual"))
    marginalised_scatter_alpha = float(speech_rate_cfg.get("marginalised_scatter_alpha", 0.35))
    marginalised_scatter_size = float(speech_rate_cfg.get("marginalised_scatter_size", 20))
    regression_line_color = str(speech_rate_cfg.get("regression_line_color", palette["other"]))
    marginalised_scatter_color = str(speech_rate_cfg.get("marginalised_scatter_color", palette["other"]))
    rng = np.random.default_rng(rng_seed)

    data = pd.read_csv(residuals_path)
    random_effects = _load_random_effects(random_effects_path)
    required_columns = [subject_col, dyad_col, rate_col, regression_x_col, regression_y_col]
    _require_columns(data, required_columns, source_path=residuals_path)
    lmm_fit = _extract_lmm_fixed_effects_from_summary(
        lmm_summary_path=lmm_summary_path,
        analysis_id=lmm_analysis_id,
    )

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
    valid_dyads = _collect_valid_dyads(
        subject_summary=subject_summary,
        dyad_column=dyad_col,
        preferred_dyads=subject_summary[dyad_col].astype(str).dropna().tolist(),
    )
    dyad_to_color = _build_dyad_palette(valid_dyads, theme_config=theme)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    ax_hist, ax_subject, ax_slope = axes

    ax_hist.hist(
        data[rate_col].to_numpy(dtype=float),
        bins=histogram_bins,
        color=str(speech_rate_cfg.get("histogram_color", theme["colors"]["neutral"])),
        edgecolor="none",
        linewidth=0.0,
    )
    ax_hist.set_title("A. Distribution of\nself speech rate", fontsize=_scaled_fontsize(11, font_scale))
    ax_hist.set_xlabel("Self speech rate", fontsize=_scaled_fontsize(10, font_scale))
    ax_hist.set_ylabel("Count", fontsize=_scaled_fontsize(10, font_scale))
    ax_hist.tick_params(axis="both", labelsize=_scaled_fontsize(8, font_scale))

    _plot_subject_median_rainplot(
        ax=ax_subject,
        subject_summary=subject_summary,
        dyad_to_color=dyad_to_color,
        point_size=point_size,
        dot_center_y=dot_center_y,
        dot_spread_y=dot_spread_y,
        jitter_sd=jitter_sd,
        connector_alpha=connector_alpha,
        dot_alpha=dot_alpha,
        rng=rng,
        font_scale=font_scale,
    )

    _plot_lmm_relationship(
        ax=ax_slope,
        dataframe=data,
        x_column=regression_x_col,
        y_column=regression_y_col,
        title="C. LMM model: fixed effect",
        xlabel="Other speech rate",
        ylabel="Self speech rate",
        line_color=regression_line_color,
        fixed_effects=lmm_fit,
        random_effects=random_effects,
        subject_column=subject_col,
        show_marginalised_scatter=show_marginalised_scatter,
        marginalised_scatter_mode=marginalised_scatter_mode,
        marginalised_scatter_color=marginalised_scatter_color,
        marginalised_scatter_alpha=marginalised_scatter_alpha,
        marginalised_scatter_size=marginalised_scatter_size,
        line_width=regression_line_width,
        font_scale=font_scale,
    )

    fig.tight_layout()

    resolved_output_stem = output_stem if output_stem is not None else out_dir / "figures" / figure_subdir / "behavioural_summary"
    resolved_output_stem.parent.mkdir(parents=True, exist_ok=True)
    written_paths: list[str] = []
    for fmt in formats:
        output_path = resolved_output_stem.with_suffix(f".{fmt}")
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        written_paths.append(str(output_path))
    plt.close(fig)

    return {
        "plot_name": "behavioural_summary",
        "source_paths": [
            str(residuals_path),
            *([str(random_effects_path)] if random_effects_path is not None else []),
            str(lmm_summary_path),
        ],
        "files": written_paths,
        "n_rows": int(len(data)),
        "n_subjects": int(data[subject_col].nunique()),
        "n_dyads": int(data[dyad_col].nunique()),
        "min_trials_per_subject": min_trials_per_subject,
        "ordered_subject_ids": subject_summary[subject_col].astype(str).tolist(),
        "ordered_dyads": subject_summary[dyad_col].astype(str).drop_duplicates().tolist(),
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


def _collect_valid_dyads(
    *,
    subject_summary: pd.DataFrame,
    dyad_column: str,
    preferred_dyads: Sequence[str] | None = None,
) -> list[str]:
    subject_dyads = subject_summary[dyad_column].dropna().astype(str).tolist()
    return list(dict.fromkeys(list(preferred_dyads or []) + subject_dyads))


def _load_random_effects(random_effects_path: Path | None) -> pd.DataFrame | None:
    if random_effects_path is None or not random_effects_path.exists():
        return None
    random_effects = pd.read_csv(random_effects_path)
    required_columns = ["grouping_factor", "level_id", "random_intercept"]
    _require_columns(random_effects, required_columns, source_path=random_effects_path)
    return random_effects


def _plot_subject_median_rainplot(
    *,
    ax: plt.Axes,
    subject_summary: pd.DataFrame,
    dyad_to_color: Mapping[str, tuple[float, float, float]],
    point_size: float,
    dot_center_y: float,
    dot_spread_y: float,
    jitter_sd: float,
    connector_alpha: float,
    dot_alpha: float,
    rng: np.random.Generator,
    font_scale: float,
) -> None:
    kde_height = max(dot_spread_y * 4.0, 0.32)
    rain_offset = max(dot_spread_y * 1.6, 0.18)
    x_min = float(subject_summary["median_rate"].min())
    x_max = float(subject_summary["median_rate"].max())
    x_padding = max((x_max - x_min) * 0.08, 0.1)
    plot_df = subject_summary.sort_values(["dyad_id", "median_rate", "subject_id"]).reset_index(drop=True)
    all_median_rates = plot_df["median_rate"].to_numpy(dtype=float)
    base_y = 0.0

    if len(all_median_rates) >= 2 and np.ptp(all_median_rates) > 0:
        x_grid = np.linspace(all_median_rates.min() - x_padding, all_median_rates.max() + x_padding, 256)
        density = gaussian_kde(all_median_rates)(x_grid)
        density = (density / np.max(density)) * kde_height
        ax.fill_between(
            x_grid,
            base_y,
            base_y + density,
            color="0.6",
            alpha=0.22,
            linewidth=0.0,
            zorder=1,
        )
        ax.plot(x_grid, base_y + density, color="0.45", linewidth=1.2, alpha=0.95, zorder=2)
    else:
        ax.vlines(
            float(all_median_rates[0]),
            ymin=base_y,
            ymax=base_y + (kde_height * 0.5),
            color="0.45",
            linewidth=1.2,
            alpha=0.9,
            zorder=2,
        )

    for dyad_id, dyad_df in plot_df.groupby("dyad_id", sort=False):
        dyad_df = dyad_df.sort_values("median_rate").reset_index(drop=True)
        color = dyad_to_color[str(dyad_id)]
        median_rates = dyad_df["median_rate"].to_numpy(dtype=float)

        if len(dyad_df) == 1:
            base_y_positions = np.array([base_y - rain_offset + dot_center_y], dtype=float)
        else:
            base_y_positions = np.linspace(
                base_y - rain_offset + dot_center_y - (dot_spread_y / 2.0),
                base_y - rain_offset + dot_center_y + (dot_spread_y / 2.0),
                len(dyad_df),
            )

        y_positions = base_y_positions + rng.normal(loc=0.0, scale=jitter_sd, size=len(dyad_df))

        if len(dyad_df) >= 2:
            order = np.argsort(median_rates)
            ax.plot(
                median_rates[order],
                y_positions[order],
                color=color,
                linewidth=1.8,
                alpha=connector_alpha,
                zorder=3,
            )

        ax.scatter(
            median_rates,
            y_positions,
            s=point_size,
            color=[color],
            edgecolor="white",
            linewidth=0.9,
            alpha=dot_alpha,
            zorder=4,
        )

    ax.set_title("B. Subject median\nspeech rates", fontsize=_scaled_fontsize(11, font_scale))
    ax.set_xlabel("Median self speech rate", fontsize=_scaled_fontsize(10, font_scale))
    ax.set_ylabel("")
    ax.set_yticks([])
    ax.tick_params(axis="both", labelsize=_scaled_fontsize(8, font_scale))
    ax.set_ylim(base_y - rain_offset - dot_spread_y - 0.08, base_y + kde_height + 0.1)
    ax.set_xlim(x_min - x_padding, x_max + x_padding)


def _scaled_fontsize(base_size: float, scale: float) -> float:
    return float(base_size) * float(scale)


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


def _extract_lmm_fixed_effects_from_summary(
    *,
    lmm_summary_path: Path,
    analysis_id: str,
) -> dict[str, dict[str, float]]:
    summary_text = lmm_summary_path.read_text(encoding="utf-8")
    blocks = re.split(r"\n={10,}\n", summary_text)
    target_block = next(
        (block for block in blocks if f"analysis_id: {analysis_id}" in block),
        None,
    )
    if target_block is None:
        raise ValueError(f"Analysis id `{analysis_id}` not found in LMM summary file: {lmm_summary_path}")

    lines = target_block.splitlines()
    try:
        fixed_effects_start = lines.index("Fixed effects:") + 1
    except ValueError as exc:
        raise ValueError(f"Fixed effects section not found for analysis `{analysis_id}` in {lmm_summary_path}") from exc

    fixed_effects: dict[str, dict[str, float]] = {}
    number_pattern = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    coefficient_pattern = re.compile(rf"^\s*(\S+)\s+({number_pattern})\s+({number_pattern})(?:\s+.*)?$")
    for line in lines[fixed_effects_start:]:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("Correlation of Fixed Effects:") or stripped.startswith("---"):
            break
        if stripped.startswith("Estimate Std. Error") or stripped.startswith("Fixed effects:"):
            continue
        match = coefficient_pattern.match(line)
        if match is None:
            continue
        fixed_effects[match.group(1)] = {
            "estimate": float(match.group(2)),
            "std_error": float(match.group(3)),
        }

    if "(Intercept)" not in fixed_effects:
        raise ValueError(f"Intercept not found in fixed effects for analysis `{analysis_id}` in {lmm_summary_path}")
    return fixed_effects


def _plot_lmm_relationship(
    *,
    ax: plt.Axes,
    dataframe: pd.DataFrame,
    x_column: str,
    y_column: str,
    title: str,
    xlabel: str,
    ylabel: str,
    line_color: str,
    fixed_effects: Mapping[str, Mapping[str, float]],
    random_effects: pd.DataFrame | None,
    subject_column: str,
    show_marginalised_scatter: bool,
    marginalised_scatter_mode: str,
    marginalised_scatter_color: str,
    marginalised_scatter_alpha: float,
    marginalised_scatter_size: float,
    line_width: float,
    font_scale: float,
) -> None:
    plot_df = dataframe.dropna(subset=[x_column, y_column]).copy()
    x_values = plot_df[x_column].to_numpy(dtype=float)
    if len(plot_df) == 0:
        raise ValueError(f"No rows available to plot `{y_column}` against `{x_column}`.")
    ax.set_facecolor((0.98, 0.98, 0.99))

    intercept = float(fixed_effects["(Intercept)"]["estimate"])
    intercept_se = float(fixed_effects["(Intercept)"]["std_error"])
    adjustment_terms = [
        term for term in fixed_effects.keys()
        if term not in {"(Intercept)", x_column} and term in plot_df.columns
    ]
    adjusted_intercept = intercept + sum(
        float(fixed_effects[term]["estimate"]) * float(plot_df[term].astype(float).mean())
        for term in adjustment_terms
    )
    slope = float(fixed_effects.get(x_column, {}).get("estimate", 0.0))
    slope_se = float(fixed_effects.get(x_column, {}).get("std_error", 0.0))
    if np.min(x_values) == np.max(x_values):
        x_line = np.asarray([float(np.min(x_values)), float(np.max(x_values))], dtype=float)
    else:
        x_line = np.linspace(float(np.min(x_values)), float(np.max(x_values)), 200)
    y_line = adjusted_intercept + (slope * x_line)

    if show_marginalised_scatter:
        if marginalised_scatter_mode == "population_level":
            y_marginalised = _compute_population_level_marginal_response(
                dataframe=plot_df,
                x_column=x_column,
                fixed_effects=fixed_effects,
            )
        elif marginalised_scatter_mode == "partial_residual":
            y_marginalised = _compute_partial_residual_response(
                dataframe=plot_df,
                y_column=y_column,
                x_column=x_column,
                fixed_effects=fixed_effects,
                random_effects=random_effects,
                subject_column=subject_column,
            )
        else:
            raise ValueError(
                "Unsupported `marginalised_scatter_mode`: "
                f"{marginalised_scatter_mode}. Expected `partial_residual` or `population_level`."
            )
        ax.scatter(
            x_values,
            y_marginalised,
            s=marginalised_scatter_size,
            alpha=marginalised_scatter_alpha,
            color=marginalised_scatter_color,
            edgecolor="white",
            linewidth=0.3,
            zorder=2,
        )

    band_half_width = 1.96 * np.sqrt((intercept_se**2) + ((x_line**2) * (slope_se**2)))
    ax.fill_between(
        x_line,
        y_line - band_half_width,
        y_line + band_half_width,
        color=line_color,
        alpha=0.16,
        linewidth=0.0,
    )
    ax.plot(x_line, y_line, linewidth=line_width, color=line_color, alpha=0.98)
    ax.grid(axis="both", color="0.90", linewidth=0.8, alpha=0.8)

    ax.set_title(title, fontsize=_scaled_fontsize(11, font_scale))
    ax.set_xlabel(xlabel, fontsize=_scaled_fontsize(10, font_scale))
    ax.set_ylabel(ylabel, fontsize=_scaled_fontsize(10, font_scale))
    ax.tick_params(axis="both", labelsize=_scaled_fontsize(8, font_scale))


def _compute_population_level_marginal_response(
    *,
    dataframe: pd.DataFrame,
    x_column: str,
    fixed_effects: Mapping[str, Mapping[str, float]],
) -> np.ndarray:
    intercept = float(fixed_effects["(Intercept)"]["estimate"])
    adjustment_terms = [
        term for term in fixed_effects.keys()
        if term not in {"(Intercept)", x_column} and term in dataframe.columns
    ]
    adjusted_intercept = intercept + sum(
        float(fixed_effects[term]["estimate"]) * float(dataframe[term].astype(float).mean())
        for term in adjustment_terms
    )
    slope = float(fixed_effects.get(x_column, {}).get("estimate", 0.0))
    x_values = dataframe[x_column].to_numpy(dtype=float)
    return adjusted_intercept + (slope * x_values)


def _compute_partial_residual_response(
    *,
    dataframe: pd.DataFrame,
    y_column: str,
    x_column: str,
    fixed_effects: Mapping[str, Mapping[str, float]],
    random_effects: pd.DataFrame | None,
    subject_column: str,
) -> np.ndarray:
    y_values = dataframe[y_column].to_numpy(dtype=float)
    partial_response = y_values.copy()
    for term, stats in fixed_effects.items():
        if term in {"(Intercept)", x_column, y_column} or term not in dataframe.columns:
            continue
        term_values = dataframe[term].to_numpy(dtype=float)
        term_mean = float(np.mean(term_values))
        partial_response = partial_response - (float(stats["estimate"]) * (term_values - term_mean))
    if random_effects is not None and subject_column in dataframe.columns:
        subject_random_intercepts = (
            random_effects.loc[random_effects["grouping_factor"].astype(str) == subject_column, ["level_id", "random_intercept"]]
            .dropna(subset=["level_id", "random_intercept"])
            .assign(level_id=lambda df: df["level_id"].astype(str))
            .drop_duplicates(subset=["level_id"], keep="last")
            .set_index("level_id")["random_intercept"]
        )
        subject_intercepts = dataframe[subject_column].astype(str).map(subject_random_intercepts).fillna(0.0).to_numpy(dtype=float)
        partial_response = partial_response - subject_intercepts
    return partial_response
