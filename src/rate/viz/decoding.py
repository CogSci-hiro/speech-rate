"""Decoding figure helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from rate.decoding.io import build_decoding_model_output_paths
from rate.viz.theme import apply_theme, get_role_palette


@dataclass(frozen=True)
class _DecodingFigurePayload:
    model_name: str
    title: str
    color: str
    times: np.ndarray
    score_matrix: np.ndarray
    significance_mask: np.ndarray | None
    selected_alpha: float | None
    alpha_selection_source: str


_MODEL_TITLES = {
    "self_temporal_generalisation": "Self rate",
    "other_temporal_generalisation": "Other rate",
    "cross_self_to_other": "Self to other",
    "cross_other_to_self": "Other to self",
}

def build_decoding_manifest(
    *,
    out_dir: Path,
    decoding_config: Mapping[str, Any],
    manifest_path: Path,
    figure_config: Mapping[str, Any] | None = None,
    theme_config: Mapping[str, Any] | None = None,
    figure_subdir: str = "decoding",
    formats: Sequence[str] = ("png", "pdf"),
    dpi: int = 300,
) -> dict[str, object]:
    """Render a combined temporal-generalisation decoding figure and write a manifest."""
    figure_config = dict(figure_config or {})
    theme = apply_theme(theme_config)
    payloads = _load_payloads(out_dir=out_dir, decoding_config=decoding_config, theme_config=theme)
    if not payloads:
        raise ValueError("No decoding figure payloads were loaded.")

    diag_auto = bool(figure_config.get("diag_auto", False))
    if diag_auto:
        diag_ymin, diag_ymax = _infer_diagonal_limits(
            payloads,
            pad_fraction=float(figure_config.get("diag_pad_fraction", 0.025)),
            min_half_range=float(figure_config.get("diag_min_half_range", 0.015)),
        )
    else:
        diag_ymin = float(figure_config.get("diag_ymin", -0.1))
        diag_ymax = float(figure_config.get("diag_ymax", 0.5))
    p_threshold = float(figure_config.get("p_threshold", 0.05))
    figure_width = float(figure_config.get("figure_width", max(12.0, 3.8 * len(payloads))))
    figure_height = float(figure_config.get("figure_height", 5.8))
    matrix_limit = figure_config.get("matrix_limit")
    matrix_limit_percentile = float(figure_config.get("matrix_limit_percentile", 100.0))
    contour_linewidth = float(figure_config.get("contour_linewidth", 0.75))
    matrix_abs_limit = (
        float(matrix_limit)
        if matrix_limit is not None
        else _infer_matrix_limit(payloads, percentile=matrix_limit_percentile)
    )

    fig, plot_metadata = _plot_decoding_summary(
        payloads,
        diag_ymin=diag_ymin,
        diag_ymax=diag_ymax,
        matrix_abs_limit=matrix_abs_limit,
        contour_linewidth=contour_linewidth,
        p_threshold=p_threshold,
        figure_width=figure_width,
        figure_height=figure_height,
    )

    output_stem = out_dir / "figures" / figure_subdir / "temporal_generalisation_summary"
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    written_paths: list[str] = []
    for fmt in formats:
        output_path = output_stem.with_suffix(f".{fmt}")
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        written_paths.append(str(output_path))
    plt.close(fig)

    manifest = {
        "status": "ok",
        "plot_count": 1,
        "plots": [
            {
                "plot_name": "temporal_generalisation_summary",
                "files": written_paths,
                "p_threshold": p_threshold,
                "diag_ylim": [diag_ymin, diag_ymax],
                "matrix_abs_limit": matrix_abs_limit,
                "models": plot_metadata,
            }
        ],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def build_decoding_results_manifest(
    *,
    out_dir: Path,
    decoding_config: Mapping[str, Any],
    manifest_path: Path,
    figure_config: Mapping[str, Any] | None = None,
    theme_config: Mapping[str, Any] | None = None,
    figure_subdir: str = "decoding_results",
    formats: Sequence[str] = ("png", "pdf"),
    dpi: int = 300,
) -> dict[str, object]:
    """Render per-model decoding stats figures from observed TG and TFCE outputs."""
    figure_config = dict(figure_config or {})
    theme = apply_theme(theme_config)
    payloads = _load_payloads(out_dir=out_dir, decoding_config=decoding_config, theme_config=theme)
    if not payloads:
        raise ValueError("No decoding results payloads were loaded.")

    observed_limit = figure_config.get("observed_matrix_limit")
    observed_abs_limit = float(observed_limit) if observed_limit is not None else _infer_matrix_limit(payloads)
    tfce_limit = figure_config.get("tfce_matrix_limit")
    p_threshold = float(figure_config.get("p_threshold", 0.05))
    plots: list[dict[str, object]] = []

    for payload in payloads:
        stats_payload = _load_decoding_stats_payload(
            out_dir=out_dir,
            decoding_config=decoding_config,
            model_name=payload.model_name,
        )
        tfce_abs_limit = float(tfce_limit) if tfce_limit is not None else _infer_abs_limit(stats_payload["tfce_matrix"], fallback=0.05)
        fig = _plot_decoding_results(
            payload=payload,
            tfce_matrix=np.asarray(stats_payload["tfce_matrix"], dtype=float),
            corrected_p=np.asarray(stats_payload["corrected_p"], dtype=float),
            observed_abs_limit=observed_abs_limit,
            tfce_abs_limit=tfce_abs_limit,
            p_threshold=p_threshold,
        )
        output_stem = out_dir / "figures" / figure_subdir / payload.model_name / "temporal_generalisation_results"
        output_stem.parent.mkdir(parents=True, exist_ok=True)
        written_paths: list[str] = []
        for fmt in formats:
            output_path = output_stem.with_suffix(f".{fmt}")
            fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
            written_paths.append(str(output_path))
        plt.close(fig)
        plots.append(
            {
                "plot_name": "temporal_generalisation_results",
                "model_name": payload.model_name,
                "title": payload.title,
                "files": written_paths,
                "p_threshold": p_threshold,
                "has_significant_samples": bool(np.any(np.asarray(stats_payload["corrected_p"], dtype=float) < p_threshold)),
                "selected_alpha": payload.selected_alpha,
                "alpha_selection_source": payload.alpha_selection_source,
            }
        )

    manifest = {"status": "ok", "plot_count": len(plots), "plots": plots}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _load_payloads(
    *,
    out_dir: Path,
    decoding_config: Mapping[str, Any],
    theme_config: Mapping[str, Any] | None = None,
) -> list[_DecodingFigurePayload]:
    role_palette = get_role_palette(theme_config)
    model_colors = {
        "self_temporal_generalisation": role_palette["self"],
        "other_temporal_generalisation": role_palette["other"],
        "cross_self_to_other": role_palette["shared"],
        "cross_other_to_self": role_palette["shared"],
    }
    estimator_cfg = dict(decoding_config.get("estimator") or {})
    alpha_values = _resolve_ridge_alphas(estimator_cfg)
    payloads: list[_DecodingFigurePayload] = []

    for model_cfg in decoding_config.get("models") or []:
        model_name = str(model_cfg["name"])
        output_paths = build_decoding_model_output_paths({"paths": {"out_dir": str(out_dir)}}, decoding_config, model_name=model_name)
        required_paths = (
            output_paths.scores_output_path,
            output_paths.fold_scores_output_path,
            output_paths.times_output_path,
        )
        missing_paths = [path for path in required_paths if not path.exists()]
        if missing_paths:
            missing_text = ", ".join(str(path) for path in missing_paths)
            raise FileNotFoundError(f"Missing decoding figure inputs for model `{model_name}`: {missing_text}")

        times = np.asarray(np.load(output_paths.times_output_path), dtype=float)
        scores = np.asarray(np.load(output_paths.scores_output_path), dtype=float)
        score_matrix = scores
        selected_alpha: float | None = None
        alpha_selection_source = "mean_scores"

        if output_paths.alpha_scores_output_path.exists():
            alpha_scores = np.asarray(np.load(output_paths.alpha_scores_output_path), dtype=float)
            if alpha_scores.ndim != 3:
                raise ValueError(
                    f"Expected alpha scores with shape (n_alphas, n_times, n_times) for `{model_name}`, "
                    f"got {alpha_scores.shape}."
                )
            if int(alpha_scores.shape[0]) != int(alpha_values.size):
                raise ValueError(
                    f"Alpha score count does not match configured ridge alphas for `{model_name}`: "
                    f"{alpha_scores.shape[0]} vs {alpha_values.size}."
                )
            alpha_diag_scores = np.nanmean(np.diagonal(alpha_scores, axis1=1, axis2=2), axis=1)
            if np.isfinite(alpha_diag_scores).any():
                best_alpha_idx = int(np.nanargmax(alpha_diag_scores))
                score_matrix = np.asarray(alpha_scores[best_alpha_idx], dtype=float)
                selected_alpha = float(alpha_values[best_alpha_idx])
                alpha_selection_source = "alpha_scores_mean_diagonal"
        elif output_paths.best_alphas_output_path.exists():
            best_alphas = np.asarray(np.load(output_paths.best_alphas_output_path), dtype=float)
            finite_best_alphas = best_alphas[np.isfinite(best_alphas)]
            if finite_best_alphas.size:
                selected_alpha = float(np.median(finite_best_alphas))
                alpha_selection_source = "best_alphas_median"

        significance_mask = None
        if output_paths.tg_p_fwer_output_path.exists():
            corrected_p = np.asarray(np.load(output_paths.tg_p_fwer_output_path), dtype=float)
            if corrected_p.shape != score_matrix.shape:
                raise ValueError(
                    f"Corrected p-value map shape mismatch for `{model_name}`: {corrected_p.shape} vs {score_matrix.shape}."
                )
            significance_mask = corrected_p < 0.05

        payloads.append(
            _DecodingFigurePayload(
                model_name=model_name,
                title=_MODEL_TITLES.get(model_name, _humanize_model_name(model_name)),
                color=model_colors.get(model_name, role_palette["shared"]),
                times=times,
                score_matrix=score_matrix,
                significance_mask=significance_mask,
                selected_alpha=selected_alpha,
                alpha_selection_source=alpha_selection_source,
            )
        )

    return payloads


def _load_decoding_stats_payload(
    *,
    out_dir: Path,
    decoding_config: Mapping[str, Any],
    model_name: str,
) -> dict[str, np.ndarray]:
    output_paths = build_decoding_model_output_paths({"paths": {"out_dir": str(out_dir)}}, decoding_config, model_name=model_name)
    required_paths = (
        output_paths.tg_matrix_output_path,
        output_paths.tg_tfce_output_path,
        output_paths.tg_p_fwer_output_path,
    )
    missing_paths = [path for path in required_paths if not path.exists()]
    if missing_paths:
        missing_text = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Missing decoding stats figure inputs for model `{model_name}`: {missing_text}")
    observed_matrix = np.asarray(np.load(output_paths.tg_matrix_output_path), dtype=float)
    tfce_matrix = np.asarray(np.load(output_paths.tg_tfce_output_path), dtype=float)
    corrected_p = np.asarray(np.load(output_paths.tg_p_fwer_output_path), dtype=float)
    if observed_matrix.shape != tfce_matrix.shape or observed_matrix.shape != corrected_p.shape:
        raise ValueError(
            f"Decoding stats figure arrays must share the same shape for `{model_name}`: "
            f"{observed_matrix.shape}, {tfce_matrix.shape}, {corrected_p.shape}."
        )
    return {
        "observed_matrix": observed_matrix,
        "tfce_matrix": tfce_matrix,
        "corrected_p": corrected_p,
    }


def _plot_decoding_summary(
    payloads: Sequence[_DecodingFigurePayload],
    *,
    diag_ymin: float,
    diag_ymax: float,
    matrix_abs_limit: float,
    contour_linewidth: float,
    p_threshold: float,
    figure_width: float,
    figure_height: float,
) -> tuple[plt.Figure, list[dict[str, object]]]:
    fig, axes = plt.subplots(
        2,
        len(payloads),
        figsize=(figure_width, figure_height),
        squeeze=False,
        gridspec_kw={"height_ratios": [1.0, 4.0], "wspace": 0.16, "hspace": 0.12},
    )
    plot_metadata: list[dict[str, object]] = []
    image_artist = None

    for column_idx, payload in enumerate(payloads):
        diagonal_ax = axes[0, column_idx]
        matrix_ax = axes[1, column_idx]

        _plot_diagonal(
            ax=diagonal_ax,
            times=payload.times,
            score_matrix=payload.score_matrix,
            color=payload.color,
            title=_format_panel_title(payload),
            show_y_axis=(column_idx == 0),
            ymin=diag_ymin,
            ymax=diag_ymax,
        )
        diagonal_ax.tick_params(axis="x", which="both", labelbottom=False)

        image_artist = _plot_generalization(
            ax=matrix_ax,
            times=payload.times,
            score_matrix=payload.score_matrix,
            significance_mask=payload.significance_mask,
            show_y_axis=(column_idx == 0),
            abs_limit=matrix_abs_limit,
            contour_linewidth=contour_linewidth,
        )

        plot_metadata.append(
            {
                "model_name": payload.model_name,
                "title": payload.title,
                "selected_alpha": payload.selected_alpha,
                "alpha_selection_source": payload.alpha_selection_source,
                "has_significant_samples": bool(payload.significance_mask is not None and np.any(payload.significance_mask)),
                "p_threshold": p_threshold,
            }
        )

    assert image_artist is not None
    colorbar = fig.colorbar(image_artist, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    colorbar.ax.set_title("r", pad=6)
    return fig, plot_metadata


def _plot_decoding_results(
    *,
    payload: _DecodingFigurePayload,
    tfce_matrix: np.ndarray,
    corrected_p: np.ndarray,
    observed_abs_limit: float,
    tfce_abs_limit: float,
    p_threshold: float,
) -> plt.Figure:
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12.5, 4.2),
        squeeze=False,
        gridspec_kw={"wspace": 0.22},
    )
    observed_ax, tfce_ax, masked_ax = axes[0]
    significance_mask = np.asarray(corrected_p, dtype=float) < p_threshold
    observed_image = _plot_generalization(
        ax=observed_ax,
        times=payload.times,
        score_matrix=payload.score_matrix,
        significance_mask=significance_mask,
        show_y_axis=True,
        abs_limit=observed_abs_limit,
    )
    observed_ax.set_title(f"{payload.title}\nObserved TG")

    tfce_image = tfce_ax.imshow(
        np.asarray(tfce_matrix, dtype=float),
        origin="lower",
        cmap="magma",
        extent=[float(payload.times[0]), float(payload.times[-1]), float(payload.times[0]), float(payload.times[-1])],
        vmin=0.0,
        vmax=tfce_abs_limit,
        aspect="auto",
    )
    tfce_ax.axvline(0.0, color="k", linewidth=1.0)
    tfce_ax.axhline(0.0, color="k", linewidth=1.0)
    tfce_ax.set_xlabel("Testing time (s)")
    tfce_ax.set_yticks([])
    tfce_ax.set_title("Observed TFCE")

    masked_matrix = np.where(significance_mask, np.asarray(payload.score_matrix, dtype=float), 0.0)
    masked_image = _plot_generalization(
        ax=masked_ax,
        times=payload.times,
        score_matrix=masked_matrix,
        significance_mask=None,
        show_y_axis=False,
        abs_limit=observed_abs_limit,
    )
    masked_ax.set_title(f"Observed TG\np < {p_threshold:g}")

    score_cbar = fig.colorbar(observed_image, ax=[observed_ax, masked_ax], fraction=0.028, pad=0.03)
    score_cbar.ax.set_title("r", pad=6)
    tfce_cbar = fig.colorbar(tfce_image, ax=[tfce_ax], fraction=0.046, pad=0.04)
    tfce_cbar.ax.set_title("TFCE", pad=6)
    return fig


def _plot_diagonal(
    *,
    ax: plt.Axes,
    times: np.ndarray,
    score_matrix: np.ndarray,
    color: str,
    title: str,
    show_y_axis: bool,
    ymin: float,
    ymax: float,
) -> None:
    diagonal = np.diagonal(np.asarray(score_matrix, dtype=float))
    ax.plot(times, diagonal, color=color, linewidth=2.2)
    ax.axhline(0.0, color="0.6", linewidth=1.0)
    ax.set_ylim(ymin, ymax)
    ax.set_title(title)
    if show_y_axis:
        ax.set_ylabel("Diagonal r")
    else:
        ax.set_yticks([])


def _plot_generalization(
    *,
    ax: plt.Axes,
    times: np.ndarray,
    score_matrix: np.ndarray,
    significance_mask: np.ndarray | None,
    show_y_axis: bool,
    abs_limit: float,
    contour_linewidth: float = 0.75,
):
    image = ax.imshow(
        np.asarray(score_matrix, dtype=float),
        origin="lower",
        cmap="RdBu_r",
        extent=[float(times[0]), float(times[-1]), float(times[0]), float(times[-1])],
        vmin=-abs_limit,
        vmax=abs_limit,
        aspect="equal",
    )
    ax.set_box_aspect(1)
    ax.axvline(0.0, color="k", linewidth=1.0)
    ax.axhline(0.0, color="k", linewidth=1.0)
    ax.set_xlabel("Testing time (s)")
    if show_y_axis:
        ax.set_ylabel("Training time (s)")
    else:
        ax.set_yticks([])

    if significance_mask is not None and np.any(significance_mask):
        big_mask = np.kron(significance_mask.astype(float), np.ones((10, 10)))
        ax.contour(
            big_mask,
            levels=[0.5],
            colors=["k"],
            linewidths=[contour_linewidth],
            origin="lower",
            extent=[float(times[0]), float(times[-1]), float(times[0]), float(times[-1])],
            corner_mask=False,
            antialiased=False,
            zorder=10,
        )

    return image


def _infer_matrix_limit(payloads: Sequence[_DecodingFigurePayload], *, percentile: float = 100.0) -> float:
    finite_chunks: list[np.ndarray] = []
    for payload in payloads:
        finite_values = np.abs(np.asarray(payload.score_matrix, dtype=float))
        if finite_values.size == 0:
            continue
        finite_values = finite_values[np.isfinite(finite_values)]
        if finite_values.size:
            finite_chunks.append(finite_values)
    if not finite_chunks:
        return 0.05
    clipped_percentile = float(np.clip(percentile, 0.0, 100.0))
    combined = np.concatenate(finite_chunks)
    return max(float(np.nanpercentile(combined, clipped_percentile)), 0.05)


def _infer_abs_limit(matrix: np.ndarray, *, fallback: float) -> float:
    finite_values = np.abs(np.asarray(matrix, dtype=float))
    if finite_values.size == 0 or not np.isfinite(finite_values).any():
        return fallback
    return max(float(np.nanmax(finite_values)), fallback)


def _infer_diagonal_limits(
    payloads: Sequence[_DecodingFigurePayload],
    *,
    pad_fraction: float = 0.025,
    min_half_range: float = 0.015,
) -> tuple[float, float]:
    diagonal_chunks: list[np.ndarray] = []
    for payload in payloads:
        diagonal = np.diagonal(np.asarray(payload.score_matrix, dtype=float))
        diagonal = diagonal[np.isfinite(diagonal)]
        if diagonal.size:
            diagonal_chunks.append(diagonal)

    if not diagonal_chunks:
        return -0.1, 0.5

    combined = np.concatenate(diagonal_chunks)
    lower = float(np.nanmin(combined))
    upper = float(np.nanmax(combined))
    span = upper - lower
    pad = max(float(min_half_range) * 0.25, span * float(pad_fraction))

    if span <= 0.0:
        half_range = float(min_half_range)
        midpoint = lower
        return midpoint - half_range, midpoint + half_range

    return lower - pad, upper + pad


def _humanize_model_name(model_name: str) -> str:
    return str(model_name).replace("_", " ").strip().title()


def _format_panel_title(payload: _DecodingFigurePayload) -> str:
    if payload.selected_alpha is None:
        return payload.title
    return f"{payload.title}\nalpha={payload.selected_alpha:g}"


def _resolve_ridge_alphas(estimator_config: Mapping[str, Any]) -> np.ndarray:
    alpha_grid = estimator_config.get("alpha_grid")
    if alpha_grid is not None:
        resolved_grid = np.asarray(alpha_grid, dtype=float)
        if resolved_grid.ndim != 1 or int(resolved_grid.size) < 1:
            raise ValueError("`decoding.estimator.alpha_grid` must resolve to a non-empty 1D array.")
        return resolved_grid
    alphas_cfg = estimator_config.get("alphas")
    if alphas_cfg is not None:
        start = float(alphas_cfg["start"])
        stop = float(alphas_cfg["stop"])
        n_alphas = int(alphas_cfg["n_alphas"])
        if n_alphas < 1:
            raise ValueError("`decoding.estimator.alphas.n_alphas` must be >= 1.")
        return np.logspace(start, stop, n_alphas, dtype=float)
    return np.asarray([float(estimator_config.get("alpha", 1.0))], dtype=float)
