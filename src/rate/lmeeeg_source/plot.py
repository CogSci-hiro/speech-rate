"""QC plotting for the experimental source-space lmeEEG branch."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from statistics import NormalDist
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Colormap

from rate.lmeeeg_source.config import DEFAULT_LMEEEG_SOURCE_SECTION, load_lmeeeg_source_settings
from rate.lmeeeg_source.pipeline import build_source_group_paths
from rate.paths import ensure_parent_dir, resolve_out_dir

LOGGER = logging.getLogger(__name__)


def build_lmeeeg_source_qc_figure(
    config: Mapping[str, Any],
    *,
    output_path: Path | None = None,
    config_section: str = DEFAULT_LMEEEG_SOURCE_SECTION,
) -> dict[str, Any]:
    """Render time-sliced source-space QC figures for self_rate and other_rate."""
    settings = load_lmeeeg_source_settings(config, config_section=config_section)
    if not settings.enabled:
        raise ValueError(f"`{config_section}.enabled` is false. Refusing to run the experimental source-space pipeline.")

    out_dir = resolve_out_dir(config)
    manifest_path = output_path or out_dir / "figures" / settings.outputs.namespace / "figure_manifest.json"
    ensure_parent_dir(manifest_path)

    predictors = tuple(settings.plotting.predictors)
    source_maps: dict[str, Any] = {}
    for predictor_name in predictors:
        group_paths = build_source_group_paths(
            config,
            predictor_name=predictor_name,
            measure_name=settings.plotting.measure,
            config_section=config_section,
        )
        summary_json = json.loads(group_paths.summary_json_path.read_text(encoding="utf-8"))
        predictor_dir = group_paths.summary_json_path.parent
        source_maps[predictor_name] = {
            "summary": summary_json,
            "beta_stc_stem": Path(str(summary_json.get("beta_full_stc_stem") or (predictor_dir / "beta_full"))),
            "t_value_stc_stem": Path(str(summary_json.get("t_value_full_stc_stem") or (predictor_dir / "t_value_full"))),
            "correction": None if summary_json.get("correction") is None else str(summary_json.get("correction")),
            "corrected_p_path": (
                Path(str(summary_json["corrected_p_values"]))
                if summary_json.get("corrected_p_values")
                else None
            ),
            "cluster_p_path": (
                Path(str(summary_json["cluster_p_values"]))
                if summary_json.get("cluster_p_values")
                else None
            ),
            "cluster_masks_path": (
                Path(str(summary_json["cluster_masks"]))
                if summary_json.get("cluster_masks")
                else None
            ),
        }

    figure_stem = out_dir / "figures" / settings.outputs.qc_subdir / "self_other_lateral"
    written_paths = _render_source_qc_panels(
        source_maps=source_maps,
        subjects_dir=settings.source_space.subjects_dir,
        fsaverage_subject=settings.source_space.fsaverage_subject,
        hemi=settings.plotting.hemi,
        view=settings.plotting.view,
        colormap=settings.plotting.colormap,
        mask_by_significance=settings.plotting.mask_by_significance,
        significance_alpha=settings.plotting.significance_alpha,
        time_window=settings.plotting.time_window,
        time_step_ms=settings.plotting.time_step_ms,
        output_stem=figure_stem,
        formats=settings.plotting.formats,
    )
    manifest = {
        "status": "ok",
        "plot_count": 1,
        "plots": [
            {
                "name": "self_other_lateral",
                "predictors": list(predictors),
                "measure": settings.plotting.measure,
                "mask_by_significance": settings.plotting.mask_by_significance,
                "significance_alpha": settings.plotting.significance_alpha,
                "time_step_ms": settings.plotting.time_step_ms,
                "files": [str(path) for path in written_paths],
            }
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _render_source_qc_panels(
    *,
    source_maps: Mapping[str, Mapping[str, Any]],
    subjects_dir: Path,
    fsaverage_subject: str,
    hemi: str,
    view: str,
    colormap: str,
    mask_by_significance: bool,
    significance_alpha: float,
    time_window: tuple[float, float] | None,
    time_step_ms: float,
    output_stem: Path,
    formats: tuple[str, ...],
) -> list[Path]:
    ensure_parent_dir(output_stem.with_suffix(".png"))
    _configure_mne_runtime()
    panel_specs = _build_panel_specs(
        source_maps=source_maps,
        time_window=time_window,
        time_step_ms=time_step_ms,
        mask_by_significance=mask_by_significance,
        significance_alpha=significance_alpha,
    )
    rendered_panels: list[np.ndarray] = []
    render_warning: str | None = None
    try:
        resolved_colormap = _resolve_colormap(colormap)
        clim = _compute_shared_clim(panel_specs)
        for panel_spec in panel_specs:
            rendered_panels.append(
                _render_lateral_brain_panel(
                    stc=panel_spec["stc"],
                    subject=fsaverage_subject,
                    subjects_dir=subjects_dir,
                    hemi=hemi,
                    view=view,
                    colormap=resolved_colormap,
                    initial_time=float(panel_spec["time_s"]),
                    clim=clim,
                )
            )
    except Exception as exc:
        render_warning = str(exc)
        LOGGER.warning(
            "Source-space brain rendering failed; writing a documented fallback QC figure instead: %s",
            exc,
        )

    if render_warning is not None:
        rendered_panels = []
        for panel_spec in panel_specs:
            rendered_panels.append(
                _render_fallback_panel(
                    stc=panel_spec["stc"],
                    predictor_name=str(panel_spec["predictor_name"]),
                    time_label=str(panel_spec["time_label"]),
                    time_index=int(panel_spec["time_index"]),
                    colormap=resolved_colormap if "resolved_colormap" in locals() else _resolve_colormap(colormap),
                    warning_text=render_warning,
                )
            )

    panel_count = len(panel_specs)
    predictor_count = len(source_maps)
    timepoint_count = max(1, panel_count // max(1, predictor_count))
    figure, axes = plt.subplots(
        predictor_count,
        timepoint_count,
        figsize=(3.2 * timepoint_count, 3.3 * predictor_count),
        squeeze=False,
    )
    for axis, image, panel_spec in zip(axes.flat, rendered_panels, panel_specs, strict=True):
        axis.imshow(image)
        axis.set_title(f"{panel_spec['predictor_name']}\n{panel_spec['time_label']}", fontsize=10)
        axis.axis("off")
    figure.suptitle(_build_qc_title(mask_by_significance=mask_by_significance, significance_alpha=significance_alpha), fontsize=14)
    figure.tight_layout()

    written_paths: list[Path] = []
    for extension in formats:
        path = output_stem.with_suffix(f".{extension}")
        figure.savefig(path, dpi=200, bbox_inches="tight")
        written_paths.append(path)
    plt.close(figure)
    return written_paths


def _render_lateral_brain_panel(
    *,
    stc,
    subject: str,
    subjects_dir: Path,
    hemi: str,
    view: str,
    colormap: Colormap | str,
    initial_time: float,
    clim: dict[str, Any],
) -> np.ndarray:
    brain = stc.plot(
        subject=subject,
        subjects_dir=str(subjects_dir),
        hemi=hemi,
        views=view,
        colorbar=True,
        time_viewer=False,
        show_traces=False,
        size=(900, 350),
        background="white",
        colormap=colormap,
        transparent=True,
        initial_time=initial_time,
        clim=clim,
    )
    try:
        return np.asarray(brain.screenshot(mode="rgb"), dtype=np.uint8)
    finally:
        brain.close()


def _render_fallback_panel(
    *,
    stc,
    predictor_name: str,
    time_label: str,
    time_index: int,
    colormap: Colormap | str,
    warning_text: str,
) -> np.ndarray:
    data = np.asarray(stc.data[:, time_index], dtype=float)
    figure, axes = plt.subplots(2, 1, figsize=(6, 4), height_ratios=[3, 1.6])
    image = data[:, np.newaxis]
    vmax = float(np.nanmax(np.abs(image))) if np.isfinite(image).any() else 1.0
    vmax = vmax if vmax > 0 else 1.0
    axes[0].imshow(image, aspect="auto", cmap=colormap, vmin=-vmax, vmax=vmax)
    axes[0].set_title(f"{predictor_name} {time_label} (fallback)")
    axes[0].set_ylabel("vertex")
    axes[0].set_xticks([])
    axes[1].axis("off")
    axes[1].text(
        0.0,
        1.0,
        "3D cortical rendering was unavailable in this environment.\n"
        f"Reason: {warning_text}\n"
        "The saved lmeEEG source maps are still available for later interactive inspection.",
        va="top",
        ha="left",
        fontsize=9,
        wrap=True,
    )
    figure.tight_layout()
    figure.canvas.draw()
    width, height = figure.canvas.get_width_height()
    buffer = np.frombuffer(figure.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)[..., :3]
    plt.close(figure)
    return buffer.copy()


def _build_panel_specs(
    *,
    source_maps: Mapping[str, Mapping[str, Any]],
    time_window: tuple[float, float] | None,
    time_step_ms: float,
    mask_by_significance: bool,
    significance_alpha: float,
) -> list[dict[str, Any]]:
    _configure_mne_runtime()
    import mne

    panel_specs: list[dict[str, Any]] = []
    t_threshold = float(NormalDist().inv_cdf(1.0 - (significance_alpha / 2.0)))
    for predictor_name in ("self_rate", "other_rate"):
        beta_stc = mne.read_source_estimate(str(source_maps[predictor_name]["beta_stc_stem"]))
        t_value_stc = mne.read_source_estimate(str(source_maps[predictor_name]["t_value_stc_stem"]))
        correction = None if source_maps[predictor_name].get("correction") is None else str(source_maps[predictor_name]["correction"]).lower()
        corrected_p_path = source_maps[predictor_name].get("corrected_p_path")
        cluster_p_path = source_maps[predictor_name].get("cluster_p_path")
        cluster_masks_path = source_maps[predictor_name].get("cluster_masks_path")
        corrected_p_values = None if corrected_p_path is None else np.asarray(np.load(corrected_p_path), dtype=float)
        cluster_p_values = None if cluster_p_path is None else np.asarray(np.load(cluster_p_path), dtype=float)
        cluster_masks = None if cluster_masks_path is None else np.asarray(np.load(cluster_masks_path), dtype=bool)
        if beta_stc.data.shape != t_value_stc.data.shape:
            raise ValueError(f"Beta and t-value source maps do not align for {predictor_name}.")
        if corrected_p_values is not None and corrected_p_values.shape != beta_stc.data.shape:
            raise ValueError(
                f"Corrected p-values shape does not align with source maps for {predictor_name}: "
                f"{corrected_p_values.shape} != {beta_stc.data.shape}"
            )
        if cluster_masks is not None and cluster_masks.ndim != 3:
            raise ValueError(f"Cluster masks must have shape (n_clusters, n_vertices, n_times) for {predictor_name}.")
        if cluster_masks is not None and cluster_masks.shape[1:] != beta_stc.data.shape:
            raise ValueError(
                f"Cluster masks shape does not align with source maps for {predictor_name}: "
                f"{cluster_masks.shape[1:]} != {beta_stc.data.shape}"
            )
        if cluster_p_values is not None and cluster_masks is not None and int(cluster_p_values.shape[0]) != int(cluster_masks.shape[0]):
            raise ValueError(
                f"Cluster p-values and masks do not align for {predictor_name}: "
                f"{cluster_p_values.shape[0]} != {cluster_masks.shape[0]}"
            )
        time_indices = _select_time_indices(beta_stc.times, time_window=time_window, time_step_ms=time_step_ms)
        for time_index in time_indices:
            significance_mask = _resolve_significance_mask(
                correction=correction,
                corrected_p_values=corrected_p_values,
                cluster_p_values=cluster_p_values,
                cluster_masks=cluster_masks,
                t_value_data=np.asarray(t_value_stc.data, dtype=float),
                time_index=int(time_index),
                significance_alpha=significance_alpha,
                t_threshold=t_threshold,
                predictor_name=predictor_name,
            )
            panel_data = np.zeros_like(beta_stc.data)
            source_values = np.asarray(beta_stc.data[:, time_index], dtype=float)
            if mask_by_significance:
                panel_data[:, time_index] = np.where(significance_mask, source_values, 0.0)
            else:
                panel_data[:, time_index] = source_values
            panel_stc = beta_stc.copy()
            panel_stc._data = panel_data
            time_ms = int(round(float(beta_stc.times[time_index]) * 1000.0))
            panel_specs.append(
                {
                    "predictor_name": predictor_name,
                    "time_index": int(time_index),
                    "time_s": float(beta_stc.times[time_index]),
                    "time_label": f"{time_ms} ms",
                    "stc": panel_stc,
                    "significance_mask": significance_mask,
                }
            )
    return panel_specs


def _resolve_significance_mask(
    *,
    correction: str | None,
    corrected_p_values: np.ndarray | None,
    cluster_p_values: np.ndarray | None,
    cluster_masks: np.ndarray | None,
    t_value_data: np.ndarray,
    time_index: int,
    significance_alpha: float,
    t_threshold: float,
    predictor_name: str,
) -> np.ndarray:
    if correction == "cluster":
        if cluster_p_values is not None and cluster_masks is not None:
            significant_clusters = np.asarray(cluster_p_values, dtype=float) < significance_alpha
            if not np.any(significant_clusters):
                return np.zeros(cluster_masks.shape[1], dtype=bool)
            return np.any(np.asarray(cluster_masks[significant_clusters, :, time_index], dtype=bool), axis=0)
        if corrected_p_values is not None:
            LOGGER.warning(
                "Cluster correction was recorded for %s, but cluster masks were unavailable; falling back to corrected_p_values.",
                predictor_name,
            )
            return np.asarray(corrected_p_values[:, time_index], dtype=float) < significance_alpha
        raise ValueError(
            f"Cluster correction was recorded for {predictor_name}, but no cluster masks were available for plotting."
        )
    if corrected_p_values is not None:
        return np.asarray(corrected_p_values[:, time_index], dtype=float) < significance_alpha
    return np.abs(np.asarray(t_value_data[:, time_index], dtype=float)) >= t_threshold


def _build_qc_title(*, mask_by_significance: bool, significance_alpha: float) -> str:
    if mask_by_significance:
        return f"Experimental source-space lmeEEG QC (lateral views, masked at p < {significance_alpha:g})"
    return "Experimental source-space lmeEEG QC (lateral views, unmasked effect maps)"


def _select_time_indices(
    times: np.ndarray,
    *,
    time_window: tuple[float, float] | None,
    time_step_ms: float,
) -> list[int]:
    if time_step_ms <= 0:
        raise ValueError("`time_step_ms` must be positive.")
    times = np.asarray(times, dtype=float)
    start = float(times[0]) if time_window is None else float(time_window[0])
    stop = float(times[-1]) if time_window is None else float(time_window[1])
    step_s = float(time_step_ms) / 1000.0
    target_times = np.arange(start, stop + (step_s * 0.5), step_s, dtype=float)
    indices = sorted({int(np.argmin(np.abs(times - target_time))) for target_time in target_times})
    return indices


def _compute_shared_clim(panel_specs: list[dict[str, Any]]) -> dict[str, Any]:
    significant_values: list[np.ndarray] = []
    for panel_spec in panel_specs:
        time_index = int(panel_spec["time_index"])
        current = np.asarray(panel_spec["stc"].data[:, time_index], dtype=float)
        nonzero = np.abs(current[np.nonzero(current)])
        if nonzero.size:
            significant_values.append(nonzero)
    if significant_values:
        stacked = np.concatenate(significant_values)
    else:
        stacked = np.concatenate([np.abs(np.asarray(spec["stc"].data, dtype=float)).ravel() for spec in panel_specs])
    vmax = float(np.nanpercentile(stacked, 98.0)) if stacked.size else 1.0
    vmax = max(vmax, 1e-12)
    return {"kind": "value", "pos_lims": [0.35 * vmax, 0.7 * vmax, vmax]}


def _resolve_colormap(colormap: str) -> Colormap | str:
    cmap_key = str(colormap).strip().lower().replace("-", "").replace("_", "")
    if cmap_key == "icefire":
        try:
            import seaborn as sns

            return sns.color_palette("icefire", as_cmap=True)
        except Exception:
            return "coolwarm"
    return colormap


def _configure_mne_runtime() -> None:
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
