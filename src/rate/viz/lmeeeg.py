"""Visualization helpers for pooled lmeEEG outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from rate.legacy.viz.glm import _sanitize_token, plot_joint_model_weights
from rate.lmeeeg.nested import resolve_lmeeeg_nested_settings


def _has_lmeeeg_fit_outputs(model_dir: Path) -> bool:
    required_paths = (
        model_dir / "t_values.npy",
        model_dir / "times.npy",
        model_dir / "channel_names.json",
        model_dir / "column_names.json",
    )
    return all(path.exists() for path in required_paths)


def _iter_lmeeeg_model_dirs(lmeeeg_root: Path) -> list[Path]:
    """Return pooled-fit directories, preferring shallow canonical locations."""
    if not lmeeeg_root.exists():
        return []

    candidates_by_name: dict[str, tuple[int, Path]] = {}
    for summary_path in sorted(lmeeeg_root.glob("**/summary.json")):
        model_dir = summary_path.parent
        if not _has_lmeeeg_fit_outputs(model_dir):
            continue
        model_name = model_dir.name
        depth = len(model_dir.relative_to(lmeeeg_root).parts)
        current = candidates_by_name.get(model_name)
        if current is None or depth < current[0]:
            candidates_by_name[model_name] = (depth, model_dir)
    return [candidate[1] for _, candidate in sorted(candidates_by_name.items())]


def build_lmeeeg_qc_manifest(
    *,
    out_dir: Path,
    lmeeeg_root: Path,
    manifest_path: Path,
    figure_subdir: str = "lmeeeg",
    formats: Sequence[str] = ("png", "pdf"),
    dpi: int = 300,
    line_width: float = 2.5,
) -> dict:
    """Render joint plots for all pooled lmeEEG effect t-maps."""
    plots: list[dict[str, object]] = []
    for model_dir in _iter_lmeeeg_model_dirs(lmeeeg_root):
        t_values_path = model_dir / "t_values.npy"
        times_path = model_dir / "times.npy"
        channel_names_path = model_dir / "channel_names.json"
        column_names_path = model_dir / "column_names.json"

        t_values = np.load(t_values_path)
        times = np.load(times_path)
        channel_names = json.loads(channel_names_path.read_text(encoding="utf-8"))
        column_names = json.loads(column_names_path.read_text(encoding="utf-8"))
        if t_values.ndim != 3:
            continue

        model_name = model_dir.name
        for kernel_index, kernel_name in enumerate(column_names):
            kernel_map = np.asarray(t_values[kernel_index], dtype=float)
            if kernel_map.ndim != 2 or not np.isfinite(kernel_map).any():
                continue
            output_stem = out_dir / "figures" / figure_subdir / "t_values" / model_name / _sanitize_token(kernel_name)
            written_paths = plot_joint_model_weights(
                kernel_map,
                times=times,
                channel_names=channel_names,
                output_stem=output_stem,
                title=f"lmeEEG t-values | {model_name} | {kernel_name}",
                formats=formats,
                dpi=dpi,
                line_width=line_width,
            )
            plots.append(
                {
                    "model_name": model_name,
                    "kernel": kernel_name,
                    "files": [str(path) for path in written_paths],
                }
            )

    manifest = {
        "status": "ok",
        "plot_count": len(plots),
        "plots": plots,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def build_lmeeeg_qc_manifest_from_model_payloads(
    *,
    out_dir: Path,
    model_payloads: dict[str, dict[str, object]],
    manifest_path: Path,
    figure_subdir: str = "lmeeeg",
    plot_metadata: dict[str, object] | None = None,
    significance_masks: dict[str, dict[str, np.ndarray]] | None = None,
    formats: Sequence[str] = ("png", "pdf"),
    dpi: int = 300,
    line_width: float = 2.5,
) -> dict:
    """Render beta and t-value joint plots from pooled lmeEEG model payloads."""
    plots: list[dict[str, object]] = []
    significance_masks = significance_masks or {}
    plot_metadata = dict(plot_metadata or {})
    for model_name, payload in sorted(model_payloads.items()):
        channel_names = list(payload["channel_names"])
        times = np.asarray(payload["times"], dtype=float)
        column_names = list(payload["column_names"])
        measures = (
            ("beta", np.asarray(payload["betas"], dtype=float)),
            ("t_value", np.asarray(payload["t_values"], dtype=float)),
        )
        for measure_name, measure_stack in measures:
            if measure_stack.ndim != 3:
                continue
            measure_dirname = "betas" if measure_name == "beta" else "t_values"
            for column_index, column_name in enumerate(column_names):
                column_map = np.asarray(measure_stack[column_index], dtype=float)
                if column_map.ndim != 2 or not np.isfinite(column_map).any():
                    continue
                significance_mask = significance_masks.get(model_name, {}).get(column_name)
                output_stem = out_dir / "figures" / figure_subdir / measure_dirname / model_name / _sanitize_token(column_name)
                written_paths = plot_joint_model_weights(
                    column_map,
                    times=times,
                    channel_names=channel_names,
                    output_stem=output_stem,
                    title=f"lmeEEG {measure_name} | {model_name} | {column_name}",
                    formats=formats,
                    dpi=dpi,
                    line_width=line_width,
                    significance_mask=significance_mask,
                )
                plot_entry = {
                    "model_name": model_name,
                    "kernel": column_name,
                    "measure": measure_name,
                    "has_significance_overlay": significance_mask is not None and bool(np.any(significance_mask)),
                    "files": [str(path) for path in written_paths],
                }
                plot_entry.update(plot_metadata)
                plots.append(plot_entry)

    manifest = {
        "status": "ok",
        "plot_count": len(plots),
        "plots": plots,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def build_lmeeeg_inference_manifest(
    *,
    out_dir: Path,
    stats_root: Path,
    lmeeeg_root: Path,
    manifest_path: Path,
    figure_subdir: str = "lmeeeg_inference",
    plot_metadata: dict[str, object] | None = None,
    formats: Sequence[str] = ("png", "pdf"),
    dpi: int = 300,
    line_width: float = 2.5,
) -> dict:
    """Render joint plots for lmeEEG inference outputs."""
    plots: list[dict[str, object]] = []
    plot_metadata = dict(plot_metadata or {})
    for effect_dir in sorted(path for path in stats_root.glob("*/*") if path.is_dir()):
        summary_path = effect_dir / "summary.json"
        if not summary_path.exists():
            continue

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        model_name = effect_dir.parent.name
        effect_name = effect_dir.name
        fit_model_dir = lmeeeg_root / model_name
        times_path = fit_model_dir / "times.npy"
        channel_names_path = fit_model_dir / "channel_names.json"
        observed_path = Path(str(summary["observed_statistic"]))
        corrected_p_path = Path(str(summary["corrected_p_values"]))
        observed_label = _format_statistic_label(summary.get("observed_statistic_kind"))
        if not all(path.exists() for path in (times_path, channel_names_path, observed_path, corrected_p_path)):
            continue

        times = np.load(times_path)
        channel_names = json.loads(channel_names_path.read_text(encoding="utf-8"))
        observed_map = np.load(observed_path)
        corrected_p_map = np.load(corrected_p_path)
        corrected_p_array = np.asarray(corrected_p_map, dtype=float)
        significance_mask = corrected_p_array < 0.05
        significance_map = np.where(significance_mask, observed_map, 0.0)

        observed_stem = out_dir / "figures" / figure_subdir / model_name / f"{_sanitize_token(effect_name)}_observed"
        observed_paths = plot_joint_model_weights(
            np.asarray(observed_map, dtype=float),
            times=times,
            channel_names=channel_names,
            output_stem=observed_stem,
            title=f"lmeEEG {observed_label} | {model_name} | {effect_name}",
            formats=formats,
            dpi=dpi,
            line_width=line_width,
            significance_mask=significance_mask,
        )
        corrected_p_stem = out_dir / "figures" / figure_subdir / model_name / f"{_sanitize_token(effect_name)}_corrected_p"
        corrected_p_paths = plot_joint_model_weights(
            1.0 - corrected_p_array,
            times=times,
            channel_names=channel_names,
            output_stem=corrected_p_stem,
            title=f"lmeEEG 1-corrected p | {model_name} | {effect_name}",
            formats=formats,
            dpi=dpi,
            line_width=line_width,
        )
        pvalue_stem = out_dir / "figures" / figure_subdir / model_name / f"{_sanitize_token(effect_name)}_p005_masked"
        pvalue_paths = plot_joint_model_weights(
            significance_map,
            times=times,
            channel_names=channel_names,
            output_stem=pvalue_stem,
            title=f"lmeEEG corrected p<0.05 | {model_name} | {effect_name}",
            formats=formats,
            dpi=dpi,
            line_width=line_width,
        )
        plot_entry = {
            "model_name": model_name,
            "effect": effect_name,
            "correction": summary.get("correction"),
            "observed_files": [str(path) for path in observed_paths],
            "corrected_p_files": [str(path) for path in corrected_p_paths],
            "significant_files": [str(path) for path in pvalue_paths],
            "has_significant_samples": bool(np.any(significance_mask)),
            "min_corrected_p": summary.get("min_corrected_p"),
            "n_significant_p_lt_0_05": summary.get("n_significant_p_lt_0_05"),
        }
        plot_entry.update(plot_metadata)
        plots.append(plot_entry)

    manifest = {
        "status": "ok",
        "plot_count": len(plots),
        "plots": plots,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def build_lmeeeg_nested_supplement_manifest(
    *,
    config: Mapping[str, Any],
    out_dir: Path,
    manifest_path: Path,
    figure_config: Mapping[str, Any] | None = None,
    config_section: str = "lmeeeg_nested",
    figure_subdir: str = "supplement",
    formats: Sequence[str] = ("png", "pdf"),
    dpi: int = 300,
    line_width: float = 2.5,
) -> dict:
    """Render supplement-ready joint plots for configured nested lmeEEG analyses."""
    settings = resolve_lmeeeg_nested_settings(config, config_section=config_section)
    section_cfg = dict(config.get(config_section) or {})
    figure_config = dict(figure_config or {})
    stats_root = out_dir / "stats" / settings.output_dirname
    lmeeeg_root = out_dir / settings.source_section
    significance_threshold = float(figure_config.get("significance_threshold", 0.05))
    configured_figures = list((figure_config.get("figures") or []))

    plots: list[dict[str, object]] = []
    default_analyses = list(settings.analyses)
    if configured_figures:
        figure_specs = configured_figures
    else:
        figure_specs = [
            {
                "figure_label": f"S{index}",
                "analysis_name": analysis.name,
                "title": f"Nested lmeEEG | {analysis.name}",
            }
            for index, analysis in enumerate(default_analyses, start=1)
        ]

    analysis_lookup = {analysis.name: analysis for analysis in default_analyses}
    for figure_spec in figure_specs:
        analysis_name = str(figure_spec["analysis_name"])
        figure_label = str(figure_spec["figure_label"])
        title = str(figure_spec.get("title", f"Nested lmeEEG | {analysis_name}"))
        figure_threshold = float(
            figure_spec.get(
                "significance_threshold",
                figure_spec.get("alpha_threshold", significance_threshold),
            )
        )
        joint_times = figure_spec.get("topomap_times_s", "peaks")
        analysis = analysis_lookup.get(analysis_name)
        if analysis is None:
            raise ValueError(f"Unknown nested analysis `{analysis_name}` in viz.figures.lmeeeg_nested_supplement.")

        effect_dir = stats_root / _sanitize_token(analysis_name)
        summary_path = effect_dir / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"Nested lmeEEG summary not found for analysis `{analysis_name}`: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

        full_model = str(summary.get("full_model", analysis.full_model))
        fit_model_dir = lmeeeg_root / _sanitize_token(full_model)
        times_path = fit_model_dir / "times.npy"
        channel_names_path = fit_model_dir / "channel_names.json"
        t_values_path = fit_model_dir / "t_values.npy"
        column_names_path = fit_model_dir / "column_names.json"
        observed_path = Path(str(summary["observed_statistic"]))
        corrected_p_path = Path(str(summary["corrected_p_values"]))
        if not all(path.exists() for path in (times_path, channel_names_path, t_values_path, column_names_path, observed_path, corrected_p_path)):
            missing = [
                str(path)
                for path in (times_path, channel_names_path, t_values_path, column_names_path, observed_path, corrected_p_path)
                if not path.exists()
            ]
            raise FileNotFoundError(f"Missing nested supplement figure inputs for `{analysis_name}`: {', '.join(missing)}")

        times = np.asarray(np.load(times_path), dtype=float)
        channel_names = json.loads(channel_names_path.read_text(encoding="utf-8"))
        column_names = json.loads(column_names_path.read_text(encoding="utf-8"))
        t_values = np.asarray(np.load(t_values_path), dtype=float)
        tested_predictor = str(summary.get("tested_predictor") or analysis.tested_predictor or "")
        effect_name = _resolve_saved_effect_name(column_names, predictor_name=tested_predictor)
        effect_index = int(column_names.index(effect_name))
        plotted_map = np.asarray(t_values[effect_index], dtype=float)
        corrected_p_map = np.asarray(np.load(corrected_p_path), dtype=float)
        significance_mask = corrected_p_map < figure_threshold

        output_stem = out_dir / "figures" / figure_subdir / f"{_sanitize_token(figure_label)}_{_sanitize_token(analysis_name)}"
        written_paths = plot_joint_model_weights(
            plotted_map,
            times=times,
            channel_names=channel_names,
            output_stem=output_stem,
            title=title,
            formats=formats,
            dpi=dpi,
            line_width=line_width,
            joint_times=joint_times,
            significance_mask=significance_mask,
        )
        plots.append(
            {
                "plot_name": "lmeeeg_nested_supplement",
                "figure_label": figure_label,
                "analysis_name": analysis_name,
                "tested_predictor": tested_predictor,
                "full_model": full_model,
                "effect_name": effect_name,
                "value_kind": "t_value",
                "correction": summary.get("correction"),
                "significance_threshold": figure_threshold,
                "joint_times_s": joint_times,
                "has_significant_samples": bool(np.any(significance_mask)),
                "n_significant_p_lt_threshold": int(np.count_nonzero(significance_mask)),
                "files": [str(path) for path in written_paths],
            }
        )

    manifest = {"status": "ok", "plot_count": len(plots), "plots": plots}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _resolve_saved_effect_name(column_names: list[str], *, predictor_name: str) -> str:
    for column_name in column_names:
        if str(column_name) == predictor_name:
            return str(column_name)
    raise ValueError(
        f"Predictor `{predictor_name}` did not resolve to a saved fixed-effect column. Available columns: {column_names}"
    )


def build_lmeeeg_qc_manifest_from_stats(
    *,
    out_dir: Path,
    stats_root: Path,
    manifest_path: Path,
    model_axes: dict[str, dict[str, object]],
    figure_subdir: str = "lmeeeg",
    plot_metadata: dict[str, object] | None = None,
    formats: Sequence[str] = ("png", "pdf"),
    dpi: int = 300,
    line_width: float = 2.5,
) -> dict:
    """Render one observed-statistic joint plot per inferred effect with significance overlays."""
    plots: list[dict[str, object]] = []
    plot_metadata = dict(plot_metadata or {})
    for effect_dir in sorted(path for path in stats_root.glob("*/*") if path.is_dir()):
        summary_path = effect_dir / "summary.json"
        if not summary_path.exists():
            continue

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        model_name = effect_dir.parent.name
        effect_name = effect_dir.name
        axes = model_axes.get(model_name)
        if not axes:
            continue

        observed_path = Path(str(summary["observed_statistic"]))
        corrected_p_path = Path(str(summary["corrected_p_values"]))
        observed_label = _format_statistic_label(summary.get("observed_statistic_kind"))
        if not observed_path.exists() or not corrected_p_path.exists():
            continue

        observed_map = np.asarray(np.load(observed_path), dtype=float)
        corrected_p_map = np.asarray(np.load(corrected_p_path), dtype=float)
        channel_names = list(axes["channel_names"])
        times = np.asarray(axes["times"], dtype=float)
        if observed_map.ndim != 2 or observed_map.shape != (len(channel_names), times.shape[0]):
            continue
        if corrected_p_map.shape != observed_map.shape:
            continue
        significance_mask = corrected_p_map < 0.05

        output_stem = out_dir / "figures" / figure_subdir / "statistics" / model_name / _sanitize_token(effect_name)
        written_paths = plot_joint_model_weights(
            observed_map,
            times=times,
            channel_names=channel_names,
            output_stem=output_stem,
            title=f"lmeEEG {observed_label} | {model_name} | {effect_name}",
            formats=formats,
            dpi=dpi,
            line_width=line_width,
            significance_mask=significance_mask,
        )
        plot_entry = {
            "model_name": model_name,
            "kernel": effect_name,
            "source": "stats_fallback",
            "has_significant_samples": bool(np.any(significance_mask)),
            "min_corrected_p": summary.get("min_corrected_p"),
            "n_significant_p_lt_0_05": summary.get("n_significant_p_lt_0_05"),
            "files": [str(path) for path in written_paths],
        }
        plot_entry.update(plot_metadata)
        plots.append(plot_entry)

    manifest = {
        "status": "ok",
        "plot_count": len(plots),
        "plots": plots,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _format_statistic_label(statistic_kind: object) -> str:
    value = str(statistic_kind or "observed_statistic").strip().lower()
    if value == "t_value":
        return "t value"
    return value.replace("_", " ")
