"""`figures-lmeeeg` CLI subcommand."""

from __future__ import annotations

import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Mapping

import json
import numpy as np
import yaml

from rate.cli.commands._workflow_common import add_common_config_arguments
from rate.design import list_model_names
from rate.lmeeeg.io import build_lmeeeg_model_output_paths
from rate.lmeeeg.mass_model import fit_lmeeeg_mass_model
from rate.lmeeeg.pooling import load_pooled_lmeeeg_trial_data, select_lmeeeg_rows
from rate.paths import resolve_out_dir
from rate.viz.lmeeeg import (
    build_lmeeeg_qc_manifest,
    build_lmeeeg_qc_manifest_from_model_payloads,
    build_lmeeeg_qc_manifest_from_stats,
)

LOGGER = logging.getLogger(__name__)
REQUIRES_CONFIG = True


def add_subparser(subparsers) -> ArgumentParser:
    parser = subparsers.add_parser("figures-lmeeeg", help="Render pooled lmeEEG QC joint plots.")
    add_common_config_arguments(parser)
    parser.add_argument("--viz-config", default="config/viz.yaml", help="Path to the figure config file.")
    parser.add_argument("--output", help="Optional explicit output path.")
    return parser


def run(args: Namespace, config: Mapping[str, Any] | None = None) -> int:
    assert config is not None
    out_dir = resolve_out_dir(config)
    output_path = Path(args.output) if args.output else out_dir / "figures" / "lmeeeg" / "figure_manifest.json"
    viz_cfg = _load_viz_config(Path(args.viz_config))
    viz_section = dict(viz_cfg.get("viz") or {})
    lmeeeg_viz_section = dict((viz_section.get("figures") or {}).get("lmeeeg") or {})
    formats = tuple(lmeeeg_viz_section.get("formats", ["png"]))
    dpi = int(viz_section.get("dpi", 300))
    lmeeeg_root = out_dir / "lmeeeg"
    stats_root = out_dir / "stats" / "lmeeeg"

    rows = select_lmeeeg_rows(config)
    model_payloads = _load_or_fit_model_payloads(config, rows=rows)
    significance_masks = _load_significance_masks(stats_root, model_payloads=model_payloads)
    primary_manifest = build_lmeeeg_qc_manifest_from_model_payloads(
        out_dir=out_dir,
        model_payloads=model_payloads,
        manifest_path=output_path,
        significance_masks=significance_masks,
        formats=formats,
        dpi=dpi,
    )

    overlay_manifest = {"plot_count": 0, "plots": []}
    if stats_root.exists():
        model_axes = {
            model_name: {
                "channel_names": payload["channel_names"],
                "times": payload["times"],
            }
            for model_name, payload in model_payloads.items()
        }
        overlay_manifest = build_lmeeeg_qc_manifest_from_stats(
            out_dir=out_dir,
            stats_root=stats_root,
            manifest_path=output_path,
            model_axes=model_axes,
            formats=formats,
            dpi=dpi,
        )

    plots = [*primary_manifest.get("plots", []), *overlay_manifest.get("plots", [])]
    manifest = {
        "status": "ok",
        "plot_count": len(plots),
        "plots": plots,
    }
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not plots:
        manifest = build_lmeeeg_qc_manifest(
            out_dir=out_dir,
            lmeeeg_root=lmeeeg_root,
            manifest_path=output_path,
            formats=formats,
            dpi=dpi,
        )
    LOGGER.info("Wrote lmeEEG QC figure manifest with %d plots: %s", int(manifest["plot_count"]), output_path)
    return 0


def _load_viz_config(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Viz config not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"Viz config must be a mapping: {path}")
    return payload


def _load_or_fit_model_payloads(
    config: Mapping[str, Any],
    *,
    rows: list[Mapping[str, Any]],
) -> dict[str, dict[str, object]]:
    out_dir = resolve_out_dir(config)
    payloads: dict[str, dict[str, object]] = {}
    model_names = list_model_names(config, section="lmeeeg")
    for model_name in model_names:
        output_paths = build_lmeeeg_model_output_paths(config, model_name=model_name)
        if all(
            path.exists()
            for path in (
                output_paths.betas_output_path,
                output_paths.t_values_output_path,
                output_paths.times_output_path,
                output_paths.channel_names_output_path,
                output_paths.column_names_output_path,
            )
        ):
            payloads[model_name] = {
                "betas": np.load(output_paths.betas_output_path),
                "t_values": np.load(output_paths.t_values_output_path),
                "times": np.load(output_paths.times_output_path),
                "channel_names": json.loads(output_paths.channel_names_output_path.read_text(encoding="utf-8")),
                "column_names": json.loads(output_paths.column_names_output_path.read_text(encoding="utf-8")),
            }
            continue

        LOGGER.info("Saved pooled lmeEEG outputs not found for model=%s under %s; refitting for QC figures", model_name, out_dir / "lmeeeg")
        trial_data = load_pooled_lmeeeg_trial_data(config, rows, model_name=model_name)
        fit_result = fit_lmeeeg_mass_model(trial_data, config, model_name=model_name)
        payloads[model_name] = {
            "betas": fit_result.betas,
            "t_values": np.asarray(fit_result.t_values if fit_result.t_values is not None else np.asarray([], dtype=float)),
            "times": fit_result.times,
            "channel_names": list(trial_data.channel_names),
            "column_names": list(fit_result.column_names),
        }
    return payloads


def _load_significance_masks(
    stats_root: Path,
    *,
    model_payloads: dict[str, dict[str, object]],
) -> dict[str, dict[str, np.ndarray]]:
    masks: dict[str, dict[str, np.ndarray]] = {}
    if not stats_root.exists():
        return masks
    for effect_dir in sorted(path for path in stats_root.glob("*/*") if path.is_dir()):
        summary_path = effect_dir / "summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        model_name = effect_dir.parent.name
        effect_name = effect_dir.name
        payload = model_payloads.get(model_name)
        corrected_p_path = Path(str(summary.get("corrected_p_values", "")))
        if payload is None or not corrected_p_path.exists():
            continue
        corrected_p_values = np.asarray(np.load(corrected_p_path), dtype=float)
        if corrected_p_values.ndim != 2:
            continue
        masks.setdefault(model_name, {})[effect_name] = corrected_p_values < 0.05
    return masks
