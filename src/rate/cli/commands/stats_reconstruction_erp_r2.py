from __future__ import annotations

import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Mapping

from rate.cli.commands._stats_config import load_stats_command_config
from rate.config import ConfigValidationError
from rate.stats import ClusterMethod, StatsConfig
from rate.stats.reconstruction_erp_r2 import (
    collect_subject_channel_r2,
    discover_reconstruction_models,
    run_reconstruction_erp_r2_vs_baseline_cluster,
    write_reconstruction_erp_r2_outputs,
)

LOGGER = logging.getLogger(__name__)
REQUIRES_CONFIG = False


def add_subparser(subparsers) -> ArgumentParser:
    parser = subparsers.add_parser(
        "stats-reconstruction-erp-r2",
        help="Spatial 1-sample cluster test on ERP reconstruction R^2 vs baseline (post-hoc; no refit).",
    )
    parser.add_argument("--analysis", default="evoked_trf", help="Analysis branch (default: evoked_trf).")
    parser.add_argument("--baseline-model", default="baseline", help="Baseline model name.")
    parser.add_argument("--models", nargs="+", default=None, help="Optional subset of models to test.")
    parser.add_argument("--config", default="config/stats.yaml", help="Path to YAML config file.")
    parser.add_argument("--schema", default="config/schema.json", help="Path to JSON schema file.")
    parser.set_defaults(command_func=run)
    return parser


def register_subparser(subparsers) -> ArgumentParser:
    return add_subparser(subparsers)


def _build_stats_config(cfg: Mapping[str, Any]) -> StatsConfig:
    stats_root = cfg.get("stats", {})
    stats_cfg = stats_root.get("kernel_cluster") or stats_root.get("kernel")
    if not stats_cfg:
        raise ValueError("Missing stats config. Expected config.stats.kernel_cluster or config.stats.kernel.")

    cluster_block = stats_cfg.get("cluster", stats_cfg)
    tfce_block = stats_cfg.get("tfce", {})
    method = ClusterMethod(stats_cfg["method"])

    threshold_p = float(cluster_block.get("threshold_p", StatsConfig.threshold_p))
    tfce_start = float(StatsConfig.tfce_start)
    tfce_step = float(StatsConfig.tfce_step)
    if method == ClusterMethod.TFCE:
        tfce_start = float(tfce_block.get("start", StatsConfig.tfce_start))
        tfce_step = float(tfce_block.get("step", StatsConfig.tfce_step))

    return StatsConfig(
        n_permutations=int(cluster_block["n_permutations"]),
        seed=int(stats_cfg.get("seed", cfg["project"]["random_seed"])),
        method=method,
        threshold_p=threshold_p,
        tfce_start=tfce_start,
        tfce_step=tfce_step,
        tail=int(stats_cfg["tail"]),
        montage_name=str(stats_cfg["adjacency"]["montage_name"]),
    )


def run(args: Namespace, config: Mapping[str, Any] | None = None) -> int:
    if config is None:
        try:
            cfg = load_stats_command_config(Path(args.config), schema_path=Path(args.schema))
        except ConfigValidationError as exc:
            LOGGER.error("%s", exc)
            return 1
    else:
        cfg = dict(config)

    out_dir = Path(cfg["paths"]["out_dir"])
    analysis = str(args.analysis)
    baseline_model = str(args.baseline_model)
    tables_format = str(cfg.get("reporting", {}).get("tables_format", "parquet")).lower()

    try:
        stats_config = _build_stats_config(cfg)
        discovered_models = discover_reconstruction_models(out_dir=out_dir, analysis=analysis)
        if not discovered_models:
            raise FileNotFoundError(
                f"No reconstruction_metrics.parquet files found under {out_dir / analysis / 'subjects'}"
            )

        requested_models = [str(m) for m in args.models] if args.models else discovered_models
        requested_models = [m for m in requested_models if m in set(discovered_models)]
        if baseline_model not in discovered_models:
            raise ValueError(
                f"Baseline model '{baseline_model}' has no reconstruction metrics. "
                f"Found models: {sorted(discovered_models)}"
            )
        models_for_load = sorted(set(requested_models + [baseline_model]))
        subject_r2 = collect_subject_channel_r2(out_dir=out_dir, analysis=analysis, models=models_for_load)

        comparison_models = [m for m in requested_models if m != baseline_model]
        if not comparison_models:
            raise ValueError("No non-baseline models selected for comparison.")

        results = []
        for model in comparison_models:
            LOGGER.info("Running reconstruction ERP R^2 cluster test: %s vs %s", model, baseline_model)
            results.append(
                run_reconstruction_erp_r2_vs_baseline_cluster(
                    subject_channel_r2=subject_r2,
                    model=model,
                    baseline_model=baseline_model,
                    config=stats_config,
                )
            )

        write_reconstruction_erp_r2_outputs(
            out_dir=out_dir,
            analysis=analysis,
            all_subject_channel_r2=subject_r2,
            comparison_results=results,
            tables_format=tables_format,
        )
    except Exception as exc:
        LOGGER.exception("stats-reconstruction-erp-r2 failed: %s", exc)
        return 1

    LOGGER.info(
        "Wrote reconstruction ERP R^2 stats for %d model comparisons under %s",
        len(comparison_models),
        out_dir / "stats" / analysis,
    )
    return 0

