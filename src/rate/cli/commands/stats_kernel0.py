from __future__ import annotations

import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
from mne.channels import make_standard_montage

from rate.config import ConfigValidationError, load_and_validate_config
from rate.stats import ClusterMethod, StatsConfig, run_two_level_kernel0
from rate.stats.reporting import write_two_level_cluster_result

LOGGER = logging.getLogger(__name__)
REQUIRES_CONFIG = True


# ==================================================================================================
#                                            CLI Registration
# ==================================================================================================


def add_subparser(subparsers) -> ArgumentParser:
    parser = subparsers.add_parser(
        "stats-kernel0",
        help="Two-level cluster test of TRF kernel vs zero.",
    )
    parser.add_argument("--analysis", required=True, help="Analysis branch (e.g. evoked_trf).")
    parser.add_argument("--model", required=True, help="Model name.")
    parser.add_argument("--feature-index", required=True, type=int, help="Feature index in kernels.")
    parser.add_argument("--feature-name", required=True, help="Feature name (for output path).")
    parser.add_argument("--config", default="config/config.yaml", help="Path to YAML config file.")
    parser.add_argument("--schema", default="config/schema.json", help="Path to JSON schema file.")
    parser.set_defaults(command_func=run)
    return parser


def register_subparser(subparsers) -> ArgumentParser:
    return add_subparser(subparsers)


# ==================================================================================================
#                                                RUN
# ==================================================================================================


def run(args: Namespace, config: Mapping[str, Any] | None = None) -> int:
    if config is None:
        try:
            cfg = load_and_validate_config(Path(args.config), schema_path=Path(args.schema))
        except ConfigValidationError as exc:
            LOGGER.error("%s", exc)
            return 1
    else:
        cfg = dict(config)

    out_dir = Path(cfg["paths"]["out_dir"])
    index_path = out_dir / "dataset" / "index.parquet"

    LOGGER.info("Reading dataset index: %s", index_path)
    df = pd.read_parquet(index_path)
    df_ok = df[df["status"] == "ok"].copy()

    if df_ok.empty:
        LOGGER.error("No valid subjects found in dataset index.")
        return 1

    subject_keys = list(zip(df_ok["dyad_id"], df_ok["subject_id"]))

    # ----------------------------------------------------------------------------------
    # Stats configuration
    # ----------------------------------------------------------------------------------

    stats_root = cfg.get("stats", {})
    stats_cfg = stats_root.get("kernel_cluster") or stats_root.get("kernel")
    if not stats_cfg:
        LOGGER.error(
            "Missing stats kernel config. Expected one of: "
            "config.stats.kernel_cluster or config.stats.kernel"
        )
        return 1

    montage_name = stats_cfg["adjacency"]["montage_name"]
    montage = make_standard_montage(montage_name)
    ch_names = montage.ch_names

    cluster_method = ClusterMethod(stats_cfg["method"])
    cluster_block = stats_cfg.get("cluster", stats_cfg)
    tfce_block = stats_cfg.get("tfce", {})

    stats_config = StatsConfig(
        n_permutations=int(cluster_block["n_permutations"]),
        seed=int(stats_cfg.get("seed", cfg["project"]["random_seed"])),
        method=cluster_method,
        threshold_p=float(cluster_block["threshold_p"]),
        tfce_start=float(tfce_block["start"]),
        tfce_step=float(tfce_block["step"]),
        tail=int(stats_cfg["tail"]),
        montage_name=montage_name,
    )

    LOGGER.info(
        "Running two-level kernel test (analysis=%s, model=%s, feature=%s)",
        args.analysis,
        args.model,
        args.feature_name,
    )

    result = run_two_level_kernel0(
        out_dir=out_dir,
        analysis=args.analysis,
        subject_keys=subject_keys,
        model=args.model,
        feature_index=args.feature_index,
        ch_names=ch_names,
        config=stats_config,
    )

    write_two_level_cluster_result(
        result=result,
        out_dir=out_dir,
        analysis=args.analysis,
        model=args.model,
        feature_name=args.feature_name,
        feature_index=args.feature_index,
        tables_format=str(cfg.get("reporting", {}).get("tables_format", "parquet")).lower(),
    )

    LOGGER.info("Two-level kernel stats completed successfully.")
    return 0
