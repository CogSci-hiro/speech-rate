from __future__ import annotations

import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Mapping
import traceback

from mne.channels import make_standard_montage
import numpy as np

from rate.config import ConfigValidationError
from rate.cli.commands._stats_config import load_stats_command_config
from rate.stats.io import (
    discover_subject_keys_with_model,
    load_subject_lags,
    stack_subject_feature_kernels,
)
from rate.stats import ClusterMethod, StatsConfig, run_two_level_kernel0
from rate.stats.cluster import ClusterPermutationResult
from rate.stats.subject_level import compute_subject_level_z_maps
from rate.stats.two_level import TwoLevelClusterResult
from rate.stats.reporting import write_two_level_cluster_result

LOGGER = logging.getLogger(__name__)
REQUIRES_CONFIG = False


_MODEL_TEMPLATE_CANDIDATES = [
    "baseline",
    "self_rate",
    "other_rate",
    "abs_diff",
    "self_rate_sq",
    "other_rate_sq",
    "abs_diff_sq",
]


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
    parser.add_argument("--config", default="config/stats.yaml", help="Path to YAML config file.")
    parser.add_argument("--schema", default="config/schema.json", help="Path to JSON schema file.")
    parser.set_defaults(command_func=run)
    return parser


def register_subparser(subparsers) -> ArgumentParser:
    return add_subparser(subparsers)


def _empty_cluster_result(n_tests: int) -> ClusterPermutationResult:
    return ClusterPermutationResult(
        t_obs=np.zeros(int(n_tests), dtype=float),
        clusters=[],
        cluster_p_values=np.asarray([], dtype=float),
        h0=np.asarray([], dtype=float),
    )


def _find_template_lags(
    *,
    out_dir: Path,
    analysis: str,
) -> np.ndarray | None:
    for candidate_model in _MODEL_TEMPLATE_CANDIDATES:
        keys = discover_subject_keys_with_model(out_dir=out_dir, analysis=analysis, model=candidate_model)
        if not keys:
            continue
        dyad_id, subject_id = keys[0]
        try:
            return load_subject_lags(
                out_dir=out_dir,
                analysis=analysis,
                dyad_id=str(dyad_id),
                subject_id=str(subject_id),
                model=str(candidate_model),
            )
        except Exception:
            continue
    return None


def _write_placeholder_stats_result(
    *,
    out_dir: Path,
    analysis: str,
    model: str,
    feature_index: int,
    feature_name: str,
    ch_names: list[str],
    lags: np.ndarray | None,
    tables_format: str,
    subject_keys: list[tuple[str, str]] | None = None,
) -> bool:
    subject_z_maps: np.ndarray | None = None

    if subject_keys:
        try:
            feature_kernels = stack_subject_feature_kernels(
                out_dir=out_dir,
                analysis=analysis,
                subject_keys=subject_keys,
                model=model,
                feature_index=feature_index,
            )
            subject_z_maps = compute_subject_level_z_maps(feature_kernels)
            subject_z_maps = np.nan_to_num(subject_z_maps, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception:
            LOGGER.warning(
                "Could not derive subject-level z maps for placeholder stats output (%s/%s/%s).",
                analysis,
                model,
                feature_name,
            )

    if lags is None:
        lags = _find_template_lags(out_dir=out_dir, analysis=analysis)
    if lags is None:
        LOGGER.error(
            "Cannot create placeholder stats outputs for analysis=%s model=%s: no template lags found.",
            analysis,
            model,
        )
        return False

    lags = np.asarray(lags, dtype=float)
    if subject_z_maps is None:
        subject_z_maps = np.zeros((1, len(ch_names), len(lags)), dtype=float)

    n_tests = int(subject_z_maps.shape[1] * subject_z_maps.shape[2])
    placeholder = TwoLevelClusterResult(
        subject_z_maps=subject_z_maps,
        group_cluster=_empty_cluster_result(n_tests),
        ch_names=list(ch_names),
    )
    write_two_level_cluster_result(
        result=placeholder,
        out_dir=out_dir,
        analysis=analysis,
        model=model,
        feature_name=feature_name,
        feature_index=feature_index,
        lags=lags,
        tables_format=tables_format,
    )
    return True


# ==================================================================================================
#                                                RUN
# ==================================================================================================


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
    tables_format = str(cfg.get("reporting", {}).get("tables_format", "parquet")).lower()
    subject_keys = discover_subject_keys_with_model(
        out_dir=out_dir,
        analysis=str(args.analysis),
        model=str(args.model),
    )

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

    if not subject_keys:
        LOGGER.warning(
            "No subject kernels found under %s for analysis=%s model=%s. "
            "Writing placeholder empty stats outputs (debug-friendly).",
            out_dir / str(args.analysis) / "subjects",
            args.analysis,
            args.model,
        )
        ok = _write_placeholder_stats_result(
            out_dir=out_dir,
            analysis=str(args.analysis),
            model=str(args.model),
            feature_index=int(args.feature_index),
            feature_name=str(args.feature_name),
            ch_names=ch_names,
            lags=None,
            tables_format=tables_format,
            subject_keys=[],
        )
        return 0 if ok else 1

    cluster_method = ClusterMethod(stats_cfg["method"])
    cluster_block = stats_cfg.get("cluster", stats_cfg)
    tfce_block = stats_cfg.get("tfce", {})

    threshold_p = float(cluster_block.get("threshold_p", StatsConfig.threshold_p))
    tfce_start = float(StatsConfig.tfce_start)
    tfce_step = float(StatsConfig.tfce_step)

    if cluster_method == ClusterMethod.TFCE:
        if "start" not in tfce_block or "step" not in tfce_block:
            LOGGER.error(
                "TFCE mode requires config.stats.kernel_cluster.tfce.start and tfce.step "
                "(or config.stats.kernel.tfce.*)."
            )
            return 1
        tfce_start = float(tfce_block["start"])
        tfce_step = float(tfce_block["step"])
    elif cluster_method == ClusterMethod.THRESHOLD:
        if "threshold_p" not in cluster_block:
            LOGGER.error(
                "Threshold mode requires config.stats.kernel_cluster.threshold_p "
                "(or config.stats.kernel.threshold_p)."
            )
            return 1

    stats_config = StatsConfig(
        n_permutations=int(cluster_block["n_permutations"]),
        seed=int(stats_cfg.get("seed", cfg["project"]["random_seed"])),
        method=cluster_method,
        threshold_p=threshold_p,
        tfce_start=tfce_start,
        tfce_step=tfce_step,
        tail=int(stats_cfg["tail"]),
        montage_name=montage_name,
    )

    LOGGER.info(
        "Running two-level kernel test (analysis=%s, model=%s, feature=%s)",
        args.analysis,
        args.model,
        args.feature_name,
    )

    try:
        result = run_two_level_kernel0(
            out_dir=out_dir,
            analysis=args.analysis,
            subject_keys=subject_keys,
            model=args.model,
            feature_index=args.feature_index,
            ch_names=ch_names,
            config=stats_config,
        )
        lags = load_subject_lags(
            out_dir=out_dir,
            analysis=args.analysis,
            dyad_id=str(subject_keys[0][0]),
            subject_id=str(subject_keys[0][1]),
            model=args.model,
        )
    except Exception as exc:
        LOGGER.warning(
            "stats-kernel0 fallback for analysis=%s model=%s feature=%s due to %s: %s",
            args.analysis,
            args.model,
            args.feature_name,
            exc.__class__.__name__,
            exc,
        )
        ok = _write_placeholder_stats_result(
            out_dir=out_dir,
            analysis=str(args.analysis),
            model=str(args.model),
            feature_index=int(args.feature_index),
            feature_name=str(args.feature_name),
            ch_names=ch_names,
            lags=None,
            tables_format=tables_format,
            subject_keys=list(subject_keys),
        )
        if ok:
            LOGGER.info("Wrote placeholder stats outputs for %s/%s/%s", args.analysis, args.model, args.feature_name)
            return 0
        LOGGER.debug("Original stats-kernel0 exception:\n%s", traceback.format_exc())
        raise

    write_two_level_cluster_result(
        result=result,
        out_dir=out_dir,
        analysis=args.analysis,
        model=args.model,
        feature_name=args.feature_name,
        feature_index=args.feature_index,
        lags=lags,
        tables_format=tables_format,
    )

    LOGGER.info("Two-level kernel stats completed successfully.")
    return 0
