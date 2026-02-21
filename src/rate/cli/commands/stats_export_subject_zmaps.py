from __future__ import annotations

import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from rate.config import ConfigValidationError, load_and_validate_config
from rate.stats.io import discover_subject_keys_with_model

LOGGER = logging.getLogger(__name__)
REQUIRES_CONFIG = True


def _resolve_feature_dir(feature_name: str, feature_index: int) -> str:
    """Resolve output directory name for a model feature."""
    if str(feature_name) == "base":
        return "base"
    return f"feature-{feature_index}"


def add_subparser(subparsers) -> ArgumentParser:
    parser = subparsers.add_parser(
        "stats-export-subject-zmaps",
        help="Export per-subject z-maps from kernel0 stats output.",
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
    subject_keys = discover_subject_keys_with_model(
        out_dir=out_dir,
        analysis=str(args.analysis),
        model=str(args.model),
    )
    if not subject_keys:
        LOGGER.error(
            "No subject kernels found under %s for analysis=%s model=%s",
            out_dir / str(args.analysis) / "subjects",
            args.analysis,
            args.model,
        )
        return 1

    feature_dir = _resolve_feature_dir(str(args.feature_name), int(args.feature_index))
    base = (
        out_dir
        / "stats"
        / str(args.analysis)
        / str(args.model)
        / feature_dir
    )
    z_maps_path = base / "subject_z_maps.npy"
    if not z_maps_path.exists():
        LOGGER.error("Missing subject z-map array: %s", z_maps_path)
        return 1

    z_maps = np.load(z_maps_path)
    if z_maps.ndim != 3:
        LOGGER.error("Expected subject_z_maps.npy ndim=3, got shape=%s", tuple(z_maps.shape))
        return 1

    n_subjects, n_channels, n_lags = z_maps.shape
    if n_subjects != len(subject_keys):
        LOGGER.error(
            "Subject count mismatch: z-maps has %d subjects but dataset index has %d ok subjects.",
            n_subjects,
            len(subject_keys),
        )
        return 1

    subject_level_dir = base / "subject_level"
    subject_level_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for i, (dyad_id, subject_id) in enumerate(subject_keys):
        file_name = f"{dyad_id}__{subject_id}.npy"
        z_map_file = subject_level_dir / file_name
        np.save(z_map_file, z_maps[i])
        rows.append(
            {
                "analysis": str(args.analysis),
                "model": str(args.model),
                "feature_name": str(args.feature_name),
                "feature_index": int(args.feature_index),
                "dyad_id": dyad_id,
                "subject_id": subject_id,
                "n_channels": int(n_channels),
                "n_lags": int(n_lags),
                "z_map_path": str(z_map_file),
            }
        )

    manifest = pd.DataFrame(rows).sort_values(["dyad_id", "subject_id"], kind="mergesort")
    table_format = str(cfg.get("reporting", {}).get("tables_format", "parquet")).lower()
    ext = "csv" if table_format == "csv" else "parquet"
    manifest_path = subject_level_dir / f"subject_z_map_index.{ext}"
    if ext == "csv":
        manifest.to_csv(manifest_path, index=False)
    else:
        manifest.to_parquet(manifest_path, index=False)

    LOGGER.info("Wrote subject-level z-map exports: %s", subject_level_dir)
    LOGGER.info("Wrote manifest: %s", manifest_path)
    return 0
