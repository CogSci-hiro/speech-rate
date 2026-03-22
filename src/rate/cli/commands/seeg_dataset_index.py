"""`seeg-dataset-index` CLI subcommand."""

from __future__ import annotations

import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml

from rate.config import ConfigValidationError, load_and_validate_config
from rate.paths import resolve_out_dir

LOGGER = logging.getLogger(__name__)
REQUIRES_CONFIG = True


def add_subparser(subparsers) -> ArgumentParser:
    parser = subparsers.add_parser(
        "seeg-dataset-index",
        help="Build a dataset index for sEEG preprocessed inputs.",
    )
    parser.add_argument("--config", default="config/config.yaml", help="Path to YAML config file.")
    parser.add_argument("--schema", default="config/schema.json", help="Path to JSON schema file.")
    parser.add_argument(
        "--seeg-config",
        default="config/seeg_preproc.yaml",
        help="Path to sEEG preprocessing config (subjects/tasks/runs).",
    )
    parser.add_argument(
        "--input-stage",
        default="line_noise",
        choices=["line_noise", "rereference"],
        help="Which sEEG preprocessing stage to index.",
    )
    parser.add_argument(
        "--output",
        help="Optional explicit output parquet path (default: <out_dir>/seeg/dataset/index.parquet).",
    )
    parser.set_defaults(command_func=run)
    return parser


def register_subparser(subparsers) -> ArgumentParser:
    return add_subparser(subparsers)


def _load_seeg_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"sEEG config not found: {path}")
    payload = yaml.safe_load(path.read_text()) or {}
    if not isinstance(payload, dict):
        raise ValueError("sEEG config must be a mapping")
    return payload


def run(args: Namespace, config: Mapping[str, Any] | None = None) -> int:
    if config is None:
        try:
            resolved = load_and_validate_config(Path(args.config), schema_path=Path(args.schema))
        except ConfigValidationError as exc:
            LOGGER.error("%s", exc)
            return 1
    else:
        resolved = dict(config)

    out_dir = resolve_out_dir(resolved)
    seeg_cfg = _load_seeg_config(Path(args.seeg_config))
    dataset_cfg = seeg_cfg.get("dataset", {}) or {}
    subjects = dataset_cfg.get("subjects", []) or []
    tasks = dataset_cfg.get("tasks", []) or []
    runs = dataset_cfg.get("runs", []) or []
    if not subjects or not tasks or not runs:
        raise ValueError("seeg_preproc.dataset subjects/tasks/runs must be non-empty")

    stage = str(args.input_stage).strip().lower()
    if stage == "rereference":
        path_pattern = (
            out_dir
            / "seeg_preproc"
            / "rereference"
            / "{subject}"
            / "{subject}_task-{task}_run-{run}_reref_raw.fif"
        )
    else:
        path_pattern = (
            out_dir
            / "seeg_preproc"
            / "line_noise"
            / "{subject}"
            / "{subject}_task-{task}_run-{run}_line_noise_raw.fif"
        )

    rows = []
    for subject in subjects:
        subject_id = str(subject).strip()
        for task in tasks:
            for run in runs:
                rows.append(
                    {
                    "dyad_id": f"seeg-{subject_id}",
                        "subject_id": subject_id,
                        "task": str(task).strip(),
                        "run": str(run).strip(),
                        "eeg_path": str(path_pattern).format(subject=subject_id, task=task, run=run),
                        "eeg_is_resampled": False,
                        "status": "ok",
                    }
                )

    df = pd.DataFrame(rows)
    output_path = Path(args.output) if args.output else out_dir / "seeg" / "dataset" / "index.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    LOGGER.info("Wrote sEEG dataset index: %s", output_path)
    return 0
