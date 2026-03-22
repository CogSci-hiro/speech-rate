"""`seeg-events-table` CLI subcommand."""

from __future__ import annotations

import logging
import re
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from rate.config import ConfigValidationError, load_and_validate_config
from rate.events.build import build_events_table
from rate.events.qc import build_ipu_qc_summary
from rate.paths import resolve_out_dir
from rate.util import write_parquet_dataframe

LOGGER = logging.getLogger(__name__)
REQUIRES_CONFIG = True

ANNOTATION_RE = re.compile(
    r"sub-(?P<subject>[^_]+)_task-(?P<task>[^_]+)_run-(?P<run>[^_-]+)[_-]ch(?P<ch>\d+)_ipu\.csv$",
    re.IGNORECASE,
)


def add_subparser(subparsers) -> ArgumentParser:
    parser = subparsers.add_parser(
        "seeg-events-table",
        help="Build sEEG events parquet from IPU annotation CSVs.",
    )
    parser.add_argument("--config", default="config/config.yaml", help="Path to YAML config file.")
    parser.add_argument("--schema", default="config/schema.json", help="Path to JSON schema file.")
    parser.add_argument(
        "--seeg-config",
        default="config/seeg_preproc.yaml",
        help="Path to sEEG config (annotations_dir/glob).",
    )
    parser.add_argument(
        "--dataset-index",
        help="Optional explicit sEEG dataset index parquet path.",
    )
    parser.add_argument(
        "--events-output",
        help="Optional explicit events parquet output path.",
    )
    parser.add_argument(
        "--qc-output",
        help="Optional explicit QC CSV output path.",
    )
    parser.set_defaults(command_func=run)
    return parser


def register_subparser(subparsers) -> ArgumentParser:
    return add_subparser(subparsers)


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    payload = yaml.safe_load(path.read_text()) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return payload


def _normalize_run(run_value: str) -> str:
    text = str(run_value).strip()
    try:
        return f"{int(float(text)):02d}"
    except ValueError:
        return text


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
    seeg_cfg = _load_yaml(Path(args.seeg_config))
    annotations_dir = Path(seeg_cfg.get("annotations_dir", "")).expanduser()
    annotations_glob = str(seeg_cfg.get("annotations_glob", "*_ipu.csv"))
    annotations_channel = str(seeg_cfg.get("annotations_channel", "ch1")).strip().lower()
    if annotations_channel in {"", "all", "both"}:
        allowed_channels = None
    else:
        allowed_channels = {annotations_channel}

    if not annotations_dir.exists():
        raise FileNotFoundError(f"annotations_dir not found: {annotations_dir}")

    dataset_index_path = Path(args.dataset_index) if args.dataset_index else out_dir / "seeg" / "dataset" / "index.parquet"
    if not dataset_index_path.exists():
        raise FileNotFoundError(f"sEEG dataset index not found: {dataset_index_path}")

    index_df = pd.read_parquet(dataset_index_path)
    keep_cols = ["subject_id", "task", "run"]
    index_units = index_df[keep_cols].dropna().copy()
    index_units["run_norm"] = index_units["run"].map(_normalize_run)

    rows = []
    matched_paths = []
    for path in annotations_dir.rglob(annotations_glob):
        match = ANNOTATION_RE.search(path.name)
        if not match:
            continue
        subject = f"sub-{match.group('subject')}"
        task = match.group("task")
        run = _normalize_run(match.group("run"))
        ch = f"ch{match.group('ch')}".lower()
        if allowed_channels is not None and ch not in allowed_channels:
            continue
        speaker = "self" if ch == "ch1" else "other"
        matched_paths.append(path)
        rows.append(
            {
                "dyad_id": f"seeg-{subject}",
                "subject_id": subject,
                "task": task,
                "run": run,
                "speaker": speaker,
                "annotations_path": str(path),
                "status": "ok",
            }
        )

    if not rows:
        LOGGER.error(
            "No sEEG annotations matched. dir=%s glob=%s",
            annotations_dir,
            annotations_glob,
        )
        raise ValueError("No sEEG annotations matched the provided pattern.")

    ds_rows = pd.DataFrame(rows)
    ds_rows["run_norm"] = ds_rows["run"].map(_normalize_run)
    ds_rows = ds_rows.merge(
        index_units[["subject_id", "task", "run_norm"]].drop_duplicates(),
        on=["subject_id", "task", "run_norm"],
        how="inner",
    ).drop(columns=["run_norm"])

    if ds_rows.empty:
        raise ValueError("No sEEG annotations matched dataset index subjects/runs/tasks.")

    events_df = build_events_table(resolved, ds_rows)
    qc_df = build_ipu_qc_summary(events_df)

    events_path = Path(args.events_output) if args.events_output else out_dir / "seeg" / "events" / "events.parquet"
    qc_path = Path(args.qc_output) if args.qc_output else out_dir / "seeg" / "events" / "qc_ipu_summary.csv"
    events_path.parent.mkdir(parents=True, exist_ok=True)
    qc_path.parent.mkdir(parents=True, exist_ok=True)
    write_parquet_dataframe(events_df, events_path)
    qc_df.to_csv(qc_path, index=False)
    LOGGER.info("Wrote sEEG events: %s (%d rows)", events_path, len(events_df))
    LOGGER.info("Wrote sEEG events QC: %s (%d rows)", qc_path, len(qc_df))
    return 0
