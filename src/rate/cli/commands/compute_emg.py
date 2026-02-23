"""`compute-emg` CLI subcommand."""

from __future__ import annotations

import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from rate.config import ConfigValidationError, get_project_downsample_hz, load_and_validate_config
from rate.features.emg import compute_emg_envelope
from rate.features.io import write_subject_emg_parquet

LOGGER = logging.getLogger(__name__)
REQUIRES_CONFIG = True


def add_subparser(subparsers) -> ArgumentParser:
    """Register the `compute-emg` subcommand."""
    parser = subparsers.add_parser(
        "compute-emg",
        help="Compute subject-level EMG envelope features aggregated across runs.",
    )
    parser.add_argument("--dyad-id", required=True, help="Dyad ID (e.g., dyad-001).")
    parser.add_argument("--subject", required=True, help="Subject ID (e.g., sub-001).")
    parser.add_argument("--config", default="config/config.yaml", help="Path to YAML config file.")
    parser.add_argument("--schema", default="config/schema.json", help="Path to JSON schema file.")
    parser.add_argument("--out-dir", help="Optional override for cfg['paths']['out_dir'].")
    parser.add_argument(
        "--dataset-index",
        help="Optional explicit dataset index parquet path. Defaults to <out_dir>/dataset/index.parquet.",
    )
    parser.add_argument(
        "--output",
        help="Optional explicit output parquet path. Defaults to <out_dir>/features/emg/<dyad>/<subject>/emg_env.parquet.",
    )
    parser.set_defaults(command_func=run)
    return parser


def register_subparser(subparsers) -> ArgumentParser:
    """Backward-compatible alias for CLI registration."""
    return add_subparser(subparsers)


def _subject_rows(ds_index: pd.DataFrame, *, dyad_id: str, subject: str) -> pd.DataFrame:
    required = {"dyad_id", "subject_id", "run", "eeg_path"}
    missing = required - set(ds_index.columns)
    if missing:
        raise ValueError(f"dataset index missing required columns: {sorted(missing)}")

    dyad_col = ds_index["dyad_id"].astype(str).str.strip()
    subj_col = ds_index["subject_id"].astype(str).str.strip()
    mask = (dyad_col == str(dyad_id).strip()) & (subj_col == str(subject).strip())
    rows = ds_index.loc[mask].copy()
    if rows.empty:
        raise ValueError(f"No dataset rows found for dyad_id={dyad_id}, subject_id={subject}")

    if "status" in rows.columns:
        rows = rows.loc[rows["status"].astype(str).str.lower() == "ok"].copy()
    rows = rows.loc[rows["eeg_path"].notna()].copy()
    rows = rows.loc[rows["eeg_path"].astype(str).str.strip() != ""].copy()
    if rows.empty:
        raise ValueError(f"No runnable dataset rows for dyad_id={dyad_id}, subject_id={subject}")

    rows["run_num"] = pd.to_numeric(rows["run"], errors="coerce")
    rows = rows.dropna(subset=["run_num"]).sort_values("run_num", kind="mergesort")
    if rows.empty:
        raise ValueError(f"No numeric run values found for dyad_id={dyad_id}, subject_id={subject}")
    return rows


def compute_subject_emg(
    *,
    cfg: Mapping[str, Any],
    out_dir: Path,
    dataset_index_path: Path,
    dyad_id: str,
    subject: str,
    output_path: Path | None = None,
) -> Path:
    if not dataset_index_path.exists():
        raise FileNotFoundError(f"dataset index not found: {dataset_index_path}")
    ds_index = pd.read_parquet(dataset_index_path)

    emg_cfg = dict(cfg.get("evoked_trf", {}).get("emg", {}) or {})
    channels = emg_cfg.get("channels", ["EMG1", "EMG2"])
    channel_patterns = emg_cfg.get("channel_patterns", [r"\bEMG\b", r"\bEXG\b", "masseter", "jaw"])
    missing_policy = str(emg_cfg.get("missing_policy", "zeros")).strip().lower()
    l_freq = float(emg_cfg.get("l_freq", 30.0))
    h_freq = float(emg_cfg.get("h_freq", 150.0))
    log_transform = bool(emg_cfg.get("log_transform", True))
    target_sfreq = float(
        emg_cfg.get(
            "target_sfreq",
            get_project_downsample_hz(cfg, fallback_section="evoked_trf"),
        )
    )

    rows = _subject_rows(ds_index, dyad_id=str(dyad_id), subject=str(subject))
    out_rows: list[pd.DataFrame] = []

    for row in rows.itertuples(index=False):
        run = int(row.run_num)
        eeg_path = Path(str(row.eeg_path)).expanduser()
        times, emg_env = compute_emg_envelope(
            eeg_path=eeg_path,
            target_sfreq=target_sfreq,
            emg_channels=channels,
            emg_channel_patterns=channel_patterns,
            missing_policy=missing_policy,
            l_freq=l_freq,
            h_freq=h_freq,
            log_transform=log_transform,
        )
        out_rows.append(
            pd.DataFrame(
                {
                    "dyad_id": [str(dyad_id)] * len(times),
                    "subject_id": [str(subject)] * len(times),
                    "run": [run] * len(times),
                    "time": times,
                    "emg_env": emg_env,
                }
            )
        )

    if not out_rows:
        raise ValueError(f"No EMG rows produced for dyad_id={dyad_id}, subject_id={subject}")

    all_rows = pd.concat(out_rows, ignore_index=True)
    output = output_path or (out_dir / "features" / "emg" / str(dyad_id) / str(subject) / "emg_env.parquet")
    write_subject_emg_parquet(output, rows=all_rows)
    return output


def run(args: Namespace, config: Mapping[str, Any] | None = None) -> int:
    """Compute subject-level EMG features and write parquet output."""
    if config is None:
        try:
            resolved = load_and_validate_config(Path(args.config), schema_path=Path(args.schema))
        except ConfigValidationError as exc:
            LOGGER.error("%s", exc)
            return 1
    else:
        resolved = dict(config)

    out_dir = Path(args.out_dir) if args.out_dir else Path(resolved["paths"]["out_dir"])
    dataset_index_path = Path(args.dataset_index) if args.dataset_index else out_dir / "dataset" / "index.parquet"
    output_path = (
        Path(args.output)
        if args.output
        else out_dir / "features" / "emg" / str(args.dyad_id) / str(args.subject) / "emg_env.parquet"
    )
    output_path = compute_subject_emg(
        cfg=resolved,
        out_dir=out_dir,
        dataset_index_path=dataset_index_path,
        dyad_id=str(args.dyad_id),
        subject=str(args.subject),
        output_path=output_path,
    )
    LOGGER.info("Wrote EMG features: %s", output_path)
    return 0
