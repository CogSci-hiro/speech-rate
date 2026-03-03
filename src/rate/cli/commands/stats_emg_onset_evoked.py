from __future__ import annotations

import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Mapping

from rate.cli.commands._stats_config import load_stats_command_config
from rate.config import ConfigValidationError
from rate.stats.emg_onset_evoked import (
    build_emg_onset_grand_average_evoked,
    emg_onset_window_from_config,
)

LOGGER = logging.getLogger(__name__)
REQUIRES_CONFIG = False


def add_subparser(subparsers) -> ArgumentParser:
    parser = subparsers.add_parser(
        "stats-emg-onset-evoked",
        help="Build grand-average EMG evoked locked to self speech onset.",
    )
    parser.add_argument("--analysis", default="evoked_trf", help="Analysis branch (default: evoked_trf).")
    parser.add_argument("--model", default="baseline", help="Model name (default: baseline).")
    parser.add_argument("--config", default="config/stats.yaml", help="Path to YAML config file.")
    parser.add_argument("--schema", default="config/schema.json", help="Path to JSON schema file.")
    parser.set_defaults(command_func=run)
    return parser


def register_subparser(subparsers) -> ArgumentParser:
    return add_subparser(subparsers)


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
    model = str(args.model)
    tmin_s, tmax_s, sfreq_hz = emg_onset_window_from_config(cfg)
    zscore_within_run = bool(
        ((cfg.get("evoked_trf", {}) or {}).get("emg", {}) or {}).get("zscore_within_run", True)
    )

    try:
        res = build_emg_onset_grand_average_evoked(
            out_dir=out_dir,
            analysis=analysis,
            model=model,
            tmin_s=tmin_s,
            tmax_s=tmax_s,
            sfreq_hz=sfreq_hz,
            zscore_within_run=zscore_within_run,
        )
    except Exception as exc:
        LOGGER.exception("stats-emg-onset-evoked failed: %s", exc)
        return 1

    LOGGER.info(
        "Wrote onset-locked EMG grand average: %s (subjects=%d/%d, epochs=%d)",
        res.output_path,
        res.n_subjects_with_epochs,
        res.n_subjects,
        res.n_epochs,
    )
    return 0

