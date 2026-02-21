# =============================================================================
#                                   CLI
# =============================================================================
#
# Entry point for the `rate` command-line interface.
#
# Thin dispatcher:
# - parse global + subcommand arguments
# - load project config once when required
# - call a single command module per subcommand
#
# Scientific logic should live in `rate.*`, not here.
#
# =============================================================================

import argparse
import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

from rate.config import ConfigValidationError, load_and_validate_config

import rate.cli.commands.clean as cmd_clean
import rate.cli.commands.dataset_index as cmd_dataset_index
import rate.cli.commands.dataset_subject_targets as cmd_dataset_subject_targets
import rate.cli.commands.events_table as cmd_events_table
import rate.cli.commands.run_tests as cmd_run_tests
import rate.cli.commands.figures as cmd_figures
import rate.cli.commands.validate_config as cmd_validate_config
import rate.cli.commands.write_manifest as cmd_write_manifest
import rate.cli.commands.resample_eeg as cmd_resample_eeg
import rate.cli.commands.compute_emg as cmd_compute_emg
import rate.cli.commands.compute_emg_all as cmd_compute_emg_all
import rate.cli.commands.evoked_trf_scores as cmd_evoked_trf_scores
import rate.cli.commands.evoked_trf_subject as cmd_evoked_trf_subject
import rate.cli.commands.evoked_trf_subject_all as cmd_evoked_trf_subject_all
import rate.cli.commands.evoked_trf_aggregate as cmd_evoked_trf_aggregate
import rate.cli.commands.stats_kernel0 as cmd_stats_kernel0
import rate.cli.commands.stats_export_subject_zmaps as cmd_stats_export_subject_zmaps

LOG_FORMAT = "%(levelname)s %(name)s: %(message)s"
LOGGER = logging.getLogger(__name__)

_COMMANDS = {
    "validate-config": cmd_validate_config,
    "write-manifest": cmd_write_manifest,
    "dataset-index": cmd_dataset_index,
    "dataset-subject-targets": cmd_dataset_subject_targets,
    "events-table": cmd_events_table,
    "resample-eeg": cmd_resample_eeg,
    "compute-emg": cmd_compute_emg,
    "compute-emg-all": cmd_compute_emg_all,
    "evoked-trf-scores": cmd_evoked_trf_scores,
    "evoked-trf-subject": cmd_evoked_trf_subject,
    "evoked-trf-subject-all": cmd_evoked_trf_subject_all,
    "evoked-trf-aggregate": cmd_evoked_trf_aggregate,
    "stats-kernel0": cmd_stats_kernel0,
    "stats-export-subject-zmaps": cmd_stats_export_subject_zmaps,
    "clean": cmd_clean,
    "run-tests": cmd_run_tests,
    "figures": cmd_figures,
}


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="rate",
        description="speech-rate SG0 CLI",
    )

    subparsers = parser.add_subparsers(dest="command", required=True, metavar="<command>")

    for name, module in _COMMANDS.items():
        add_subparser = getattr(module, "add_subparser", None) or getattr(module, "register_subparser", None)
        if add_subparser is None:
            raise RuntimeError(f"CLI command module for '{name}' is missing add_subparser().")
        add_subparser(subparsers)

    return parser


def _load_command_config(args: argparse.Namespace, module: object) -> Mapping[str, Any] | None:
    """Load validated config for commands that require it."""
    if not bool(getattr(module, "REQUIRES_CONFIG", False)):
        return None

    if not hasattr(args, "config"):
        raise RuntimeError("Internal error: subcommand args missing --config.")

    schema_path = Path(getattr(args, "schema", "config/schema.json"))
    return load_and_validate_config(Path(args.config), schema_path=schema_path)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    command_name = str(args.command)
    module = _COMMANDS.get(command_name)
    if module is None:
        raise RuntimeError(f"Unknown command: {command_name}")

    run = getattr(module, "run", None)
    if run is None:
        raise RuntimeError(f"CLI command module for '{command_name}' is missing run().")

    try:
        cfg = _load_command_config(args, module)
    except ConfigValidationError as exc:
        LOGGER.error("%s", exc)
        return 1

    exit_code = run(args, cfg)
    return int(exit_code) if exit_code is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
