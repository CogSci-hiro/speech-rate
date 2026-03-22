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
import importlib
import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

from rate.config import ConfigValidationError, load_and_validate_config

LOG_FORMAT = "%(levelname)s %(name)s: %(message)s"
LOGGER = logging.getLogger(__name__)

_COMMANDS = {
    "validate-config": "rate.cli.commands.validate_config",
    "write-manifest": "rate.cli.commands.write_manifest",
    "dataset-index": "rate.cli.commands.dataset_index",
    "dataset-subject-targets": "rate.cli.commands.dataset_subject_targets",
    "events-table": "rate.cli.commands.events_table",
    "resample-eeg": "rate.cli.commands.resample_eeg",
    "compute-emg": "rate.cli.commands.compute_emg",
    "compute-emg-all": "rate.cli.commands.compute_emg_all",
    "fooof": "rate.cli.commands.fooof",
    "evoked-trf-scores": "rate.cli.commands.evoked_trf_scores",
    "evoked-trf-subject": "rate.cli.commands.evoked_trf_subject",
    "evoked-trf-subject-all": "rate.cli.commands.evoked_trf_subject_all",
    "evoked-trf-aggregate": "rate.cli.commands.evoked_trf_aggregate",
    "evoked-trf-reconstruction-aggregate": "rate.cli.commands.evoked_trf_reconstruction_aggregate",
    "evoked-trf-similarity-aggregate": "rate.cli.commands.evoked_trf_similarity_aggregate",
    "induced-trf-subject": "rate.cli.commands.induced_trf_subject",
    "induced-trf-sliding-fooof-subject": "rate.cli.commands.induced_trf_sliding_fooof_subject",
    "induced-trf-subject-all": "rate.cli.commands.induced_trf_subject_all",
    "induced-trf-aggregate": "rate.cli.commands.induced_trf_aggregate",
    "stats-kernel0": "rate.cli.commands.stats_kernel0",
    "stats-emg-onset-evoked": "rate.cli.commands.stats_emg_onset_evoked",
    "stats-reconstruction-erp-r2": "rate.cli.commands.stats_reconstruction_erp_r2",
    "stats-export-subject-zmaps": "rate.cli.commands.stats_export_subject_zmaps",
    "clean": "rate.cli.commands.clean",
    "run-tests": "rate.cli.commands.run_tests",
    "figures": "rate.cli.commands.figures",
}


def _load_command_module(module_path: str) -> object | None:
    try:
        return importlib.import_module(module_path)
    except Exception as exc:
        LOGGER.warning("Skipping CLI command module %s: %s", module_path, exc)
        return None


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="rate",
        description="speech-rate SG0 CLI",
    )

    subparsers = parser.add_subparsers(dest="command", required=True, metavar="<command>")

    available_commands: dict[str, object] = {}
    for name, module_path in _COMMANDS.items():
        module = _load_command_module(module_path)
        if module is None:
            continue
        available_commands[name] = module
        add_subparser = getattr(module, "add_subparser", None) or getattr(module, "register_subparser", None)
        if add_subparser is None:
            raise RuntimeError(f"CLI command module for '{name}' is missing add_subparser().")
        add_subparser(subparsers)

    parser.set_defaults(_available_commands=available_commands)
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
    available_commands = getattr(args, "_available_commands", {})
    module = available_commands.get(command_name)
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
