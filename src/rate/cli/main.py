"""Entry point for the repository CLI."""

from __future__ import annotations

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
    "build-events": "rate.cli.commands.build_events",
    "preprocess-subject": "rate.cli.commands.preprocess_subject",
    "epoch-subject": "rate.cli.commands.epoch_subject",
    "glm-subject": "rate.legacy.cli.commands.glm_subject",
    "glm-group-stats": "rate.legacy.cli.commands.glm_group_stats",
    "fooof-epochs-subject": "rate.legacy.cli.commands.fooof_epochs_subject",
    "fooof-mass-subject": "rate.legacy.cli.commands.fooof_mass_subject",
    "fooof-mass-group-stats": "rate.legacy.cli.commands.fooof_mass_group_stats",
    "lmeeeg-subject": "rate.cli.commands.lmeeeg_subject",
    "lmeeeg-group-stats": "rate.cli.commands.lmeeeg_group_stats",
    "lmeeeg-variance-partition-fit": "rate.cli.commands.lmeeeg_variance_partition_fit",
    "lmeeeg-variance-partition-group-stats": "rate.cli.commands.lmeeeg_variance_partition_group_stats",
    "decode-temporal-generalisation": "rate.cli.commands.decode_temporal_generalisation",
    "decode-temporal-generalisation-stats": "rate.cli.commands.decode_temporal_generalisation_stats",
    "figures-behavioural": "rate.cli.commands.figures_behavioural",
    "figures-decoding": "rate.cli.commands.figures_decoding",
    "figures-glm": "rate.legacy.cli.commands.figures_glm",
    "figures-lmeeeg": "rate.cli.commands.figures_lmeeeg",
    "figures-lmeeeg-fooof": "rate.legacy.cli.commands.figures_lmeeeg_fooof",
    "figures-lmeeeg-inference": "rate.cli.commands.figures_lmeeeg_inference",
    "figures-lmeeeg-fooof-inference": "rate.legacy.cli.commands.figures_lmeeeg_fooof_inference",
    "model-weights-glm": "rate.legacy.cli.commands.model_weights_glm",
    "figures-fooof": "rate.legacy.cli.commands.figures_fooof",
}


def _load_command_module(module_path: str) -> object | None:
    try:
        return importlib.import_module(module_path)
    except Exception as exc:
        LOGGER.warning("Skipping CLI command module %s: %s", module_path, exc)
        return None


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the top-level parser with registered subcommands."""
    parser = argparse.ArgumentParser(prog="rate", description="speech-rate mass-univariate scaffold")
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
    """Load validated config for commands that require the main config."""
    if not bool(getattr(module, "REQUIRES_CONFIG", False)):
        return None
    if not hasattr(args, "config"):
        raise RuntimeError("Internal error: subcommand args missing --config.")
    return load_and_validate_config(Path(args.config), schema_path=Path(getattr(args, "schema", "config/schema.json")))


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI."""
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    available_commands = getattr(args, "_available_commands", {})
    module = available_commands.get(str(args.command))
    if module is None:
        raise RuntimeError(f"Unknown command: {args.command}")

    run = getattr(module, "run", None)
    if run is None:
        raise RuntimeError(f"CLI command module for '{args.command}' is missing run().")

    try:
        cfg = _load_command_config(args, module)
    except ConfigValidationError as exc:
        LOGGER.error("%s", exc)
        return 1

    exit_code = run(args, cfg)
    return int(exit_code) if exit_code is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
