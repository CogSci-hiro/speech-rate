"""CLI dispatcher for the `rate` package."""

from __future__ import annotations

import logging
from argparse import ArgumentParser
from typing import Sequence

from cli.commands import run_tests, validate_config, write_manifest

LOG_FORMAT = "%(levelname)s %(name)s: %(message)s"


def build_parser() -> ArgumentParser:
    """Construct the CLI parser."""
    parser = ArgumentParser(prog="rate", description="speech-rate SG0 CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_config.register_subparser(subparsers)
    write_manifest.register_subparser(subparsers)
    run_tests.register_subparser(subparsers)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint."""
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    parser = build_parser()
    args = parser.parse_args(argv)
    command_func = args.command_func
    return int(command_func(args))


if __name__ == "__main__":
    raise SystemExit(main())
