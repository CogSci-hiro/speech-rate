"""`run-tests` CLI subcommand."""

import subprocess
from argparse import ArgumentParser, Namespace
from typing import Any, Mapping


REQUIRES_CONFIG = False


def add_subparser(subparsers) -> ArgumentParser:
    """Register the `run-tests` subcommand."""
    parser = subparsers.add_parser("run-tests", help="Run project tests via pytest.")
    parser.add_argument("pytest_args", nargs="*", help="Optional additional pytest arguments.")
    parser.set_defaults(command_func=run)
    return parser


def register_subparser(subparsers) -> ArgumentParser:
    """Backward-compatible alias for CLI registration."""
    return add_subparser(subparsers)


def run(args: Namespace, config: Mapping[str, Any] | None = None) -> int:
    """Execute the project test suite through ``pytest``."""
    command = ["pytest", "-q", *args.pytest_args]
    completed = subprocess.run(command, check=False)
    return completed.returncode
