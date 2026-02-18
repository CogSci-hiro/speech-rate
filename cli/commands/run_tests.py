"""`run-tests` CLI subcommand."""

from __future__ import annotations

import subprocess
from argparse import ArgumentParser, Namespace, _SubParsersAction


def register_subparser(subparsers: _SubParsersAction[ArgumentParser]) -> ArgumentParser:
    """Register the `run-tests` subcommand."""
    parser = subparsers.add_parser("run-tests", help="Run project tests via pytest.")
    parser.add_argument("pytest_args", nargs="*", help="Optional additional pytest arguments.")
    parser.set_defaults(command_func=run)
    return parser


def run(args: Namespace) -> int:
    """Execute the `run-tests` command."""
    command = ["pytest", "-q", *args.pytest_args]
    completed = subprocess.run(command, check=False)
    return completed.returncode
