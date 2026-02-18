"""`run-tests` CLI subcommand."""

from __future__ import annotations

import subprocess
from argparse import ArgumentParser, Namespace, _SubParsersAction


def register_subparser(subparsers: _SubParsersAction) -> ArgumentParser:
    """Register the `run-tests` subcommand.

    Parameters
    ----------
    subparsers
        Parent subparser collection from the root CLI parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser configured for the ``run-tests`` command.
    """
    parser = subparsers.add_parser("run-tests", help="Run project tests via pytest.")
    parser.add_argument("pytest_args", nargs="*", help="Optional additional pytest arguments.")
    parser.set_defaults(command_func=run)
    return parser


def run(args: Namespace) -> int:
    """Execute the project test suite through ``pytest``.

    Parameters
    ----------
    args
        Parsed CLI namespace for this command.

    Returns
    -------
    int
        Exit code returned by the ``pytest`` process.
    """
    command = ["pytest", "-q", *args.pytest_args]
    completed = subprocess.run(command, check=False)
    return completed.returncode
