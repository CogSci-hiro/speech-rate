"""`clean` CLI subcommand for local cache cleanup."""

from __future__ import annotations

import logging
import shutil
from argparse import ArgumentParser, Namespace, _SubParsersAction
from pathlib import Path

from rate.config import ConfigValidationError, load_and_validate_config

LOGGER = logging.getLogger(__name__)

CACHE_DIR_NAME = "__pycache__"
PYTEST_CACHE_NAME = ".pytest_cache"
SNAKEMAKE_CACHE_NAME = ".snakemake"


def register_subparser(subparsers: _SubParsersAction) -> ArgumentParser:
    """Register the `clean` subcommand."""
    parser = subparsers.add_parser("clean", help="Remove local cache/generated metadata in this repository.")
    parser.add_argument("--root", default=".", help="Repository root directory.")
    parser.add_argument("--config", default="config/config.yaml", help="Config path used to protect out_dir.")
    parser.add_argument(
        "--include-snakemake",
        action="store_true",
        help="Also remove .snakemake cache directory.",
    )
    parser.set_defaults(command_func=run)
    return parser


def _is_within(path: Path, parent: Path) -> bool:
    """Return True when `path` is within `parent`."""
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _load_out_dir(config_path: Path) -> Path | None:
    """Load configured out_dir when config exists and validates."""
    if not config_path.exists():
        return None
    try:
        resolved = load_and_validate_config(config_path)
    except ConfigValidationError as exc:
        LOGGER.warning("Could not validate config while resolving out_dir: %s", exc)
        return None

    paths = resolved.get("paths", {})
    out_dir = paths.get("out_dir") if isinstance(paths, dict) else None
    return Path(str(out_dir)).resolve() if out_dir else None


def _collect_targets(root: Path, include_snakemake: bool) -> list[Path]:
    """Collect removable cache files/directories under root."""
    targets: list[Path] = []
    targets.extend(path for path in root.rglob(CACHE_DIR_NAME) if path.is_dir())
    targets.extend(path for path in root.rglob("*.pyc") if path.is_file())
    targets.extend(path for path in root.rglob(PYTEST_CACHE_NAME) if path.is_dir())
    targets.extend(path for path in root.rglob("*.egg-info") if path.is_dir())
    if include_snakemake:
        targets.extend(path for path in root.rglob(SNAKEMAKE_CACHE_NAME) if path.is_dir())
    return sorted({path.resolve() for path in targets})


def run(args: Namespace) -> int:
    """Execute the `clean` command."""
    root = Path(args.root).resolve()
    out_dir = _load_out_dir((root / args.config).resolve())

    removed: list[Path] = []
    skipped: list[Path] = []
    targets = _collect_targets(root, include_snakemake=bool(args.include_snakemake))
    for target in targets:
        if out_dir and _is_within(target, out_dir):
            skipped.append(target)
            continue

        try:
            if target.is_dir():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()
            removed.append(target)
        except OSError as exc:
            LOGGER.warning("Failed to remove %s: %s", target, exc)

    if not removed:
        print("No cache files removed.")
    else:
        print("Removed cache paths:")
        for path in removed:
            print(f"- {path}")

    if skipped:
        print("Skipped paths under out_dir:")
        for path in skipped:
            print(f"- {path}")

    if not args.include_snakemake:
        print("Note: .snakemake was not removed. Re-run with --include-snakemake to remove it.")

    return 0
