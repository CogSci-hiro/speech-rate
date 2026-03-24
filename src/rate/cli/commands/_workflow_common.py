"""Shared helpers for scaffold CLI subcommands."""

from __future__ import annotations

import json
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Mapping

from rate.paths import ensure_parent_dir, resolve_out_dir


def add_common_config_arguments(parser: ArgumentParser) -> None:
    """Attach standard config arguments to a parser."""
    parser.add_argument("--config", default="config/config.yaml", help="Path to the main YAML config file.")
    parser.add_argument("--schema", default="config/schema.json", help="Path to the JSON schema file.")


def add_subject_arguments(parser: ArgumentParser) -> None:
    """Attach standard subject selection arguments to a parser."""
    parser.add_argument("--subject-id", help="Optional subject identifier.")
    parser.add_argument("--all", action="store_true", dest="run_all", help="Write a stage-level placeholder instead.")


def resolve_output_path(args: Namespace, config: Mapping[str, Any], *, default_relative_path: str) -> Path:
    """Resolve command output path from CLI args or the configured output root."""
    if getattr(args, "output", None):
        return Path(str(args.output))
    return resolve_out_dir(config) / default_relative_path


def build_stage_payload(
    *,
    stage: str,
    args: Namespace,
    config: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a minimal JSON payload for scaffold outputs."""
    payload = {
        "stage": stage,
        "status": "placeholder",
        "subject_id": getattr(args, "subject_id", None),
        "run_all": bool(getattr(args, "run_all", False)),
        "message": "Scaffold output created during repository refactor. Full analysis logic is pending.",
        "out_dir": str(resolve_out_dir(config)),
    }
    if extra:
        payload.update(dict(extra))
    return payload


def write_placeholder_output(output_path: Path, payload: Mapping[str, Any]) -> Path:
    """Write a placeholder JSON artifact to disk."""
    ensure_parent_dir(output_path)
    output_path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path
