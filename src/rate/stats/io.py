"""IO helpers for thin stats wrappers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from rate.paths import ensure_parent_dir


def write_stats_summary(output_path: Path, payload: Mapping[str, Any]) -> Path:
    """Write a small JSON summary for a stats stage."""
    ensure_parent_dir(output_path)
    output_path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path
