"""Helpers for target-specific lmeEEG analyses."""

from __future__ import annotations

from typing import Any, Mapping


def section_supports_targets(config: Mapping[str, Any], *, config_section: str) -> bool:
    """Return whether a config section declares target-specific analyses."""
    section_cfg = config.get(config_section) or {}
    targets = section_cfg.get("targets")
    return isinstance(targets, list) and len(targets) > 0


def resolve_target_names(
    config: Mapping[str, Any],
    *,
    config_section: str,
    target_name: str | None = None,
) -> list[str | None]:
    """Resolve configured targets for one lmeEEG section."""
    if not section_supports_targets(config, config_section=config_section):
        return [None]
    if target_name is not None:
        return [str(target_name)]
    targets = [str(value) for value in (config.get(config_section) or {}).get("targets", [])]
    if not targets:
        raise ValueError(f"No `{config_section}.targets` were configured for target-specific inference.")
    return targets
