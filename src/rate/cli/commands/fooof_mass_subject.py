"""Compatibility wrapper for the legacy FOOOF subject command."""

from __future__ import annotations

from argparse import Namespace
from typing import Any, Mapping

from rate.legacy.cli.commands import fooof_mass_subject as _legacy

REQUIRES_CONFIG = _legacy.REQUIRES_CONFIG
add_subparser = _legacy.add_subparser
fit_fooof_tensor = _legacy.fit_fooof_tensor
extract_fooof_features = _legacy.extract_fooof_features


def run(args: Namespace, config: Mapping[str, Any] | None = None) -> int:
    _legacy.fit_fooof_tensor = fit_fooof_tensor
    _legacy.extract_fooof_features = extract_fooof_features
    return _legacy.run(args, config)
