"""Shared visualization theme helpers."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


DEFAULT_THEME_NAME = "paper"

_DEFAULT_THEME: dict[str, Any] = {
    "name": DEFAULT_THEME_NAME,
    "colors": {
        "self": "#8C2D2D",
        "other": "#1F4E79",
        "shared": "#2F6B3B",
        "neutral": "#4B5563",
        "grid": "#D7DCE2",
        "spine": "#9CA3AF",
        "reference": "#6B7280",
        "background": "#FFFFFF",
    },
    "rcparams": {
        "figure.figsize": (6, 4),
        "figure.facecolor": "#FFFFFF",
        "axes.facecolor": "#FFFFFF",
        "axes.edgecolor": "#9CA3AF",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 12,
        "axes.titleweight": "semibold",
        "axes.labelsize": 11,
        "axes.labelcolor": "#111827",
        "axes.grid": False,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "xtick.color": "#374151",
        "ytick.color": "#374151",
        "grid.color": "#D7DCE2",
        "grid.linewidth": 0.8,
        "grid.alpha": 0.45,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "font.family": "serif",
        "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "#FFFFFF",
        "savefig.transparent": False,
    },
}


def resolve_theme_config(theme_config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Merge the provided theme config onto the default paper theme."""
    resolved = deepcopy(_DEFAULT_THEME)
    if not theme_config:
        return resolved

    incoming_name = theme_config.get("name")
    if incoming_name is not None:
        resolved["name"] = str(incoming_name)

    incoming_colors = theme_config.get("colors")
    if isinstance(incoming_colors, Mapping):
        resolved["colors"].update({str(key): value for key, value in incoming_colors.items()})

    incoming_rcparams = theme_config.get("rcparams")
    if isinstance(incoming_rcparams, Mapping):
        resolved["rcparams"].update({str(key): value for key, value in incoming_rcparams.items()})

    return resolved


def apply_theme(theme_config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Apply theme rcParams globally and return the resolved theme."""
    resolved = resolve_theme_config(theme_config)
    mpl.rcParams.update(resolved["rcparams"])
    return resolved


def get_theme_color(name: str, theme_config: Mapping[str, Any] | None = None, *, fallback: str | None = None) -> str:
    """Resolve a semantic theme color."""
    resolved = resolve_theme_config(theme_config)
    colors = resolved["colors"]
    if name in colors:
        return str(colors[name])
    if fallback is not None:
        return fallback
    raise KeyError(f"Unknown theme color: {name}")


def get_role_palette(theme_config: Mapping[str, Any] | None = None) -> dict[str, str]:
    """Return the main semantic palette used across figures."""
    resolved = resolve_theme_config(theme_config)
    colors = resolved["colors"]
    return {
        "self": str(colors["self"]),
        "other": str(colors["other"]),
        "shared": str(colors["shared"]),
    }


def make_dyad_palette(valid_dyads: list[str] | tuple[str, ...], theme_config: Mapping[str, Any] | None = None) -> dict[str, tuple[float, float, float]]:
    """Build a distinct categorical palette with a HUSL palette."""
    if not valid_dyads:
        return {}
    apply_theme(theme_config)
    colors = sns.color_palette("husl", n_colors=len(valid_dyads))
    return {dyad: tuple(colors[index][:3]) for index, dyad in enumerate(valid_dyads)}
