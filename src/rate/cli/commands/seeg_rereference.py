"""`seeg-rereference` CLI subcommand."""

from __future__ import annotations

import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from rate.config import ConfigValidationError, load_and_validate_config
from rate.seeg_preproc.rereference import RereferenceSpec, apply_rereference, build_bipolar_pairs

try:
    import mne
except Exception as exc:  # pragma: no cover
    mne = None
    _MNE_IMPORT_ERROR = exc
else:  # pragma: no cover
    _MNE_IMPORT_ERROR = None

LOGGER = logging.getLogger(__name__)
REQUIRES_CONFIG = True


def add_subparser(subparsers) -> ArgumentParser:
    parser = subparsers.add_parser(
        "seeg-rereference",
        help="Apply sEEG rereferencing to FIF data and save a rereferenced FIF.",
    )
    parser.add_argument("--input", required=True, help="Input FIF file (line-noise filtered).")
    parser.add_argument("--output", required=True, help="Output rereferenced FIF path.")
    parser.add_argument("--config", default="config/config.yaml", help="Path to YAML config file.")
    parser.add_argument("--schema", default="config/schema.json", help="Path to JSON schema file.")
    parser.add_argument(
        "--mode",
        default="bipolar",
        choices=["monopolar", "bipolar", "laplacian"],
        help="Rereference mode.",
    )
    parser.add_argument(
        "--laplacian-neighbors",
        help="Optional JSON mapping of channel name -> neighbor names (laplacian mode).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output.")
    parser.set_defaults(command_func=run)
    return parser


def register_subparser(subparsers) -> ArgumentParser:
    return add_subparser(subparsers)


def _load_laplacian_neighbors(path: str | None, ch_names: list[str]) -> dict[int, list[int]]:
    if not path:
        return {}
    import json

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("laplacian-neighbors JSON must be an object mapping channel -> neighbors")
    name_to_index = {name: idx for idx, name in enumerate(ch_names)}
    neighbors: dict[int, list[int]] = {}
    for key, value in payload.items():
        if key not in name_to_index:
            continue
        if not isinstance(value, list):
            raise ValueError(f"Neighbors for {key!r} must be a list")
        idx = name_to_index[key]
        neighbors[idx] = [name_to_index[n] for n in value if n in name_to_index]
    return neighbors


def run(args: Namespace, config: Mapping[str, Any] | None = None) -> int:
    if config is None:
        try:
            _ = load_and_validate_config(Path(args.config), schema_path=Path(args.schema))
        except ConfigValidationError as exc:
            LOGGER.error("%s", exc)
            return 1

    if mne is None:
        raise ImportError("mne is required for reading FIF data") from _MNE_IMPORT_ERROR

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input FIF not found: {input_path}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Reading FIF: %s", input_path)
    raw = mne.io.read_raw_fif(input_path, preload=True, verbose=False)
    data = raw.get_data()
    ch_names = list(raw.ch_names)

    spec = RereferenceSpec(mode=str(args.mode))
    bipolar_pairs: list[tuple[int, int]] | None = None
    laplacian_neighbors: dict[int, list[int]] | None = None
    new_names: list[str] | None = None

    mode = spec.mode.lower()

    if mode == "bipolar":
        bipolar_pairs, new_names = build_bipolar_pairs(ch_names)
        if not bipolar_pairs:
            raise ValueError("No bipolar pairs found from channel names; check channel naming.")
    elif mode == "laplacian":
        laplacian_neighbors = _load_laplacian_neighbors(args.laplacian_neighbors, ch_names)
        if not laplacian_neighbors:
            raise ValueError("laplacian-neighbors is required for laplacian mode")

    cleaned = apply_rereference(
        data,
        spec=spec,
        bipolar_pairs=bipolar_pairs,
        laplacian_neighbors=laplacian_neighbors,
    )

    if mode == "bipolar":
        assert bipolar_pairs is not None
        assert new_names is not None
        ch_types = [raw.get_channel_types(picks=[pair[0]])[0] for pair in bipolar_pairs]
        info = mne.create_info(new_names, sfreq=raw.info["sfreq"], ch_types=ch_types)
        new_raw = mne.io.RawArray(np.asarray(cleaned, dtype=float), info)
    else:
        new_raw = raw.copy()
        new_raw._data = np.asarray(cleaned, dtype=float)

    new_raw.set_annotations(raw.annotations)

    LOGGER.info("Writing rereferenced FIF: %s", output_path)
    new_raw.save(output_path, overwrite=bool(args.overwrite))
    return 0
