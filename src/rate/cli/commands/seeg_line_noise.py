"""`seeg-line-noise` CLI subcommand."""

from __future__ import annotations

import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from rate.config import ConfigValidationError, load_and_validate_config
from rate.seeg_preproc.line_noise import LineNoiseSpec, apply_line_noise

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
        "seeg-line-noise",
        help="Apply line-noise removal to sEEG BrainVision (.vhdr) data and save FIF.",
    )
    parser.add_argument("--input", required=True, help="Input BrainVision .vhdr file.")
    parser.add_argument("--output", required=True, help="Output FIF path.")
    parser.add_argument("--config", default="config/config.yaml", help="Path to YAML config file.")
    parser.add_argument("--schema", default="config/schema.json", help="Path to JSON schema file.")
    parser.add_argument("--mode", default="notch", choices=["notch", "zapline"], help="Line-noise mode.")
    parser.add_argument("--sfreq", type=float, help="Optional override sampling rate (Hz).")
    parser.add_argument("--line-freq", type=float, default=60.0, help="Line frequency for zapline (Hz).")
    parser.add_argument("--notch-hz", type=float, default=60.0, help="Notch frequency (Hz).")
    parser.add_argument("--bandwidth-hz", type=float, default=2.0, help="Notch bandwidth (Hz).")
    parser.add_argument("--segment-sec", type=float, default=10.0, help="Zapline segment length (sec).")
    parser.add_argument("--overlap-sec", type=float, default=1.0, help="Zapline overlap (sec).")
    parser.add_argument("--nfft", type=int, default=None, help="Optional zapline FFT length.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output.")
    parser.set_defaults(command_func=run)
    return parser


def register_subparser(subparsers) -> ArgumentParser:
    return add_subparser(subparsers)


def run(args: Namespace, config: Mapping[str, Any] | None = None) -> int:
    if config is None:
        try:
            _ = load_and_validate_config(Path(args.config), schema_path=Path(args.schema))
        except ConfigValidationError as exc:
            LOGGER.error("%s", exc)
            return 1

    if mne is None:
        raise ImportError("mne is required for reading BrainVision data") from _MNE_IMPORT_ERROR

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input .vhdr not found: {input_path}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Reading BrainVision: %s", input_path)
    raw = mne.io.read_raw_brainvision(input_path, preload=True, verbose=False)

    sfreq = float(args.sfreq) if args.sfreq else float(raw.info["sfreq"])
    spec = LineNoiseSpec(
        mode=str(args.mode),
        notch_hz=float(args.notch_hz),
        bandwidth_hz=float(args.bandwidth_hz),
        line_freq=float(args.line_freq),
        segment_sec=float(args.segment_sec),
        overlap_sec=float(args.overlap_sec),
        nfft=int(args.nfft) if args.nfft is not None else None,
    )

    data = raw.get_data()
    cleaned = apply_line_noise(data, sfreq=sfreq, spec=spec)
    raw._data = np.asarray(cleaned, dtype=float)

    LOGGER.info("Writing line-noise filtered FIF: %s", output_path)
    raw.save(output_path, overwrite=bool(args.overwrite))
    return 0
