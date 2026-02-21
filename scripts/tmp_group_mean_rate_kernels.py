#!/usr/bin/env python3
"""Build group-mean Evoked kernels for self_rate/other_rate/abs_diff."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import mne
import pandas as pd

from rate.config import load_and_validate_config

LOGGER = logging.getLogger("tmp_group_mean_rate_kernels")


REGRESSOR_FILES = {
    "self_rate": ("self_rate", "self_rate-ave.fif"),
    "other_rate": ("other_rate", "other_rate-ave.fif"),
    "abs_diff": ("abs_diff", "abs_diff-ave.fif"),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create group-mean Evoked objects for self_rate/other_rate/abs_diff."
    )
    parser.add_argument("--config", default="config/config.yaml", help="Path to YAML config.")
    parser.add_argument("--schema", default="config/schema.json", help="Path to JSON schema.")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Optional override for cfg['paths']['out_dir'].",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional explicit output dir (default: <out_dir>/evoked_trf/temp_group_kernels).",
    )
    return parser.parse_args()


def _collect_subject_evokeds(
    *,
    subjects_root: Path,
    model_name: str,
    evoked_file: str,
) -> tuple[list[mne.Evoked], list[Path]]:
    pattern = f"dyad-*/sub-*/models/{model_name}/{evoked_file}"
    files = sorted(subjects_root.glob(pattern))

    evokeds: list[mne.Evoked] = []
    used_files: list[Path] = []
    for fpath in files:
        try:
            ev = mne.read_evokeds(fpath, verbose="ERROR")[0]
        except Exception as exc:
            LOGGER.warning("Failed to read evoked file %s: %s", fpath, exc)
            continue
        evokeds.append(ev)
        used_files.append(fpath)
    return evokeds, used_files


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parse_args()

    cfg = load_and_validate_config(Path(args.config), schema_path=Path(args.schema))
    out_dir = Path(args.out_dir) if args.out_dir else Path(cfg["paths"]["out_dir"])
    subjects_root = out_dir / "evoked_trf" / "subjects"
    output_dir = Path(args.output_dir) if args.output_dir else out_dir / "evoked_trf" / "temp_group_kernels"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not subjects_root.exists():
        raise FileNotFoundError(f"Subjects root not found: {subjects_root}")

    summary_rows: list[dict[str, object]] = []
    wrote = 0

    for reg_name, (model_name, evoked_file) in REGRESSOR_FILES.items():
        evokeds, files = _collect_subject_evokeds(
            subjects_root=subjects_root,
            model_name=model_name,
            evoked_file=evoked_file,
        )
        if not evokeds:
            LOGGER.warning("No subject evokeds found for %s (%s/%s).", reg_name, model_name, evoked_file)
            continue

        grand = mne.grand_average(evokeds, interpolate_bads=False, drop_bads=False)
        grand.comment = f"{reg_name}_group_mean"
        out_path = output_dir / f"{reg_name}-group-mean-ave.fif"
        grand.save(out_path, overwrite=True)
        wrote += 1

        summary_rows.append(
            {
                "regressor": reg_name,
                "model": model_name,
                "n_subjects": int(len(evokeds)),
                "sfreq_hz": float(grand.info["sfreq"]),
                "n_channels": int(len(grand.ch_names)),
                "n_times": int(grand.data.shape[1]),
                "path": str(out_path),
            }
        )

        files_df = pd.DataFrame({"source_path": [str(p) for p in files]})
        files_df.to_csv(output_dir / f"{reg_name}_source_files.csv", index=False)
        LOGGER.info("Wrote group evoked for %s: %s (n=%d)", reg_name, out_path, len(evokeds))

    pd.DataFrame(summary_rows).to_csv(output_dir / "group_kernel_summary.csv", index=False)

    if wrote != 3:
        LOGGER.warning("Wrote %d/3 group evoked files.", wrote)
    else:
        LOGGER.info("Wrote all 3 group evoked files.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
