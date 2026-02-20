#!/usr/bin/env python3
"""Audit resampled EEG outputs against direct in-memory MNE resampling."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rate.config import load_and_validate_config

LOGGER = logging.getLogger("tmp_audit_resample_outputs")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit resampled outputs vs direct resample.")
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--schema", default="config/schema.json")
    p.add_argument("--out-dir", default=None, help="Override cfg['paths']['out_dir']")
    p.add_argument("--max-rows", type=int, default=10, help="Rows to audit (ignored when --all).")
    p.add_argument("--all", action="store_true", help="Audit all rows.")
    p.add_argument("--seed", type=int, default=1234, help="Sampling seed for row selection.")
    p.add_argument("--dyad-id", default=None)
    p.add_argument("--subject-id", default=None)
    p.add_argument("--run", type=int, default=None)
    p.add_argument(
        "--output-csv",
        default=None,
        help="Path to output CSV report (default: <out_dir>/eeg_resampled/qc/resample_audit.csv).",
    )
    return p.parse_args()


def _read_raw_auto(path: Path) -> Any:
    import mne

    suffixes = [s.lower() for s in path.suffixes]
    # Handle .fif and .fif.gz
    if suffixes[-2:] == [".fif", ".gz"] or suffixes[-1:] == [".fif"]:
        return mne.io.read_raw_fif(path, preload=True, verbose="ERROR")
    if suffixes[-1:] == [".edf"]:
        return mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    if suffixes[-1:] == [".bdf"]:
        return mne.io.read_raw_bdf(path, preload=True, verbose="ERROR")
    if suffixes[-1:] == [".vhdr"]:
        return mne.io.read_raw_brainvision(path, preload=True, verbose="ERROR")
    raise ValueError(f"Unsupported EEG file extension for: {path}")


def _norm_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()


def _select_rows(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = df.copy()
    if args.dyad_id is not None:
        out = out.loc[_norm_series(out["dyad_id"]) == str(args.dyad_id).strip()]
    if args.subject_id is not None:
        out = out.loc[_norm_series(out["subject_id"]) == str(args.subject_id).strip()]
    if args.run is not None:
        out = out.loc[pd.to_numeric(out["run"], errors="coerce") == int(args.run)]

    if args.all:
        return out.reset_index(drop=True)
    if len(out) <= args.max_rows:
        return out.reset_index(drop=True)
    return out.sample(n=args.max_rows, random_state=int(args.seed)).reset_index(drop=True)


def _compare_one(in_path: Path, out_path: Path, target_hz: float) -> dict[str, object]:
    row: dict[str, object] = {
        "input_exists": in_path.exists(),
        "output_exists": out_path.exists(),
    }
    if not in_path.exists() or not out_path.exists():
        row.update(
            {
                "status": "missing_file",
                "sfreq_in": np.nan,
                "sfreq_out": np.nan,
                "sfreq_direct": np.nan,
                "n_chan_in": np.nan,
                "n_chan_out": np.nan,
                "n_chan_direct": np.nan,
                "n_times_out": np.nan,
                "n_times_direct": np.nan,
                "max_abs_diff": np.nan,
                "mean_abs_diff": np.nan,
                "rms_diff": np.nan,
                "chan_name_match": False,
            }
        )
        return row

    raw_in = _read_raw_auto(in_path)
    raw_out = _read_raw_auto(out_path)
    raw_direct = raw_in.copy()
    if not np.isclose(float(raw_direct.info["sfreq"]), float(target_hz), atol=1e-6):
        raw_direct.resample(float(target_hz), npad="auto", verbose="ERROR")

    data_out = raw_out.get_data()
    data_direct = raw_direct.get_data()

    chan_name_match = raw_out.ch_names == raw_direct.ch_names
    same_shape = data_out.shape == data_direct.shape
    if same_shape:
        diff = data_out - data_direct
        max_abs = float(np.max(np.abs(diff)))
        mean_abs = float(np.mean(np.abs(diff)))
        rms = float(np.sqrt(np.mean(diff**2)))
    else:
        max_abs = np.nan
        mean_abs = np.nan
        rms = np.nan

    row.update(
        {
            "status": "ok" if (chan_name_match and same_shape) else "shape_or_channel_mismatch",
            "sfreq_in": float(raw_in.info["sfreq"]),
            "sfreq_out": float(raw_out.info["sfreq"]),
            "sfreq_direct": float(raw_direct.info["sfreq"]),
            "n_chan_in": int(len(raw_in.ch_names)),
            "n_chan_out": int(len(raw_out.ch_names)),
            "n_chan_direct": int(len(raw_direct.ch_names)),
            "n_times_out": int(raw_out.n_times),
            "n_times_direct": int(raw_direct.n_times),
            "max_abs_diff": max_abs,
            "mean_abs_diff": mean_abs,
            "rms_diff": rms,
            "chan_name_match": bool(chan_name_match),
        }
    )
    return row


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parse_args()
    cfg = load_and_validate_config(Path(args.config), schema_path=Path(args.schema))
    out_dir = Path(args.out_dir) if args.out_dir else Path(cfg["paths"]["out_dir"])
    target_hz = float(cfg["evoked_trf"]["downsample_hz"])

    index_in = out_dir / "dataset" / "index.parquet"
    index_out = out_dir / "dataset" / "index_resampled.parquet"
    if not index_in.exists():
        raise FileNotFoundError(index_in)
    if not index_out.exists():
        raise FileNotFoundError(index_out)

    df_in = pd.read_parquet(index_in).copy()
    df_out = pd.read_parquet(index_out).copy()

    for df in (df_in, df_out):
        df["dyad_id"] = _norm_series(df["dyad_id"])
        df["subject_id"] = _norm_series(df["subject_id"])
        df["run"] = pd.to_numeric(df["run"], errors="coerce").astype("Int64")

    key_cols = ["dyad_id", "subject_id", "run"]
    merged = df_out.merge(
        df_in[key_cols + ["eeg_path"]].rename(columns={"eeg_path": "eeg_path_in"}),
        on=key_cols,
        how="left",
    ).rename(columns={"eeg_path": "eeg_path_out"})

    ok = merged.loc[merged.get("status", "ok").astype(str).str.lower() == "ok"].copy()
    sel = _select_rows(ok, args)
    LOGGER.info("Auditing %d rows (from %d ok rows).", len(sel), len(ok))

    out_rows: list[dict[str, object]] = []
    for r in sel.itertuples(index=False):
        in_path = Path(str(r.eeg_path_in)).expanduser()
        out_path = Path(str(r.eeg_path_out)).expanduser()
        cmp_row = _compare_one(in_path=in_path, out_path=out_path, target_hz=target_hz)
        out_rows.append(
            {
                "dyad_id": str(r.dyad_id),
                "subject_id": str(r.subject_id),
                "run": int(r.run),
                "eeg_path_in": str(in_path),
                "eeg_path_out": str(out_path),
                **cmp_row,
            }
        )

    report = pd.DataFrame(out_rows)
    report = report.sort_values(["dyad_id", "subject_id", "run"], kind="mergesort").reset_index(drop=True)

    out_csv = Path(args.output_csv) if args.output_csv else out_dir / "eeg_resampled" / "qc" / "resample_audit.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out_csv, index=False)

    LOGGER.info("Wrote resample audit report: %s", out_csv)
    if not report.empty:
        LOGGER.info(
            "Summary: status_counts=%s, max_abs_diff_max=%s",
            report["status"].value_counts(dropna=False).to_dict(),
            float(report["max_abs_diff"].max(skipna=True)) if report["max_abs_diff"].notna().any() else np.nan,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
