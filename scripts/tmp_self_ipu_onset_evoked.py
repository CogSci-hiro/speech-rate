#!/usr/bin/env python3
"""Build per-subject evoked responses from self IPU onset events."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from rate.config import load_and_validate_config

LOGGER = logging.getLogger("tmp_self_ipu_onset_evoked")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build per-subject evoked from self IPU onsets.")
    parser.add_argument("--config", default="config/config.yaml", help="Path to YAML config.")
    parser.add_argument("--schema", default="config/schema.json", help="Path to config schema.")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Optional override for cfg['paths']['out_dir'].",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional explicit output dir (default: <out_dir>/evoked_trf/temp_self_ipu_onset).",
    )
    parser.add_argument("--tmin", type=float, default=-2.0, help="Epoch start (s).")
    parser.add_argument("--tmax", type=float, default=0.5, help="Epoch end (s).")
    parser.add_argument(
        "--baseline-start",
        type=float,
        default=-2.0,
        help="Baseline window start (s).",
    )
    parser.add_argument(
        "--baseline-end",
        type=float,
        default=0.5,
        help="Baseline window end (s).",
    )
    return parser.parse_args()


def _subject_pairs(events: pd.DataFrame) -> list[tuple[str, str]]:
    pairs = {
        (str(d).strip(), str(s).strip())
        for d, s in events[["dyad_id", "subject_id"]].itertuples(index=False)
    }
    return sorted(pairs)


def _run_rows_for_subject(ds_index: pd.DataFrame, dyad_id: str, subject_id: str) -> pd.DataFrame:
    dyad_col = ds_index["dyad_id"].astype(str).str.strip()
    subj_col = ds_index["subject_id"].astype(str).str.strip()
    run_col = pd.to_numeric(ds_index["run"], errors="coerce")
    rows = ds_index.loc[(dyad_col == dyad_id) & (subj_col == subject_id)].copy()
    rows["run_num"] = pd.to_numeric(run_col.loc[rows.index], errors="coerce")
    rows = rows.dropna(subset=["run_num"]).sort_values("run_num")
    return rows


def _events_for_run(events: pd.DataFrame, dyad_id: str, subject_id: str, run: int) -> np.ndarray:
    ev = events.loc[
        (events["dyad_id"].astype(str).str.strip() == dyad_id)
        & (events["subject_id"].astype(str).str.strip() == subject_id)
        & (pd.to_numeric(events["run"], errors="coerce") == int(run))
    ].copy()
    if ev.empty:
        return np.empty((0,), dtype=float)
    t = pd.to_numeric(ev["t_onset"], errors="coerce").to_numpy(dtype=float)
    t = t[np.isfinite(t)]
    return np.sort(t)


def _make_epochs(
    *,
    raw: mne.io.BaseRaw,
    onset_times_s: np.ndarray,
    tmin: float,
    tmax: float,
    baseline: tuple[float, float],
) -> mne.Epochs | None:
    if onset_times_s.size == 0:
        return None
    sample_idx = np.round(onset_times_s * float(raw.info["sfreq"])).astype(int)
    sample_idx = sample_idx[(sample_idx >= 0) & (sample_idx < raw.n_times)]
    if sample_idx.size == 0:
        return None
    events_mne = np.column_stack(
        [
            sample_idx,
            np.zeros(sample_idx.size, dtype=int),
            np.ones(sample_idx.size, dtype=int),
        ]
    )
    return mne.Epochs(
        raw=raw,
        events=events_mne,
        event_id={"self_onset": 1},
        tmin=float(tmin),
        tmax=float(tmax),
        baseline=baseline,
        preload=True,
        reject_by_annotation=False,
        verbose=False,
        detrend=1
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parse_args()

    cfg = load_and_validate_config(Path(args.config), schema_path=Path(args.schema))
    out_dir = Path(args.out_dir) if args.out_dir else Path(cfg["paths"]["out_dir"])
    output_dir = Path(args.output_dir) if args.output_dir else out_dir / "evoked_trf" / "temp_self_ipu_onset"
    output_dir.mkdir(parents=True, exist_ok=True)

    events_path = out_dir / "events" / "events.parquet"
    index_path = out_dir / "dataset" / "index_resampled.parquet"
    events = pd.read_parquet(events_path)
    ds_index = pd.read_parquet(index_path)

    downsample_hz = float(cfg["evoked_trf"]["downsample_hz"])
    baseline = (float(args.baseline_start), float(args.baseline_end))

    pairs = _subject_pairs(events)
    LOGGER.info("Found %d subject pairs in events table.", len(pairs))

    wrote = 0
    subject_evokeds: list[mne.Evoked] = []
    for dyad_id, subject_id in pairs:
        run_rows = _run_rows_for_subject(ds_index, dyad_id, subject_id)
        if run_rows.empty:
            LOGGER.warning("No dataset rows for %s/%s; skipping.", dyad_id, subject_id)
            continue

        epochs_all: list[mne.Epochs] = []
        for row in run_rows.itertuples(index=False):
            run = int(row.run_num)
            eeg_path = Path(str(row.eeg_path)).expanduser()
            if not eeg_path.exists():
                LOGGER.warning("Missing EEG file for %s/%s run %d: %s", dyad_id, subject_id, run, eeg_path)
                continue

            onset_times_s = _events_for_run(events, dyad_id, subject_id, run)
            if onset_times_s.size == 0:
                continue

            raw = mne.io.read_raw(eeg_path, preload=True, verbose="ERROR")
            raw.pick("eeg")
            if not np.isclose(float(raw.info["sfreq"]), downsample_hz, atol=1e-9):
                raw.resample(downsample_hz, npad="auto", verbose="ERROR")

            epochs = _make_epochs(
                raw=raw,
                onset_times_s=onset_times_s,
                tmin=float(args.tmin),
                tmax=float(args.tmax),
                baseline=baseline,
            )
            if epochs is not None and len(epochs) > 0:
                epochs_all.append(epochs)

        if not epochs_all:
            LOGGER.warning("No usable epochs for %s/%s; skipping.", dyad_id, subject_id)
            continue

        if len(epochs_all) == 1:
            merged = epochs_all[0]
        else:
            merged = mne.concatenate_epochs(epochs_all, add_offset=True, verbose=False)

        evoked = merged.average()
        evoked.comment = "self_onset"
        subject_evokeds.append(evoked.copy())

        subj_dir = output_dir / dyad_id / subject_id
        subj_dir.mkdir(parents=True, exist_ok=True)
        evoked_path = subj_dir / "self_onset-ave.fif"
        evoked.save(evoked_path, overwrite=True)

        meta = pd.DataFrame(
            [
                {
                    "dyad_id": dyad_id,
                    "subject_id": subject_id,
                    "n_epochs": int(len(merged)),
                    "sfreq_hz": float(evoked.info["sfreq"]),
                    "tmin": float(args.tmin),
                    "tmax": float(args.tmax),
                    "baseline_start": float(args.baseline_start),
                    "baseline_end": float(args.baseline_end),
                    "path": str(evoked_path),
                }
            ]
        )
        meta.to_csv(subj_dir / "meta.csv", index=False)
        wrote += 1

    if subject_evokeds:
        grand = mne.grand_average(subject_evokeds, interpolate_bads=False, drop_bads=False)
        grand.comment = "self_onset_grand_average"
        grand_path = output_dir / "grand_average_self_onset-ave.fif"
        grand.save(grand_path, overwrite=True)

        grand_meta = pd.DataFrame(
            [
                {
                    "n_subjects": int(len(subject_evokeds)),
                    "sfreq_hz": float(grand.info["sfreq"]),
                    "tmin": float(args.tmin),
                    "tmax": float(args.tmax),
                    "baseline_start": float(args.baseline_start),
                    "baseline_end": float(args.baseline_end),
                    "path": str(grand_path),
                }
            ]
        )
        grand_meta.to_csv(output_dir / "grand_average_meta.csv", index=False)
        LOGGER.info("Wrote grand average: %s", grand_path)
    else:
        LOGGER.warning("No subject evokeds were created; skipping grand average.")

    LOGGER.info("Wrote %d subject evoked files under %s", wrote, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
