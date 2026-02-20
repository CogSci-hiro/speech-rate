#!/usr/bin/env python3
"""Dump evoked TRF event-derived feature matrices from the current pipeline."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from rate.config import load_and_validate_config
from rate.evoked_trf.fit import build_design_matrix, load_eeg_run
from rate.evoked_trf.score import parse_evoked_trf_config, select_eeg_path, validate_events_table
from rate.evoked_trf.spec import get_models

LOGGER = logging.getLogger("tmp_dump_evoked_trf_features")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump per-run evoked TRF feature matrices.")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config YAML.")
    parser.add_argument("--schema", default="config/schema.json", help="Path to config schema.")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Optional override for cfg['paths']['out_dir'].",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Default: <out_dir>/evoked_trf/temp_features",
    )
    parser.add_argument("--picks", default="eeg", help="MNE picks passed to load_eeg_run().")
    return parser.parse_args()


def _norm_str(v: object) -> str:
    return str(v).strip()


def _subject_run_table(events: pd.DataFrame) -> pd.DataFrame:
    tmp = events.copy()
    tmp["dyad_id"] = tmp["dyad_id"].map(_norm_str)
    tmp["subject_id"] = tmp["subject_id"].map(_norm_str)
    tmp["run"] = pd.to_numeric(tmp["run"], errors="coerce")
    tmp = tmp.dropna(subset=["run"])
    tmp["run"] = tmp["run"].astype(int)
    rows = (
        tmp[["dyad_id", "subject_id", "run"]]
        .drop_duplicates()
        .sort_values(["dyad_id", "subject_id", "run"], kind="mergesort")
        .reset_index(drop=True)
    )
    return rows


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parse_args()

    cfg = load_and_validate_config(Path(args.config), schema_path=Path(args.schema))
    out_dir = Path(args.out_dir) if args.out_dir else Path(cfg["paths"]["out_dir"])
    output_dir = Path(args.output_dir) if args.output_dir else out_dir / "evoked_trf" / "temp_features"
    output_dir.mkdir(parents=True, exist_ok=True)

    trf_cfg = parse_evoked_trf_config(cfg)
    events_path = out_dir / "events" / "events.parquet"
    index_path = out_dir / "dataset" / "index_resampled.parquet"

    events = pd.read_parquet(events_path)
    events = validate_events_table(events)
    ds_index = pd.read_parquet(index_path)

    models = list(get_models())
    model_names = [m.model_name for m in models]
    runs_df = _subject_run_table(events)
    LOGGER.info("Found %d dyad/subject/run rows.", len(runs_df))

    metadata_rows: list[dict[str, object]] = []
    for row in runs_df.itertuples(index=False):
        dyad_id = str(row.dyad_id)
        subject = str(row.subject_id)
        run = int(row.run)

        eeg_path = select_eeg_path(ds_index, subject=subject, dyad_id=dyad_id, run=run)
        run_data = load_eeg_run(eeg_path, downsample_hz=trf_cfg.downsample_hz, picks=args.picks)
        sfreq_hz = float(run_data.sfreq_hz)
        n_samples = int(run_data.y.shape[0])

        ev_run = events.loc[
            (events["dyad_id"].astype(str).str.strip() == dyad_id)
            & (events["subject_id"].astype(str).str.strip() == subject)
            & (pd.to_numeric(events["run"], errors="coerce") == run)
        ].copy()

        run_dir = output_dir / dyad_id / subject / f"run-{run:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)

        arrays_for_npz: dict[str, np.ndarray] = {}
        feature_names_out: dict[str, list[str]] = {}
        for model in models:
            X, feature_names = build_design_matrix(
                events=ev_run,
                model=model,
                sfreq_hz=sfreq_hz,
                n_samples=n_samples,
                sample_t0_s=float(run_data.times_s[0]),
                return_feature_names=True,
            )
            arrays_for_npz[model.model_name] = X
            feature_names_out[model.model_name] = feature_names

            per_model_path = run_dir / f"{model.model_name}__X.npy"
            np.save(per_model_path, X)
            metadata_rows.append(
                {
                    "dyad_id": dyad_id,
                    "subject_id": subject,
                    "run": run,
                    "model": model.model_name,
                    "n_samples": n_samples,
                    "n_regressors": int(X.shape[1]),
                    "sfreq_hz": sfreq_hz,
                    "feature_names": "|".join(feature_names),
                    "path_npy": str(per_model_path),
                    "eeg_path": str(eeg_path),
                }
            )

        np.savez_compressed(run_dir / "features_by_model.npz", **arrays_for_npz)
        (run_dir / "feature_names.json").write_text(
            json.dumps(feature_names_out, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    meta_df = pd.DataFrame(metadata_rows)
    meta_df = meta_df.sort_values(
        ["dyad_id", "subject_id", "run", "model"],
        kind="mergesort",
    ).reset_index(drop=True)
    meta_df.to_csv(output_dir / "feature_index.csv", index=False)

    summary = {
        "n_rows": int(len(meta_df)),
        "n_runs": int(len(runs_df)),
        "models": model_names,
        "output_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    LOGGER.info("Wrote feature dump: %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
