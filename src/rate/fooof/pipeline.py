from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd

from rate.fooof.aggregate import aggregate_to_rois, prepare_ddc_channel_table
from rate.fooof.features import extract_band_features
from rate.fooof.fit import fit_fooof_to_window
from rate.fooof.io import (
    write_channel_ddc_table,
    write_qc_summary,
    write_roi_lmm_table,
    write_settings,
    write_trial_channels,
    write_trial_rois,
)
from rate.fooof.qc import apply_fit_qc
from rate.fooof.spec import FooofSpec, build_fooof_spec
from rate.fooof.windows import iter_event_windows
from rate.util import resolve_random_seed

LOGGER = logging.getLogger(__name__)

CANONICAL_CHANNEL_COLUMNS = [
    "dyad_id",
    "subject_id",
    "run",
    "event_id",
    "channel",
    "window_start",
    "window_end",
    "window_length",
    "offset",
    "exponent",
    "knee",
    "theta_power",
    "alpha_power",
    "beta_power",
    "theta_cf",
    "alpha_cf",
    "beta_cf",
    "r_squared",
    "error",
    "fit_success",
]


def _run_token(value: object) -> str:
    text = str(value).strip()
    try:
        return str(int(float(text)))
    except ValueError:
        return text


def _read_raw_eeg(path: Path) -> mne.io.BaseRaw:
    suffix = path.suffix.lower()
    if suffix == ".fif":
        return mne.io.read_raw_fif(path, preload=False, verbose="ERROR")
    if suffix == ".edf":
        return mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
    if suffix == ".bdf":
        return mne.io.read_raw_bdf(path, preload=False, verbose="ERROR")
    if suffix == ".vhdr":
        return mne.io.read_raw_brainvision(path, preload=False, verbose="ERROR")
    raise ValueError(f"Unsupported EEG file extension for FOOOF pipeline: {path}")


def _validate_inputs(dataset_index: pd.DataFrame, events_df: pd.DataFrame) -> None:
    ds_required = {"dyad_id", "subject_id", "run", "eeg_path"}
    ev_required = {"dyad_id", "subject_id", "run", "event_id", "t_onset"}
    ds_missing = ds_required - set(dataset_index.columns)
    ev_missing = ev_required - set(events_df.columns)
    if ds_missing:
        raise ValueError(f"dataset_index missing required columns: {sorted(ds_missing)}")
    if ev_missing:
        raise ValueError(f"events_df missing required columns: {sorted(ev_missing)}")


def _events_for_dataset_row(events_df: pd.DataFrame, ds_row: pd.Series) -> pd.DataFrame:
    run_key = _run_token(ds_row["run"])
    mask = (
        (events_df["dyad_id"].astype(str) == str(ds_row["dyad_id"]))
        & (events_df["subject_id"].astype(str) == str(ds_row["subject_id"]))
        & (events_df["run"].map(_run_token) == run_key)
    )
    if "task" in events_df.columns and "task" in ds_row.index and pd.notna(ds_row["task"]):
        mask = mask & (events_df["task"].astype(str) == str(ds_row["task"]))
    return events_df.loc[mask].copy()


def _window_bounds_from_event(event_onset_s: float, spec: FooofSpec) -> tuple[float, float]:
    return (float(event_onset_s + spec.window_tmin_s), float(event_onset_s + spec.window_tmax_s))


def _boundary_excluded_event_count(events_df: pd.DataFrame, raw: mne.io.BaseRaw, spec: FooofSpec) -> int:
    if events_df.empty:
        return 0
    sfreq = float(raw.info["sfreq"])
    excluded = 0
    for onset in pd.to_numeric(events_df["t_onset"], errors="coerce"):
        if not np.isfinite(onset):
            excluded += 1
            continue
        start = int(np.round((float(onset) + spec.window_tmin_s) * sfreq))
        stop = int(np.round((float(onset) + spec.window_tmax_s) * sfreq))
        if start < 0 or stop > raw.n_times or stop <= start:
            excluded += 1
    return excluded


def _finalize_channel_df(records: list[dict[str, Any]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=CANONICAL_CHANNEL_COLUMNS)
    df = pd.DataFrame.from_records(records)
    for col in CANONICAL_CHANNEL_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df = df.sort_values(
        by=["dyad_id", "subject_id", "run", "event_id", "channel"],
        kind="mergesort",
    ).reset_index(drop=True)
    return df


def run_fooof_pipeline(
    config: dict,
    dataset_index: pd.DataFrame,
    events_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Run the deterministic event-aligned FOOOF feature extraction pipeline."""
    np.random.seed(resolve_random_seed(config))
    _validate_inputs(dataset_index, events_df)
    spec = build_fooof_spec(config)

    if "status" in dataset_index.columns:
        ds_rows = dataset_index.loc[dataset_index["status"].astype(str).str.lower() == "ok"].copy()
    else:
        ds_rows = dataset_index.copy()

    records: list[dict[str, Any]] = []
    qc_meta_rows: list[dict[str, Any]] = []

    ds_rows = ds_rows.sort_values(by=["dyad_id", "subject_id", "run"], kind="mergesort").reset_index(drop=True)

    for _, ds_row in ds_rows.iterrows():
        eeg_path = Path(str(ds_row["eeg_path"])).expanduser()
        if not eeg_path.exists():
            qc_meta_rows.append(
                {
                    "dyad_id": str(ds_row["dyad_id"]),
                    "subject_id": str(ds_row["subject_id"]),
                    "run": _run_token(ds_row["run"]),
                    "qc_stage": "input",
                    "metric": "missing_eeg_file",
                    "value": 1,
                }
            )
            LOGGER.warning("Skipping missing EEG file: %s", eeg_path)
            continue

        row_events = _events_for_dataset_row(events_df, ds_row)
        if row_events.empty:
            qc_meta_rows.append(
                {
                    "dyad_id": str(ds_row["dyad_id"]),
                    "subject_id": str(ds_row["subject_id"]),
                    "run": _run_token(ds_row["run"]),
                    "qc_stage": "windows",
                    "metric": "events_for_row",
                    "value": 0,
                }
            )
            continue

        raw = _read_raw_eeg(eeg_path)
        sfreq = float(raw.info["sfreq"])
        run_int = int(float(ds_row["run"])) if str(ds_row["run"]).strip().replace(".", "", 1).isdigit() else int(ds_row["run"])

        boundary_excluded_events = _boundary_excluded_event_count(row_events, raw, spec)
        n_eeg_channels = len(mne.pick_types(raw.info, eeg=True, exclude=[]))
        qc_meta_rows.extend(
            [
                {
                    "dyad_id": str(ds_row["dyad_id"]),
                    "subject_id": str(ds_row["subject_id"]),
                    "run": _run_token(ds_row["run"]),
                    "qc_stage": "windows",
                    "metric": "events_for_row",
                    "value": int(len(row_events)),
                },
                {
                    "dyad_id": str(ds_row["dyad_id"]),
                    "subject_id": str(ds_row["subject_id"]),
                    "run": _run_token(ds_row["run"]),
                    "qc_stage": "windows",
                    "metric": "boundary_excluded_events",
                    "value": int(boundary_excluded_events),
                },
                {
                    "dyad_id": str(ds_row["dyad_id"]),
                    "subject_id": str(ds_row["subject_id"]),
                    "run": _run_token(ds_row["run"]),
                    "qc_stage": "windows",
                    "metric": "eeg_channels",
                    "value": int(n_eeg_channels),
                },
            ]
        )

        onset_by_event = (
            row_events.loc[:, ["event_id", "t_onset"]]
            .drop_duplicates(subset=["event_id"])
            .set_index("event_id")["t_onset"]
            .to_dict()
        )

        for subject_id, run, event_id, channel_name, window_data in iter_event_windows(
            raw=raw,
            events_df=row_events,
            spec=spec,
            subject_id=str(ds_row["subject_id"]),
            run=run_int,
        ):
            event_onset_s = float(onset_by_event[str(event_id)])
            window_start, window_end = _window_bounds_from_event(event_onset_s, spec)

            fit_result = fit_fooof_to_window(window_data=window_data, sfreq=sfreq, spec=spec)
            band_features = extract_band_features(fit_result, spec)

            records.append(
                {
                    "dyad_id": str(ds_row["dyad_id"]),
                    "subject_id": str(subject_id),
                    "run": int(run),
                    "event_id": str(event_id),
                    "channel": str(channel_name),
                    "window_start": float(window_start),
                    "window_end": float(window_end),
                    "window_length": float(spec.window_length_s),
                    "offset": float(fit_result["offset"]),
                    "exponent": float(fit_result["exponent"]),
                    "knee": float(fit_result["knee"]) if pd.notna(fit_result["knee"]) else np.nan,
                    "theta_power": float(band_features["theta_power"]),
                    "alpha_power": float(band_features["alpha_power"]),
                    "beta_power": float(band_features["beta_power"]),
                    "theta_cf": float(band_features["theta_cf"]) if pd.notna(band_features["theta_cf"]) else np.nan,
                    "alpha_cf": float(band_features["alpha_cf"]) if pd.notna(band_features["alpha_cf"]) else np.nan,
                    "beta_cf": float(band_features["beta_cf"]) if pd.notna(band_features["beta_cf"]) else np.nan,
                    "r_squared": float(fit_result["r_squared"]),
                    "error": float(fit_result["error"]),
                    "fit_success": bool(fit_result["fit_success"]),
                }
            )

    trial_channels_df = _finalize_channel_df(records)
    if not trial_channels_df.empty:
        trial_channels_df = trial_channels_df.apply(apply_fit_qc, axis=1)

    write_trial_channels(trial_channels_df, out_dir)

    trial_rois_df = aggregate_to_rois(trial_channels_df, spec)
    write_trial_rois(trial_rois_df, out_dir)

    ddc_channel_df = prepare_ddc_channel_table(trial_channels_df)
    write_channel_ddc_table(ddc_channel_df, out_dir)
    write_roi_lmm_table(trial_rois_df, out_dir)

    qc_summary_df = trial_channels_df.copy()
    if qc_meta_rows:
        meta_df = pd.DataFrame.from_records(qc_meta_rows)
        meta_csv_path = Path(out_dir) / "fooof" / "meta" / "qc_pipeline_counts.csv"
        meta_csv_path.parent.mkdir(parents=True, exist_ok=True)
        meta_df.to_csv(meta_csv_path, index=False)
    write_qc_summary(qc_summary_df, out_dir)
    write_settings(spec, out_dir)
