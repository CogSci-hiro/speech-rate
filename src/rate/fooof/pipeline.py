from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from rate.fooof.aggregate import aggregate_to_rois, prepare_ddc_channel_table
from rate.fooof.features import extract_band_features
from rate.fooof.fit import fit_fooof_to_window
from rate.fooof.io import (
    write_channel_ddc_table,
    write_qc_summary,
    write_settings,
    write_trial_psd_spectra,
    write_subject_psd_spectra,
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


class _NoOpProgressBar:
    def __iter__(self):
        return iter(())

    def close(self) -> None:
        return


@dataclass(frozen=True)
class SubjectRunFooofResult:
    trial_channels: pd.DataFrame
    trial_psd_spectra: pd.DataFrame
    run_psd_spectra: pd.DataFrame


def _is_truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def _assert_non_resampled_eeg_inputs(dataset_index: pd.DataFrame) -> None:
    if "eeg_is_resampled" in dataset_index.columns and dataset_index["eeg_is_resampled"].map(_is_truthy).any():
        raise ValueError(
            "FOOOF pipeline received resampled EEG rows (eeg_is_resampled=True). "
            "FOOOF must run on non-downsampled EEG inputs."
        )
    if "eeg_path" in dataset_index.columns:
        eeg_paths = dataset_index["eeg_path"].astype(str)
        if eeg_paths.str.contains(r"(?:^|[/\\\\])eeg_resampled(?:[/\\\\]|$)", regex=True, na=False).any():
            raise ValueError(
                "FOOOF pipeline received eeg_resampled file paths. "
                "FOOOF must run on non-downsampled EEG inputs."
            )


def _finite_float_or_none(value: object) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _serialize_peak_params_json(peaks: object) -> str:
    if not isinstance(peaks, list) or not peaks:
        return "[]"

    rows: list[dict[str, float | None]] = []
    for peak in peaks:
        if not isinstance(peak, dict):
            continue
        rows.append(
            {
                "center_frequency": _finite_float_or_none(peak.get("center_frequency")),
                "power": _finite_float_or_none(peak.get("power")),
                "bandwidth": _finite_float_or_none(peak.get("bandwidth")),
            }
        )
    return json.dumps(rows, sort_keys=True, separators=(",", ":"))


def _window_progress(iterable, *, enabled: bool, desc: str, total: int | None = None):
    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm  # type: ignore
    except Exception:
        return iterable
    return tqdm(
        iterable,
        total=total,
        desc=desc,
        unit="chanwin",
        leave=False,
        ascii=True,
        dynamic_ncols=True,
    )


def _outer_progress(iterable, *, enabled: bool, total: int | None = None):
    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm  # type: ignore
    except Exception:
        return iterable
    return tqdm(
        iterable,
        total=total,
        desc="fooof subject-runs",
        unit="subrun",
        leave=True,
        ascii=True,
        dynamic_ncols=True,
    )


@contextmanager
def _joblib_outer_progress(*, enabled: bool, total: int | None = None):
    """Attach a tqdm progress bar to joblib task completion callbacks."""
    if not enabled:
        yield
        return
    try:
        from tqdm.auto import tqdm  # type: ignore
        import joblib.parallel as joblib_parallel
    except Exception:
        yield
        return

    pbar = tqdm(
        total=total,
        desc="fooof subject-runs",
        unit="subrun",
        leave=True,
        ascii=True,
        dynamic_ncols=True,
    )
    original_cb = joblib_parallel.BatchCompletionCallBack

    class _TqdmBatchCompletionCallBack(original_cb):  # type: ignore[misc, valid-type]
        def __call__(self, *args, **kwargs):
            try:
                pbar.update(getattr(self, "batch_size", 1))
            except Exception:
                pbar.update(1)
            return super().__call__(*args, **kwargs)

    joblib_parallel.BatchCompletionCallBack = _TqdmBatchCompletionCallBack  # type: ignore[assignment]
    try:
        yield
    finally:
        joblib_parallel.BatchCompletionCallBack = original_cb  # type: ignore[assignment]
        pbar.close()


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
    _assert_non_resampled_eeg_inputs(dataset_index)


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
        by=["subject_id", "run", "event_id", "channel"],
        kind="mergesort",
    ).reset_index(drop=True)
    return df


def _finalize_channel_concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    nonempty = [df for df in frames if not df.empty]
    if not nonempty:
        return pd.DataFrame(columns=CANONICAL_CHANNEL_COLUMNS)
    df = pd.concat(nonempty, ignore_index=True)
    for col in CANONICAL_CHANNEL_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    ordered_cols = [*CANONICAL_CHANNEL_COLUMNS, *[c for c in df.columns if c not in CANONICAL_CHANNEL_COLUMNS]]
    df = df.loc[:, ordered_cols]
    df = df.sort_values(
        by=["subject_id", "run", "event_id", "channel"],
        kind="mergesort",
    ).reset_index(drop=True)
    return df


def _empty_run_psd_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "dyad_id",
            "subject_id",
            "subject_key",
            "run",
            "freq_hz",
            "psd_raw",
            "psd_aperiodic",
            "psd_periodic",
            "n_windows",
        ]
    )


def _empty_trial_psd_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "dyad_id",
            "subject_id",
            "subject_key",
            "run",
            "event_id",
            "trial_key",
            "freq_hz",
            "psd_raw",
            "psd_aperiodic",
            "psd_periodic",
            "n_channels",
            "exponent",
            "offset",
            "theta_power",
            "theta_cf",
        ]
    )


def _finalize_run_psd_concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    nonempty = [df for df in frames if not df.empty]
    if not nonempty:
        return _empty_run_psd_df()
    df = pd.concat(nonempty, ignore_index=True)
    return df.sort_values(["subject_key", "run", "freq_hz"], kind="mergesort").reset_index(drop=True)


def _finalize_trial_psd_concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    nonempty = [df for df in frames if not df.empty]
    if not nonempty:
        return _empty_trial_psd_df()
    df = pd.concat(nonempty, ignore_index=True)
    return df.sort_values(["subject_key", "run", "event_id", "freq_hz"], kind="mergesort").reset_index(drop=True)


def _aggregate_subject_psd_spectra(
    *,
    run_psd_df: pd.DataFrame,
    trial_channels_df: pd.DataFrame,
) -> pd.DataFrame:
    if run_psd_df.empty:
        return pd.DataFrame(
            columns=[
                "dyad_id",
                "subject_id",
                "subject_key",
                "freq_hz",
                "psd_raw",
                "psd_aperiodic",
                "psd_periodic",
                "exponent",
                "offset",
                "theta_power",
                "theta_cf",
            ]
        )

    run_df = run_psd_df.copy()
    for col in ("freq_hz", "psd_raw", "psd_aperiodic", "psd_periodic", "n_windows"):
        run_df[col] = pd.to_numeric(run_df[col], errors="coerce")
    run_df = run_df.dropna(subset=["subject_key", "freq_hz", "psd_raw", "psd_aperiodic", "psd_periodic"])
    if run_df.empty:
        return pd.DataFrame()

    # Weighted subject average across runs using contributing channel-window counts.
    run_df["n_windows"] = pd.to_numeric(run_df["n_windows"], errors="coerce").fillna(0).clip(lower=0)
    run_df["w_raw"] = run_df["psd_raw"] * run_df["n_windows"]
    run_df["w_ap"] = run_df["psd_aperiodic"] * run_df["n_windows"]
    run_df["w_per"] = run_df["psd_periodic"] * run_df["n_windows"]
    subj_psd = (
        run_df.groupby(["dyad_id", "subject_id", "subject_key", "freq_hz"], as_index=False)
        .agg(
            sum_w=("n_windows", "sum"),
            w_raw=("w_raw", "sum"),
            w_ap=("w_ap", "sum"),
            w_per=("w_per", "sum"),
        )
        .sort_values(["subject_key", "freq_hz"], kind="mergesort")
        .reset_index(drop=True)
    )
    valid_w = subj_psd["sum_w"] > 0
    subj_psd = subj_psd.loc[valid_w].copy()
    if subj_psd.empty:
        return pd.DataFrame()
    subj_psd["psd_raw"] = subj_psd["w_raw"] / subj_psd["sum_w"]
    subj_psd["psd_aperiodic"] = subj_psd["w_ap"] / subj_psd["sum_w"]
    subj_psd["psd_periodic"] = subj_psd["w_per"] / subj_psd["sum_w"]

    subj_targets = (
        trial_channels_df.assign(
            subject_key=lambda d: d["dyad_id"].astype(str) + "::" + d["subject_id"].astype(str),
            exponent=lambda d: pd.to_numeric(d["exponent"], errors="coerce"),
            offset=lambda d: pd.to_numeric(d["offset"], errors="coerce"),
            theta_power=lambda d: pd.to_numeric(d["theta_power"], errors="coerce"),
            theta_cf=lambda d: pd.to_numeric(d["theta_cf"], errors="coerce"),
        )
        .groupby(["dyad_id", "subject_id", "subject_key"], as_index=False)
        .agg(
            exponent=("exponent", "mean"),
            offset=("offset", "mean"),
            theta_power=("theta_power", "mean"),
            theta_cf=("theta_cf", "mean"),
        )
    )
    out = subj_psd.merge(
        subj_targets,
        on=["dyad_id", "subject_id", "subject_key"],
        how="left",
        validate="many_to_one",
    )
    return out[
        [
            "dyad_id",
            "subject_id",
            "subject_key",
            "freq_hz",
            "psd_raw",
            "psd_aperiodic",
            "psd_periodic",
            "exponent",
            "offset",
            "theta_power",
            "theta_cf",
        ]
    ].sort_values(["subject_key", "freq_hz"], kind="mergesort").reset_index(drop=True)


def _run_to_int(run_value: object) -> int:
    text = str(run_value).strip()
    try:
        return int(float(text))
    except ValueError as exc:
        raise ValueError(f"Run value must be integer-like for FOOOF pipeline: {run_value!r}") from exc


def process_subject_run(
    dyad_id: str,
    subject_id: str,
    run: int,
    dataset_index: pd.DataFrame,
    events_df: pd.DataFrame,
    spec: FooofSpec,
    config: dict,
) -> SubjectRunFooofResult:
    """Process one (dyad_id, subject_id, run) block and return channel rows + run-level mean PSD."""
    np.random.seed(resolve_random_seed(config))

    run_key = _run_token(run)
    ds_rows = dataset_index.loc[
        (dataset_index["dyad_id"].astype(str) == str(dyad_id))
        & (dataset_index["subject_id"].astype(str) == str(subject_id))
        & (dataset_index["run"].map(_run_token) == run_key)
    ].copy()
    if ds_rows.empty:
        return SubjectRunFooofResult(
            trial_channels=pd.DataFrame(columns=CANONICAL_CHANNEL_COLUMNS),
            trial_psd_spectra=_empty_trial_psd_df(),
            run_psd_spectra=_empty_run_psd_df(),
        )

    worker_sort_cols = [col for col in ("dyad_id", "subject_id", "run", "task") if col in ds_rows.columns]
    ds_rows = ds_rows.sort_values(by=worker_sort_cols, kind="mergesort").reset_index(drop=True)

    records: list[dict[str, Any]] = []
    raw_cache: dict[str, mne.io.BaseRaw] = {}
    psd_sum_raw: np.ndarray | None = None
    psd_sum_ap: np.ndarray | None = None
    psd_sum_per: np.ndarray | None = None
    psd_freqs: np.ndarray | None = None
    psd_count = 0
    dyad_for_run: str | None = str(dyad_id)
    event_psd_accum: dict[str, dict[str, Any]] = {}

    for _, ds_row in ds_rows.iterrows():
        eeg_path = Path(str(ds_row["eeg_path"])).expanduser()
        if not eeg_path.exists():
            continue

        row_events = _events_for_dataset_row(events_df, ds_row)
        if row_events.empty:
            continue

        eeg_key = str(eeg_path)
        raw = raw_cache.get(eeg_key)
        if raw is None:
            raw = _read_raw_eeg(eeg_path)
            raw_cache[eeg_key] = raw

        sfreq = float(raw.info["sfreq"])
        run_int = _run_to_int(ds_row["run"])
        n_eeg_channels = len(mne.pick_types(raw.info, eeg=True, exclude=[]))
        boundary_excluded_events = _boundary_excluded_event_count(row_events, raw, spec)
        valid_events = max(0, int(len(row_events) - boundary_excluded_events))
        total_chan_windows = int(valid_events * n_eeg_channels)

        onset_by_event = (
            row_events.loc[:, ["event_id", "t_onset"]]
            .drop_duplicates(subset=["event_id"])
            .set_index("event_id")["t_onset"]
            .to_dict()
        )

        window_iter = iter_event_windows(
            raw=raw,
            events_df=row_events,
            spec=spec,
            subject_id=str(ds_row["subject_id"]),
            run=run_int,
        )
        window_iter = _window_progress(
            window_iter,
            enabled=bool(spec.n_jobs == 1),
            desc=f"fooof {subject_id}/run-{run_int}",
            total=total_chan_windows,
        )
        for _, window_run, event_id, channel_name, window_data in window_iter:
            event_onset_s = float(onset_by_event[str(event_id)])
            window_start, window_end = _window_bounds_from_event(event_onset_s, spec)

            fit_result = fit_fooof_to_window(window_data=window_data, sfreq=sfreq, spec=spec)
            band_features = extract_band_features(fit_result, spec)
            freqs_arr = fit_result.get("freqs_hz")
            raw_psd_arr = fit_result.get("psd_raw")
            ap_psd_arr = fit_result.get("psd_aperiodic")
            per_psd_arr = fit_result.get("psd_periodic")
            if (
                freqs_arr is not None
                and raw_psd_arr is not None
                and ap_psd_arr is not None
                and per_psd_arr is not None
            ):
                freqs_vec = np.asarray(freqs_arr, dtype=float).reshape(-1)
                raw_vec = np.asarray(raw_psd_arr, dtype=float).reshape(-1)
                ap_vec = np.asarray(ap_psd_arr, dtype=float).reshape(-1)
                per_vec = np.asarray(per_psd_arr, dtype=float).reshape(-1)
                if (
                    freqs_vec.size > 0
                    and raw_vec.shape == freqs_vec.shape
                    and ap_vec.shape == freqs_vec.shape
                    and per_vec.shape == freqs_vec.shape
                    and np.all(np.isfinite(freqs_vec))
                ):
                    if psd_freqs is None:
                        psd_freqs = freqs_vec.copy()
                        psd_sum_raw = np.zeros_like(raw_vec, dtype=float)
                        psd_sum_ap = np.zeros_like(ap_vec, dtype=float)
                        psd_sum_per = np.zeros_like(per_vec, dtype=float)
                    if psd_freqs is not None and np.allclose(psd_freqs, freqs_vec, atol=1e-9, rtol=1e-6):
                        psd_sum_raw = psd_sum_raw + raw_vec  # type: ignore[operator]
                        psd_sum_ap = psd_sum_ap + ap_vec  # type: ignore[operator]
                        psd_sum_per = psd_sum_per + per_vec  # type: ignore[operator]
                        psd_count += 1
                        dyad_for_run = str(ds_row["dyad_id"])
                        ev_key = str(event_id)
                        rec = event_psd_accum.get(ev_key)
                        if rec is None:
                            rec = {
                                "freq_hz": freqs_vec.copy(),
                                "sum_raw": np.zeros_like(raw_vec, dtype=float),
                                "sum_ap": np.zeros_like(ap_vec, dtype=float),
                                "sum_per": np.zeros_like(per_vec, dtype=float),
                                "n_channels": 0,
                            }
                            event_psd_accum[ev_key] = rec
                        if np.allclose(np.asarray(rec["freq_hz"], dtype=float), freqs_vec, atol=1e-9, rtol=1e-6):
                            rec["sum_raw"] = np.asarray(rec["sum_raw"], dtype=float) + raw_vec
                            rec["sum_ap"] = np.asarray(rec["sum_ap"], dtype=float) + ap_vec
                            rec["sum_per"] = np.asarray(rec["sum_per"], dtype=float) + per_vec
                            rec["n_channels"] = int(rec["n_channels"]) + 1

            records.append(
                {
                    "dyad_id": str(ds_row["dyad_id"]),
                    "subject_id": str(subject_id),
                    "run": int(window_run),
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
                    # Additive trial-level metadata for later model-spectrum reconstruction.
                    "fooof_aperiodic_mode": str(spec.aperiodic_mode),
                    "fooof_peak_params_json": _serialize_peak_params_json(fit_result.get("peaks", [])),
                }
            )

    trial_df = _finalize_channel_df(records)
    trial_psd_df = _empty_trial_psd_df()
    if event_psd_accum and not trial_df.empty:
        target_by_trial = (
            trial_df.assign(
                exponent=lambda d: pd.to_numeric(d["exponent"], errors="coerce"),
                offset=lambda d: pd.to_numeric(d["offset"], errors="coerce"),
                theta_power=lambda d: pd.to_numeric(d["theta_power"], errors="coerce"),
                theta_cf=lambda d: pd.to_numeric(d["theta_cf"], errors="coerce"),
            )
            .groupby(["dyad_id", "subject_id", "run", "event_id"], as_index=False)
            .agg(
                exponent=("exponent", "mean"),
                offset=("offset", "mean"),
                theta_power=("theta_power", "mean"),
                theta_cf=("theta_cf", "mean"),
            )
        )
        rows: list[pd.DataFrame] = []
        for ev_key, rec in event_psd_accum.items():
            n_channels_ev = int(rec.get("n_channels", 0))
            if n_channels_ev <= 0:
                continue
            freq_ev = np.asarray(rec["freq_hz"], dtype=float)
            frame = pd.DataFrame(
                {
                    "dyad_id": str(dyad_for_run) if dyad_for_run is not None else str(dyad_id),
                    "subject_id": str(subject_id),
                    "subject_key": (f"{dyad_for_run}::{subject_id}" if dyad_for_run is not None else str(subject_id)),
                    "run": int(run),
                    "event_id": str(ev_key),
                    "trial_key": f"{int(run)}::{str(ev_key)}",
                    "freq_hz": freq_ev,
                    "psd_raw": np.asarray(rec["sum_raw"], dtype=float) / float(n_channels_ev),
                    "psd_aperiodic": np.asarray(rec["sum_ap"], dtype=float) / float(n_channels_ev),
                    "psd_periodic": np.asarray(rec["sum_per"], dtype=float) / float(n_channels_ev),
                    "n_channels": int(n_channels_ev),
                }
            )
            rows.append(frame)
        if rows:
            trial_psd_df = pd.concat(rows, ignore_index=True)
            target_by_trial_x = target_by_trial.copy()
            target_by_trial_x["event_id"] = target_by_trial_x["event_id"].astype(str)
            trial_psd_df = trial_psd_df.merge(
                target_by_trial_x,
                on=["dyad_id", "subject_id", "run", "event_id"],
                how="left",
                validate="many_to_one",
            )
    run_psd_df = _empty_run_psd_df()
    if psd_count > 0 and psd_freqs is not None and psd_sum_raw is not None and psd_sum_ap is not None and psd_sum_per is not None:
        run_psd_df = pd.DataFrame(
            {
                "dyad_id": str(dyad_for_run) if dyad_for_run is not None else "",
                "subject_id": str(subject_id),
                "subject_key": (f"{dyad_for_run}::{subject_id}" if dyad_for_run is not None else str(subject_id)),
                "run": int(run),
                "freq_hz": np.asarray(psd_freqs, dtype=float),
                "psd_raw": np.asarray(psd_sum_raw, dtype=float) / float(psd_count),
                "psd_aperiodic": np.asarray(psd_sum_ap, dtype=float) / float(psd_count),
                "psd_periodic": np.asarray(psd_sum_per, dtype=float) / float(psd_count),
                "n_windows": int(psd_count),
            }
        )
    return SubjectRunFooofResult(
        trial_channels=trial_df,
        trial_psd_spectra=trial_psd_df,
        run_psd_spectra=run_psd_df,
    )


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

    qc_meta_rows: list[dict[str, Any]] = []

    ds_rows = ds_rows.sort_values(by=["dyad_id", "subject_id", "run"], kind="mergesort").reset_index(drop=True)
    # Main-process QC/meta accounting remains deterministic and avoids worker logging noise.
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

    subject_run_pairs = sorted(
        {
            (str(row.dyad_id), str(row.subject_id), _run_to_int(row.run))
            for row in ds_rows.itertuples(index=False)
            if pd.notna(row.dyad_id) and pd.notna(row.subject_id)
        },
        key=lambda item: (item[0], item[1], item[2]),
    )

    if spec.n_jobs == 1:
        serial_pairs = _outer_progress(
            subject_run_pairs,
            enabled=True,
            total=len(subject_run_pairs),
        )
        results = [
            process_subject_run(
                dyad_id=dyad_id,
                subject_id=subject_id,
                run=run,
                dataset_index=ds_rows,
                events_df=events_df,
                spec=spec,
                config=config,
            )
            for dyad_id, subject_id, run in serial_pairs
        ]
    else:
        with _joblib_outer_progress(enabled=True, total=len(subject_run_pairs)):
            results = Parallel(
                n_jobs=spec.n_jobs,
                backend="loky",
                verbose=0,
            )(
                delayed(process_subject_run)(
                    dyad_id=dyad_id,
                    subject_id=subject_id,
                    run=run,
                    dataset_index=ds_rows,
                    events_df=events_df,
                    spec=spec,
                    config=config,
                )
                for dyad_id, subject_id, run in subject_run_pairs
            )
        

    LOGGER.info("FOOOF post-processing 1/8: concatenating worker outputs")
    trial_channels_df = _finalize_channel_concat([r.trial_channels for r in results])
    trial_psd_df = _finalize_trial_psd_concat([r.trial_psd_spectra for r in results])
    run_psd_df = _finalize_run_psd_concat([r.run_psd_spectra for r in results])
    LOGGER.info(
        "FOOOF post-processing 1/8 complete: trial_channels=%d rows, trial_psd=%d rows, run_psd=%d rows",
        int(len(trial_channels_df)),
        int(len(trial_psd_df)),
        int(len(run_psd_df)),
    )
    if not trial_channels_df.empty:
        LOGGER.info("FOOOF post-processing 2/8: applying QC flags to channel rows")
        trial_channels_df = trial_channels_df.apply(apply_fit_qc, axis=1)
        LOGGER.info("FOOOF post-processing 2/8 complete")
    else:
        LOGGER.info("FOOOF post-processing 2/8: skipped QC (no trial channel rows)")

    LOGGER.info("FOOOF post-processing 3/8: writing trial-level channel table")
    write_trial_channels(trial_channels_df, out_dir)
    LOGGER.info("FOOOF post-processing 3/8 complete")

    LOGGER.info("FOOOF post-processing 4/8: aggregating + writing trial ROI table")
    trial_rois_df = aggregate_to_rois(trial_channels_df, spec)
    write_trial_rois(trial_rois_df, out_dir)
    LOGGER.info("FOOOF post-processing 4/8 complete: trial_rois=%d rows", int(len(trial_rois_df)))

    LOGGER.info("FOOOF post-processing 5/8: building + writing DDC channel table")
    ddc_channel_df = prepare_ddc_channel_table(trial_channels_df)
    write_channel_ddc_table(ddc_channel_df, out_dir)
    LOGGER.info("FOOOF post-processing 5/8 complete: ddc_channel=%d rows", int(len(ddc_channel_df)))

    LOGGER.info("FOOOF post-processing 6/8: writing trial PSD spectra table")
    write_trial_psd_spectra(trial_psd_df, out_dir)
    LOGGER.info("FOOOF post-processing 6/8 complete: trial_psd=%d rows", int(len(trial_psd_df)))

    LOGGER.info("FOOOF post-processing 7/8: aggregating + writing subject PSD spectra table")
    subject_psd_spectra_df = _aggregate_subject_psd_spectra(run_psd_df=run_psd_df, trial_channels_df=trial_channels_df)
    write_subject_psd_spectra(subject_psd_spectra_df, out_dir)
    LOGGER.info(
        "FOOOF post-processing 7/8 complete: subject_psd=%d rows",
        int(len(subject_psd_spectra_df)),
    )

    LOGGER.info("FOOOF post-processing 8/8: writing QC summary + settings")
    qc_summary_df = trial_channels_df.copy()
    if qc_meta_rows:
        meta_df = pd.DataFrame.from_records(qc_meta_rows)
        meta_csv_path = Path(out_dir) / "fooof" / "meta" / "qc_pipeline_counts.csv"
        meta_csv_path.parent.mkdir(parents=True, exist_ok=True)
        meta_df.to_csv(meta_csv_path, index=False)
    write_qc_summary(qc_summary_df, out_dir)
    write_settings(spec, out_dir)
    LOGGER.info("FOOOF post-processing 8/8 complete")
