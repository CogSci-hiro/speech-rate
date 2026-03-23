from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from rate.config import get_project_downsample_hz, load_yaml_config
from rate.eeg.io import read_raw_eeg
from rate.evoked_trf.fit import _configure_mne_logging
from rate.evoked_trf.score import select_eeg_path
from rate.induced_trf.aperiodic import SlidingAperiodicConfig, compute_sliding_aperiodic


def parse_sliding_fooof_config(cfg: Mapping[str, Any]) -> SlidingAperiodicConfig:
    root = dict(cfg)
    sliding = root.get("fooof_sliding")
    if sliding is None and isinstance(root.get("fooof"), Mapping):
        sliding = root["fooof"].get("fooof_sliding")
    if sliding is None:
        raise ValueError("Missing fooof_sliding config")
    if not isinstance(sliding, Mapping):
        raise ValueError("fooof_sliding must be a mapping")

    required = ("window_s", "fit_range_hz")
    missing = [key for key in required if key not in sliding]
    if missing:
        raise ValueError(f"fooof_sliding must define window_s and fit_range_hz (missing: {missing})")

    fit_range_hz = sliding.get("fit_range_hz")
    if not isinstance(fit_range_hz, (list, tuple)) or len(fit_range_hz) != 2:
        raise ValueError("fooof_sliding.fit_range_hz must be [fmin, fmax]")
    peak_width_limits_hz = sliding.get("peak_width_limits_hz", [1.0, 12.0])
    if not isinstance(peak_width_limits_hz, (list, tuple)) or len(peak_width_limits_hz) != 2:
        raise ValueError("fooof_sliding.peak_width_limits_hz must be [fmin, fmax]")
    downsample_hz = get_project_downsample_hz(cfg)
    spectrum_method = str(sliding.get("spectrum_method", sliding.get("method", "welch"))).strip().lower()
    if spectrum_method not in {"welch", "multitaper"}:
        raise ValueError("fooof_sliding.spectrum_method must be 'welch' or 'multitaper'")

    return SlidingAperiodicConfig(
        window_s=float(sliding["window_s"]),
        step_s=float(1.0 / float(downsample_hz)),
        fit_range_hz=(float(fit_range_hz[0]), float(fit_range_hz[1])),
        edge_drop=str(sliding.get("edge_drop", "none")),
        spectrum_method=spectrum_method,
        welch_overlap=float(sliding.get("welch_overlap", 0.0)),
        multitaper_bandwidth_hz=(
            None
            if sliding.get("multitaper_bandwidth_hz") is None
            else float(sliding.get("multitaper_bandwidth_hz"))
        ),
        multitaper_adaptive=bool(sliding.get("multitaper_adaptive", False)),
        multitaper_low_bias=bool(sliding.get("multitaper_low_bias", True)),
        aperiodic_mode=str(sliding.get("aperiodic_mode", "fixed")).strip().lower(),
        peak_width_limits_hz=(float(peak_width_limits_hz[0]), float(peak_width_limits_hz[1])),
        max_n_peaks=int(sliding.get("max_n_peaks", 6)),
        min_peak_height=float(sliding.get("min_peak_height", 0.0)),
        peak_threshold=(float(sliding["peak_threshold"]) if ("peak_threshold" in sliding) else None),
    )


def load_sliding_fooof_config(config_path: Path | str) -> SlidingAperiodicConfig:
    return parse_sliding_fooof_config(load_yaml_config(config_path))


def subject_aperiodic_dir(*, out_dir: Path, band: str, dyad_id: str, subject: str) -> Path:
    return out_dir / f"induced_trf_{band}_residual" / "subjects" / str(dyad_id) / str(subject) / "targets" / "aperiodic"


def run_aperiodic_path(*, out_dir: Path, band: str, dyad_id: str, subject: str, run: int) -> Path:
    return subject_aperiodic_dir(out_dir=out_dir, band=band, dyad_id=dyad_id, subject=subject) / f"run-{int(run):02d}.parquet"


def done_marker_path(*, out_dir: Path, band: str, dyad_id: str, subject: str) -> Path:
    return subject_aperiodic_dir(out_dir=out_dir, band=band, dyad_id=dyad_id, subject=subject) / "_done.txt"


def write_subject_sliding_fooof_outputs(
    *,
    sliding_cfg: SlidingAperiodicConfig,
    out_dir: Path,
    dyad_id: str,
    subject: str,
    band: str,
    band_hz: tuple[float, float],
    dataset_index: pd.DataFrame,
    picks: str | list[str] = "eeg",
) -> list[Path]:
    ds = dataset_index.loc[
        (dataset_index["dyad_id"].astype(str).str.strip() == str(dyad_id).strip())
        & (dataset_index["subject_id"].astype(str).str.strip() == str(subject).strip())
    ].copy()
    if "status" in ds.columns:
        ds = ds.loc[ds["status"].astype(str).str.strip().str.lower() == "ok"].copy()
    ds = ds.loc[ds["eeg_path"].notna()].copy()
    ds = ds.loc[~ds["eeg_path"].astype(str).str.strip().str.lower().isin({"", "none", "nan"})].copy()
    runs = sorted(ds["run"].astype(int).unique().tolist())
    if not runs:
        raise ValueError(f"No runnable EEG runs found in dataset index for {dyad_id}/{subject}")

    out_paths: list[Path] = []
    for run in runs:
        eeg_path = select_eeg_path(dataset_index, subject=subject, dyad_id=dyad_id, run=int(run))
        _configure_mne_logging()
        raw = read_raw_eeg(Path(eeg_path), preload=False, verbose="ERROR")
        if picks is not None:
            raw = raw.copy().pick(picks)
        aperiodic_df = compute_sliding_aperiodic(
            raw.get_data(),
            float(raw.info["sfreq"]),
            raw.times,
            sliding_cfg,
            band_hz=band_hz,
            channel_names=tuple(str(ch) for ch in raw.ch_names),
        )
        out_path = run_aperiodic_path(out_dir=out_dir, band=band, dyad_id=dyad_id, subject=subject, run=int(run))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        aperiodic_df.insert(0, "run", int(run))
        aperiodic_df.insert(0, "subject", str(subject))
        aperiodic_df.insert(0, "dyad_id", str(dyad_id))
        aperiodic_df.to_parquet(out_path, index=False)
        out_paths.append(out_path)
    return out_paths
