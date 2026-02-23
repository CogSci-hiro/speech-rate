from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rate.config import get_project_downsample_hz

from rate.evoked_trf.feature_sources import emg_for_run_aligned, load_subject_emg
from rate.evoked_trf.fit import fit_receptive_field, predict_receptive_field
from rate.evoked_trf.score import (
    _canonical_lags_s,
    _extract_rf_coef_and_lags,
    _fit_full_data_kernel,
    _validate_scores_table,
    build_run_matrices,
    load_input_tables,
    select_eeg_path,
    validate_events_table,
)
from rate.evoked_trf.utils import (
    concatenate_with_gap,
    max_lag_samples,
    score_multichannel_mean_r,
    standardize_design_matrix_foldwise,
)
from rate.targets.induced import load_induced_band_run_target
from rate.trf.design import DesignModelSpec, parse_evoked_models_config
from rate.trf.engine import TRFResult, run_trf
from rate.trf.io_contract import write_trf_artifacts
from rate.trf.reconstruction import compute_reconstruction_r2
from rate.trf import scoring as trf_scoring


@dataclass(frozen=True)
class InducedTrfConfig:
    downsample_hz: float
    bands: tuple[str, ...]
    bands_hz: dict[str, tuple[float, float]]
    edge_drop: str
    edge_window_s: float | None
    tmin_s: float
    tmax_s: float
    regularization_type: str
    ridge_alpha_grid: list[float]
    ridge_alpha_default: float
    models: tuple[DesignModelSpec, ...]


def parse_induced_trf_config(cfg: dict[str, Any]) -> InducedTrfConfig:
    ind = dict(cfg.get("induced_trf") or {})
    ev = dict(cfg.get("evoked_trf") or {})
    if not ind:
        raise ValueError("Missing induced_trf config")

    bands_hz_cfg = ind.get("bands_hz") or {}
    if not isinstance(bands_hz_cfg, dict) or not bands_hz_cfg:
        raise ValueError("induced_trf.bands_hz must be a non-empty mapping")
    bands_hz: dict[str, tuple[float, float]] = {}
    for name, rng in bands_hz_cfg.items():
        if not isinstance(rng, (list, tuple)) or len(rng) != 2:
            raise ValueError(f"induced_trf.bands_hz.{name} must be [fmin, fmax]")
        bands_hz[str(name)] = (float(rng[0]), float(rng[1]))

    bands_cfg = ind.get("bands")
    if bands_cfg is None:
        bands = tuple(sorted([b for b in ("theta", "alpha", "beta") if b in bands_hz]))
    else:
        if not isinstance(bands_cfg, list) or not bands_cfg:
            raise ValueError("induced_trf.bands must be a non-empty list")
        bands = tuple(str(b) for b in bands_cfg)
    for b in bands:
        if b not in bands_hz:
            raise ValueError(f"Band '{b}' listed in induced_trf.bands but missing in induced_trf.bands_hz")

    fooof_sliding = ind.get("fooof_sliding") or {}
    if fooof_sliding is None:
        fooof_sliding = {}
    if not isinstance(fooof_sliding, dict):
        raise ValueError("induced_trf.fooof_sliding must be a mapping")
    edge_drop = str(fooof_sliding.get("edge_drop", "none"))
    edge_window_s = float(fooof_sliding["window_s"]) if "window_s" in fooof_sliding else None

    lags_cfg = ev.get("lags_s") or {}
    if not isinstance(lags_cfg, dict):
        raise ValueError("evoked_trf.lags_s must be a mapping (reused for induced TRF)")
    reg_cfg = ev.get("regularization") or {}
    if not isinstance(reg_cfg, dict):
        raise ValueError("evoked_trf.regularization must be a mapping (reused for induced TRF)")
    regularization_type = str(reg_cfg.get("type", "ridge")).strip().lower()

    grid_cfg = ev.get("ridge_alpha_grid", None)
    if grid_cfg is None:
        ridge_alpha_grid = [0.1, 1.0, 10.0, 100.0, 1000.0]
    elif isinstance(grid_cfg, dict):
        ridge_alpha_grid = np.logspace(
            float(grid_cfg["min"]),
            float(grid_cfg["max"]),
            num=int(grid_cfg["n_alpha"]),
        ).astype(float).tolist()
    else:
        ridge_alpha_grid = [float(a) for a in grid_cfg]

    return InducedTrfConfig(
        downsample_hz=get_project_downsample_hz(cfg, fallback_section="induced_trf"),
        bands=bands,
        bands_hz=bands_hz,
        edge_drop=edge_drop,
        edge_window_s=edge_window_s,
        tmin_s=float(lags_cfg["tmin"]),
        tmax_s=float(lags_cfg["tmax"]),
        regularization_type=regularization_type,
        ridge_alpha_grid=ridge_alpha_grid,
        ridge_alpha_default=float(ev.get("ridge_alpha", 1.0)),
        models=parse_evoked_models_config(ind.get("models")),
    )


def _analysis_name_for_band(band: str) -> str:
    return f"induced_trf_{str(band)}"


def _subject_dir(*, out_dir: Path, band: str, dyad_id: str, subject: str) -> Path:
    return out_dir / _analysis_name_for_band(band) / "subjects" / str(dyad_id) / str(subject)


def _prepare_induced_subject_runs(
    *,
    cfg: InducedTrfConfig,
    out_dir: Path,
    dyad_id: str,
    subject: str,
    band: str,
    events: pd.DataFrame,
    ds_index: pd.DataFrame,
    picks: str | list[str],
) -> tuple[list[int], dict[int, np.ndarray], dict[int, np.ndarray], dict[int, dict[str, np.ndarray]], dict[int, dict[str, list[str]]], float]:
    ev_subj = events.loc[
        (events["dyad_id"].astype(str).str.strip() == str(dyad_id).strip())
        & (events["subject_id"].astype(str).str.strip() == str(subject).strip())
    ].copy()
    runs = sorted(ev_subj["run"].astype(int).unique().tolist())
    if len(runs) < 2:
        raise ValueError(f"leave_one_run_out requires >=2 runs, got {len(runs)} for {dyad_id}/{subject}")

    lags_s = _canonical_lags_s(sfreq_hz=cfg.downsample_hz, tmin_s=cfg.tmin_s, tmax_s=cfg.tmax_s)

    per_run_events: dict[int, np.ndarray] = {}
    per_run_y: dict[int, np.ndarray] = {}
    per_run_X: dict[int, dict[str, np.ndarray]] = {}
    per_run_feature_names: dict[int, dict[str, list[str]]] = {}
    sfreqs: set[float] = set()
    emg_subject = load_subject_emg(out_dir=out_dir, dyad_id=dyad_id, subject=subject)

    # induced models can still use event-derived regressors and optional emg if provided later
    for run in runs:
        eeg_path = select_eeg_path(ds_index, subject=subject, dyad_id=dyad_id, run=int(run))
        run_target = load_induced_band_run_target(
            eeg_path,
            band=band,
            bands_hz=cfg.bands_hz,
            downsample_hz=cfg.downsample_hz,
            edge_drop=cfg.edge_drop,
            edge_window_s=cfg.edge_window_s,
            picks=picks,
        )
        sfreqs.add(float(run_target.sfreq_hz))
        ev_run = ev_subj.loc[ev_subj["run"].astype(int) == int(run)].copy()
        emg_run = emg_for_run_aligned(
            emg_subject=emg_subject,
            run=int(run),
            n_samples=int(run_target.y.shape[0]),
            sfreq_hz=float(run_target.sfreq_hz),
            sample_t0_s=float(run_target.times_s[0]),
            zscore_within_run=True,
        )
        X_by_model, names_by_model = build_run_matrices(
            ev_run=ev_run,
            sfreq_hz=float(run_target.sfreq_hz),
            lags_s=lags_s,
            n_samples=int(run_target.y.shape[0]),
            run_id=int(run),
            sample_t0_s=float(run_target.times_s[0]),
            models=cfg.models,
            continuous_regressors={"emg": emg_run},
        )
        per_run_events[int(run)] = ev_run
        per_run_y[int(run)] = np.asarray(run_target.y, dtype=float)
        per_run_X[int(run)] = X_by_model
        per_run_feature_names[int(run)] = names_by_model

    if len(sfreqs) != 1:
        raise ValueError(f"Expected identical induced sfreq across runs; got {sorted(sfreqs)}")
    sfreq_hz = float(next(iter(sfreqs)))
    return runs, per_run_events, per_run_y, per_run_X, per_run_feature_names, sfreq_hz


def write_subject_output(
    *,
    cfg: dict[str, Any],
    out_dir: Path,
    dyad_id: str,
    subject: str,
    band: str,
    picks: str | list[str] = "eeg",
) -> Path:
    trf_cfg = parse_induced_trf_config(cfg)
    if str(band) not in trf_cfg.bands:
        raise ValueError(f"Band '{band}' not enabled in induced_trf.bands={list(trf_cfg.bands)}")

    events, ds_index = load_input_tables(out_dir)
    events = validate_events_table(events)
    runs, per_run_events, per_run_y, per_run_X, per_run_feature_names, sfreq_hz = _prepare_induced_subject_runs(
        cfg=trf_cfg,
        out_dir=out_dir,
        dyad_id=str(dyad_id),
        subject=str(subject),
        band=str(band),
        events=events,
        ds_index=ds_index,
        picks=picks,
    )
    lags_s = _canonical_lags_s(sfreq_hz=sfreq_hz, tmin_s=trf_cfg.tmin_s, tmax_s=trf_cfg.tmax_s)
    gap_samples = max_lag_samples(sfreq_hz=sfreq_hz, tmin_s=trf_cfg.tmin_s, tmax_s=trf_cfg.tmax_s) + 1

    inner_rows: list[pd.DataFrame] = []
    outer_rows: list[pd.DataFrame] = []
    best_alpha_rows: list[pd.DataFrame] = []
    trf_results_by_model: dict[str, TRFResult] = {}

    for model in trf_cfg.models:
        result = run_trf(
            X_by_run={r: per_run_X[r][model.model_name] for r in runs},
            y_by_run={r: per_run_y[r] for r in runs},
            lags=np.asarray(lags_s, dtype=float),
            alpha_grid=np.asarray(trf_cfg.ridge_alpha_grid, dtype=float),
            cv_config={
                "gap_samples": int(gap_samples),
                "alpha_default": float(trf_cfg.ridge_alpha_default),
                "concat_fn": concatenate_with_gap,
                "standardize_fn": standardize_design_matrix_foldwise,
                "fit_fn": lambda *, X_train, y_train, alpha: fit_receptive_field(
                    X_train=X_train,
                    y_train=y_train,
                    sfreq_hz=sfreq_hz,
                    tmin_s=trf_cfg.tmin_s,
                    tmax_s=trf_cfg.tmax_s,
                    ridge_alpha=float(alpha),
                    regularization_type=trf_cfg.regularization_type,
                ),
                "predict_fn": predict_receptive_field,
                "score_fn": score_multichannel_mean_r,
                "extract_kernel_fn": lambda rf: _extract_rf_coef_and_lags(
                    rf=rf,
                    sfreq_hz=sfreq_hz,
                    tmin_s=trf_cfg.tmin_s,
                    tmax_s=trf_cfg.tmax_s,
                ),
                "table_meta": {"dyad_id": str(dyad_id), "subject": str(subject), "model": model.model_name},
                "outer_progress_enabled": True,
                "outer_progress_desc": f"outer-cv {dyad_id}/{subject}/{model.model_name}/{band}",
                "outer_progress_unit": "fold",
                "enable_reconstruction": True,
                "reconstruction_fn": compute_reconstruction_r2,
                "events_by_run": per_run_events,
                "reconstruction_kwargs": {"sfreq": sfreq_hz, "tmin": trf_cfg.tmin_s, "tmax": trf_cfg.tmax_s},
            },
        )
        trf_results_by_model[model.model_name] = result
        if result.inner_scores is not None:
            inner_rows.append(result.inner_scores)
        outer_rows.append(result.outer_scores)
        if result.best_alpha_table is not None:
            best_alpha_rows.append(result.best_alpha_table)

    inner_scores = pd.concat(inner_rows, ignore_index=True) if inner_rows else pd.DataFrame()
    outer_scores = pd.concat(outer_rows, ignore_index=True) if outer_rows else pd.DataFrame()
    scores = trf_scoring.summarize_scores(dyad_id=str(dyad_id), subject=str(subject), outer_scores=outer_scores)
    _validate_scores_table(scores=scores)

    subject_tables = trf_scoring.build_subject_score_tables(
        dyad_id=str(dyad_id),
        subject=str(subject),
        scores=scores,
        inner_scores=inner_scores,
        outer_scores=outer_scores,
    )

    subject_dir = _subject_dir(out_dir=out_dir, band=str(band), dyad_id=str(dyad_id), subject=str(subject))
    subject_dir.mkdir(parents=True, exist_ok=True)
    out_path = subject_dir / "scores.parquet"
    subject_tables.summary_fold_sorted.to_parquet(out_path, index=False)
    subject_tables.alpha_grid.to_parquet(subject_dir / "alpha_grid.parquet", index=False)
    subject_tables.inner_cv.to_parquet(subject_dir / "inner_cv.parquet", index=False)
    subject_tables.model_comparison.to_parquet(subject_dir / "model_comparison.parquet", index=False)
    subject_tables.summary.to_parquet(subject_dir / "summary.parquet", index=False)
    subject_tables.alpha_grid.to_parquet(subject_dir / "alpha_grid_scores.parquet", index=False)
    subject_tables.inner_sorted.to_parquet(subject_dir / "inner_cv_scores.parquet", index=False)
    subject_tables.outer_sorted.to_parquet(subject_dir / "outer_cv_scores.parquet", index=False)

    for model in trf_cfg.models:
        model_name = model.model_name
        model_dir = subject_dir / "models" / model_name
        model_outer = subject_tables.outer_fold_scores.loc[subject_tables.outer_fold_scores["model"] == model_name].copy()
        model_outer = model_outer[
            ["dyad_id", "subject", "model", "test_run", "alpha", "r", "baseline_r", "delta_r"]
        ].sort_values(["test_run"], kind="mergesort").reset_index(drop=True)
        trf_result = trf_results_by_model[model_name]
        write_trf_artifacts(
            TRFResult(
                kernels_outer=np.asarray(trf_result.kernels_outer),
                chosen_alpha=np.asarray(trf_result.chosen_alpha, dtype=float),
                outer_scores=model_outer,
                lags=np.asarray(trf_result.lags, dtype=float),
                predictions_by_fold=None,
                reconstruction_r2=trf_result.reconstruction_r2,
            ),
            model_dir,
        )

        # Optional full-data refit parity check (keeps lags contract stable, no additional outputs).
        if not subject_tables.best_alpha_by_model.empty:
            row = subject_tables.best_alpha_by_model.loc[subject_tables.best_alpha_by_model["model"] == model_name]
            if not row.empty:
                alpha = float(row.iloc[0]["best_alpha"])
                _, lags_full = _fit_full_data_kernel(
                    model_name=model_name,
                    alpha=alpha,
                    per_run_y=per_run_y,
                    per_run_X=per_run_X,
                    sfreq_hz=sfreq_hz,
                    tmin_s=trf_cfg.tmin_s,
                    tmax_s=trf_cfg.tmax_s,
                    regularization_type=trf_cfg.regularization_type,
                )
                if not np.allclose(np.asarray(trf_result.lags, dtype=float), np.asarray(lags_full, dtype=float)):
                    raise ValueError(f"Lag grid mismatch in induced TRF full refit for model={model_name}")

    return out_path


def compute_scores(model, X: np.ndarray, y: np.ndarray, metadata: dict) -> pd.DataFrame:
    # Backward-compatible lightweight helper retained.
    r = float(model.score(X, y))
    baseline = 0.0
    return pd.DataFrame([{**metadata, "r": r, "r_baseline": baseline, "delta_r": r - baseline}])


def aggregate_subject_scores(*, out_dir: Path) -> pd.DataFrame:
    subject_paths = sorted(out_dir.glob("induced_trf_*/subjects/*/*/scores.parquet"))
    if not subject_paths:
        raise ValueError(f"No induced TRF subject score files found under: {out_dir}")

    frames: list[pd.DataFrame] = []
    for path in subject_paths:
        try:
            df = pd.read_parquet(path)
        except Exception as exc:
            raise ValueError(f"Failed to read induced TRF subject scores parquet: {path}") from exc
        analysis_name = str(path.parents[3].name)
        if analysis_name.startswith("induced_trf_"):
            df = df.copy()
            df["band"] = analysis_name.removeprefix("induced_trf_")
        frames.append(df)

    scores = pd.concat(frames, ignore_index=True)
    sort_cols = [c for c in ["band", "dyad_id", "subject", "run", "model", "fold_id"] if c in scores.columns]
    if sort_cols:
        scores = scores.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    return scores


def write_aggregate_output(*, out_dir: Path) -> Path:
    scores = aggregate_subject_scores(out_dir=out_dir)
    _validate_scores_table(scores=scores)
    out_path = out_dir / "induced_trf" / "scores" / "scores.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(out_path, index=False)
    return out_path
