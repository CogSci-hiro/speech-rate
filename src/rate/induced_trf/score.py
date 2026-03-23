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
    select_eeg_path,
    validate_events_table,
)
from rate.evoked_trf.utils import (
    concatenate_with_gap,
    max_lag_samples,
    score_multichannel_mean_r,
    standardize_design_matrix_foldwise,
)
from rate.induced_trf.aperiodic import SlidingAperiodicConfig
from rate.induced_trf.sliding_fooof import parse_sliding_fooof_config, run_aperiodic_path
from rate.targets.induced import load_induced_band_run_target
from rate.targets.induced import build_residual_target_from_aperiodic_df
from rate.trf.design import DesignModelSpec, parse_evoked_models_config
from rate.trf.engine import TRFResult, run_trf
from rate.trf.io_contract import write_trf_artifacts
from rate.trf.reconstruction import compute_reconstruction_r2
from rate.trf import scoring as trf_scoring


@dataclass(frozen=True)
class InducedTrfConfig:
    downsample_hz: float
    bands: tuple[str, ...]
    target_kinds: tuple[str, ...]
    residual_bands: tuple[str, ...]
    bands_hz: dict[str, tuple[float, float]]
    edge_drop: str
    edge_window_s: float | None
    fooof_sliding_cfg: SlidingAperiodicConfig | None
    tmin_s: float
    tmax_s: float
    regularization_type: str
    ridge_alpha_grid: list[float]
    ridge_alpha_default: float
    models: tuple[DesignModelSpec, ...]

    def bands_for_target_kind(self, target_kind: str) -> tuple[str, ...]:
        kind = str(target_kind).strip().lower()
        if kind == "residual":
            return self.residual_bands
        return self.bands


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
    if bands_cfg is not None:
        if not isinstance(bands_cfg, (list, tuple)) or not bands_cfg:
            raise ValueError("induced_trf.bands must be a non-empty list when provided")
        bands = tuple(str(v).strip() for v in bands_cfg if str(v).strip())
    else:
        bands = tuple(sorted(bands_hz.keys()))
    if not bands:
        raise ValueError("induced_trf must enable at least one band")
    missing_bands = [band for band in bands if band not in bands_hz]
    if missing_bands:
        raise ValueError(f"Missing induced_trf.bands_hz entries for enabled bands: {missing_bands}")

    target_kinds_cfg = ind.get("target_kinds")
    if target_kinds_cfg is not None:
        target_list = [str(v).strip().lower() for v in target_kinds_cfg]
        allowed_target_kinds = {"raw", "residual"}
        if (not target_list) or any(kind not in allowed_target_kinds for kind in target_list):
            raise ValueError(
                "induced_trf.target_kinds must be a non-empty subset of ['raw', 'residual']"
            )
        target_kinds = tuple(dict.fromkeys(target_list))
    else:
        target_kinds = ("raw", "residual")

    residual_bands_cfg = ind.get("residual_bands")
    if "residual" in set(target_kinds):
        if residual_bands_cfg is not None:
            residual_list = [str(v).strip() for v in residual_bands_cfg if str(v).strip()]
            if not residual_list:
                raise ValueError("induced_trf.residual_bands must be non-empty when residual is enabled")
            residual_bands = tuple(dict.fromkeys(residual_list))
        else:
            residual_bands = bands
        invalid_residual_bands = [band for band in residual_bands if band not in set(bands)]
        if invalid_residual_bands:
            raise ValueError(
                "induced_trf.residual_bands must be a subset of induced_trf.bands; "
                f"got invalid values {invalid_residual_bands}"
            )
    else:
        residual_bands = tuple()

    fooof_sliding_cfg: SlidingAperiodicConfig | None = None
    if ("residual" in set(target_kinds)) and residual_bands:
        fooof_sliding_cfg = parse_sliding_fooof_config(cfg)
    edge_drop = str(fooof_sliding_cfg.edge_drop if fooof_sliding_cfg is not None else "none")
    edge_window_s = float(fooof_sliding_cfg.window_s) if fooof_sliding_cfg is not None else None

    lags_cfg = ev.get("lags_s") or {}
    if not isinstance(lags_cfg, dict):
        raise ValueError("evoked_trf.lags_s must be a mapping (reused for induced TRF)")
    reg_cfg = ind.get("regularization")
    if reg_cfg is None:
        reg_cfg = ev.get("regularization") or {}
    if not isinstance(reg_cfg, dict):
        raise ValueError("induced_trf.regularization must be a mapping")
    regularization_type = str(reg_cfg.get("type", "ridge")).strip().lower()

    grid_cfg = ind.get("ridge_alpha_grid", None)
    if grid_cfg is None:
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
        target_kinds=target_kinds,
        residual_bands=residual_bands,
        bands_hz=bands_hz,
        edge_drop=edge_drop,
        edge_window_s=edge_window_s,
        fooof_sliding_cfg=fooof_sliding_cfg,
        tmin_s=float(lags_cfg["tmin"]),
        tmax_s=float(lags_cfg["tmax"]),
        regularization_type=regularization_type,
        ridge_alpha_grid=ridge_alpha_grid,
        ridge_alpha_default=float(ind.get("ridge_alpha", ev.get("ridge_alpha", 1.0))),
        models=parse_evoked_models_config(ind.get("models")),
    )


def _analysis_name_for_band(band: str, *, target_kind: str) -> str:
    band_name = str(band)
    kind = str(target_kind).strip().lower()
    if kind == "raw":
        return f"induced_trf_{band_name}"
    return f"induced_trf_{band_name}_{kind}"


def _subject_dir(*, out_dir: Path, band: str, target_kind: str, dyad_id: str, subject: str) -> Path:
    return out_dir / _analysis_name_for_band(band, target_kind=target_kind) / "subjects" / str(dyad_id) / str(subject)


def load_induced_input_tables(out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    events_path = out_dir / "events" / "events.parquet"
    ds_index_path = out_dir / "dataset" / "index.parquet"
    return pd.read_parquet(events_path), pd.read_parquet(ds_index_path)


def _prepare_induced_subject_runs(
    *,
    cfg: InducedTrfConfig,
    out_dir: Path,
    dyad_id: str,
    subject: str,
    band: str,
    target_kind: str,
    events: pd.DataFrame,
    ds_index: pd.DataFrame,
    picks: str | list[str],
) -> tuple[
    list[int],
    dict[int, np.ndarray],
    dict[int, np.ndarray],
    dict[int, dict[str, np.ndarray]],
    dict[int, dict[str, list[str]]],
    dict[int, pd.DataFrame],
    tuple[str, ...],
    float,
]:
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
    per_run_aperiodic: dict[int, pd.DataFrame] = {}
    channel_names_ref: tuple[str, ...] | None = None
    sfreqs: set[float] = set()
    emg_subject = load_subject_emg(out_dir=out_dir, dyad_id=dyad_id, subject=subject)

    # induced models can still use event-derived regressors and optional emg if provided later
    for run in runs:
        if str(target_kind).strip().lower() == "residual":
            aperiodic_path = run_aperiodic_path(
                out_dir=out_dir,
                band=band,
                dyad_id=dyad_id,
                subject=subject,
                run=int(run),
            )
            if not aperiodic_path.exists():
                raise FileNotFoundError(
                    f"Missing precomputed sliding FOOOF target: {aperiodic_path}. "
                    "Run the sliding FOOOF rule before fitting residual induced TRF."
                )
            aperiodic_df = pd.read_parquet(aperiodic_path)
            run_target = build_residual_target_from_aperiodic_df(
                aperiodic_df,
                downsample_hz=cfg.downsample_hz,
                edge_drop=cfg.edge_drop,
                edge_window_s=cfg.edge_window_s,
            )
        else:
            eeg_path = select_eeg_path(ds_index, subject=subject, dyad_id=dyad_id, run=int(run))
            run_target = load_induced_band_run_target(
                eeg_path,
                band=band,
                bands_hz=cfg.bands_hz,
                downsample_hz=cfg.downsample_hz,
                target_kind=target_kind,
                fooof_sliding_cfg=cfg.fooof_sliding_cfg,
                edge_drop=cfg.edge_drop,
                edge_window_s=cfg.edge_window_s,
                picks=picks,
            )
        if channel_names_ref is None:
            channel_names_ref = tuple(str(ch) for ch in run_target.channel_names)
        elif tuple(str(ch) for ch in run_target.channel_names) != channel_names_ref:
            raise ValueError("Induced target channel names are inconsistent across runs for one subject.")
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
        if run_target.aperiodic_time_series is not None:
            per_run_aperiodic[int(run)] = run_target.aperiodic_time_series.copy()

    if len(sfreqs) != 1:
        raise ValueError(f"Expected identical induced sfreq across runs; got {sorted(sfreqs)}")
    if channel_names_ref is None:
        raise ValueError("No channel names found for induced subject runs.")
    sfreq_hz = float(next(iter(sfreqs)))
    return runs, per_run_events, per_run_y, per_run_X, per_run_feature_names, per_run_aperiodic, channel_names_ref, sfreq_hz


def write_subject_output(
    *,
    cfg: dict[str, Any],
    out_dir: Path,
    dyad_id: str,
    subject: str,
    band: str,
    target_kind: str = "raw",
    picks: str | list[str] = "eeg",
) -> Path:
    trf_cfg = parse_induced_trf_config(cfg)
    band_name = str(band)
    target_mode = str(target_kind).strip().lower()
    if band_name not in trf_cfg.bands:
        raise ValueError(f"Band '{band}' not enabled in induced_trf.bands={list(trf_cfg.bands)}")
    if target_mode not in set(trf_cfg.target_kinds):
        raise ValueError(
            f"Target kind '{target_kind}' not enabled in induced_trf.target_kinds={list(trf_cfg.target_kinds)}"
        )
    allowed_bands_for_kind = set(trf_cfg.bands_for_target_kind(target_mode))
    if band_name not in allowed_bands_for_kind:
        if target_mode == "residual":
            raise ValueError(
                "Band "
                f"'{band}' not enabled for induced_trf.target_kind='residual'; "
                f"set induced_trf.residual_bands to include it (current={list(trf_cfg.residual_bands)})"
            )
        raise ValueError(f"Band '{band}' is not enabled for induced_trf.target_kind='{target_mode}'")

    events, ds_index = load_induced_input_tables(out_dir)
    events = validate_events_table(events)
    runs, per_run_events, per_run_y, per_run_X, per_run_feature_names, per_run_aperiodic, channel_names, sfreq_hz = _prepare_induced_subject_runs(
        cfg=trf_cfg,
        out_dir=out_dir,
        dyad_id=str(dyad_id),
        subject=str(subject),
        band=band_name,
        target_kind=target_mode,
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

    subject_dir = _subject_dir(
        out_dir=out_dir,
        band=str(band),
        target_kind=str(target_kind),
        dyad_id=str(dyad_id),
        subject=str(subject),
    )
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

    if per_run_aperiodic and target_mode != "residual":
        aperiodic_dir = subject_dir / "targets" / "aperiodic"
        aperiodic_dir.mkdir(parents=True, exist_ok=True)
        for run in runs:
            if run not in per_run_aperiodic:
                continue
            run_df = per_run_aperiodic[run].copy()
            run_df.insert(0, "run", int(run))
            run_df.insert(0, "subject", str(subject))
            run_df.insert(0, "dyad_id", str(dyad_id))
            run_df.to_parquet(aperiodic_dir / f"run-{int(run):02d}.parquet", index=False)

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
                channel_names=tuple(channel_names),
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
            remainder = analysis_name.removeprefix("induced_trf_")
            if remainder.endswith("_residual"):
                df["target_kind"] = "residual"
                df["band"] = remainder[: -len("_residual")]
            else:
                df["target_kind"] = "raw"
                df["band"] = remainder
        frames.append(df)

    scores = pd.concat(frames, ignore_index=True)
    sort_cols = [c for c in ["band", "target_kind", "dyad_id", "subject", "run", "model", "fold_id"] if c in scores.columns]
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
