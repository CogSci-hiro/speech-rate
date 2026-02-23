import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rate.evoked_trf.fit import (
    fit_receptive_field,
    load_eeg_run,
    predict_receptive_field,
)
from rate.evoked_trf.feature_sources import emg_for_run_aligned, load_subject_emg
from rate.evoked_trf.results_io import DetailedPaths, save_receptive_field_npz, write_parquet
from rate.evoked_trf.spec import add_timelock_columns
from rate.evoked_trf.utils import (
    concatenate_with_gap,
    max_lag_samples,
    score_multichannel_mean_r,
    standardize_design_matrix_foldwise,
)
from rate.trf.engine import TRFResult, run_trf
from rate.trf.io_contract import write_trf_artifacts
from rate.trf.reconstruction import compute_reconstruction_metrics_df
from rate.trf.similarity import cross_kernel_temporal_similarity
from rate.trf import scoring as trf_scoring
from rate.trf.design import DesignModelSpec, build_design_matrix as build_design_matrix_from_config, parse_evoked_models_config

LOGGER = logging.getLogger(__name__)

PHASE5_PRIMARY_MODELS = ("baseline", "self_rate", "other_rate", "abs_diff")


# =============================================================================
# Config
# =============================================================================


@dataclass(frozen=True)
class EvokedTrfConfig:
    downsample_hz: float
    tmin_s: float
    tmax_s: float
    scheme: str
    regularization_type: str
    ridge_alpha_grid: list[float]
    ridge_alpha_default: float
    emg_zscore_within_run: bool
    models: tuple[DesignModelSpec, ...]


def parse_evoked_trf_config(cfg: dict[str, Any]) -> EvokedTrfConfig:
    ev = cfg["evoked_trf"]
    cv_cfg = ev["cv"]
    scheme = str(cv_cfg.get("scheme", "leave_one_run_out"))
    reg_cfg = ev.get("regularization", {})
    if reg_cfg is None:
        reg_cfg = {}
    if not isinstance(reg_cfg, dict):
        raise ValueError("evoked_trf.regularization must be a mapping when provided")
    regularization_type = str(reg_cfg.get("type", "ridge")).strip().lower()
    if regularization_type not in {"ridge", "laplacian"}:
        raise ValueError("evoked_trf.regularization.type must be 'ridge' or 'laplacian'")

    grid_cfg = ev.get("ridge_alpha_grid", None)
    if grid_cfg is None:
        grid = [0.1, 1.0, 10.0, 100.0, 1000.0]
    elif isinstance(grid_cfg, dict):
        missing = {"min", "max", "n_alpha"} - set(grid_cfg.keys())
        if missing:
            raise ValueError(f"ridge_alpha_grid mapping missing keys: {sorted(missing)}")
        alpha_min = float(grid_cfg["min"])
        alpha_max = float(grid_cfg["max"])
        n_alpha = int(grid_cfg["n_alpha"])
        if n_alpha < 1:
            raise ValueError("ridge_alpha_grid.n_alpha must be >= 1")
        grid = np.logspace(alpha_min, alpha_max, num=n_alpha).astype(float).tolist()
    else:
        grid = [float(a) for a in grid_cfg]

    emg_cfg = ev.get("emg", {}) or {}
    if not isinstance(emg_cfg, dict):
        raise ValueError("evoked_trf.emg must be a mapping when provided")

    return EvokedTrfConfig(
        downsample_hz=float(ev["downsample_hz"]),
        tmin_s=float(ev["lags_s"]["tmin"]),
        tmax_s=float(ev["lags_s"]["tmax"]),
        scheme=scheme,
        regularization_type=regularization_type,
        ridge_alpha_grid=grid,
        ridge_alpha_default=float(ev.get("ridge_alpha", 1.0)),
        emg_zscore_within_run=bool(emg_cfg.get("zscore_within_run", True)),
        models=parse_evoked_models_config(ev.get("models")),
    )


# =============================================================================
# IO helpers
# =============================================================================


def load_input_tables(out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    events_path = out_dir / "events" / "events.parquet"
    ds_index_path = out_dir / "dataset" / "index_resampled.parquet"
    return pd.read_parquet(events_path), pd.read_parquet(ds_index_path)


def validate_events_table(events: pd.DataFrame) -> pd.DataFrame:
    required = {
        "dyad_id",
        "subject_id",
        "run",
        "t_onset",
        "t_offset",
    }
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"events table missing required columns: {sorted(missing)}")
    return add_timelock_columns(events)


def select_eeg_path(ds_index: pd.DataFrame, *, subject: str, dyad_id: str, run: int) -> Path:
    required = {"dyad_id", "subject_id", "run", "eeg_path"}
    missing = required - set(ds_index.columns)
    if missing:
        raise ValueError(f"dataset index missing required columns: {sorted(missing)}")

    dyad_col = ds_index["dyad_id"].astype(str).str.strip()
    subj_col = ds_index["subject_id"].astype(str).str.strip()
    run_col = pd.to_numeric(ds_index["run"], errors="coerce")

    mask = (dyad_col == str(dyad_id).strip()) & (subj_col == str(subject).strip()) & (run_col == int(run))
    sub = ds_index.loc[mask].copy()
    if sub.empty:
        raise ValueError(f"No dataset index row for dyad_id={dyad_id}, subject_id={subject}, run={run}")

    eeg_path = Path(str(sub.iloc[0]["eeg_path"])).expanduser()
    return eeg_path


# =============================================================================
# Model building
# =============================================================================


def list_models(models: tuple[DesignModelSpec, ...]) -> list[str]:
    return [m.model_name for m in models]


def build_run_matrices(
    *,
    ev_run: pd.DataFrame,
    sfreq_hz: float,
    lags_s: np.ndarray,
    n_samples: int,
    run_id: int,
    sample_t0_s: float,
    models: tuple[DesignModelSpec, ...],
    continuous_regressors: dict[str, np.ndarray] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
    X_by_model: dict[str, np.ndarray] = {}
    names_by_model: dict[str, list[str]] = {}
    for m in models:
        X, feature_names = build_design_matrix_from_config(
            model_spec=m,
            events_df=ev_run,
            run_id=int(run_id),
            sfreq=float(sfreq_hz),
            lags=np.asarray(lags_s, dtype=float),
            n_samples=int(n_samples),
            sample_t0_s=sample_t0_s,
            continuous_regressors=continuous_regressors,
            return_feature_names=True,
        )
        X_by_model[m.model_name] = X
        names_by_model[m.model_name] = feature_names
    return X_by_model, names_by_model


# =============================================================================
# CV + scoring
# =============================================================================


def _canonical_lags_s(*, sfreq_hz: float, tmin_s: float, tmax_s: float) -> np.ndarray:
    lags_samples = np.arange(
        int(np.floor(float(tmin_s) * float(sfreq_hz))),
        int(np.ceil(float(tmax_s) * float(sfreq_hz))) + 1,
    )
    return lags_samples / float(sfreq_hz)


def _reconstruction_event_lock_for_model(model_name: str) -> str:
    model = str(model_name)
    if model in {"self_rate", "abs_diff"}:
        return "self_IPU_onset"
    if model == "other_rate":
        return "other_IPU_offset"
    # Baseline bookkeeping only; default to self onset.
    return "self_IPU_onset"


# =============================================================================
# Per-subject worker (parallel-safe)
# =============================================================================


def _compute_one_dyad_subject(
    *,
    out_dir: Path,
    dyad_id: str,
    subject: str,
    events: pd.DataFrame,
    ds_index: pd.DataFrame,
    trf_cfg: EvokedTrfConfig,
    picks: str | list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, TRFResult]]:
    _, runs, per_run_y, per_run_X, per_run_feature_names, sfreq_hz = _prepare_subject_run_data(
        out_dir=out_dir,
        dyad_id=dyad_id,
        subject=subject,
        events=events,
        ds_index=ds_index,
        trf_cfg=trf_cfg,
        picks=picks,
    )
    ev_subj = events.loc[
        (events["dyad_id"].astype(str).str.strip() == str(dyad_id).strip())
        & (events["subject_id"].astype(str).str.strip() == str(subject).strip())
    ].copy()
    per_run_events = {
        int(run): ev_subj.loc[ev_subj["run"].astype(int) == int(run)].copy() for run in runs
    }

    models = list_models(trf_cfg.models)
    inner_rows: list[pd.DataFrame] = []
    outer_rows: list[pd.DataFrame] = []
    best_alpha_frames: list[pd.DataFrame] = []
    trf_results_by_model: dict[str, TRFResult] = {}

    lags_s = _canonical_lags_s(sfreq_hz=sfreq_hz, tmin_s=trf_cfg.tmin_s, tmax_s=trf_cfg.tmax_s)
    gap_samples = max_lag_samples(sfreq_hz=sfreq_hz, tmin_s=trf_cfg.tmin_s, tmax_s=trf_cfg.tmax_s) + 1

    for model_name in models:
        trf_result = run_trf(
                X_by_run={r: per_run_X[r][model_name] for r in runs},  # type: ignore[arg-type]
                y_by_run={r: per_run_y[r] for r in runs},  # type: ignore[arg-type]
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
                    "table_meta": {
                        "dyad_id": dyad_id,
                        "subject": subject,
                        "model": model_name,
                    },
                    "outer_progress_enabled": True,
                    "outer_progress_desc": f"outer-cv {dyad_id}/{subject}/{model_name}",
                    "outer_progress_unit": "fold",
                    "collect_predictions": False,
                    "enable_reconstruction": True,
                    "reconstruction_fn": compute_reconstruction_metrics_df,
                    "events_by_run": {r: per_run_events[r] for r in runs},
                    "reconstruction_kwargs": {
                        "sfreq": float(sfreq_hz),
                        "tmin": float(trf_cfg.tmin_s),
                        "tmax": float(trf_cfg.tmax_s),
                        "event_lock": _reconstruction_event_lock_for_model(model_name),
                    },
                },
            )
        trf_results_by_model[model_name] = trf_result
        if trf_result.inner_scores is not None:
            inner_rows.append(trf_result.inner_scores)
        outer_rows.append(trf_result.outer_scores)
        if trf_result.best_alpha_table is not None:
            best_alpha_frames.append(trf_result.best_alpha_table)

    inner_scores = pd.concat(inner_rows, ignore_index=True) if inner_rows else pd.DataFrame()
    if not inner_scores.empty:
        inner_scores = inner_scores.sort_values(
            ["dyad_id", "subject", "outer_fold", "model", "inner_fold", "alpha"],
            kind="mergesort",
        ).reset_index(drop=True)
    outer_scores = pd.concat(outer_rows, ignore_index=True) if outer_rows else pd.DataFrame()
    if not outer_scores.empty:
        outer_scores = outer_scores.sort_values(
            ["dyad_id", "subject", "fold_id", "model", "alpha"],
            kind="mergesort",
        ).reset_index(drop=True)
    best_alpha = pd.concat(best_alpha_frames, ignore_index=True) if best_alpha_frames else pd.DataFrame()

    # Summary delta_r using chosen_alpha score for each fold/model vs baseline chosen_alpha
    summary = trf_scoring.summarize_scores(dyad_id=dyad_id, subject=subject, outer_scores=outer_scores)

    # Best model per subject = max mean delta_r (excluding baseline)
    best_model = trf_scoring.best_model_table(dyad_id=dyad_id, subject=subject, summary=summary)

    # Save final RF params for the best model only (trained on all runs)
    final_model_rows = fit_and_package_best_model(
        dyad_id=dyad_id,
        subject=subject,
        best_model=best_model,
        per_run_y=per_run_y,
        per_run_X=per_run_X,
        per_run_feature_names=per_run_feature_names,
        sfreq_hz=sfreq_hz,
        tmin_s=trf_cfg.tmin_s,
        tmax_s=trf_cfg.tmax_s,
        regularization_type=trf_cfg.regularization_type,
    )

    return summary, inner_scores, outer_scores, pd.concat([best_alpha, final_model_rows], ignore_index=True), trf_results_by_model


def _prepare_subject_run_data(
    *,
    out_dir: Path,
    dyad_id: str,
    subject: str,
    events: pd.DataFrame,
    ds_index: pd.DataFrame,
    trf_cfg: EvokedTrfConfig,
    picks: str | list[str],
) -> tuple[pd.DataFrame, list[int], dict[int, np.ndarray], dict[int, dict[str, np.ndarray]], dict[int, dict[str, list[str]]], float]:
    dyad_norm = str(dyad_id).strip()
    subject_norm = str(subject).strip()
    ev_subj = events.loc[
        (events["dyad_id"].astype(str).str.strip() == dyad_norm)
        & (events["subject_id"].astype(str).str.strip() == subject_norm)
    ].copy()

    runs = sorted(ev_subj["run"].astype(int).unique().tolist())
    if len(runs) < 2:
        raise ValueError(
            f"leave_one_run_out requires >=2 runs, got {len(runs)} for dyad_id={dyad_id}, subject={subject}"
        )

    per_run_y: dict[int, np.ndarray] = {}
    per_run_X: dict[int, dict[str, np.ndarray]] = {}
    per_run_feature_names: dict[int, dict[str, list[str]]] = {}
    sfreqs: set[float] = set()
    emg_subject = load_subject_emg(out_dir=out_dir, dyad_id=dyad_id, subject=subject)

    for run in runs:
        eeg_path = select_eeg_path(ds_index, subject=subject, dyad_id=dyad_id, run=int(run))
        run_data = load_eeg_run(eeg_path, downsample_hz=trf_cfg.downsample_hz, picks=picks)

        y = run_data.y
        sfreq_hz = float(run_data.sfreq_hz)
        sfreqs.add(sfreq_hz)

        ev_run = ev_subj.loc[ev_subj["run"].astype(int) == int(run)].copy()
        emg_run = emg_for_run_aligned(
            emg_subject=emg_subject,
            run=int(run),
            n_samples=y.shape[0],
            sfreq_hz=sfreq_hz,
            sample_t0_s=float(run_data.times_s[0]),
            zscore_within_run=trf_cfg.emg_zscore_within_run,
        )
        X_by_model, names_by_model = build_run_matrices(
            ev_run=ev_run,
            sfreq_hz=sfreq_hz,
            lags_s=_canonical_lags_s(sfreq_hz=sfreq_hz, tmin_s=trf_cfg.tmin_s, tmax_s=trf_cfg.tmax_s),
            n_samples=y.shape[0],
            run_id=int(run),
            sample_t0_s=float(run_data.times_s[0]),
            models=trf_cfg.models,
            continuous_regressors={"emg": emg_run},
        )

        per_run_y[int(run)] = y
        per_run_X[int(run)] = X_by_model
        per_run_feature_names[int(run)] = names_by_model

    if len(sfreqs) != 1:
        raise ValueError(f"Expected identical sfreq across runs after resample; got {sorted(sfreqs)}")
    sfreq_hz = float(next(iter(sfreqs)))

    return ev_subj, runs, per_run_y, per_run_X, per_run_feature_names, sfreq_hz


# =============================================================================
# Summaries + final model packaging
# =============================================================================


def summarize_scores(*, dyad_id: str, subject: str, outer_scores: pd.DataFrame) -> pd.DataFrame:
    return trf_scoring.summarize_scores(dyad_id=dyad_id, subject=subject, outer_scores=outer_scores)


def best_model_table(*, dyad_id: str, subject: str, summary: pd.DataFrame) -> pd.DataFrame:
    return trf_scoring.best_model_table(dyad_id=dyad_id, subject=subject, summary=summary)


def _mode_alpha(values: list[float]) -> float:
    if not values:
        raise ValueError("values must be non-empty")
    vals = pd.Series(values)
    mode = vals.mode()
    return float(mode.iloc[0]) if not mode.empty else float(vals.iloc[0])


def fit_and_package_best_model(
    *,
    dyad_id: str,
    subject: str,
    best_model: pd.DataFrame,
    per_run_y: dict[int, np.ndarray],
    per_run_X: dict[int, dict[str, np.ndarray]],
    per_run_feature_names: dict[int, dict[str, list[str]]],
    sfreq_hz: float,
    tmin_s: float,
    tmax_s: float,
    regularization_type: str,
) -> pd.DataFrame:
    """
    Fit the best model on ALL runs using a stable alpha (mode of chosen alphas),
    return a one-row table with reproduction metadata and NPZ path placeholder.
    """
    best_model_name = str(best_model.iloc[0]["best_model"])

    # Use the baseline-ish grid selection summary by taking mode of chosen alphas in outer table later.
    # Here we just store the model; actual alpha is written by the caller from best_alpha table.
    # We still fit a final model using a conservative alpha=1.0 by default unless the caller updates.
    alpha = 1.0

    gap = max_lag_samples(sfreq_hz=sfreq_hz, tmin_s=tmin_s, tmax_s=tmax_s) + 1
    runs = sorted(per_run_y.keys())

    y_all = concatenate_with_gap([per_run_y[r] for r in runs], gap_samples=gap)
    X_all = concatenate_with_gap([per_run_X[r][best_model_name] for r in runs], gap_samples=gap)

    # Standardize using all data (final fit artifact; not for evaluation)
    X_all, _ = standardize_design_matrix_foldwise(X_all, X_all)

    rf = fit_receptive_field(
        X_train=X_all,
        y_train=y_all,
        sfreq_hz=sfreq_hz,
        tmin_s=tmin_s,
        tmax_s=tmax_s,
        ridge_alpha=float(alpha),
        regularization_type=regularization_type,
    )

    # Pull coefficients in a library-agnostic way
    coef = getattr(rf, "coef_", None)
    if coef is None:
        coef = getattr(getattr(rf, "estimator_", None), "coef_", None)
    if coef is None:
        raise ValueError("Could not extract coefficients from ReceptiveField object")

    intercept = getattr(getattr(rf, "estimator_", None), "intercept_", None)
    delays = getattr(rf, "delays_", None)
    if delays is None:
        lags_samples = np.arange(
            int(np.floor(float(tmin_s) * float(sfreq_hz))),
            int(np.ceil(float(tmax_s) * float(sfreq_hz))) + 1,
        )
        lags_s = lags_samples / float(sfreq_hz)
    else:
        lags_s = np.asarray(delays, dtype=float) / float(sfreq_hz)

    # Feature names (assume consistent across runs for a model)
    feature_names = per_run_feature_names[runs[0]][best_model_name]

    # NPZ written by caller (needs out_dir paths); we return metadata now
    return pd.DataFrame(
        [
            {
                "dyad_id": dyad_id,
                "subject": subject,
                "packaged_model": best_model_name,
                "packaged_alpha": float(alpha),
                "sfreq_hz": float(sfreq_hz),
                "tmin_s": float(tmin_s),
                "tmax_s": float(tmax_s),
                "feature_names": feature_names,
                "coef": np.asarray(coef),
                "intercept": np.asarray(intercept) if intercept is not None else None,
                "lags_s": np.asarray(lags_s, dtype=float),
                "rf_model": rf,
            }
        ]
    )


# =============================================================================
# Public API
# =============================================================================


def compute_scores_and_splits(
    *,
    cfg: dict[str, Any],
    out_dir: Path,
    picks: str | list[str] = "eeg",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main entry: computes summary scores and cv_splits (run-based).

    cv_splits schema (LORO)
    -----------------------
    dyad_id, subject, fold_id, run, split
    """
    trf_cfg = parse_evoked_trf_config(cfg)
    if trf_cfg.scheme != "leave_one_run_out":
        raise ValueError(f"Unsupported evoked_trf.cv.scheme: {trf_cfg.scheme}")

    events, ds_index = load_input_tables(out_dir)
    events = validate_events_table(events)

    pairs = sorted({(str(d), str(s)) for d, s in events[["dyad_id", "subject_id"]].itertuples(index=False)})
    results = []
    for dyad_id, subject in pairs:
        result = _compute_one_dyad_subject(
            out_dir=out_dir,
            dyad_id=dyad_id,
            subject=subject,
            events=events,
            ds_index=ds_index,
            trf_cfg=trf_cfg,
            picks=picks,
        )
        results.append(result)

    summaries = [r[0] for r in results]
    inner_all = [r[1] for r in results]
    outer_all = [r[2] for r in results]
    packaged_all = [r[3] for r in results]

    scores_df = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()

    # cv_splits (run-based) derived from score table
    splits_df = make_run_splits_from_scores(scores_df)

    # Write detailed package here (side-effect) to keep CLI simple
    write_detailed_package(
        out_dir=out_dir,
        inner_scores=pd.concat(inner_all, ignore_index=True) if inner_all else pd.DataFrame(),
        outer_scores=pd.concat(outer_all, ignore_index=True) if outer_all else pd.DataFrame(),
        packaged=packaged_all,
    )

    return scores_df, splits_df


def compute_subject_scores(
    *,
    cfg: dict[str, Any],
    out_dir: Path,
    dyad_id: str,
    subject: str,
    picks: str | list[str] = "eeg",
) -> pd.DataFrame:
    summary, _, _, _, _ = _compute_subject_artifacts(
        cfg=cfg,
        out_dir=out_dir,
        dyad_id=dyad_id,
        subject=subject,
        picks=picks,
    )
    return summary

def _compute_subject_artifacts(
    *,
    cfg: dict[str, Any],
    out_dir: Path,
    dyad_id: str,
    subject: str,
    picks: str | list[str] = "eeg",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, TRFResult]]:
    trf_cfg = parse_evoked_trf_config(cfg)
    if trf_cfg.scheme != "leave_one_run_out":
        raise ValueError(f"Unsupported evoked_trf.cv.scheme: {trf_cfg.scheme}")

    events, ds_index = load_input_tables(out_dir)
    events = validate_events_table(events)

    summary, inner_scores, outer_scores, packaged, trf_results_by_model = _compute_one_dyad_subject(
        out_dir=out_dir,
        dyad_id=str(dyad_id),
        subject=str(subject),
        events=events,
        ds_index=ds_index,
        trf_cfg=trf_cfg,
        picks=picks,
    )
    return summary, inner_scores, outer_scores, packaged, trf_results_by_model


def _subject_scores_output_path(*, out_dir: Path, dyad_id: str, subject: str) -> Path:
    return out_dir / "evoked_trf" / "subjects" / str(dyad_id) / str(subject) / "scores.parquet"


def _extract_rf_coef_and_lags(*, rf: Any, sfreq_hz: float, tmin_s: float, tmax_s: float) -> tuple[np.ndarray, np.ndarray]:
    coef = getattr(rf, "coef_", None)
    if coef is None:
        coef = getattr(getattr(rf, "estimator_", None), "coef_", None)
    if coef is None:
        raise ValueError("Could not extract coefficients from ReceptiveField object")

    delays = getattr(rf, "delays_", None)
    if delays is None:
        lags_samples = np.arange(
            int(np.floor(float(tmin_s) * float(sfreq_hz))),
            int(np.ceil(float(tmax_s) * float(sfreq_hz))) + 1,
        )
        lags_s = lags_samples / float(sfreq_hz)
    else:
        lags_s = np.asarray(delays, dtype=float) / float(sfreq_hz)
    return np.asarray(coef), np.asarray(lags_s, dtype=float)


def _fit_full_data_kernel(
    *,
    model_name: str,
    alpha: float,
    per_run_y: dict[int, np.ndarray],
    per_run_X: dict[int, dict[str, np.ndarray]],
    sfreq_hz: float,
    tmin_s: float,
    tmax_s: float,
    regularization_type: str,
) -> tuple[np.ndarray, np.ndarray]:
    gap = max_lag_samples(sfreq_hz=sfreq_hz, tmin_s=tmin_s, tmax_s=tmax_s) + 1
    runs = sorted(per_run_y.keys())
    y_all = concatenate_with_gap([per_run_y[r] for r in runs], gap_samples=gap)
    X_all = concatenate_with_gap([per_run_X[r][model_name] for r in runs], gap_samples=gap)
    X_all, _ = standardize_design_matrix_foldwise(X_all, X_all)
    rf = fit_receptive_field(
        X_train=X_all,
        y_train=y_all,
        sfreq_hz=sfreq_hz,
        tmin_s=tmin_s,
        tmax_s=tmax_s,
        ridge_alpha=float(alpha),
        regularization_type=regularization_type,
    )
    return _extract_rf_coef_and_lags(rf=rf, sfreq_hz=sfreq_hz, tmin_s=tmin_s, tmax_s=tmax_s)


def _subject_mne_info(
    *,
    ds_index: pd.DataFrame,
    dyad_id: str,
    subject: str,
    picks: str | list[str],
    sfreq_hz: float,
) -> Any:
    import mne

    dyad_col = ds_index["dyad_id"].astype(str).str.strip()
    subj_col = ds_index["subject_id"].astype(str).str.strip()
    run_col = pd.to_numeric(ds_index["run"], errors="coerce")
    rows = ds_index.loc[(dyad_col == str(dyad_id).strip()) & (subj_col == str(subject).strip())].copy()
    if rows.empty:
        raise ValueError(f"No dataset rows found for dyad_id={dyad_id}, subject={subject}")

    if "status" in rows.columns:
        rows = rows.loc[rows["status"].astype(str).str.lower() == "ok"].copy()
    rows = rows.loc[rows["eeg_path"].notna()].copy()
    rows = rows.loc[rows["eeg_path"].astype(str).str.strip() != ""].copy()
    if rows.empty:
        raise ValueError(f"No valid EEG paths for dyad_id={dyad_id}, subject={subject}")

    rows["run_num"] = pd.to_numeric(run_col.loc[rows.index], errors="coerce")
    rows = rows.dropna(subset=["run_num"]).sort_values("run_num")
    if rows.empty:
        raise ValueError(f"No runnable rows after run parsing for dyad_id={dyad_id}, subject={subject}")
    eeg_path = Path(str(rows.iloc[0]["eeg_path"])).expanduser()
    raw = mne.io.read_raw(eeg_path, preload=True, verbose="ERROR")
    if picks is not None:
        raw = raw.copy().pick(picks)
    if not np.isclose(float(raw.info["sfreq"]), float(sfreq_hz), atol=1e-9):
        raw.resample(float(sfreq_hz), npad="auto", verbose="ERROR")
    return raw.info.copy()


def _evoked_fname_for_regressor(regressor_name: str) -> str:
    if regressor_name == "self_onset_impulse":
        return "self_onset-ave.fif"
    if regressor_name == "other_offset_impulse":
        return "other_offset-ave.fif"
    return f"{regressor_name}-ave.fif"


def write_subject_output(
    *,
    cfg: dict[str, Any],
    out_dir: Path,
    dyad_id: str,
    subject: str,
    picks: str | list[str] = "eeg",
) -> Path:
    scores, inner_scores, outer_scores, _, trf_results_by_model = _compute_subject_artifacts(
        cfg=cfg,
        out_dir=out_dir,
        dyad_id=dyad_id,
        subject=subject,
        picks=picks,
    )
    _validate_scores_table(scores=scores)

    out_path = _subject_scores_output_path(out_dir=out_dir, dyad_id=dyad_id, subject=subject)
    subject_dir = out_path.parent
    subject_dir.mkdir(parents=True, exist_ok=True)

    trf_cfg = parse_evoked_trf_config(cfg)
    events, ds_index = load_input_tables(out_dir)
    events = validate_events_table(events)
    _, runs, per_run_y, per_run_X, per_run_feature_names, sfreq_hz = _prepare_subject_run_data(
        out_dir=out_dir,
        dyad_id=str(dyad_id),
        subject=str(subject),
        events=events,
        ds_index=ds_index,
        trf_cfg=trf_cfg,
        picks=picks,
    )
    info = _subject_mne_info(
        ds_index=ds_index,
        dyad_id=str(dyad_id),
        subject=str(subject),
        picks=picks,
        sfreq_hz=sfreq_hz,
    )

    subject_tables = trf_scoring.build_subject_score_tables(
        dyad_id=str(dyad_id),
        subject=str(subject),
        scores=scores,
        inner_scores=inner_scores,
        outer_scores=outer_scores,
    )

    # Keep existing subject-level score output for aggregation compatibility.
    subject_tables.summary_fold_sorted.to_parquet(out_path, index=False)
    subject_tables.alpha_grid.to_parquet(subject_dir / "alpha_grid.parquet", index=False)
    subject_tables.inner_cv.to_parquet(subject_dir / "inner_cv.parquet", index=False)
    subject_tables.model_comparison.to_parquet(subject_dir / "model_comparison.parquet", index=False)
    subject_tables.summary.to_parquet(subject_dir / "summary.parquet", index=False)
    # Backward-compatible detailed filenames retained.
    subject_tables.alpha_grid.to_parquet(subject_dir / "alpha_grid_scores.parquet", index=False)
    subject_tables.inner_sorted.to_parquet(subject_dir / "inner_cv_scores.parquet", index=False)
    subject_tables.outer_sorted.to_parquet(subject_dir / "outer_cv_scores.parquet", index=False)

    model_to_regs = {m.model_name: [r.name for r in m.regressors] for m in trf_cfg.models}
    for model_name in model_to_regs:
        model_dir = subject_dir / "models" / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        model_outer = subject_tables.outer_fold_scores.loc[subject_tables.outer_fold_scores["model"] == model_name].copy()
        model_outer = model_outer[
            ["dyad_id", "subject", "model", "test_run", "alpha", "r", "baseline_r", "delta_r"]
        ].sort_values(["test_run"], kind="mergesort")
        model_trf = trf_results_by_model.get(model_name)
        if model_trf is None:
            raise ValueError(f"Missing TRF result for model={model_name}")
        write_trf_artifacts(
            TRFResult(
                kernels_outer=np.asarray(model_trf.kernels_outer),
                chosen_alpha=np.asarray(model_trf.chosen_alpha, dtype=float),
                outer_scores=model_outer,
                lags=np.asarray(model_trf.lags, dtype=float),
                predictions_by_fold=None,
                reconstruction_r2=(
                    model_trf.reconstruction_r2.sort_values(
                        [c for c in ["fold_id", "channel", "time_s"] if c in model_trf.reconstruction_r2.columns],
                        kind="mergesort",
                    ).reset_index(drop=True)
                    if model_trf.reconstruction_r2 is not None and not model_trf.reconstruction_r2.empty
                    else model_trf.reconstruction_r2
                ),
            ),
            model_dir,
        )

        best_alpha_row = subject_tables.best_alpha_by_model.loc[subject_tables.best_alpha_by_model["model"] == model_name]
        if best_alpha_row.empty:
            chosen = np.asarray(model_trf.chosen_alpha, dtype=float).ravel()
            finite = chosen[np.isfinite(chosen)]
            if finite.size > 0:
                best_alpha = float(pd.Series(finite.tolist()).mode().iloc[0])
                LOGGER.warning(
                    "Missing best_alpha table row for model=%s; using mode(chosen_alpha)=%.6g fallback.",
                    model_name,
                    best_alpha,
                )
            else:
                best_alpha = float(trf_cfg.ridge_alpha_default)
                LOGGER.warning(
                    "Missing best_alpha table row for model=%s and no finite chosen_alpha; using default alpha=%.6g.",
                    model_name,
                    best_alpha,
                )
        else:
            best_alpha = float(best_alpha_row.iloc[0]["best_alpha"])
        coef_full, lags_full = _fit_full_data_kernel(
            model_name=model_name,
            alpha=best_alpha,
            per_run_y=per_run_y,
            per_run_X=per_run_X,
            sfreq_hz=sfreq_hz,
            tmin_s=trf_cfg.tmin_s,
            tmax_s=trf_cfg.tmax_s,
            regularization_type=trf_cfg.regularization_type,
        )
        if not np.allclose(np.asarray(model_trf.lags, dtype=float), np.asarray(lags_full, dtype=float)):
            raise ValueError(f"Refit lag grid mismatch for model={model_name}")
        regs = per_run_feature_names[runs[0]][model_name]
        for reg_idx, reg_name in enumerate(regs):
            reg_data = np.asarray(coef_full)[:, reg_idx, :]
            evoked_name = _evoked_fname_for_regressor(reg_name)
            import mne

            evoked = mne.EvokedArray(
                data=reg_data,
                info=info.copy(),
                tmin=float(lags_full[0]),
                comment=str(reg_name),
                nave=len(runs),
                verbose=False,
            )
            evoked.save(model_dir / evoked_name, overwrite=True)

    # Phase 5 confirmatory sidecar: cross-kernel temporal similarity (self_rate vs other_rate).
    if {"self_rate", "other_rate"}.issubset(set(model_to_regs)):
        self_res = trf_results_by_model.get("self_rate")
        other_res = trf_results_by_model.get("other_rate")
        if self_res is not None and other_res is not None:
            self_regs = per_run_feature_names[runs[0]].get("self_rate", [])
            other_regs = per_run_feature_names[runs[0]].get("other_rate", [])
            if "self_rate" in self_regs and "other_rate" in other_regs:
                self_idx = int(self_regs.index("self_rate"))
                other_idx = int(other_regs.index("other_rate"))
                lags = np.asarray(self_res.lags, dtype=float)
                if lags.ndim == 1 and lags.size > 0:
                    t0_index = int(np.argmin(np.abs(lags - 0.0)))
                    sim = cross_kernel_temporal_similarity(
                        kernel_a=np.asarray(self_res.kernels_outer)[:, :, self_idx, :],
                        kernel_b=np.asarray(other_res.kernels_outer)[:, :, other_idx, :],
                        t0_index=t0_index,
                    )
                    sim_arr = np.stack(
                        [
                            np.asarray(sim["self_self_corr"], dtype=float),
                            np.asarray(sim["self_other_corr"], dtype=float),
                        ],
                        axis=0,
                    )
                    np.save(subject_dir / "models" / "self_rate" / "cross_kernel_similarity.npy", sim_arr)

    return out_path


def aggregate_subject_scores(*, out_dir: Path) -> pd.DataFrame:
    subjects_root = out_dir / "evoked_trf" / "subjects"
    subject_paths = sorted(subjects_root.rglob("scores.parquet"))
    if not subject_paths:
        raise ValueError(f"No subject score files found under: {subjects_root}")

    frames = []
    for path in subject_paths:
        try:
            frames.append(pd.read_parquet(path))
        except Exception as exc:
            raise ValueError(f"Failed to read subject scores parquet: {path}") from exc
    scores = pd.concat(frames, ignore_index=True)
    return scores.sort_values(
        ["dyad_id", "subject", "run", "model", "fold_id"],
        kind="mergesort",
    ).reset_index(drop=True)


def write_aggregate_output(*, out_dir: Path) -> Path:
    scores = aggregate_subject_scores(out_dir=out_dir)
    _validate_scores_table(scores=scores)

    out_path = out_dir / "evoked_trf" / "scores" / "scores.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(out_path, index=False)
    return out_path


def write_reconstruction_aggregate(*, out_dir: Path) -> list[Path]:
    subjects_root = out_dir / "evoked_trf" / "subjects"
    paths = sorted(subjects_root.rglob("models/*/reconstruction_metrics.parquet"))
    by_model: dict[str, list[pd.DataFrame]] = {}
    for path in paths:
        model = str(path.parent.name)
        df = pd.read_parquet(path)
        # Inject identifiers from path if absent.
        if "dyad_id" not in df.columns:
            df["dyad_id"] = str(path.parents[3].name)
        if "subject" not in df.columns:
            df["subject"] = str(path.parents[2].name)
        if "model" not in df.columns:
            df["model"] = model
        by_model.setdefault(model, []).append(df)
    out_paths: list[Path] = []
    agg_root = out_dir / "evoked_trf" / "aggregate" / "reconstruction"
    for model, frames in sorted(by_model.items()):
        out_path = agg_root / f"{model}_group.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(frames, ignore_index=True).to_parquet(out_path, index=False)
        out_paths.append(out_path)
    return out_paths


def write_similarity_aggregate(*, out_dir: Path) -> Path:
    subjects_root = out_dir / "evoked_trf" / "subjects"
    sim_paths = sorted(subjects_root.rglob("models/self_rate/cross_kernel_similarity.npy"))
    if not sim_paths:
        raise ValueError(f"No subject similarity files found under {subjects_root}")
    arrays = [np.asarray(np.load(p), dtype=float) for p in sim_paths]
    stacked = np.stack(arrays, axis=0)  # (subject, 2, lag)
    out_path = out_dir / "evoked_trf" / "aggregate" / "similarity" / "self_vs_other.npy"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, stacked)
    return out_path


def make_run_splits_from_scores(scores: pd.DataFrame) -> pd.DataFrame:
    if scores.empty:
        return pd.DataFrame(columns=["dyad_id", "subject", "fold_id", "run", "split"])

    runs = scores[["dyad_id", "subject", "fold_id", "run"]].drop_duplicates()
    rows: list[dict[str, object]] = []
    for (dyad_id, subject, fold_id), g in runs.groupby(["dyad_id", "subject", "fold_id"], sort=True):
        test_run = int(g["run"].iloc[0])
        all_runs = sorted(scores.loc[(scores["dyad_id"] == dyad_id) & (scores["subject"] == subject), "run"].unique())
        for r in all_runs:
            rows.append(
                {
                    "dyad_id": dyad_id,
                    "subject": subject,
                    "fold_id": int(fold_id),
                    "run": int(r),
                    "split": "test" if int(r) == test_run else "train",
                }
            )
    return pd.DataFrame(rows)


def write_detailed_package(
    *,
    out_dir: Path,
    inner_scores: pd.DataFrame,
    outer_scores: pd.DataFrame,
    packaged: list[pd.DataFrame],
) -> None:
    paths = DetailedPaths.from_out_dir(out_dir)

    write_parquet(inner_scores, paths.inner_scores)
    write_parquet(outer_scores, paths.outer_scores)

    # best alpha per dyad/subject/model: mode across outer folds
    if not outer_scores.empty and "chosen_alpha" in outer_scores.columns:
        best_alpha = (
            outer_scores[["dyad_id", "subject", "model", "chosen_alpha"]]
            .drop_duplicates()
            .groupby(["dyad_id", "subject", "model"], as_index=False)["chosen_alpha"]
            .agg(lambda x: float(pd.Series(list(x)).mode().iloc[0]))
            .rename(columns={"chosen_alpha": "best_alpha"})
        )
    else:
        best_alpha = pd.DataFrame(columns=["dyad_id", "subject", "model", "best_alpha"])
    write_parquet(best_alpha, paths.best_alpha)

    # best model per subject
    scores_path = out_dir / "evoked_trf" / "scores" / "scores.parquet"
    if scores_path.exists():
        scores = pd.read_parquet(scores_path)
        means = scores.groupby(["dyad_id", "subject", "model"], as_index=False)["r"].mean(numeric_only=True)
        best = (
            means.sort_values(["dyad_id", "subject", "r"], ascending=[True, True, False])
            .groupby(["dyad_id", "subject"], as_index=False)
            .head(1)
            .rename(columns={"model": "best_model", "r": "mean_r"})
        )
    else:
        best = pd.DataFrame(columns=["dyad_id", "subject", "best_model", "mean_r"])
    write_parquet(best, paths.best_model)

    # Save packaged model params (best model per subject; one file per subject)
    for df in packaged:
        if df.empty:
            continue
        packaged_rows = df.loc[df["packaged_model"].notna()] if "packaged_model" in df.columns else pd.DataFrame()
        if packaged_rows.empty:
            continue
        row = packaged_rows.iloc[0]
        dyad_id = str(row["dyad_id"])
        subject = str(row["subject"])
        model = str(row["packaged_model"])
        alpha = float(row["packaged_alpha"])
        sfreq_hz = float(row["sfreq_hz"])
        tmin_s = float(row["tmin_s"])
        tmax_s = float(row["tmax_s"])
        feature_names = list(row["feature_names"])
        coef = np.asarray(row["coef"])
        intercept = row["intercept"]
        intercept_arr = np.asarray(intercept) if intercept is not None else None

        out_path = paths.models_dir / dyad_id / subject / f"{model}.npz"
        save_receptive_field_npz(
            path=out_path,
            dyad_id=dyad_id,
            subject=subject,
            model=model,
            sfreq_hz=sfreq_hz,
            tmin_s=tmin_s,
            tmax_s=tmax_s,
            alpha=alpha,
            feature_names=feature_names,
            coef=coef,
            intercept=intercept_arr,
        )


def write_outputs(*, cfg: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:

    scores, splits = compute_scores_and_splits(cfg=cfg, out_dir=out_dir)

    scores_out = out_dir / "evoked_trf" / "scores" / "scores.parquet"
    splits_out = out_dir / "evoked_trf" / "qc" / "cv_splits.parquet"
    scores_out.parent.mkdir(parents=True, exist_ok=True)
    splits_out.parent.mkdir(parents=True, exist_ok=True)

    _validate_output_tables(scores=scores, splits=splits)

    scores.to_parquet(scores_out, index=False)
    splits.to_parquet(splits_out, index=False)
    return scores_out, splits_out


def _validate_output_tables(*, scores: pd.DataFrame, splits: pd.DataFrame) -> None:
    _validate_scores_table(scores=scores)

    required_splits = {"dyad_id", "subject", "fold_id", "run", "split"}
    missing_splits = required_splits - set(splits.columns)
    if missing_splits:
        raise ValueError(f"cv_splits table missing columns: {sorted(missing_splits)}")


def _validate_scores_table(*, scores: pd.DataFrame) -> None:
    required_scores = {"subject", "dyad_id", "run", "model", "fold_id", "r"}
    missing_scores = required_scores - set(scores.columns)
    if missing_scores:
        raise ValueError(f"scores table missing columns: {sorted(missing_scores)}")

    if "delta_r" in scores.columns:
        nonbaseline = scores["model"] != "baseline"
        bad = scores.loc[nonbaseline & scores["r"].notna() & scores["delta_r"].isna()]
        if not bad.empty:
            raise ValueError("delta_r has missing values where r exists (non-baseline rows).")
