import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rate.evoked_trf.fit import (
    build_design_matrix,
    fit_receptive_field,
    load_eeg_run,
    predict_receptive_field,
)
from rate.evoked_trf.results_io import DetailedPaths, save_receptive_field_npz, write_parquet
from rate.evoked_trf.spec import add_timelock_columns, get_models
from rate.evoked_trf.tuning import select_alpha_nested_loro
from rate.evoked_trf.utils import (
    concatenate_with_gap,
    max_lag_samples,
    score_multichannel_mean_r,
    standardize_design_matrix_foldwise,
)

LOGGER = logging.getLogger(__name__)


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

    return EvokedTrfConfig(
        downsample_hz=float(ev["downsample_hz"]),
        tmin_s=float(ev["lags_s"]["tmin"]),
        tmax_s=float(ev["lags_s"]["tmax"]),
        scheme=scheme,
        regularization_type=regularization_type,
        ridge_alpha_grid=grid,
        ridge_alpha_default=float(ev.get("ridge_alpha", 1.0)),
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
        "self_rate",
        "other_rate",
        "abs_diff",
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


def list_models() -> list[str]:
    return [m.model_name for m in get_models()]


def build_run_matrices(
    *,
    ev_run: pd.DataFrame,
    sfreq_hz: float,
    n_samples: int,
    sample_t0_s: float,
) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
    X_by_model: dict[str, np.ndarray] = {}
    names_by_model: dict[str, list[str]] = {}
    for m in get_models():
        X, feature_names = build_design_matrix(
            events=ev_run,
            model=m,
            sfreq_hz=sfreq_hz,
            n_samples=n_samples,
            sample_t0_s=sample_t0_s,
            return_feature_names=True,
        )
        X_by_model[m.model_name] = X
        names_by_model[m.model_name] = feature_names
    return X_by_model, names_by_model


# =============================================================================
# CV + scoring
# =============================================================================


def _outer_folds_from_runs(runs: list[int]) -> list[tuple[list[int], int, int]]:
    folds: list[tuple[list[int], int, int]] = []
    for fold_id, test_run in enumerate(runs):
        train_runs = [r for r in runs if r != test_run]
        folds.append((train_runs, test_run, fold_id))
    return folds


def _score_alpha_from_prepared(
    *,
    alpha: float,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    sfreq_hz: float,
    tmin_s: float,
    tmax_s: float,
    regularization_type: str,
) -> float:
    rf = fit_receptive_field(
        X_train=X_train,
        y_train=y_train,
        sfreq_hz=sfreq_hz,
        tmin_s=tmin_s,
        tmax_s=tmax_s,
        ridge_alpha=float(alpha),
        regularization_type=regularization_type,
    )
    y_hat = predict_receptive_field(rf, X_test)
    return score_multichannel_mean_r(y_test, y_hat)


def _outer_scores_full_table(
    *,
    dyad_id: str,
    subject: str,
    model_name: str,
    fold_id: int,
    train_runs: list[int],
    test_run: int,
    alphas: list[float],
    per_run_y: dict[int, np.ndarray],
    per_run_X: dict[int, dict[str, np.ndarray]],
    sfreq_hz: float,
    tmin_s: float,
    tmax_s: float,
    regularization_type: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    gap = max_lag_samples(sfreq_hz=sfreq_hz, tmin_s=tmin_s, tmax_s=tmax_s) + 1

    # 🔹 PREPARE FOLD ONCE
    y_train = concatenate_with_gap([per_run_y[r] for r in train_runs], gap_samples=gap)
    y_test = per_run_y[test_run]

    X_train = concatenate_with_gap([per_run_X[r][model_name] for r in train_runs], gap_samples=gap)
    X_test = per_run_X[test_run][model_name]

    X_train, X_test = standardize_design_matrix_foldwise(X_train, X_test)

    # 🔹 LOOP ONLY ALPHA FITS
    for alpha in alphas:
        r = _score_alpha_from_prepared(
            alpha=float(alpha),
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            sfreq_hz=sfreq_hz,
            tmin_s=tmin_s,
            tmax_s=tmax_s,
            regularization_type=regularization_type,
        )

        rows.append(
            {
                "dyad_id": dyad_id,
                "subject": subject,
                "model": model_name,
                "fold_id": int(fold_id),
                "test_run": int(test_run),
                "alpha": float(alpha),
                "r": float(r),
            }
        )

    return pd.DataFrame(rows)


def _choose_best_alpha_inner(
    *,
    dyad_id: str,
    subject: str,
    model_name: str,
    train_runs: list[int],
    per_run_y: dict[int, np.ndarray],
    per_run_X: dict[int, dict[str, np.ndarray]],
    alphas: list[float],
    sfreq_hz: float,
    tmin_s: float,
    tmax_s: float,
    alpha_default: float,
    regularization_type: str,
) -> tuple[float, pd.DataFrame]:
    if len(train_runs) < 2:
        df = pd.DataFrame(
            [
                {
                    "dyad_id": dyad_id,
                    "subject": subject,
                    "model": model_name,
                    "alpha": float(alpha_default),
                    "inner_fold": 0,
                    "test_run": None,
                    "r": float("nan"),
                    "note": "inner_cv_skipped_only_one_training_run",
                }
            ]
        )
        return float(alpha_default), df

    res = select_alpha_nested_loro(
        dyad_id=dyad_id,
        subject=subject,
        model_name=model_name,
        runs_train_outer=train_runs,
        per_run_y=per_run_y,
        per_run_X=per_run_X,
        alphas=alphas,
        sfreq_hz=sfreq_hz,
        tmin_s=tmin_s,
        tmax_s=tmax_s,
        regularization_type=regularization_type,
    )
    return float(res.best_alpha), res.table


# =============================================================================
# Per-subject worker (parallel-safe)
# =============================================================================


def _compute_one_dyad_subject(
    *,
    dyad_id: str,
    subject: str,
    events: pd.DataFrame,
    ds_index: pd.DataFrame,
    trf_cfg: EvokedTrfConfig,
    picks: str | list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _, runs, per_run_y, per_run_X, per_run_feature_names, sfreq_hz = _prepare_subject_run_data(
        dyad_id=dyad_id,
        subject=subject,
        events=events,
        ds_index=ds_index,
        trf_cfg=trf_cfg,
        picks=picks,
    )

    models = list_models()
    outer_folds = _outer_folds_from_runs(runs)

    inner_rows: list[pd.DataFrame] = []
    outer_rows: list[pd.DataFrame] = []
    best_alpha_rows: list[dict[str, object]] = []

    for train_runs, test_run, fold_id in outer_folds:
        for model_name in models:
            best_alpha, inner_df = _choose_best_alpha_inner(
                dyad_id=dyad_id,
                subject=subject,
                model_name=model_name,
                train_runs=train_runs,
                per_run_y=per_run_y,
                per_run_X=per_run_X,
                alphas=trf_cfg.ridge_alpha_grid,
                sfreq_hz=sfreq_hz,
                tmin_s=trf_cfg.tmin_s,
                tmax_s=trf_cfg.tmax_s,
                alpha_default=trf_cfg.ridge_alpha_default,
                regularization_type=trf_cfg.regularization_type,
            )
            inner_df = inner_df.assign(outer_fold=int(fold_id), outer_test_run=int(test_run))
            inner_rows.append(inner_df)

            best_alpha_rows.append(
                {
                    "dyad_id": dyad_id,
                    "subject": subject,
                    "model": model_name,
                    "fold_id": int(fold_id),
                    "test_run": int(test_run),
                    "best_alpha": float(best_alpha),
                }
            )

            outer_full = _outer_scores_full_table(
                dyad_id=dyad_id,
                subject=subject,
                model_name=model_name,
                fold_id=int(fold_id),
                train_runs=train_runs,
                test_run=int(test_run),
                alphas=trf_cfg.ridge_alpha_grid,
                per_run_y=per_run_y,
                per_run_X=per_run_X,
                sfreq_hz=sfreq_hz,
                tmin_s=trf_cfg.tmin_s,
                tmax_s=trf_cfg.tmax_s,
                regularization_type=trf_cfg.regularization_type,
            )
            outer_full["chosen_alpha"] = float(best_alpha)
            outer_rows.append(outer_full)

    inner_scores = pd.concat(inner_rows, ignore_index=True) if inner_rows else pd.DataFrame()
    outer_scores = pd.concat(outer_rows, ignore_index=True) if outer_rows else pd.DataFrame()
    best_alpha = pd.DataFrame(best_alpha_rows)

    # Summary delta_r using chosen_alpha score for each fold/model vs baseline chosen_alpha
    summary = summarize_scores(dyad_id=dyad_id, subject=subject, outer_scores=outer_scores)

    # Best model per subject = max mean delta_r (excluding baseline)
    best_model = best_model_table(dyad_id=dyad_id, subject=subject, summary=summary)

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

    return summary, inner_scores, outer_scores, pd.concat([best_alpha, final_model_rows], ignore_index=True)


def _prepare_subject_run_data(
    *,
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

    for run in runs:
        eeg_path = select_eeg_path(ds_index, subject=subject, dyad_id=dyad_id, run=int(run))
        run_data = load_eeg_run(eeg_path, downsample_hz=trf_cfg.downsample_hz, picks=picks)

        y = run_data.y
        sfreq_hz = float(run_data.sfreq_hz)
        sfreqs.add(sfreq_hz)

        ev_run = ev_subj.loc[ev_subj["run"].astype(int) == int(run)].copy()
        X_by_model, names_by_model = build_run_matrices(
            ev_run=ev_run,
            sfreq_hz=sfreq_hz,
            n_samples=y.shape[0],
            sample_t0_s=float(run_data.times_s[0]),
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
    """
    Produce the main scores.parquet table:
    subject, dyad_id, run, model, fold_id, r, r_baseline, delta_r

    We use r at chosen_alpha for each (model, fold).

    DataFrame format example
    ------------------------
    dyad_id   subject   run   model      fold_id   r     r_baseline  delta_r
    dyad-001  sub-001   1     baseline   0         0.01  0.01        0.00
    dyad-001  sub-001   1     self_rate  0         0.02  0.01        0.01
    """
    required = {"dyad_id", "subject", "model", "fold_id", "test_run", "alpha", "r", "chosen_alpha"}
    missing = required - set(outer_scores.columns)
    if missing:
        raise ValueError(f"outer_scores missing columns: {sorted(missing)}")

    chosen = outer_scores.loc[outer_scores["alpha"] == outer_scores["chosen_alpha"]].copy()
    chosen = chosen.rename(columns={"test_run": "run"})
    chosen = chosen[["dyad_id", "subject", "run", "model", "fold_id", "r"]]

    base = chosen.loc[chosen["model"] == "baseline", ["dyad_id", "subject", "run", "fold_id", "r"]].rename(
        columns={"r": "r_baseline"}
    )

    merged = chosen.merge(base, on=["dyad_id", "subject", "run", "fold_id"], how="left")
    merged["delta_r"] = merged["r"] - merged["r_baseline"]
    return merged


def best_model_table(*, dyad_id: str, subject: str, summary: pd.DataFrame) -> pd.DataFrame:
    """
    Best model per subject by mean delta_r across folds (excluding baseline).
    """
    nonbase = summary.loc[summary["model"] != "baseline"].copy()
    means = nonbase.groupby("model", as_index=False)["delta_r"].mean(numeric_only=True)
    means = means.sort_values(["delta_r", "model"], ascending=[False, True])
    best = means.iloc[0].to_dict()
    return pd.DataFrame(
        [
            {
                "dyad_id": dyad_id,
                "subject": subject,
                "best_model": str(best["model"]),
                "mean_delta_r": float(best["delta_r"]),
            }
        ]
    )


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
    summary, _, _, _ = _compute_subject_artifacts(
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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trf_cfg = parse_evoked_trf_config(cfg)
    if trf_cfg.scheme != "leave_one_run_out":
        raise ValueError(f"Unsupported evoked_trf.cv.scheme: {trf_cfg.scheme}")

    events, ds_index = load_input_tables(out_dir)
    events = validate_events_table(events)

    summary, inner_scores, outer_scores, packaged = _compute_one_dyad_subject(
        dyad_id=str(dyad_id),
        subject=str(subject),
        events=events,
        ds_index=ds_index,
        trf_cfg=trf_cfg,
        picks=picks,
    )
    return summary, inner_scores, outer_scores, packaged


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


def _fit_outer_fold_kernel(
    *,
    model_name: str,
    train_runs: list[int],
    alpha: float,
    per_run_y: dict[int, np.ndarray],
    per_run_X: dict[int, dict[str, np.ndarray]],
    sfreq_hz: float,
    tmin_s: float,
    tmax_s: float,
    regularization_type: str,
) -> tuple[np.ndarray, np.ndarray]:
    gap = max_lag_samples(sfreq_hz=sfreq_hz, tmin_s=tmin_s, tmax_s=tmax_s) + 1
    y_train = concatenate_with_gap([per_run_y[r] for r in train_runs], gap_samples=gap)
    X_train = concatenate_with_gap([per_run_X[r][model_name] for r in train_runs], gap_samples=gap)
    X_train, _ = standardize_design_matrix_foldwise(X_train, X_train)
    rf = fit_receptive_field(
        X_train=X_train,
        y_train=y_train,
        sfreq_hz=sfreq_hz,
        tmin_s=tmin_s,
        tmax_s=tmax_s,
        ridge_alpha=float(alpha),
        regularization_type=regularization_type,
    )
    return _extract_rf_coef_and_lags(rf=rf, sfreq_hz=sfreq_hz, tmin_s=tmin_s, tmax_s=tmax_s)


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
    scores, inner_scores, outer_scores, _ = _compute_subject_artifacts(
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

    summary_fold_sorted = scores.sort_values(["dyad_id", "subject", "run", "model", "fold_id"], kind="mergesort").reset_index(
        drop=True
    )
    inner_sorted = (
        inner_scores.sort_values(["dyad_id", "subject", "model", "outer_fold", "inner_fold", "alpha"], kind="mergesort")
        .reset_index(drop=True)
        if not inner_scores.empty
        else inner_scores
    )
    outer_sorted = (
        outer_scores.sort_values(["dyad_id", "subject", "model", "fold_id", "alpha"], kind="mergesort").reset_index(
            drop=True
        )
        if not outer_scores.empty
        else outer_scores
    )

    # All alpha x outer-fold evaluations (no reduction).
    alpha_grid = outer_sorted.copy()

    # Inner CV transparency: mean score per alpha.
    if inner_sorted.empty:
        inner_cv = pd.DataFrame(columns=["dyad_id", "subject", "model", "alpha", "mean_r"])
    else:
        inner_cv = (
            inner_sorted.groupby(["dyad_id", "subject", "model", "alpha"], as_index=False)["r"]
            .mean(numeric_only=True)
            .rename(columns={"r": "mean_r"})
            .sort_values(["dyad_id", "subject", "model", "alpha"], kind="mergesort")
            .reset_index(drop=True)
        )

    # Per-fold outer scores at chosen alpha with baseline comparison.
    selected = outer_sorted.loc[np.isclose(outer_sorted["alpha"], outer_sorted["chosen_alpha"])].copy()
    selected = selected[["dyad_id", "subject", "model", "fold_id", "test_run", "alpha", "r"]]
    base = selected.loc[selected["model"] == "baseline", ["fold_id", "test_run", "r"]].rename(columns={"r": "baseline_r"})
    outer_fold_scores = selected.merge(base, on=["fold_id", "test_run"], how="left")
    outer_fold_scores["delta_r"] = outer_fold_scores["r"] - outer_fold_scores["baseline_r"]
    outer_fold_scores = outer_fold_scores.sort_values(["model", "test_run"], kind="mergesort").reset_index(drop=True)

    # One row per model summary for aggregation/reporting.
    best_alpha_by_model = (
        selected.groupby("model", as_index=False)["alpha"]
        .agg(lambda x: float(pd.Series(list(x)).mode().iloc[0]))
        .rename(columns={"alpha": "best_alpha"})
    )
    model_comparison = (
        outer_fold_scores.groupby("model", as_index=False)
        .agg(
            mean_r=("r", "mean"),
            mean_baseline_r=("baseline_r", "mean"),
            mean_delta_r=("delta_r", "mean"),
        )
        .merge(best_alpha_by_model, on="model", how="left")
        .sort_values(["mean_delta_r", "model"], ascending=[False, True], kind="mergesort")
        .reset_index(drop=True)
    )
    summary = model_comparison.assign(dyad_id=str(dyad_id), subject=str(subject))[
        ["dyad_id", "subject", "model", "mean_r", "mean_baseline_r", "mean_delta_r", "best_alpha"]
    ]

    # Keep existing subject-level score output for aggregation compatibility.
    summary_fold_sorted.to_parquet(out_path, index=False)
    alpha_grid.to_parquet(subject_dir / "alpha_grid.parquet", index=False)
    inner_cv.to_parquet(subject_dir / "inner_cv.parquet", index=False)
    model_comparison.to_parquet(subject_dir / "model_comparison.parquet", index=False)
    summary.to_parquet(subject_dir / "summary.parquet", index=False)
    # Backward-compatible detailed filenames retained.
    alpha_grid.to_parquet(subject_dir / "alpha_grid_scores.parquet", index=False)
    inner_sorted.to_parquet(subject_dir / "inner_cv_scores.parquet", index=False)
    outer_sorted.to_parquet(subject_dir / "outer_cv_scores.parquet", index=False)

    model_to_regs = {m.model_name: [r.name for r in m.regressors] for m in get_models()}
    for model_name in model_to_regs:
        model_dir = subject_dir / "models" / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        model_outer = outer_fold_scores.loc[outer_fold_scores["model"] == model_name].copy()
        model_outer = model_outer[
            ["dyad_id", "subject", "model", "test_run", "alpha", "r", "baseline_r", "delta_r"]
        ].sort_values(["test_run"], kind="mergesort")
        model_outer.to_parquet(model_dir / "outer_scores.parquet", index=False)

        fold_kernels: list[np.ndarray] = []
        fold_lags: np.ndarray | None = None
        for row in model_outer.itertuples(index=False):
            test_run = int(row.test_run)
            train_runs = [r for r in runs if int(r) != test_run]
            coef_fold, lags_fold = _fit_outer_fold_kernel(
                model_name=model_name,
                train_runs=train_runs,
                alpha=float(row.alpha),
                per_run_y=per_run_y,
                per_run_X=per_run_X,
                sfreq_hz=sfreq_hz,
                tmin_s=trf_cfg.tmin_s,
                tmax_s=trf_cfg.tmax_s,
                regularization_type=trf_cfg.regularization_type,
            )
            # kernels_outer.npy axis order: (outer_fold, channel, regressor, lag)
            fold_kernels.append(coef_fold)
            if fold_lags is None:
                fold_lags = lags_fold
            elif not np.allclose(fold_lags, lags_fold):
                raise ValueError(f"Inconsistent lag grid across folds for model={model_name}")
        if not fold_kernels:
            raise ValueError(f"No outer-fold kernels generated for model={model_name}")
        np.save(model_dir / "kernels_outer.npy", np.stack(fold_kernels, axis=0))
        np.save(model_dir / "lags.npy", np.asarray(fold_lags, dtype=float))

        best_alpha_row = best_alpha_by_model.loc[best_alpha_by_model["model"] == model_name]
        if best_alpha_row.empty:
            raise ValueError(f"Missing best_alpha for model={model_name}")
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
        if not np.allclose(np.asarray(fold_lags, dtype=float), np.asarray(lags_full, dtype=float)):
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

    return out_path


def aggregate_subject_scores(*, out_dir: Path) -> pd.DataFrame:
    subjects_root = out_dir / "evoked_trf" / "subjects"
    subject_paths = sorted(subjects_root.rglob("scores.parquet"))
    if not subject_paths:
        raise ValueError(f"No subject score files found under: {subjects_root}")

    frames = [pd.read_parquet(path) for path in subject_paths]
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
        nonbase = scores.loc[scores["model"] != "baseline"].copy()
        means = nonbase.groupby(["dyad_id", "subject", "model"], as_index=False)["delta_r"].mean(numeric_only=True)
        best = (
            means.sort_values(["dyad_id", "subject", "delta_r"], ascending=[True, True, False])
            .groupby(["dyad_id", "subject"], as_index=False)
            .head(1)
            .rename(columns={"model": "best_model", "delta_r": "mean_delta_r"})
        )
    else:
        best = pd.DataFrame(columns=["dyad_id", "subject", "best_model", "mean_delta_r"])
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
    required_scores = {"subject", "dyad_id", "run", "model", "fold_id", "r", "r_baseline", "delta_r"}
    missing_scores = required_scores - set(scores.columns)
    if missing_scores:
        raise ValueError(f"scores table missing columns: {sorted(missing_scores)}")

    nonbaseline = scores["model"] != "baseline"
    bad = scores.loc[nonbaseline & scores["r"].notna() & scores["delta_r"].isna()]
    if not bad.empty:
        raise ValueError("delta_r has missing values where r exists (non-baseline rows).")
