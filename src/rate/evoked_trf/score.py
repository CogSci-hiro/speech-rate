from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from rate.evoked_trf.cv import BlockedWithinRunCV, bins_to_sample_mask
from rate.evoked_trf.fit import (
    build_design_matrix,
    fit_receptive_field,
    load_eeg_run,
    predict_receptive_field,
)
from rate.evoked_trf.spec import add_timelock_columns, get_models


# =============================================================================
# Scoring utilities
# =============================================================================


def pearsonr_1d(a: np.ndarray, b: np.ndarray) -> float:
    """
    Pearson correlation for 1D arrays. Returns NaN if variance is zero.

    Usage example
    -------------
        r = pearsonr_1d(y_true, y_pred)
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size != b.size:
        raise ValueError("a and b must have the same length")

    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt(np.sum(a * a) * np.sum(b * b))
    if denom == 0.0:
        return float("nan")
    return float(np.sum(a * b) / denom)


def score_multichannel_mean_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Score as mean Pearson r across channels.

    Parameters
    ----------
    y_true, y_pred
        Arrays of shape (n_samples, n_channels).

    Usage example
    -------------
        r = score_multichannel_mean_r(y_test, y_hat)
    """
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    rs: list[float] = []
    for ch in range(y_true.shape[1]):
        rs.append(pearsonr_1d(y_true[:, ch], y_pred[:, ch]))
    return float(np.nanmean(rs))


def standardize_design_matrix_foldwise(
    X_train: np.ndarray, X_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Standardize X using training statistics only (no leakage).

    Any constant feature in training is left unchanged (std set to 1).

    Usage example
    -------------
        Xtr, Xte = standardize_design_matrix_foldwise(X_train, X_test)
    """
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std = np.where(std == 0.0, 1.0, std)
    return (X_train - mean) / std, (X_test - mean) / std


# =============================================================================
# Config + IO
# =============================================================================


@dataclass(frozen=True)
class EvokedTrfConfig:
    """
    Minimal configuration required for evoked TRF scoring.

    Usage example
    -------------
        c = EvokedTrfConfig(
            downsample_hz=100.0, tmin_s=-0.2, tmax_s=0.8,
            k_folds=5, time_bin_s=10.0, scheme="blocked_kfold_within_run",
            ridge_alpha=1.0, seed=0
        )
    """

    downsample_hz: float
    tmin_s: float
    tmax_s: float
    scheme: str
    k_folds: int
    time_bin_s: float
    ridge_alpha: float
    seed: int


def parse_evoked_trf_config(cfg: dict[str, Any]) -> EvokedTrfConfig:
    """
    Parse and validate evoked_trf config block.

    Usage example
    -------------
        trf_cfg = parse_evoked_trf_config(cfg)
    """
    ev = cfg["evoked_trf"]
    cv_cfg = ev["cv"]
    seed = int(cv_cfg.get("seed", cfg.get("project", {}).get("random_seed", 0)))
    return EvokedTrfConfig(
        downsample_hz=float(ev["downsample_hz"]),
        tmin_s=float(ev["lags_s"]["tmin"]),
        tmax_s=float(ev["lags_s"]["tmax"]),
        scheme=str(cv_cfg.get("scheme")),
        k_folds=int(cv_cfg["k_folds"]),
        time_bin_s=float(cv_cfg["time_bin_s"]),
        ridge_alpha=float(ev.get("ridge_alpha", 1.0)),
        seed=seed,
    )


def load_input_tables(out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load required input tables.

    Usage example
    -------------
        events, ds_index = load_input_tables(Path(cfg["paths"]["out_dir"]))
    """
    events_path = Path(out_dir) / "events" / "events.parquet"
    ds_index_path = Path(out_dir) / "dataset" / "index.parquet"
    return pd.read_parquet(events_path), pd.read_parquet(ds_index_path)


def validate_events_table(events: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and augment the events table.

    Usage example
    -------------
        events = validate_events_table(events)
    """
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
    """
    Resolve EEG path for a specific (dyad_id, subject_id, run) from dataset index.

    Usage example
    -------------
        p = select_eeg_path(ds_index, subject="sub-001", dyad_id="dyad-01", run=1)
    """
    sub = ds_index.loc[
        (ds_index["subject_id"] == subject)
        & (ds_index["dyad_id"] == dyad_id)
        & (ds_index["run"] == int(run))
    ]
    if sub.empty:
        raise ValueError(f"No dataset index row for dyad_id={dyad_id}, subject_id={subject}, run={run}")
    if "eeg_path" not in sub.columns:
        raise ValueError("dataset index missing required column: eeg_path")
    return Path(sub.iloc[0]["eeg_path"])


# =============================================================================
# Core computation
# =============================================================================


def make_cv_splits(
    *,
    trf_cfg: EvokedTrfConfig,
    dyad_id: str,
    run: int,
    times_s: np.ndarray,
    sfreq_hz: float,
) -> pd.DataFrame:
    """
    Create blocked within-run CV splits and return split audit table.

    Usage example
    -------------
        splits = make_cv_splits(trf_cfg=trf_cfg, dyad_id="d1", run=1, times_s=t, sfreq_hz=100.0)
    """
    run_duration_s = float(times_s[-1] + 1.0 / sfreq_hz)
    cv = BlockedWithinRunCV(k_folds=trf_cfg.k_folds, time_bin_s=trf_cfg.time_bin_s, seed=trf_cfg.seed)
    return cv.make_splits(run_duration_s=run_duration_s, dyad_id=dyad_id, run=int(run))


def build_design_matrices_for_run(
    *,
    ev_run: pd.DataFrame,
    sfreq_hz: float,
    n_samples: int,
) -> tuple[Any, dict[str, np.ndarray]]:
    """
    Build non-lagged design matrices for all models (RF expands lags internally).

    Returns
    -------
    baseline_model, X_by_model

    Usage example
    -------------
        baseline_model, X_by_model = build_design_matrices_for_run(ev_run=ev, sfreq_hz=sfreq, n_samples=n)
    """
    models = get_models()
    baseline_model = next(m for m in models if m.model_name == "baseline")
    X_by_model: dict[str, np.ndarray] = {}
    for model in models:
        X_by_model[model.model_name] = build_design_matrix(
            events=ev_run,
            model=model,
            sfreq_hz=sfreq_hz,
            n_samples=n_samples,
        )
    return baseline_model, X_by_model


def get_fold_masks(
    *, splits: pd.DataFrame, fold_id: int, times_s: np.ndarray, context: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert split table into (train_mask, test_mask) for a fold.

    Usage example
    -------------
        train_mask, test_mask = get_fold_masks(splits=splits, fold_id=0, times_s=t, context="dyad-01/run1")
    """
    train_mask = bins_to_sample_mask(splits=splits, fold_id=fold_id, split="train", times_s=times_s)
    test_mask = bins_to_sample_mask(splits=splits, fold_id=fold_id, split="test", times_s=times_s)
    if np.any(train_mask & test_mask):
        raise RuntimeError(f"Leakage detected: overlapping samples in train and test ({context}, fold={fold_id})")
    return train_mask, test_mask


def fit_and_score_one_model(
    *,
    X: np.ndarray,
    y: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    trf_cfg: EvokedTrfConfig,
    sfreq_hz: float,
) -> float:
    """
    Fit ReceptiveField on train split, predict on test, and return mean channel r.

    Usage example
    -------------
        r = fit_and_score_one_model(X=X, y=y, train_mask=tr, test_mask=te, trf_cfg=cfg, sfreq_hz=sfreq)
    """
    y_train = y[train_mask, :]
    y_test = y[test_mask, :]

    X_train = X[train_mask, :]
    X_test = X[test_mask, :]

    X_train, X_test = standardize_design_matrix_foldwise(X_train, X_test)

    rf = fit_receptive_field(
        X_train=X_train,
        y_train=y_train,
        sfreq_hz=sfreq_hz,
        tmin_s=trf_cfg.tmin_s,
        tmax_s=trf_cfg.tmax_s,
        ridge_alpha=trf_cfg.ridge_alpha,
    )
    y_hat = predict_receptive_field(rf, X_test)
    return score_multichannel_mean_r(y_test, y_hat)


def run_scores_for_one_recording(
    *,
    dyad_id: str,
    subject: str,
    run: int,
    ev_run: pd.DataFrame,
    ds_index: pd.DataFrame,
    trf_cfg: EvokedTrfConfig,
    picks: str | list[str],
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    """
    Compute all fold scores for one (dyad_id, subject_id, run).

    Returns
    -------
    score_rows, splits_df

    Usage example
    -------------
        rows, splits = run_scores_for_one_recording(...)
    """
    eeg_path = select_eeg_path(ds_index, subject=subject, dyad_id=dyad_id, run=run)
    run_data = load_eeg_run(eeg_path, downsample_hz=trf_cfg.downsample_hz, picks=picks)

    y = run_data.y
    sfreq_hz = run_data.sfreq_hz
    times_s = run_data.times_s

    splits = make_cv_splits(trf_cfg=trf_cfg, dyad_id=dyad_id, run=run, times_s=times_s, sfreq_hz=sfreq_hz)

    _, X_by_model = build_design_matrices_for_run(ev_run=ev_run, sfreq_hz=sfreq_hz, n_samples=y.shape[0])
    baseline_X = X_by_model["baseline"]
    other_models = [k for k in X_by_model.keys() if k != "baseline"]

    rows: list[dict[str, object]] = []
    context = f"dyad_id={dyad_id}, subject={subject}, run={run}"

    for fold_id in range(trf_cfg.k_folds):
        train_mask, test_mask = get_fold_masks(splits=splits, fold_id=fold_id, times_s=times_s, context=context)

        r_base = fit_and_score_one_model(
            X=baseline_X,
            y=y,
            train_mask=train_mask,
            test_mask=test_mask,
            trf_cfg=trf_cfg,
            sfreq_hz=sfreq_hz,
        )
        rows.append(
            {
                "subject": subject,
                "dyad_id": dyad_id,
                "run": int(run),
                "model": "baseline",
                "fold_id": int(fold_id),
                "r": float(r_base),
                "r_baseline": float(r_base),
                "delta_r": float(0.0),
            }
        )

        for model_name in other_models:
            r = fit_and_score_one_model(
                X=X_by_model[model_name],
                y=y,
                train_mask=train_mask,
                test_mask=test_mask,
                trf_cfg=trf_cfg,
                sfreq_hz=sfreq_hz,
            )
            delta = float(r - r_base) if np.isfinite(r) and np.isfinite(r_base) else float("nan")
            rows.append(
                {
                    "subject": subject,
                    "dyad_id": dyad_id,
                    "run": int(run),
                    "model": str(model_name),
                    "fold_id": int(fold_id),
                    "r": float(r),
                    "r_baseline": float(r_base),
                    "delta_r": delta,
                }
            )

    return rows, splits


def compute_scores_and_splits(
    *,
    cfg: dict[str, Any],
    out_dir: Path,
    picks: str | list[str] = "eeg",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute evoked TRF scores and split audit tables.

    Outputs
    -------
    scores
        subject, dyad_id, run, model, fold_id, r, r_baseline, delta_r
    cv_splits
        dyad_id, run, fold_id, bin_id, bin_t0, bin_t1, split

    Usage example
    -------------
        scores, splits = compute_scores_and_splits(cfg=cfg, out_dir=Path(cfg["paths"]["out_dir"]))
    """
    trf_cfg = parse_evoked_trf_config(cfg)
    if trf_cfg.scheme != "blocked_kfold_within_run":
        raise ValueError(f"Unsupported evoked_trf.cv.scheme: {trf_cfg.scheme}")

    events, ds_index = load_input_tables(out_dir)
    events = validate_events_table(events)

    all_score_rows: list[dict[str, object]] = []
    all_splits: list[pd.DataFrame] = []

    grouped = events.groupby(["dyad_id", "subject_id", "run"], sort=True)
    for (dyad_id, subject, run), ev_run in grouped:
        rows, splits = run_scores_for_one_recording(
            dyad_id=str(dyad_id),
            subject=str(subject),
            run=int(run),
            ev_run=ev_run,
            ds_index=ds_index,
            trf_cfg=trf_cfg,
            picks=picks,
        )
        all_score_rows.extend(rows)
        all_splits.append(splits)

    scores_df = pd.DataFrame(all_score_rows)
    splits_df = pd.concat(all_splits, axis=0, ignore_index=True) if all_splits else pd.DataFrame()
    return scores_df, splits_df


# =============================================================================
# Outputs
# =============================================================================


def write_outputs(*, cfg: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    """
    Write parquet outputs to:
    - {out_dir}/evoked_trf/scores/scores.parquet
    - {out_dir}/evoked_trf/qc/cv_splits.parquet

    Usage example
    -------------
        scores_path, splits_path = write_outputs(cfg=cfg, out_dir=Path(cfg["paths"]["out_dir"]))
    """
    scores, splits = compute_scores_and_splits(cfg=cfg, out_dir=out_dir)

    scores_out = Path(out_dir) / "evoked_trf" / "scores" / "scores.parquet"
    splits_out = Path(out_dir) / "evoked_trf" / "qc" / "cv_splits.parquet"
    scores_out.parent.mkdir(parents=True, exist_ok=True)
    splits_out.parent.mkdir(parents=True, exist_ok=True)

    _validate_output_tables(scores=scores, splits=splits)

    scores.to_parquet(scores_out, index=False)
    splits.to_parquet(splits_out, index=False)
    return scores_out, splits_out


def _validate_output_tables(*, scores: pd.DataFrame, splits: pd.DataFrame) -> None:
    """
    Validate output schemas and basic missingness constraints.

    Usage example
    -------------
        _validate_output_tables(scores=scores, splits=splits)
    """
    required_scores = {"subject", "dyad_id", "run", "model", "fold_id", "r", "r_baseline", "delta_r"}
    missing_scores = required_scores - set(scores.columns)
    if missing_scores:
        raise ValueError(f"scores table missing columns: {sorted(missing_scores)}")

    required_splits = {"dyad_id", "run", "fold_id", "bin_id", "bin_t0", "bin_t1", "split"}
    missing_splits = required_splits - set(splits.columns)
    if missing_splits:
        raise ValueError(f"cv_splits table missing columns: {sorted(missing_splits)}")

    # delta_r should exist whenever r exists for non-baseline models
    nonbaseline = scores["model"] != "baseline"
    if nonbaseline.any():
        bad = scores.loc[nonbaseline & scores["r"].notna() & scores["delta_r"].isna()]
        if not bad.empty:
            raise ValueError("delta_r has missing values where r exists (non-baseline rows).")
