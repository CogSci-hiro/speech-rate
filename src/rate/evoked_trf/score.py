from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

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
    k_folds: int | None
    time_bin_s: float | None
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

    scheme = str(cv_cfg.get("scheme"))
    k_folds = cv_cfg.get("k_folds", None)
    time_bin_s = cv_cfg.get("time_bin_s", None)

    return EvokedTrfConfig(
        downsample_hz=float(ev["downsample_hz"]),
        tmin_s=float(ev["lags_s"]["tmin"]),
        tmax_s=float(ev["lags_s"]["tmax"]),
        scheme=scheme,
        k_folds=int(k_folds) if k_folds is not None else None,
        time_bin_s=float(time_bin_s) if time_bin_s is not None else None,
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

    This function normalizes types and strips whitespace to avoid false negatives
    when parquet stores columns as category/object/int inconsistently.

    Usage example
    -------------
        p = select_eeg_path(ds_index, subject="sub-001", dyad_id="dyad-01", run=1)
    """
    required = {"dyad_id", "subject_id", "run", "eeg_path"}
    missing = required - set(ds_index.columns)
    if missing:
        raise ValueError(f"dataset index missing required columns: {sorted(missing)}")

    # Normalize matching keys (string columns may be categorical, may contain spaces)
    dyad_col = ds_index["dyad_id"].astype(str).str.strip()
    subj_col = ds_index["subject_id"].astype(str).str.strip()

    # Normalize run (may be string/float/etc.)
    run_col = pd.to_numeric(ds_index["run"], errors="coerce")

    target_dyad = str(dyad_id).strip()
    target_subj = str(subject).strip()
    target_run = int(run)

    mask = (dyad_col == target_dyad) & (subj_col == target_subj) & (run_col == target_run)
    sub = ds_index.loc[mask].copy()

    if sub.empty:
        # Helpful debugging: show near-misses for this dyad/run
        dyad_run = ds_index.loc[(dyad_col == target_dyad) & (run_col == target_run)].copy()
        available_subjects = (
            dyad_run["subject_id"].astype(str).str.strip().drop_duplicates().sort_values().tolist()
            if not dyad_run.empty
            else []
        )
        raise ValueError(
            f"No dataset index row for dyad_id={target_dyad}, subject_id={target_subj}, run={target_run}. "
            f"Available subjects for dyad/run: {available_subjects}"
        )

    eeg_path = Path(str(sub.iloc[0]["eeg_path"])).expanduser()
    return eeg_path



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


def dyad_subject_loro_scores(
    *,
    dyad_id: str,
    subject: str,
    events: pd.DataFrame,
    ds_index: pd.DataFrame,
    trf_cfg: EvokedTrfConfig,
    picks: str | list[str],
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    """
    Leave-one-run-out scores for a single (dyad_id, subject_id) across all runs.
    """
    ev_subj = events.loc[
        (events["dyad_id"].astype(str) == dyad_id)
        & (events["subject_id"].astype(str) == subject)
    ].copy()
    runs = sorted(ev_subj["run"].astype(int).unique().tolist())
    if len(runs) < 2:
        raise ValueError(f"LORO requires >=2 runs, got {len(runs)} for dyad_id={dyad_id}, subject={subject}")

    models = get_models()
    baseline_model = next(m for m in models if m.model_name == "baseline")
    model_names = [m.model_name for m in models]

    per_run: dict[int, dict[str, np.ndarray]] = {}
    per_run_y: dict[int, np.ndarray] = {}
    per_run_sfreq: dict[int, float] = {}

    # Load + build X/y per run
    for run in runs:
        eeg_path = select_eeg_path(ds_index, subject=subject, dyad_id=dyad_id, run=run)
        run_data = load_eeg_run(eeg_path, downsample_hz=trf_cfg.downsample_hz, picks=picks)
        y = run_data.y
        sfreq_hz = run_data.sfreq_hz
        per_run_y[run] = y
        per_run_sfreq[run] = sfreq_hz

        ev_run = ev_subj.loc[ev_subj["run"].astype(int) == run].copy()
        X_by_model: dict[str, np.ndarray] = {}
        for m in models:
            X_by_model[m.model_name] = build_design_matrix(
                events=ev_run,
                model=m,
                sfreq_hz=sfreq_hz,
                n_samples=y.shape[0],
            )
        per_run[run] = X_by_model

    # Sanity: sfreq must be identical across runs after resample
    sfreqs = {float(v) for v in per_run_sfreq.values()}
    if len(sfreqs) != 1:
        raise ValueError(f"Expected one sfreq after resample, got: {sorted(sfreqs)}")
    sfreq_hz = float(next(iter(sfreqs)))

    gap = _max_lag_samples(sfreq_hz=sfreq_hz, tmin_s=trf_cfg.tmin_s, tmax_s=trf_cfg.tmax_s) + 1

    rows: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []

    for fold_id, test_run in tqdm(list(enumerate(runs)), desc=f"{dyad_id}/{subject} folds", leave=False):
        train_runs = [r for r in runs if r != test_run]

        # CV audit table (new schema, see below)
        for r in runs:
            split_rows.append(
                {
                    "dyad_id": dyad_id,
                    "subject": subject,
                    "fold_id": int(fold_id),
                    "run": int(r),
                    "split": "test" if r == test_run else "train",
                }
            )

        # Fit baseline (train runs concatenated with gaps)
        y_train = concatenate_with_gap([per_run_y[r] for r in train_runs], gap_samples=gap)
        y_test = per_run_y[test_run]

        Xb_train = concatenate_with_gap([per_run[r]["baseline"] for r in train_runs], gap_samples=gap)
        Xb_test = per_run[test_run]["baseline"]

        # fold-wise standardization (Option B) using TRAIN only
        Xb_train, Xb_test = standardize_design_matrix_foldwise(Xb_train, Xb_test)

        rf_base = fit_receptive_field(
            X_train=Xb_train,
            y_train=y_train,
            sfreq_hz=sfreq_hz,
            tmin_s=trf_cfg.tmin_s,
            tmax_s=trf_cfg.tmax_s,
            ridge_alpha=trf_cfg.ridge_alpha,
        )
        y_hat_base = predict_receptive_field(rf_base, Xb_test)
        r_base = score_multichannel_mean_r(y_test, y_hat_base)

        rows.append(
            {
                "subject": subject,
                "dyad_id": dyad_id,
                "run": int(test_run),
                "model": "baseline",
                "fold_id": int(fold_id),
                "r": float(r_base),
                "r_baseline": float(r_base),
                "delta_r": float(0.0),
            }
        )

        # Other models (always compare to baseline fitted on same fold)
        for model_name in model_names:
            if model_name == "baseline":
                continue

            Xm_train = concatenate_with_gap([per_run[r][model_name] for r in train_runs], gap_samples=gap)
            Xm_test = per_run[test_run][model_name]

            Xm_train, Xm_test = standardize_design_matrix_foldwise(Xm_train, Xm_test)

            rf = fit_receptive_field(
                X_train=Xm_train,
                y_train=y_train,
                sfreq_hz=sfreq_hz,
                tmin_s=trf_cfg.tmin_s,
                tmax_s=trf_cfg.tmax_s,
                ridge_alpha=trf_cfg.ridge_alpha,
            )
            y_hat = predict_receptive_field(rf, Xm_test)
            r = score_multichannel_mean_r(y_test, y_hat)

            rows.append(
                {
                    "subject": subject,
                    "dyad_id": dyad_id,
                    "run": int(test_run),
                    "model": model_name,
                    "fold_id": int(fold_id),
                    "r": float(r),
                    "r_baseline": float(r_base),
                    "delta_r": float(r - r_base) if np.isfinite(r) and np.isfinite(r_base) else float("nan"),
                }
            )

    return rows, pd.DataFrame(split_rows)


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
    if trf_cfg.scheme not in {"blocked_kfold_within_run", "leave_one_run_out"}:
        raise ValueError(f"Unsupported evoked_trf.cv.scheme: {trf_cfg.scheme}")

    events, ds_index = load_input_tables(out_dir)
    events = validate_events_table(events)

    all_score_rows: list[dict[str, object]] = []
    all_splits: list[pd.DataFrame] = []

    grouped = events.groupby(["dyad_id", "subject_id"], sort=True)
    for (dyad_id, subject), _ in tqdm(grouped, desc="evoked_trf dyad×subject"):
        rows, splits = dyad_subject_loro_scores(
            dyad_id=str(dyad_id),
            subject=str(subject),
            events=events,
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

    required_splits = {"dyad_id", "subject", "fold_id", "run", "split"}
    missing_splits = required_splits - set(splits.columns)
    if missing_splits:
        raise ValueError(f"cv_splits table missing columns: {sorted(missing_splits)}")

    # delta_r should exist whenever r exists for non-baseline models
    nonbaseline = scores["model"] != "baseline"
    if nonbaseline.any():
        bad = scores.loc[nonbaseline & scores["r"].notna() & scores["delta_r"].isna()]
        if not bad.empty:
            raise ValueError("delta_r has missing values where r exists (non-baseline rows).")


def _max_lag_samples(*, sfreq_hz: float, tmin_s: float, tmax_s: float) -> int:
    return int(np.ceil(max(abs(tmin_s), abs(tmax_s)) * sfreq_hz))


def concatenate_with_gap(
    arrays: list[np.ndarray],
    *,
    gap_samples: int,
    fill_value: float = 0.0,
) -> np.ndarray:
    """
    Concatenate [A, gap, B, gap, C, ...] along axis=0.
    arrays must be (n_samples, n_features) or (n_samples, n_channels).
    """
    if not arrays:
        raise ValueError("arrays must be non-empty")
    if gap_samples < 0:
        raise ValueError("gap_samples must be >= 0")

    parts: list[np.ndarray] = []
    for i, a in enumerate(arrays):
        parts.append(a)
        if i < len(arrays) - 1 and gap_samples > 0:
            gap_shape = (gap_samples,) + a.shape[1:]
            parts.append(np.full(gap_shape, fill_value, dtype=float))
    return np.concatenate(parts, axis=0)

