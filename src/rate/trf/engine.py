from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd


ArrayDict = dict[Any, np.ndarray]


@dataclass(frozen=True)
class TRFResult:
    kernels_outer: np.ndarray
    chosen_alpha: np.ndarray
    outer_scores: pd.DataFrame
    lags: np.ndarray
    predictions_by_fold: dict[int, np.ndarray] | None = None
    inner_scores: pd.DataFrame | None = None
    best_alpha_table: pd.DataFrame | None = None
    reconstruction_r2: pd.DataFrame | None = None


def _outer_folds_from_runs(runs: list[Any]) -> list[tuple[list[Any], Any, int]]:
    folds: list[tuple[list[Any], Any, int]] = []
    for fold_id, test_run in enumerate(runs):
        train_runs = [r for r in runs if r != test_run]
        folds.append((train_runs, test_run, fold_id))
    return folds


def _inner_folds_from_runs(runs: list[Any]) -> list[tuple[list[Any], Any]]:
    if len(runs) < 2:
        return []
    folds: list[tuple[list[Any], Any]] = []
    for test_run in runs:
        train_runs = [r for r in runs if r != test_run]
        folds.append((train_runs, test_run))
    return folds


def run_trf(
    X_by_run: dict[str, np.ndarray],
    y_by_run: dict[str, np.ndarray],
    lags: np.ndarray,
    alpha_grid: np.ndarray,
    cv_config: dict,
) -> TRFResult:
    """
    Generic nested-CV TRF engine.

    Required cv_config keys:
    - gap_samples
    - alpha_default
    - concat_fn(arrays, gap_samples=...) -> ndarray
    - standardize_fn(X_train, X_test) -> (X_train_std, X_test_std)
    - fit_fn(X_train=..., y_train=..., alpha=...) -> fitted model
    - predict_fn(model, X_test) -> y_hat
    - score_fn(y_true, y_pred) -> float
    - extract_kernel_fn(model) -> (coef, lags)

    Optional:
    - table_meta: {"dyad_id": ..., "subject": ..., "model": ...}
    - collect_predictions: bool
    - enable_reconstruction: bool
    - reconstruction_fn(y_test_raw, y_test_pred, events_test, sfreq, tmin, tmax) -> DataFrame
    - events_by_run: dict[run, DataFrame]
    - reconstruction_kwargs: dict with sfreq, tmin, tmax
    """
    concat_fn: Callable[..., np.ndarray] = cv_config["concat_fn"]
    standardize_fn: Callable[..., tuple[np.ndarray, np.ndarray]] = cv_config["standardize_fn"]
    fit_fn: Callable[..., Any] = cv_config["fit_fn"]
    predict_fn: Callable[..., np.ndarray] = cv_config["predict_fn"]
    score_fn: Callable[..., float] = cv_config["score_fn"]
    extract_kernel_fn: Callable[..., tuple[np.ndarray, np.ndarray]] = cv_config["extract_kernel_fn"]
    gap_samples = int(cv_config["gap_samples"])
    alpha_default = float(cv_config.get("alpha_default", np.asarray(alpha_grid, dtype=float)[0]))
    table_meta = dict(cv_config.get("table_meta", {}))
    collect_predictions = bool(cv_config.get("collect_predictions", False))
    enable_reconstruction = bool(cv_config.get("enable_reconstruction", False))
    reconstruction_fn: Callable[..., pd.DataFrame] | None = cv_config.get("reconstruction_fn")
    events_by_run: dict[Any, pd.DataFrame] | None = cv_config.get("events_by_run")
    reconstruction_kwargs = dict(cv_config.get("reconstruction_kwargs", {}))
    reconstruction_rows: list[pd.DataFrame] = []
    if enable_reconstruction and reconstruction_fn is None:
        raise ValueError("enable_reconstruction=True requires reconstruction_fn")
    if enable_reconstruction and events_by_run is None:
        raise ValueError("enable_reconstruction=True requires events_by_run")

    runs = sorted(X_by_run.keys())
    if sorted(y_by_run.keys()) != runs:
        raise ValueError("X_by_run and y_by_run must share the same run keys")
    if len(runs) < 2:
        raise ValueError(f"leave_one_run_out requires >=2 runs, got {len(runs)}")

    alphas = [float(a) for a in np.asarray(alpha_grid, dtype=float).tolist()]
    outer_folds = _outer_folds_from_runs(runs)

    inner_rows: list[pd.DataFrame] = []
    outer_rows: list[pd.DataFrame] = []
    best_alpha_rows: list[dict[str, object]] = []
    fold_kernels: list[np.ndarray] = []
    chosen_alphas: list[float] = []
    predictions_by_fold: dict[int, np.ndarray] | None = {} if collect_predictions else None
    lags_out: np.ndarray | None = None

    for train_runs, test_run, fold_id in outer_folds:
        if len(train_runs) < 2:
            inner_df = pd.DataFrame(
                [
                    {
                        **table_meta,
                        "alpha": float(alpha_default),
                        "inner_fold": 0,
                        "test_run": None,
                        "r": float("nan"),
                        "note": "inner_cv_skipped_only_one_training_run",
                    }
                ]
            )
            best_alpha = float(alpha_default)
        else:
            inner_fold_rows: list[dict[str, object]] = []
            for inner_fold_id, (inner_train_runs, inner_test_run) in enumerate(_inner_folds_from_runs(train_runs)):
                y_train = concat_fn([y_by_run[r] for r in inner_train_runs], gap_samples=gap_samples)
                y_test = y_by_run[inner_test_run]
                X_train = concat_fn([X_by_run[r] for r in inner_train_runs], gap_samples=gap_samples)
                X_test = X_by_run[inner_test_run]
                X_train, X_test = standardize_fn(X_train, X_test)

                for alpha in alphas:
                    model = fit_fn(X_train=X_train, y_train=y_train, alpha=float(alpha))
                    y_hat = predict_fn(model, X_test)
                    r = score_fn(y_test, y_hat)
                    inner_fold_rows.append(
                        {
                            **table_meta,
                            "alpha": float(alpha),
                            "inner_fold": int(inner_fold_id),
                            "test_run": int(inner_test_run),
                            "r": float(r),
                            "note": "",
                        }
                    )

            inner_df = pd.DataFrame(inner_fold_rows)
            mean_r = inner_df.groupby("alpha", as_index=False)["r"].mean(numeric_only=True)
            mean_r = mean_r.sort_values(["r", "alpha"], ascending=[False, True])
            best_alpha = float(mean_r.iloc[0]["alpha"])

        inner_df = inner_df.assign(outer_fold=int(fold_id), outer_test_run=int(test_run))
        inner_rows.append(inner_df)
        best_alpha_rows.append(
            {
                **table_meta,
                "fold_id": int(fold_id),
                "test_run": int(test_run),
                "best_alpha": float(best_alpha),
            }
        )
        chosen_alphas.append(float(best_alpha))

        # Outer scores across all alphas (uses foldwise standardization against outer test data).
        y_train_outer = concat_fn([y_by_run[r] for r in train_runs], gap_samples=gap_samples)
        y_test_outer = y_by_run[test_run]
        X_train_outer = concat_fn([X_by_run[r] for r in train_runs], gap_samples=gap_samples)
        X_test_outer = X_by_run[test_run]
        X_train_outer_std, X_test_outer_std = standardize_fn(X_train_outer, X_test_outer)

        outer_fold_rows: list[dict[str, object]] = []
        for alpha in alphas:
            model = fit_fn(X_train=X_train_outer_std, y_train=y_train_outer, alpha=float(alpha))
            y_hat = predict_fn(model, X_test_outer_std)
            r = score_fn(y_test_outer, y_hat)
            outer_fold_rows.append(
                {
                    **table_meta,
                    "fold_id": int(fold_id),
                    "test_run": int(test_run),
                    "alpha": float(alpha),
                    "r": float(r),
                }
            )
            if collect_predictions and np.isclose(float(alpha), float(best_alpha)):
                assert predictions_by_fold is not None
                predictions_by_fold[int(fold_id)] = np.asarray(y_hat)
            if enable_reconstruction and np.isclose(float(alpha), float(best_alpha)):
                assert reconstruction_fn is not None
                assert events_by_run is not None
                recon = reconstruction_fn(
                    y_test_raw=np.asarray(y_test_outer),
                    y_test_pred=np.asarray(y_hat),
                    events_test=events_by_run[test_run],
                    **reconstruction_kwargs,
                )
                recon = recon.copy()
                for k, v in table_meta.items():
                    recon[k] = v
                recon["fold_id"] = int(fold_id)
                recon["test_run"] = int(test_run)
                recon["alpha"] = float(best_alpha)
                reconstruction_rows.append(recon)
        outer_df = pd.DataFrame(outer_fold_rows)
        outer_df["chosen_alpha"] = float(best_alpha)
        outer_rows.append(outer_df)

        # Kernel refit matches historical artifact path behavior (train standardized against itself).
        X_train_kernel, _ = standardize_fn(X_train_outer, X_train_outer)
        kernel_model = fit_fn(X_train=X_train_kernel, y_train=y_train_outer, alpha=float(best_alpha))
        coef, lags_fold = extract_kernel_fn(kernel_model)
        fold_kernels.append(np.asarray(coef))
        if lags_out is None:
            lags_out = np.asarray(lags_fold, dtype=float)
        elif not np.allclose(np.asarray(lags_out, dtype=float), np.asarray(lags_fold, dtype=float)):
            raise ValueError("Inconsistent lag grid across outer folds")

    return TRFResult(
        kernels_outer=np.stack(fold_kernels, axis=0),
        chosen_alpha=np.asarray(chosen_alphas, dtype=float),
        outer_scores=pd.concat(outer_rows, ignore_index=True) if outer_rows else pd.DataFrame(),
        lags=np.asarray(lags_out if lags_out is not None else lags, dtype=float),
        predictions_by_fold=predictions_by_fold,
        inner_scores=pd.concat(inner_rows, ignore_index=True) if inner_rows else pd.DataFrame(),
        best_alpha_table=pd.DataFrame(best_alpha_rows),
        reconstruction_r2=pd.concat(reconstruction_rows, ignore_index=True) if reconstruction_rows else None,
    )
