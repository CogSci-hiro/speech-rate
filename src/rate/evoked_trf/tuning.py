from dataclasses import dataclass

import numpy as np
import pandas as pd

from rate.evoked_trf.fit import fit_receptive_field, predict_receptive_field
from rate.evoked_trf.utils import (
    concatenate_with_gap,
    max_lag_samples,
    score_multichannel_mean_r,
    standardize_design_matrix_foldwise,
)


@dataclass(frozen=True)
class AlphaSelectionResult:
    """
    Alpha selection outcome for one (dyad, subject, model, outer-fold).

    Usage example
    -------------
        best = AlphaSelectionResult(best_alpha=10.0, table=df)
    """

    best_alpha: float
    table: pd.DataFrame


def _inner_folds_from_runs(runs: list[int]) -> list[tuple[list[int], int]]:
    """
    Inner folds = leave-one-run-out within the provided runs.

    Usage example
    -------------
        folds = _inner_folds_from_runs([2,3,4])
    """
    if len(runs) < 2:
        return []
    folds: list[tuple[list[int], int]] = []
    for test_run in runs:
        train_runs = [r for r in runs if r != test_run]
        folds.append((train_runs, test_run))
    return folds


def select_alpha_nested_loro(
    *,
    dyad_id: str,
    subject: str,
    model_name: str,
    runs_train_outer: list[int],
    per_run_y: dict[int, np.ndarray],
    per_run_X: dict[int, dict[str, np.ndarray]],
    alphas: list[float],
    sfreq_hz: float,
    tmin_s: float,
    tmax_s: float,
) -> AlphaSelectionResult:
    """
    Choose alpha using inner LORO on the OUTER training runs only.

    Returns a table with inner scores per alpha and fold and the selected alpha.

    Usage example
    -------------
        res = select_alpha_nested_loro(...)
        best_alpha = res.best_alpha
    """
    inner_folds = _inner_folds_from_runs(runs_train_outer)
    if not inner_folds:
        table = pd.DataFrame(
            [
                {
                    "dyad_id": dyad_id,
                    "subject": subject,
                    "model": model_name,
                    "alpha": float(alphas[0]),
                    "inner_fold": 0,
                    "test_run": None,
                    "r": float("nan"),
                    "note": "inner_cv_skipped_only_one_training_run",
                }
            ]
        )
        return AlphaSelectionResult(best_alpha=float(alphas[0]), table=table)

    gap = max_lag_samples(sfreq_hz=sfreq_hz, tmin_s=tmin_s, tmax_s=tmax_s) + 1

    rows: list[dict[str, object]] = []
    for alpha in alphas:
        for inner_fold_id, (train_runs, test_run) in enumerate(inner_folds):
            y_train = concatenate_with_gap([per_run_y[r] for r in train_runs], gap_samples=gap)
            y_test = per_run_y[test_run]

            X_train = concatenate_with_gap([per_run_X[r][model_name] for r in train_runs], gap_samples=gap)
            X_test = per_run_X[test_run][model_name]

            X_train, X_test = standardize_design_matrix_foldwise(X_train, X_test)

            rf = fit_receptive_field(
                X_train=X_train,
                y_train=y_train,
                sfreq_hz=sfreq_hz,
                tmin_s=tmin_s,
                tmax_s=tmax_s,
                ridge_alpha=float(alpha),
            )
            y_hat = predict_receptive_field(rf, X_test)
            r = score_multichannel_mean_r(y_test, y_hat)

            rows.append(
                {
                    "dyad_id": dyad_id,
                    "subject": subject,
                    "model": model_name,
                    "alpha": float(alpha),
                    "inner_fold": int(inner_fold_id),
                    "test_run": int(test_run),
                    "r": float(r),
                    "note": "",
                }
            )

    table = pd.DataFrame(rows)

    # Pick alpha by mean inner r (ignoring NaNs)
    mean_r = table.groupby("alpha", as_index=False)["r"].mean(numeric_only=True)
    mean_r = mean_r.sort_values(["r", "alpha"], ascending=[False, True])
    best_alpha = float(mean_r.iloc[0]["alpha"])
    return AlphaSelectionResult(best_alpha=best_alpha, table=table)
