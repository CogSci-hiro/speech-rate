from dataclasses import dataclass

import numpy as np
import pandas as pd


def summarize_scores(*, dyad_id: str, subject: str, outer_scores: pd.DataFrame) -> pd.DataFrame:
    required = {"dyad_id", "subject", "model", "fold_id", "test_run", "alpha", "r", "chosen_alpha"}
    missing = required - set(outer_scores.columns)
    if missing:
        raise ValueError(f"outer_scores missing columns: {sorted(missing)}")

    chosen = outer_scores.loc[outer_scores["alpha"] == outer_scores["chosen_alpha"]].copy()
    chosen = chosen.rename(columns={"test_run": "run"})
    chosen = chosen[["dyad_id", "subject", "run", "model", "fold_id", "r"]]

    base = chosen.loc[chosen["model"] == "baseline", ["dyad_id", "subject", "run", "fold_id", "r"]].rename(columns={"r": "r_baseline"})
    merged = chosen.merge(base, on=["dyad_id", "subject", "run", "fold_id"], how="left")
    merged["delta_r"] = np.where(merged["r_baseline"].notna(), merged["r"] - merged["r_baseline"], np.nan)
    return merged


def best_model_table(*, dyad_id: str, subject: str, summary: pd.DataFrame) -> pd.DataFrame:
    nonbase = summary.loc[summary["model"] != "baseline"].copy()
    if nonbase.empty:
        means = summary.groupby("model", as_index=False)["r"].mean(numeric_only=True)
        if means.empty:
            return pd.DataFrame(
                [
                    {
                        "dyad_id": dyad_id,
                        "subject": subject,
                        "best_model": "baseline",
                        "mean_delta_r": 0.0,
                    }
                ]
            )
        means = means.sort_values(["r", "model"], ascending=[False, True], kind="mergesort")
        best = means.iloc[0].to_dict()
        return pd.DataFrame(
            [
                {
                    "dyad_id": dyad_id,
                    "subject": subject,
                    "best_model": str(best["model"]),
                    "mean_delta_r": float("nan"),
                }
            ]
        )
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


@dataclass(frozen=True)
class SubjectScoreTables:
    summary_fold_sorted: pd.DataFrame
    inner_sorted: pd.DataFrame
    outer_sorted: pd.DataFrame
    alpha_grid: pd.DataFrame
    inner_cv: pd.DataFrame
    outer_fold_scores: pd.DataFrame
    best_alpha_by_model: pd.DataFrame
    model_comparison: pd.DataFrame
    summary: pd.DataFrame


def build_subject_score_tables(
    *,
    dyad_id: str,
    subject: str,
    scores: pd.DataFrame,
    inner_scores: pd.DataFrame,
    outer_scores: pd.DataFrame,
) -> SubjectScoreTables:
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

    alpha_grid = outer_sorted.copy()

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

    selected = outer_sorted.loc[np.isclose(outer_sorted["alpha"], outer_sorted["chosen_alpha"])].copy()
    selected = selected[["dyad_id", "subject", "model", "fold_id", "test_run", "alpha", "r"]]
    base = selected.loc[selected["model"] == "baseline", ["fold_id", "test_run", "r"]].rename(columns={"r": "baseline_r"})
    outer_fold_scores = selected.merge(base, on=["fold_id", "test_run"], how="left")
    outer_fold_scores["delta_r"] = np.where(
        outer_fold_scores["baseline_r"].notna(),
        outer_fold_scores["r"] - outer_fold_scores["baseline_r"],
        np.nan,
    )
    outer_fold_scores = outer_fold_scores.sort_values(["model", "test_run"], kind="mergesort").reset_index(drop=True)

    best_alpha_by_model = selected.groupby("model", as_index=False).agg(
        best_alpha=("alpha", lambda x: float(pd.Series(list(x)).mode().iloc[0]))
    )
    model_comparison = (
        outer_fold_scores.groupby("model", as_index=False)
        .agg(
            mean_r=("r", "mean"),
            mean_baseline_r=("baseline_r", "mean"),
            mean_delta_r=("delta_r", "mean"),
        )
        .merge(best_alpha_by_model, on="model", how="left")
        .sort_values(["mean_r", "model"], ascending=[False, True], kind="mergesort")
        .reset_index(drop=True)
    )
    summary = model_comparison.assign(dyad_id=str(dyad_id), subject=str(subject))[
        ["dyad_id", "subject", "model", "mean_r", "mean_baseline_r", "mean_delta_r", "best_alpha"]
    ]

    return SubjectScoreTables(
        summary_fold_sorted=summary_fold_sorted,
        inner_sorted=inner_sorted,
        outer_sorted=outer_sorted,
        alpha_grid=alpha_grid,
        inner_cv=inner_cv,
        outer_fold_scores=outer_fold_scores,
        best_alpha_by_model=best_alpha_by_model,
        model_comparison=model_comparison,
        summary=summary,
    )
