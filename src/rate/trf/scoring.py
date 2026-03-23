from dataclasses import dataclass

import numpy as np
import pandas as pd


def summarize_scores(*, dyad_id: str, subject: str, outer_scores: pd.DataFrame) -> pd.DataFrame:
    required = {"dyad_id", "subject", "model", "fold_id", "test_run", "alpha", "r", "chosen_alpha"}
    missing = required - set(outer_scores.columns)
    if missing:
        raise ValueError(f"outer_scores missing columns: {sorted(missing)}")

    has_target = "target" in outer_scores.columns
    chosen_cols = ["dyad_id", "subject", "run", "model", "fold_id", "r"]
    base_merge_cols = ["dyad_id", "subject", "run", "fold_id"]
    if has_target:
        chosen_cols.insert(3, "target")
        base_merge_cols.insert(2, "target")

    chosen = outer_scores.loc[outer_scores["alpha"] == outer_scores["chosen_alpha"]].copy()
    chosen = chosen.rename(columns={"test_run": "run"})
    chosen = chosen[chosen_cols]

    base_cols = base_merge_cols + ["r"]
    base = chosen.loc[chosen["model"] == "baseline", base_cols].rename(columns={"r": "r_baseline"})
    merged = chosen.merge(base, on=base_merge_cols, how="left")
    merged["delta_r"] = np.where(merged["r_baseline"].notna(), merged["r"] - merged["r_baseline"], np.nan)
    return merged


def best_model_table(*, dyad_id: str, subject: str, summary: pd.DataFrame) -> pd.DataFrame:
    has_target = "target" in summary.columns
    group_cols = ["target"] if has_target else []
    rows: list[dict[str, object]] = []

    if has_target:
        groups = summary.groupby("target", sort=True, dropna=False)
    else:
        groups = [(None, summary)]

    for target_value, group in groups:
        nonbase = group.loc[group["model"] != "baseline"].copy()
        row: dict[str, object] = {"dyad_id": dyad_id, "subject": subject}
        if has_target:
            row["target"] = target_value
        if nonbase.empty:
            means = group.groupby(group_cols + ["model"], as_index=False)["r"].mean(numeric_only=True)
            if means.empty:
                row["best_model"] = "baseline"
                row["mean_delta_r"] = 0.0
                rows.append(row)
                continue
            means = means.sort_values(["r", "model"], ascending=[False, True], kind="mergesort")
            best = means.iloc[0].to_dict()
            row["best_model"] = str(best["model"])
            row["mean_delta_r"] = float("nan")
            rows.append(row)
            continue
        means = nonbase.groupby(group_cols + ["model"], as_index=False)["delta_r"].mean(numeric_only=True)
        means = means.sort_values(["delta_r", "model"], ascending=[False, True], kind="mergesort")
        best = means.iloc[0].to_dict()
        row["best_model"] = str(best["model"])
        row["mean_delta_r"] = float(best["delta_r"])
        rows.append(row)

    return pd.DataFrame(rows)


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
    has_target = "target" in scores.columns or "target" in inner_scores.columns or "target" in outer_scores.columns
    sort_prefix = ["dyad_id", "subject"] + (["target"] if has_target else [])
    summary_fold_sorted = scores.sort_values(sort_prefix + ["run", "model", "fold_id"], kind="mergesort").reset_index(drop=True)
    inner_sorted = (
        inner_scores.sort_values(sort_prefix + ["model", "outer_fold", "inner_fold", "alpha"], kind="mergesort")
        .reset_index(drop=True)
        if not inner_scores.empty
        else inner_scores
    )
    outer_sorted = (
        outer_scores.sort_values(sort_prefix + ["model", "fold_id", "alpha"], kind="mergesort").reset_index(drop=True)
        if not outer_scores.empty
        else outer_scores
    )

    alpha_grid = outer_sorted.copy()

    if inner_sorted.empty:
        inner_cols = ["dyad_id", "subject"] + (["target"] if has_target else []) + ["model", "alpha", "mean_r"]
        inner_cv = pd.DataFrame(columns=inner_cols)
    else:
        inner_cv = (
            inner_sorted.groupby(sort_prefix + ["model", "alpha"], as_index=False)["r"]
            .mean(numeric_only=True)
            .rename(columns={"r": "mean_r"})
            .sort_values(sort_prefix + ["model", "alpha"], kind="mergesort")
            .reset_index(drop=True)
        )

    selected = outer_sorted.loc[np.isclose(outer_sorted["alpha"], outer_sorted["chosen_alpha"])].copy()
    selected_cols = ["dyad_id", "subject"] + (["target"] if has_target else []) + ["model", "fold_id", "test_run", "alpha", "r"]
    selected = selected[selected_cols]
    base_merge_cols = (["target"] if has_target else []) + ["fold_id", "test_run"]
    base = selected.loc[selected["model"] == "baseline", base_merge_cols + ["r"]].rename(columns={"r": "baseline_r"})
    outer_fold_scores = selected.merge(base, on=base_merge_cols, how="left")
    outer_fold_scores["delta_r"] = np.where(
        outer_fold_scores["baseline_r"].notna(),
        outer_fold_scores["r"] - outer_fold_scores["baseline_r"],
        np.nan,
    )
    outer_fold_scores = outer_fold_scores.sort_values((["target"] if has_target else []) + ["model", "test_run"], kind="mergesort").reset_index(drop=True)

    best_alpha_by_model = selected.groupby((["target"] if has_target else []) + ["model"], as_index=False).agg(
        best_alpha=("alpha", lambda x: float(pd.Series(list(x)).mode().iloc[0]))
    )
    model_comparison = (
        outer_fold_scores.groupby((["target"] if has_target else []) + ["model"], as_index=False)
        .agg(
            mean_r=("r", "mean"),
            mean_baseline_r=("baseline_r", "mean"),
            mean_delta_r=("delta_r", "mean"),
        )
        .merge(best_alpha_by_model, on=(["target"] if has_target else []) + ["model"], how="left")
        .sort_values((["target"] if has_target else []) + ["mean_r", "model"], ascending=([True] if has_target else []) + [False, True], kind="mergesort")
        .reset_index(drop=True)
    )
    summary_cols = ["dyad_id", "subject"] + (["target"] if has_target else []) + [
        "model",
        "mean_r",
        "mean_baseline_r",
        "mean_delta_r",
        "best_alpha",
    ]
    summary = model_comparison.assign(dyad_id=str(dyad_id), subject=str(subject))[summary_cols]

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
