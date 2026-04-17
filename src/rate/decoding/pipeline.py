"""Residualised temporal-generalisation decoding pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from time import perf_counter
import sys
from typing import Any, Mapping

import numpy as np
import pandas as pd
from mne.decoding import GeneralizingEstimator
from numpy.lib.format import open_memmap
from pandas import CategoricalDtype
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from sklearn.metrics import make_scorer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from rate.decoding.cache import build_prepared_data_cache_key, load_prepared_data_cache, save_prepared_data_cache
from rate.decoding.io import build_decoding_model_output_paths, save_decoding_outputs
from rate.epochs.io import build_epoch_output_paths
from rate.lmeeeg.io import load_lmeeeg_inputs
from rate.lmeeeg.pooling import select_lmeeeg_rows
from rate.paths import ensure_parent_dir

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TemporalGeneralizationResult:
    """Observed TG outputs for one model."""

    fold_scores: np.ndarray
    mean_scores: np.ndarray
    alpha_scores: np.ndarray | None
    best_alphas: np.ndarray


@dataclass(frozen=True)
class PreparedDecodingData:
    """Residualised decoding inputs for one model."""

    model_name: str
    train_target: str
    test_target: str
    metadata_df: pd.DataFrame
    eeg_residuals: np.ndarray
    train_target_residuals: np.ndarray
    test_target_residuals: np.ndarray
    fold_indices: list[tuple[np.ndarray, np.ndarray]]
    times: np.ndarray
    channel_names: list[str]
    sfreq: float
    summary: dict[str, Any]


def run_decoding_model(
    config: Mapping[str, Any],
    decoding_config: Mapping[str, Any],
    *,
    model_name: str,
    dataset_index: str | None = None,
    show_progress: bool = False,
) -> dict[str, Any]:
    """Run one decoding model and persist its outputs."""
    prepared = prepare_decoding_model_data(
        config,
        decoding_config,
        model_name=model_name,
        dataset_index=dataset_index,
    )
    tg_result = _run_temporal_generalisation(
        prepared.eeg_residuals,
        prepared.train_target_residuals,
        prepared.test_target_residuals,
        estimator_config=decoding_config["estimator"],
        cv_config=decoding_config["cv"],
        nested_cv_config=decoding_config.get("nested_cv") or {},
        scoring_config=decoding_config["scoring"],
        fold_indices=prepared.fold_indices,
        groups=prepared.metadata_df[str(decoding_config["cv"]["group_column"])].to_numpy(),
        model_name=str(prepared.model_name),
        show_progress=show_progress,
        save_alpha_scores=bool(decoding_config["estimator"].get("save_alpha_scores", True)),
        alpha_scores_output_path=build_decoding_model_output_paths(config, decoding_config, model_name=model_name).alpha_scores_output_path,
    )

    selection_df = prepared.metadata_df.copy()
    selection_df.loc[:, "residual_train_target"] = prepared.train_target_residuals
    selection_df.loc[:, "residual_test_target"] = prepared.test_target_residuals
    selection_df.loc[:, "residual_train_target_name"] = str(prepared.train_target)
    selection_df.loc[:, "residual_test_target_name"] = str(prepared.test_target)

    output_paths = build_decoding_model_output_paths(config, decoding_config, model_name=model_name)
    summary = dict(prepared.summary)
    summary["status"] = "ok"
    save_decoding_outputs(
        output_paths=output_paths,
        mean_scores=tg_result.mean_scores,
        fold_scores=tg_result.fold_scores,
        alpha_scores=tg_result.alpha_scores,
        best_alphas=tg_result.best_alphas,
        times=prepared.times,
        channel_names=prepared.channel_names,
        selection_df=selection_df,
        summary=summary,
    )
    LOGGER.info("Wrote decoding outputs for model=%s -> %s", model_name, output_paths.summary_output_path)
    return summary


def prepare_decoding_model_data(
    config: Mapping[str, Any],
    decoding_config: Mapping[str, Any],
    *,
    model_name: str,
    dataset_index: str | None = None,
    use_prepared_data_cache: bool = True,
) -> PreparedDecodingData:
    """Load, filter, residualise, and package one decoding model's inputs."""
    model_cfg = _resolve_model_config(decoding_config, model_name)
    rows = select_lmeeeg_rows(config, dataset_index=dataset_index)
    cache_key = build_prepared_data_cache_key(
        config,
        decoding_config,
        model_cfg=model_cfg,
        dataset_index=dataset_index,
        rows=rows,
    )
    if use_prepared_data_cache:
        cached_payload = load_prepared_data_cache(cache_key)
        if cached_payload is not None:
            LOGGER.info(
                "Loaded prepared decoding data from cache for model=%s (%s)",
                model_name,
                cache_key.cache_key[:12],
            )
            return PreparedDecodingData(
                model_name=str(cached_payload["model_name"]),
                train_target=str(cached_payload["train_target"]),
                test_target=str(cached_payload["test_target"]),
                metadata_df=cached_payload["metadata_df"],
                eeg_residuals=cached_payload["eeg_residuals"],
                train_target_residuals=cached_payload["train_target_residuals"],
                test_target_residuals=cached_payload["test_target_residuals"],
                fold_indices=cached_payload["fold_indices"],
                times=cached_payload["times"],
                channel_names=list(cached_payload["channel_names"]),
                sfreq=float(cached_payload["summary"]["decoding_sfreq"]),
                summary=dict(cached_payload["summary"]),
            )

    prep_start = perf_counter()
    data_bundle = _load_decoding_data(config, decoding_config, rows=rows)
    filtered_df, filtered_eeg = _filter_model_rows(data_bundle["metadata"], data_bundle["eeg"], decoding_config, model_cfg=model_cfg)

    eeg_residuals_start = perf_counter()
    eeg_residuals = _residualise_eeg(
        filtered_df,
        filtered_eeg,
        random_effect=str(decoding_config["eeg_residualisation"]["random_effect"]),
        controls=list(decoding_config["eeg_residualisation"]["controls"]),
    )
    LOGGER.info("Prepared EEG residuals for model=%s in %.2fs", model_name, perf_counter() - eeg_residuals_start)
    target_residuals_start = perf_counter()
    train_target_residuals = _residualise_target(
        filtered_df,
        target_name=str(model_cfg["train_target"]),
        random_effect=str(decoding_config["target_residualisation"]["random_effect"]),
        controls=list(decoding_config["target_residualisation"]["controls"]),
    )
    test_target_residuals = _residualise_target(
        filtered_df,
        target_name=str(model_cfg["test_target"]),
        random_effect=str(decoding_config["target_residualisation"]["random_effect"]),
        controls=list(decoding_config["target_residualisation"]["controls"]),
    )
    LOGGER.info("Prepared target residuals for model=%s in %.2fs", model_name, perf_counter() - target_residuals_start)

    fold_indices = _build_group_folds(
        filtered_df[str(decoding_config["cv"]["group_column"])],
        n_splits=int(decoding_config["cv"]["n_splits"]),
        shuffle=bool(decoding_config["cv"].get("shuffle", True)),
        random_state=int(decoding_config.get("random_state", 42)),
    )
    prepared = PreparedDecodingData(
        model_name=str(model_cfg["name"]),
        train_target=str(model_cfg["train_target"]),
        test_target=str(model_cfg["test_target"]),
        metadata_df=filtered_df,
        eeg_residuals=eeg_residuals,
        train_target_residuals=train_target_residuals,
        test_target_residuals=test_target_residuals,
        fold_indices=fold_indices,
        times=np.asarray(data_bundle["times"], dtype=float),
        channel_names=list(data_bundle["channel_names"]),
        sfreq=float(data_bundle["sfreq"]),
        summary=_build_decoding_summary(
            decoding_config,
            filtered_df=filtered_df,
            eeg_residuals=eeg_residuals,
            model_cfg=model_cfg,
            dataset_index=dataset_index,
            sfreq=float(data_bundle["sfreq"]),
        ),
    )
    if use_prepared_data_cache:
        save_prepared_data_cache(
            cache_key,
            model_name=prepared.model_name,
            train_target=prepared.train_target,
            test_target=prepared.test_target,
            metadata_df=prepared.metadata_df,
            eeg_residuals=prepared.eeg_residuals,
            train_target_residuals=prepared.train_target_residuals,
            test_target_residuals=prepared.test_target_residuals,
            fold_indices=prepared.fold_indices,
            times=prepared.times,
            channel_names=prepared.channel_names,
            summary=prepared.summary,
        )
        LOGGER.info(
            "Wrote prepared decoding data cache for model=%s (%s)",
            model_name,
            cache_key.cache_key[:12],
        )
    LOGGER.info("Prepared decoding inputs for model=%s in %.2fs", model_name, perf_counter() - prep_start)
    return prepared


def _resolve_model_config(decoding_config: Mapping[str, Any], model_name: str) -> dict[str, str]:
    for model_cfg in decoding_config.get("models") or []:
        if str(model_cfg.get("name")) == str(model_name):
            return {"name": str(model_cfg["name"]), "train_target": str(model_cfg["train_target"]), "test_target": str(model_cfg["test_target"])}
    raise ValueError(f"Unknown decoding model: {model_name}")


def _load_decoding_data(
    config: Mapping[str, Any],
    decoding_config: Mapping[str, Any],
    *,
    rows: list[Mapping[str, Any]] | None = None,
    dataset_index: str | None = None,
) -> dict[str, Any]:
    rows = rows if rows is not None else select_lmeeeg_rows(config, dataset_index=dataset_index)
    pooled_eeg: list[np.ndarray] = []
    pooled_metadata: list[pd.DataFrame] = []
    reference_channel_names: list[str] | None = None
    reference_times: np.ndarray | None = None
    sfreq_value: float | None = None
    epochs_section = str(decoding_config["input"]["epochs_section"])

    for row in rows:
        epoch_paths = build_epoch_output_paths(config, row, epochs_section=epochs_section)
        epochs, metadata_df = load_lmeeeg_inputs(epoch_paths.epochs_output_path, epoch_paths.metadata_output_path)
        epochs = _prepare_epochs(epochs, decoding_config)
        for field_name in ("dyad_id", "subject_id", "task", "run"):
            if field_name in row and field_name not in metadata_df.columns:
                metadata_df.loc[:, field_name] = row[field_name]
        eeg_data = epochs.get_data(copy=True)
        times = np.asarray(epochs.times, dtype=float)
        channel_names = list(epochs.ch_names)

        if reference_channel_names is None:
            reference_channel_names = channel_names
            reference_times = times
            sfreq_value = float(epochs.info["sfreq"])
        else:
            if channel_names != reference_channel_names:
                raise ValueError("All decoding inputs must share the same channel order.")
            if not np.allclose(times, reference_times):
                raise ValueError("All decoding inputs must share the same time axis.")

        pooled_eeg.append(np.asarray(eeg_data, dtype=float))
        pooled_metadata.append(metadata_df.reset_index(drop=True))

    assert reference_channel_names is not None
    assert reference_times is not None
    assert sfreq_value is not None
    return {
        "eeg": np.concatenate(pooled_eeg, axis=0),
        "metadata": pd.concat(pooled_metadata, axis=0, ignore_index=True),
        "channel_names": reference_channel_names,
        "times": reference_times,
        "sfreq": sfreq_value,
    }


def _prepare_epochs(epochs, decoding_config: Mapping[str, Any]):
    epochs = epochs.copy()
    channels_cfg = decoding_config["channels"]
    include = [str(name) for name in channels_cfg.get("include") or []]
    exclude = {str(name) for name in channels_cfg.get("exclude") or []}
    picks = channels_cfg.get("picks", "eeg")
    if picks == "eeg":
        epochs.pick("eeg")
    if include:
        epochs.pick(include)
    if exclude:
        keep = [name for name in epochs.ch_names if name not in exclude]
        epochs.pick(keep)

    target_sfreq = decoding_config.get("sfreq")
    if target_sfreq is not None and not np.isclose(float(epochs.info["sfreq"]), float(target_sfreq)):
        epochs.resample(float(target_sfreq))

    time_cfg = decoding_config.get("time_window") or {}
    tmin = time_cfg.get("tmin_s")
    tmax = time_cfg.get("tmax_s")
    if tmin is not None or tmax is not None:
        epochs.crop(tmin=tmin, tmax=tmax)
    return epochs


def _filter_model_rows(
    metadata_df: pd.DataFrame,
    eeg_data: np.ndarray,
    decoding_config: Mapping[str, Any],
    *,
    model_cfg: Mapping[str, str],
) -> tuple[pd.DataFrame, np.ndarray]:
    required_columns = {
        str(model_cfg["train_target"]),
        str(model_cfg["test_target"]),
        str(decoding_config["cv"]["group_column"]),
        str(decoding_config["eeg_residualisation"]["random_effect"]),
        str(decoding_config["target_residualisation"]["random_effect"]),
        *[str(name) for name in decoding_config["eeg_residualisation"]["controls"]],
        *[str(name) for name in decoding_config["target_residualisation"]["controls"]],
    }
    missing_columns = [column_name for column_name in sorted(required_columns) if column_name not in metadata_df.columns]
    if missing_columns:
        raise ValueError(f"Decoding metadata is missing required columns: {', '.join(missing_columns)}")

    keep_mask = metadata_df.loc[:, sorted(required_columns)].notna().all(axis=1).to_numpy(dtype=bool, copy=False)
    filtered_df = metadata_df.loc[keep_mask].copy().reset_index(drop=True)
    filtered_eeg = np.asarray(eeg_data, dtype=float)[keep_mask, :, :]
    if filtered_df.empty:
        raise ValueError(f"No decoding trials remain after filtering for model `{model_cfg['name']}`.")
    return filtered_df, filtered_eeg


def _residualise_eeg(
    metadata_df: pd.DataFrame,
    eeg_data: np.ndarray,
    *,
    random_effect: str,
    controls: list[str],
) -> np.ndarray:
    # We remove nuisance variance from EEG only, never the decoding target itself.
    design_matrix = _build_residualisation_design(metadata_df, random_effect=random_effect, controls=controls)
    eeg_flat = np.asarray(eeg_data, dtype=float).reshape(eeg_data.shape[0], -1)
    coefficients, _, _, _ = np.linalg.lstsq(design_matrix, eeg_flat, rcond=None)
    residual_flat = eeg_flat - (design_matrix @ coefficients)
    return residual_flat.reshape(eeg_data.shape)


def _residualise_target(
    metadata_df: pd.DataFrame,
    *,
    target_name: str,
    random_effect: str,
    controls: list[str],
) -> np.ndarray:
    target_values = metadata_df[target_name].to_numpy(dtype=float, copy=True)
    nuisance_controls = [control_name for control_name in controls if str(control_name) != str(target_name)]
    design_matrix = _build_residualisation_design(metadata_df, random_effect=random_effect, controls=nuisance_controls)
    coefficients, _, _, _ = np.linalg.lstsq(design_matrix, target_values, rcond=None)
    return target_values - (design_matrix @ coefficients)


def _build_residualisation_design(metadata_df: pd.DataFrame, *, random_effect: str, controls: list[str]) -> np.ndarray:
    design_columns: list[pd.Series] = [pd.Series(np.ones(len(metadata_df), dtype=float), name="Intercept")]
    design_columns.extend(_encode_column(metadata_df, random_effect, force_categorical=True))
    for control_name in controls:
        design_columns.extend(_encode_column(metadata_df, control_name, force_categorical=(str(control_name) == "run")))

    design_df = pd.concat(design_columns, axis=1)
    return design_df.to_numpy(dtype=float, copy=False)


def _encode_column(metadata_df: pd.DataFrame, column_name: str, *, force_categorical: bool) -> list[pd.Series]:
    series = metadata_df[column_name]
    if force_categorical or pd.api.types.is_object_dtype(series) or isinstance(series.dtype, CategoricalDtype):
        dummies = pd.get_dummies(series.astype(str), prefix=column_name, drop_first=True, dtype=float)
        return [dummies.loc[:, column].astype(float) for column in dummies.columns]
    return [pd.to_numeric(series, errors="raise").astype(float).rename(column_name)]


def _build_group_folds(
    groups: pd.Series,
    *,
    n_splits: int,
    shuffle: bool,
    random_state: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    unique_groups = groups.astype(str).drop_duplicates().to_numpy()
    if unique_groups.shape[0] < n_splits:
        raise ValueError(f"Need at least {n_splits} unique groups for grouped decoding CV; found {unique_groups.shape[0]}.")
    ordered_groups = unique_groups.copy()
    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(ordered_groups)
    group_splits = [split for split in np.array_split(ordered_groups, n_splits) if len(split) > 0]
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    all_groups = groups.astype(str).to_numpy()
    for test_groups in group_splits:
        test_mask = np.isin(all_groups, test_groups)
        train_indices = np.flatnonzero(~test_mask)
        test_indices = np.flatnonzero(test_mask)
        if train_indices.size == 0 or test_indices.size == 0:
            continue
        folds.append((train_indices, test_indices))
    if not folds:
        raise ValueError("Unable to construct any non-empty grouped decoding folds.")
    return folds


def _run_temporal_generalisation(
    eeg_data: np.ndarray,
    train_targets: np.ndarray,
    test_targets: np.ndarray,
    *,
    estimator_config: Mapping[str, Any],
    cv_config: Mapping[str, Any],
    nested_cv_config: Mapping[str, Any],
    scoring_config: Mapping[str, Any],
    fold_indices: list[tuple[np.ndarray, np.ndarray]],
    groups: np.ndarray,
    model_name: str,
    show_progress: bool,
    save_alpha_scores: bool = False,
    alpha_scores_output_path=None,
    fixed_alpha: float | None = None,
) -> TemporalGeneralizationResult:
    estimator_name = str(estimator_config.get("name", "ridge"))
    if estimator_name != "ridge":
        raise ValueError(f"Unsupported decoding estimator `{estimator_name}`. Only `ridge` is implemented.")
    scoring_name = str(scoring_config.get("name", "pearson_r"))
    if scoring_name != "pearson_r":
        raise ValueError(f"Unsupported decoding scoring metric `{scoring_name}`. Only `pearson_r` is implemented.")

    n_times = int(eeg_data.shape[2])
    n_folds = len(fold_indices)
    fold_scores = np.full((n_folds, n_times, n_times), np.nan, dtype=float)
    ridge_alphas = resolve_ridge_alphas(estimator_config)
    scorer = make_scorer(_pearson_r, greater_is_better=True)
    fold_best_alphas = np.full(n_folds, np.nan, dtype=float)
    alpha_scores_sum = None
    alpha_scores_memmap = None
    alpha_count = int(ridge_alphas.size)
    groups = np.asarray(groups)
    if save_alpha_scores:
        if alpha_scores_output_path is not None:
            ensure_parent_dir(alpha_scores_output_path)
            alpha_scores_memmap = open_memmap(
                alpha_scores_output_path,
                mode="w+",
                dtype=float,
                shape=(alpha_count, n_times, n_times),
            )
            alpha_scores_memmap[:] = 0.0
        else:
            alpha_scores_sum = np.zeros((alpha_count, n_times, n_times), dtype=float)

    progress = _build_progress()
    task_id: int | None = None
    total_steps = n_folds * (1 + (alpha_count if save_alpha_scores else 0))
    if show_progress:
        progress.start()
        task_id = progress.add_task(f"Decoding {model_name}", total=total_steps)

    estimator_cfg = dict(estimator_config)
    cv_cfg = dict(cv_config)
    nested_cfg = dict(nested_cv_config or {})
    scoring_cfg = dict(scoring_config)
    final_scoring_start = perf_counter()
    try:
        for fold_idx, (train_indices, test_indices) in enumerate(fold_indices):
            y_train = train_targets[train_indices]
            y_test = test_targets[test_indices]
            train_eeg = eeg_data[train_indices]
            test_eeg = eeg_data[test_indices]
            if fixed_alpha is not None:
                # Permutation inference may optionally condition on a fixed observed-run alpha
                # to avoid re-running inner CV in every permutation. The observed decoding run
                # still uses nested CV and should keep `fixed_alpha=None`.
                best_alpha = float(fixed_alpha)
            else:
                alpha_selection_start = perf_counter()
                best_alpha = _select_outer_fold_alpha(
                    train_eeg,
                    y_train,
                    outer_train_groups=groups[train_indices],
                    estimator_config=estimator_cfg,
                    cv_config=cv_cfg,
                    nested_cv_config=nested_cfg,
                    scoring_config=scoring_cfg,
                )
                LOGGER.info(
                    "Selected alpha for model=%s fold=%d/%d in %.2fs",
                    model_name,
                    fold_idx + 1,
                    n_folds,
                    perf_counter() - alpha_selection_start,
                )
            estimator = _make_generalizing_estimator(best_alpha, scorer)
            estimator.fit(train_eeg, y_train)
            best_alpha_scores = np.asarray(estimator.score(test_eeg, y_test), dtype=float)
            fold_scores[fold_idx] = best_alpha_scores
            fold_best_alphas[fold_idx] = float(best_alpha)
            if task_id is not None:
                progress.update(
                    task_id,
                    advance=1,
                    description=f"Decoding {model_name} (cv fold {fold_idx + 1}/{n_folds})",
                )
            if save_alpha_scores:
                alpha_export_start = perf_counter()
                for alpha_idx, alpha_value in enumerate(ridge_alphas):
                    if np.isclose(float(alpha_value), float(best_alpha)):
                        alpha_matrix = best_alpha_scores
                    else:
                        alpha_estimator = _make_generalizing_estimator(alpha_value, scorer)
                        alpha_estimator.fit(train_eeg, y_train)
                        alpha_matrix = np.asarray(alpha_estimator.score(test_eeg, y_test), dtype=float)
                    if alpha_scores_memmap is not None:
                        alpha_scores_memmap[alpha_idx] += alpha_matrix
                    else:
                        assert alpha_scores_sum is not None
                        alpha_scores_sum[alpha_idx] += alpha_matrix
                    if task_id is not None:
                        progress.update(
                            task_id,
                            advance=1,
                            description=(
                                f"Decoding {model_name} "
                                f"(alpha {alpha_idx + 1}/{alpha_count}, fold {fold_idx + 1}/{n_folds})"
                            ),
                        )
                LOGGER.info(
                    "Exported alpha TG scores for model=%s fold=%d/%d in %.2fs",
                    model_name,
                    fold_idx + 1,
                    n_folds,
                    perf_counter() - alpha_export_start,
                )
    finally:
        if show_progress:
            progress.stop()
    LOGGER.info("Completed final scoring for model=%s in %.2fs", model_name, perf_counter() - final_scoring_start)

    mean_scores = np.nanmean(fold_scores, axis=0)
    best_alphas = fold_best_alphas
    alpha_scores = None
    if save_alpha_scores:
        denominator = float(n_folds)
        if alpha_scores_memmap is not None:
            alpha_scores_memmap[:] = alpha_scores_memmap[:] / denominator
            alpha_scores_memmap.flush()
        else:
            assert alpha_scores_sum is not None
            alpha_scores = alpha_scores_sum / denominator
    return TemporalGeneralizationResult(
        fold_scores=fold_scores,
        mean_scores=mean_scores,
        alpha_scores=alpha_scores,
        best_alphas=best_alphas,
    )


def _build_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} train-times"),
        TimeElapsedColumn(),
        console=Console(file=sys.stderr, force_terminal=True),
        refresh_per_second=4,
        transient=False,
    )


def _build_decoding_summary(
    decoding_config: Mapping[str, Any],
    *,
    filtered_df: pd.DataFrame,
    eeg_residuals: np.ndarray,
    model_cfg: Mapping[str, str],
    dataset_index: str | None,
    sfreq: float,
) -> dict[str, Any]:
    return {
        "analysis_name": str(decoding_config.get("analysis_name", "decoding")),
        "model_name": str(model_cfg["name"]),
        "train_target": str(model_cfg["train_target"]),
        "test_target": str(model_cfg["test_target"]),
        "n_trials_used": int(filtered_df.shape[0]),
        "n_subjects": int(filtered_df[str(decoding_config["cv"]["group_column"])].nunique()),
        "n_channels": int(eeg_residuals.shape[1]),
        "n_times": int(eeg_residuals.shape[2]),
        "cv": {
            "n_splits": int(decoding_config["cv"]["n_splits"]),
            "group_column": str(decoding_config["cv"]["group_column"]),
            "shuffle": bool(decoding_config["cv"].get("shuffle", True)),
            "random_state": int(decoding_config["cv"].get("random_state", decoding_config.get("random_state", 42))),
        },
        "nested_cv": {
            "enabled": bool((decoding_config.get("nested_cv") or {}).get("enabled", True)),
            "inner_n_splits": int((decoding_config.get("nested_cv") or {}).get("inner_n_splits", 4)),
            "selection_metric": str((decoding_config.get("nested_cv") or {}).get("selection_metric", "mean_diagonal_pearson_r")),
            "select_single_alpha_per_outer_fold": bool(
                (decoding_config.get("nested_cv") or {}).get("select_single_alpha_per_outer_fold", True)
            ),
        },
        "scoring": {"name": str(decoding_config["scoring"]["name"])},
        "estimator": {
            "name": str(decoding_config["estimator"]["name"]),
            "alpha": None
            if (
                decoding_config["estimator"].get("alphas") is not None
                or decoding_config["estimator"].get("alpha_grid") is not None
            )
            else float(decoding_config["estimator"]["alpha"]),
            "alphas": resolve_ridge_alphas(decoding_config["estimator"]).tolist(),
        },
        "random_state": int(decoding_config.get("random_state", 42)),
        "decoding_sfreq": float(sfreq),
        "input": {
            "epochs_section": str(decoding_config["input"]["epochs_section"]),
            "events_table": decoding_config["input"].get("events_table"),
            "dataset_index": dataset_index,
        },
        "eeg_residualisation": {
            "random_effect": str(decoding_config["eeg_residualisation"]["random_effect"]),
            "controls": list(decoding_config["eeg_residualisation"]["controls"]),
            "target_excluded": True,
        },
        "target_residualisation": {
            "random_effect": str(decoding_config["target_residualisation"]["random_effect"]),
            "controls": list(decoding_config["target_residualisation"]["controls"]),
            "exclude_target_from_own_nuisance_model": True,
        },
        "interpretation": (
            "Temporal generalisation scores quantify how well residual speech-rate fluctuations "
            "can be decoded from residual EEG fluctuations after removing subject intercepts "
            "and fixed-effect nuisance controls from both signals."
        ),
    }


def resolve_ridge_alphas(estimator_config: Mapping[str, Any]) -> np.ndarray:
    """Resolve ridge alpha candidates from config."""
    alpha_grid = estimator_config.get("alpha_grid")
    if alpha_grid is not None:
        resolved_grid = np.asarray(alpha_grid, dtype=float)
        if resolved_grid.ndim != 1 or resolved_grid.size < 1:
            raise ValueError("`decoding.estimator.alpha_grid` must resolve to a non-empty 1D array.")
        return resolved_grid
    alphas_cfg = estimator_config.get("alphas")
    if alphas_cfg is not None:
        start = float(alphas_cfg["start"])
        stop = float(alphas_cfg["stop"])
        n_alphas = int(alphas_cfg["n_alphas"])
        if n_alphas < 1:
            raise ValueError("`decoding.estimator.alphas.n_alphas` must be >= 1.")
        return np.logspace(start, stop, n_alphas, dtype=float)
    return np.asarray([float(estimator_config.get("alpha", 1.0))], dtype=float)


def _build_ridge_estimator(alpha: float):
    return make_pipeline(StandardScaler(), Ridge(alpha=float(alpha)))


def _make_generalizing_estimator(alpha: float, scorer) -> GeneralizingEstimator:
    return GeneralizingEstimator(
        _build_ridge_estimator(float(alpha)),
        scoring=scorer,
        n_jobs=1,
        verbose=False,
    )


def _select_outer_fold_alpha(
    outer_train_eeg: np.ndarray,
    outer_train_targets: np.ndarray,
    *,
    outer_train_groups: np.ndarray,
    estimator_config: Mapping[str, Any],
    cv_config: Mapping[str, Any],
    nested_cv_config: Mapping[str, Any],
    scoring_config: Mapping[str, Any],
) -> float:
    candidate_alphas = resolve_ridge_alphas(estimator_config)
    if int(candidate_alphas.size) == 1:
        return float(candidate_alphas[0])

    nested_cfg = dict(nested_cv_config or {})
    if not bool(nested_cfg.get("enabled", True)):
        return float(candidate_alphas[0])

    unique_group_count = int(pd.Series(np.asarray(outer_train_groups)).astype(str).nunique())
    if unique_group_count < 2:
        raise ValueError("Nested decoding CV requires at least 2 unique groups in each outer-train split.")
    effective_inner_splits = min(int(nested_cfg.get("inner_n_splits", 4)), unique_group_count)
    inner_fold_indices = _build_group_folds(
        pd.Series(np.asarray(outer_train_groups)),
        n_splits=effective_inner_splits,
        shuffle=bool(cv_config.get("shuffle", True)),
        random_state=int(cv_config.get("random_state", 42)),
    )

    best_score = -np.inf
    best_alpha = float(candidate_alphas[0])
    for alpha_value in candidate_alphas:
        inner_score = _score_alpha_with_inner_cv(
            outer_train_eeg,
            outer_train_targets,
            alpha=float(alpha_value),
            scoring_config=scoring_config,
            inner_fold_indices=inner_fold_indices,
        )
        if inner_score > best_score:
            best_score = inner_score
            best_alpha = float(alpha_value)
    return best_alpha


def _score_alpha_with_inner_cv(
    eeg_data: np.ndarray,
    targets: np.ndarray,
    *,
    alpha: float,
    scoring_config: Mapping[str, Any],
    inner_fold_indices: list[tuple[np.ndarray, np.ndarray]],
) -> float:
    scoring_name = str(scoring_config.get("name", "pearson_r"))
    if scoring_name != "pearson_r":
        raise ValueError(f"Unsupported decoding scoring metric `{scoring_name}`. Only `pearson_r` is implemented.")

    scorer = make_scorer(_pearson_r, greater_is_better=True)
    fold_scores: list[float] = []
    for train_indices, test_indices in inner_fold_indices:
        estimator = _make_generalizing_estimator(alpha, scorer)
        estimator.fit(eeg_data[train_indices], targets[train_indices])
        tg_scores = np.asarray(estimator.score(eeg_data[test_indices], targets[test_indices]), dtype=float)
        fold_scores.append(float(np.nanmean(np.diag(tg_scores))))
    return float(np.nanmean(fold_scores))


def _pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    centered_true = np.asarray(y_true, dtype=float) - np.mean(y_true)
    centered_pred = np.asarray(y_pred, dtype=float) - np.mean(y_pred)
    denominator = float(np.linalg.norm(centered_true) * np.linalg.norm(centered_pred))
    if np.isclose(denominator, 0.0):
        return 0.0
    return float(np.dot(centered_true, centered_pred) / denominator)
