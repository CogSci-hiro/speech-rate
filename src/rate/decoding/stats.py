"""Permutation-based TFCE inference for temporal-generalisation matrices."""

from __future__ import annotations

import json
import logging
from typing import Any, Mapping

import numpy as np

from rate.decoding.io import build_decoding_model_output_paths
from rate.decoding.pipeline import PreparedDecodingData, _build_progress, _run_temporal_generalisation, prepare_decoding_model_data
from rate.paths import ensure_parent_dir

LOGGER = logging.getLogger(__name__)


def run_decoding_model_stats(
    config: Mapping[str, Any],
    decoding_config: Mapping[str, Any],
    *,
    model_name: str,
    dataset_index: str | None = None,
    show_progress: bool = False,
) -> dict[str, Any]:
    """Run permutation TFCE inference for one temporal-generalisation model."""
    stats_cfg = dict(decoding_config.get("stats") or {})
    if not bool(stats_cfg.get("enabled", True)):
        raise ValueError("Decoding stats are disabled in the decoding config.")
    if int(stats_cfg.get("n_jobs", 1)) != 1:
        raise ValueError("Decoding permutation stats currently support only `n_jobs: 1`.")

    prepared = prepare_decoding_model_data(
        config,
        decoding_config,
        model_name=model_name,
        dataset_index=dataset_index,
    )
    output_paths = build_decoding_model_output_paths(config, decoding_config, model_name=model_name)
    if output_paths.scores_output_path.exists():
        observed_matrix = np.load(output_paths.scores_output_path)
    else:
        observed_matrix = _run_temporal_generalisation(
            prepared.eeg_residuals,
            prepared.train_target_residuals,
            prepared.test_target_residuals,
            estimator_config=decoding_config["estimator"],
            cv_config=decoding_config["cv"],
            nested_cv_config=decoding_config.get("nested_cv") or {},
            scoring_config=decoding_config["scoring"],
            fold_indices=prepared.fold_indices,
            groups=prepared.metadata_df[str(decoding_config["cv"]["group_column"])].to_numpy(),
            model_name=prepared.model_name,
            show_progress=show_progress,
            save_alpha_scores=False,
        ).mean_scores
    tfce_params = dict(stats_cfg.get("tfce") or {})

    observed_tfce = compute_tfce_2d(
        observed_matrix,
        h_power=float(tfce_params.get("h_power", 2.0)),
        e_power=float(tfce_params.get("e_power", 0.5)),
        h_start=float(tfce_params.get("h_start", 0.0)),
        dh=float(tfce_params.get("dh", 0.1)),
        tail=str(tfce_params.get("tail", "positive")),
    )
    null_max_tfce = _compute_null_max_tfce(
        prepared,
        decoding_config=decoding_config,
        stats_config=stats_cfg,
        show_progress=show_progress,
    )
    p_fwer = _compute_fwer_p_values(observed_tfce, null_max_tfce)

    for output_path in (
        output_paths.tg_matrix_output_path,
        output_paths.tg_tfce_output_path,
        output_paths.tg_p_fwer_output_path,
        output_paths.tg_null_max_tfce_output_path,
        output_paths.tg_stats_metadata_output_path,
    ):
        ensure_parent_dir(output_path)

    np.save(output_paths.tg_matrix_output_path, np.asarray(observed_matrix, dtype=float))
    np.save(output_paths.tg_tfce_output_path, np.asarray(observed_tfce, dtype=float))
    np.save(output_paths.tg_p_fwer_output_path, np.asarray(p_fwer, dtype=float))
    np.save(output_paths.tg_null_max_tfce_output_path, np.asarray(null_max_tfce, dtype=float))

    metadata = {
        "status": "ok",
        "model_name": prepared.model_name,
        "train_target": prepared.train_target,
        "test_target": prepared.test_target,
        "scoring_metric": str((decoding_config.get("scoring") or {}).get("name", "pearson_r")),
        "permutation_count": int(stats_cfg.get("n_permutations", 1000)),
        "permutation_scheme": str(stats_cfg.get("permutation_scheme", "within_subject")),
        "tfce": {
            "enabled": bool(tfce_params.get("enabled", True)),
            "h_power": float(tfce_params.get("h_power", 2.0)),
            "e_power": float(tfce_params.get("e_power", 0.5)),
            "h_start": float(tfce_params.get("h_start", 0.0)),
            "dh": float(tfce_params.get("dh", 0.1)),
            "tail": str(tfce_params.get("tail", "positive")),
        },
        "random_seed": int(stats_cfg.get("random_state", decoding_config.get("random_state", 42))),
        "n_trials": int(prepared.metadata_df.shape[0]),
        "n_subjects": int(prepared.metadata_df[str((decoding_config.get("cv") or {}).get("group_column", "subject_id"))].nunique()),
        "n_splits": int((decoding_config.get("cv") or {}).get("n_splits", 5)),
        "decoder": dict(prepared.summary["estimator"]),
        "residualisation_description": {
            "eeg": prepared.summary["eeg_residualisation"],
            "targets": prepared.summary["target_residualisation"],
            "interpretation": prepared.summary["interpretation"],
        },
        "null_hypothesis": _null_hypothesis_text(prepared),
    }
    output_paths.tg_stats_metadata_output_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    LOGGER.info("Wrote decoding TFCE stats for model=%s -> %s", model_name, output_paths.tg_stats_metadata_output_path)
    return metadata


def compute_tfce_2d(
    score_matrix: np.ndarray,
    *,
    h_power: float,
    e_power: float,
    h_start: float,
    dh: float,
    tail: str = "positive",
) -> np.ndarray:
    """Compute a positive-tail TFCE transform on a 2D TG matrix."""
    if tail != "positive":
        raise ValueError("Only positive-tail TFCE is implemented for decoding stats.")
    scores = np.maximum(np.asarray(score_matrix, dtype=float), 0.0)
    tfce = np.zeros_like(scores, dtype=float)
    max_height = float(np.nanmax(scores))
    if np.isclose(max_height, 0.0):
        return tfce

    threshold = max(float(h_start), 0.0)
    if np.isclose(threshold, 0.0):
        threshold = float(dh)
    while threshold <= (max_height + (0.5 * dh)):
        supra_threshold = scores >= threshold
        for component in _connected_components_2d(supra_threshold):
            extent = float(len(component))
            increment = (threshold ** h_power) * (extent ** e_power) * dh
            for row_idx, col_idx in component:
                tfce[row_idx, col_idx] += increment
        threshold += dh
    return tfce


def _compute_null_max_tfce(
    prepared: PreparedDecodingData,
    *,
    decoding_config: Mapping[str, Any],
    stats_config: Mapping[str, Any],
    show_progress: bool,
) -> np.ndarray:
    n_permutations = int(stats_config.get("n_permutations", 1000))
    rng = np.random.default_rng(int(stats_config.get("random_state", decoding_config.get("random_state", 42))))
    permutation_scheme = str(stats_config.get("permutation_scheme", "within_subject"))
    tfce_cfg = dict(stats_config.get("tfce") or {})
    null_max = np.zeros(n_permutations, dtype=float)

    progress = _build_progress()
    task_id: int | None = None
    if show_progress:
        progress.start()
        task_id = progress.add_task(f"TFCE perms {prepared.model_name}", total=n_permutations)

    try:
        for permutation_idx in range(n_permutations):
            permuted_train, permuted_test = permute_decoding_targets(
                prepared,
                scheme=permutation_scheme,
                rng=rng,
            )
            permuted_scores = _run_temporal_generalisation(
                prepared.eeg_residuals,
                permuted_train,
                permuted_test,
                estimator_config=decoding_config["estimator"],
                cv_config=decoding_config["cv"],
                nested_cv_config=decoding_config.get("nested_cv") or {},
                scoring_config=decoding_config["scoring"],
                fold_indices=prepared.fold_indices,
                groups=prepared.metadata_df[str(decoding_config["cv"]["group_column"])].to_numpy(),
                model_name=f"{prepared.model_name} perm {permutation_idx + 1}",
                show_progress=False,
                save_alpha_scores=False,
            ).mean_scores
            permuted_tfce = compute_tfce_2d(
                permuted_scores,
                h_power=float(tfce_cfg.get("h_power", 2.0)),
                e_power=float(tfce_cfg.get("e_power", 0.5)),
                h_start=float(tfce_cfg.get("h_start", 0.0)),
                dh=float(tfce_cfg.get("dh", 0.1)),
                tail=str(tfce_cfg.get("tail", "positive")),
            )
            null_max[permutation_idx] = float(np.max(permuted_tfce))
            if task_id is not None:
                progress.update(task_id, advance=1, description=f"TFCE perms {prepared.model_name} ({permutation_idx + 1}/{n_permutations})")
    finally:
        if show_progress:
            progress.stop()
    return null_max


def permute_decoding_targets(
    prepared: PreparedDecodingData,
    *,
    scheme: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Permute targets while respecting within-subject dependence structure."""
    train_values = np.asarray(prepared.train_target_residuals, dtype=float)
    test_values = np.asarray(prepared.test_target_residuals, dtype=float)
    permuted_train = train_values.copy()
    permuted_test = test_values.copy()

    if scheme == "within_subject":
        group_keys = prepared.metadata_df["subject_id"].astype(str)
    elif scheme == "within_subject_within_run":
        if "run" not in prepared.metadata_df.columns:
            raise ValueError("`within_subject_within_run` permutation requires a `run` column in decoding metadata.")
        group_keys = prepared.metadata_df["subject_id"].astype(str) + "::" + prepared.metadata_df["run"].astype(str)
    else:
        raise ValueError(f"Unsupported decoding permutation scheme `{scheme}`.")

    unique_keys = group_keys.drop_duplicates().tolist()
    for group_key in unique_keys:
        group_indices = np.flatnonzero(group_keys.to_numpy(dtype=str) == str(group_key))
        if group_indices.size <= 1:
            continue
        shuffled_indices = rng.permutation(group_indices)
        permuted_train[group_indices] = train_values[shuffled_indices]
        if prepared.train_target == prepared.test_target:
            # Same-target decoding permutes the single residual target array used in both train and test splits.
            permuted_test[group_indices] = test_values[shuffled_indices]
        else:
            # Cross-condition decoding permutes only the training target.
            # This tests a null of no decodable mapping from EEG to the training target,
            # not a null of no shared representation beyond behavioural covariance.
            permuted_test[group_indices] = test_values[group_indices]
    return permuted_train, permuted_test


def _compute_fwer_p_values(observed_tfce: np.ndarray, null_max_tfce: np.ndarray) -> np.ndarray:
    null_max = np.asarray(null_max_tfce, dtype=float)
    observed = np.asarray(observed_tfce, dtype=float)
    comparisons = null_max[:, np.newaxis, np.newaxis] >= observed[np.newaxis, :, :]
    return (1.0 + comparisons.sum(axis=0)) / (float(null_max.shape[0]) + 1.0)


def _connected_components_2d(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    visited = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    n_rows, n_cols = mask.shape
    for row_idx in range(n_rows):
        for col_idx in range(n_cols):
            if not bool(mask[row_idx, col_idx]) or bool(visited[row_idx, col_idx]):
                continue
            stack = [(row_idx, col_idx)]
            visited[row_idx, col_idx] = True
            component: list[tuple[int, int]] = []
            while stack:
                current_row, current_col = stack.pop()
                component.append((current_row, current_col))
                for delta_row, delta_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    next_row = current_row + delta_row
                    next_col = current_col + delta_col
                    if next_row < 0 or next_row >= n_rows or next_col < 0 or next_col >= n_cols:
                        continue
                    if not bool(mask[next_row, next_col]) or bool(visited[next_row, next_col]):
                        continue
                    visited[next_row, next_col] = True
                    stack.append((next_row, next_col))
            components.append(component)
    return components


def _null_hypothesis_text(prepared: PreparedDecodingData) -> str:
    if prepared.train_target == prepared.test_target:
        return (
            "Permutation inference tests a null of no decodable mapping from residual EEG to the residual "
            f"training target `{prepared.train_target}` by refitting the full decoder after within-group target shuffling."
        )
    return (
        "Permutation inference tests a null of no decodable mapping from residual EEG to the residual "
        f"training target `{prepared.train_target}` while evaluating against the unpermuted test target `{prepared.test_target}`. "
        "It does not isolate a null of no shared representation beyond behavioural covariance."
    )
