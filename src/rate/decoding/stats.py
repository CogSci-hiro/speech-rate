"""Permutation-based TFCE inference for temporal-generalisation matrices.

The default permutation path re-runs nested alpha selection inside each
permutation. That remains the most principled option because it propagates
hyperparameter-selection uncertainty into the null distribution.

An optional shortcut is also supported for computationally heavy analyses:
permutations may condition on a single alpha chosen from the observed decoding
run. This yields a conditional null for a fixed decoder regularisation, which
is faster but slightly less conservative than full per-permutation retuning.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Mapping

import numpy as np
from scipy import ndimage

from rate.decoding.io import build_decoding_model_output_paths
from rate.decoding.pipeline import PreparedDecodingData, _build_progress, _run_temporal_generalisation, prepare_decoding_model_data
from rate.paths import ensure_parent_dir

LOGGER = logging.getLogger(__name__)

_TFCE_CONNECTIVITY_2D = ndimage.generate_binary_structure(2, 1)


@dataclass(frozen=True)
class PermutationGrouping:
    """Immutable lookup structure for repeated within-group shuffling."""

    group_labels: tuple[str, ...]
    group_indices: tuple[np.ndarray, ...]


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

    prep_start = perf_counter()
    prepared = prepare_decoding_model_data(
        config,
        decoding_config,
        model_name=model_name,
        dataset_index=dataset_index,
    )
    LOGGER.info("Loaded prepared stats inputs for model=%s in %.2fs", model_name, perf_counter() - prep_start)
    output_paths = build_decoding_model_output_paths(config, decoding_config, model_name=model_name)
    observed_tg_result = None
    if output_paths.scores_output_path.exists():
        observed_matrix = np.load(output_paths.scores_output_path)
    else:
        observed_start = perf_counter()
        observed_tg_result = _run_temporal_generalisation(
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
        )
        observed_matrix = observed_tg_result.mean_scores
        LOGGER.info("Computed observed TG scores for model=%s in %.2fs", model_name, perf_counter() - observed_start)

    observed_best_alphas: np.ndarray | None = None
    if output_paths.best_alphas_output_path.exists():
        observed_best_alphas = np.asarray(np.load(output_paths.best_alphas_output_path), dtype=float)
    elif observed_tg_result is not None:
        observed_best_alphas = np.asarray(observed_tg_result.best_alphas, dtype=float)

    permutation_alpha_mode = str(stats_cfg.get("permutation_alpha_mode", "retune_per_permutation"))
    fixed_alpha_source = str(stats_cfg.get("fixed_alpha_source", "median_observed_outer_folds"))
    fixed_permutation_alpha = _resolve_permutation_fixed_alpha(
        observed_best_alphas,
        permutation_alpha_mode=permutation_alpha_mode,
        fixed_alpha_source=fixed_alpha_source,
    )
    tfce_params = dict(stats_cfg.get("tfce") or {})

    tfce_thresholds = _build_tfce_thresholds(
        observed_matrix,
        h_start=float(tfce_params.get("h_start", 0.0)),
        dh=float(tfce_params.get("dh", 0.1)),
    )
    observed_tfce_start = perf_counter()
    observed_tfce = compute_tfce_2d(
        observed_matrix,
        h_power=float(tfce_params.get("h_power", 2.0)),
        e_power=float(tfce_params.get("e_power", 0.5)),
        h_start=float(tfce_params.get("h_start", 0.0)),
        dh=float(tfce_params.get("dh", 0.1)),
        tail=str(tfce_params.get("tail", "positive")),
        thresholds=tfce_thresholds,
    )
    LOGGER.info("Computed observed TFCE for model=%s in %.2fs", model_name, perf_counter() - observed_tfce_start)
    null_max_tfce = _compute_null_max_tfce(
        prepared,
        decoding_config=decoding_config,
        stats_config=stats_cfg,
        fixed_alpha=fixed_permutation_alpha,
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
        "permutation_alpha_mode": permutation_alpha_mode,
        "fixed_alpha_source": fixed_alpha_source if permutation_alpha_mode == "fixed_alpha" else None,
        "fixed_alpha": float(fixed_permutation_alpha) if fixed_permutation_alpha is not None else None,
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
        "permutation_inference_notes": _permutation_inference_notes(
            permutation_alpha_mode=permutation_alpha_mode,
            fixed_alpha_source=fixed_alpha_source,
            fixed_alpha=fixed_permutation_alpha,
        ),
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
    thresholds: np.ndarray | None = None,
) -> np.ndarray:
    """Compute a positive-tail TFCE transform on a 2D TG matrix."""
    if tail != "positive":
        raise ValueError("Only positive-tail TFCE is implemented for decoding stats.")
    scores = np.maximum(np.asarray(score_matrix, dtype=float), 0.0)
    tfce = np.zeros_like(scores, dtype=float)
    max_height = float(np.nanmax(scores))
    if np.isclose(max_height, 0.0):
        return tfce

    threshold_grid = thresholds if thresholds is not None else _build_tfce_thresholds(scores, h_start=h_start, dh=dh)
    if threshold_grid.size == 0:
        return tfce

    for threshold in threshold_grid:
        if threshold > (max_height + (0.5 * dh)):
            break
        supra_threshold = scores >= threshold
        if not np.any(supra_threshold):
            continue
        labeled_components, component_count = ndimage.label(supra_threshold, structure=_TFCE_CONNECTIVITY_2D)
        if component_count == 0:
            continue
        component_extents = np.bincount(labeled_components.ravel(), minlength=component_count + 1)
        component_increments = ((float(threshold) ** h_power) * (component_extents.astype(float) ** e_power) * dh)
        component_increments[0] = 0.0
        tfce += component_increments[labeled_components]
    return tfce


def _compute_null_max_tfce(
    prepared: PreparedDecodingData,
    *,
    decoding_config: Mapping[str, Any],
    stats_config: Mapping[str, Any],
    fixed_alpha: float | None,
    show_progress: bool,
) -> np.ndarray:
    n_permutations = int(stats_config.get("n_permutations", 1000))
    rng = np.random.default_rng(int(stats_config.get("random_state", decoding_config.get("random_state", 42))))
    permutation_scheme = str(stats_config.get("permutation_scheme", "within_subject"))
    tfce_cfg = dict(stats_config.get("tfce") or {})
    estimator_cfg = decoding_config["estimator"]
    cv_cfg = decoding_config["cv"]
    nested_cv_cfg = decoding_config.get("nested_cv") or {}
    scoring_cfg = decoding_config["scoring"]
    group_column = str(cv_cfg["group_column"])
    groups = prepared.metadata_df[group_column].to_numpy()
    permutation_grouping = build_permutation_grouping(prepared, scheme=permutation_scheme)
    null_max = np.zeros(n_permutations, dtype=float)

    progress = _build_progress()
    task_id: int | None = None
    if show_progress:
        progress.start()
        task_id = progress.add_task(f"TFCE perms {prepared.model_name}", total=n_permutations)

    permutation_start = perf_counter()
    try:
        for permutation_idx in range(n_permutations):
            permuted_train, permuted_test = permute_decoding_targets(
                prepared,
                scheme=permutation_scheme,
                rng=rng,
                grouping=permutation_grouping,
            )
            permuted_scores = _run_temporal_generalisation(
                prepared.eeg_residuals,
                permuted_train,
                permuted_test,
                estimator_config=estimator_cfg,
                cv_config=cv_cfg,
                nested_cv_config=nested_cv_cfg,
                scoring_config=scoring_cfg,
                fold_indices=prepared.fold_indices,
                groups=groups,
                model_name=f"{prepared.model_name} perm {permutation_idx + 1}",
                show_progress=False,
                save_alpha_scores=False,
                fixed_alpha=fixed_alpha,
            ).mean_scores
            permuted_thresholds = _build_tfce_thresholds(
                permuted_scores,
                h_start=float(tfce_cfg.get("h_start", 0.0)),
                dh=float(tfce_cfg.get("dh", 0.1)),
            )
            permuted_tfce = compute_tfce_2d(
                permuted_scores,
                h_power=float(tfce_cfg.get("h_power", 2.0)),
                e_power=float(tfce_cfg.get("e_power", 0.5)),
                h_start=float(tfce_cfg.get("h_start", 0.0)),
                dh=float(tfce_cfg.get("dh", 0.1)),
                tail=str(tfce_cfg.get("tail", "positive")),
                thresholds=permuted_thresholds,
            )
            null_max[permutation_idx] = float(np.max(permuted_tfce))
            if task_id is not None:
                progress.update(task_id, advance=1, description=f"TFCE perms {prepared.model_name} ({permutation_idx + 1}/{n_permutations})")
    finally:
        if show_progress:
            progress.stop()
    LOGGER.info(
        "Computed %d decoding permutations for model=%s in %.2fs",
        n_permutations,
        prepared.model_name,
        perf_counter() - permutation_start,
    )
    return null_max


def _resolve_permutation_fixed_alpha(
    observed_best_alphas: np.ndarray | None,
    *,
    permutation_alpha_mode: str,
    fixed_alpha_source: str,
) -> float | None:
    """Resolve the optional fixed alpha used only for permutation inference.

    `retune_per_permutation` is the gold-standard option because it re-runs
    nested CV inside every permutation. `fixed_alpha` is a computational
    shortcut that conditions the null on a single observed-run regularisation
    choice and therefore may slightly underestimate uncertainty.
    """
    if permutation_alpha_mode == "retune_per_permutation":
        return None
    if permutation_alpha_mode != "fixed_alpha":
        raise ValueError(f"Unsupported decoding permutation alpha mode `{permutation_alpha_mode}`.")
    if observed_best_alphas is None:
        raise ValueError(
            "Fixed-alpha decoding permutations require observed outer-fold alphas from the observed decoding run."
        )
    observed_best_alphas = np.asarray(observed_best_alphas, dtype=float)
    finite_alphas = observed_best_alphas[np.isfinite(observed_best_alphas)]
    if finite_alphas.size == 0:
        raise ValueError("Observed decoding outer-fold alphas are empty or non-finite; cannot run fixed-alpha permutations.")
    if fixed_alpha_source == "median_observed_outer_folds":
        return float(np.median(finite_alphas))
    raise ValueError(f"Unsupported decoding fixed alpha source `{fixed_alpha_source}`.")


def _permutation_inference_notes(
    *,
    permutation_alpha_mode: str,
    fixed_alpha_source: str,
    fixed_alpha: float | None,
) -> dict[str, Any]:
    if permutation_alpha_mode == "retune_per_permutation":
        return {
            "summary": "Permutation inference re-runs nested alpha selection inside each permutation.",
            "methodological_status": "gold_standard",
            "conditioning": "Null distribution propagates hyperparameter-selection uncertainty.",
        }
    return {
        "summary": (
            "Permutation inference uses a fixed alpha from the observed decoding run as a computational shortcut "
            "instead of re-running nested CV inside each permutation."
        ),
        "methodological_status": "computational_approximation",
        "conditioning": "Null distribution is conditional on the chosen regularisation value.",
        "fixed_alpha_source": fixed_alpha_source,
        "fixed_alpha": None if fixed_alpha is None else float(fixed_alpha),
        "caveat": "This shortcut may slightly underestimate uncertainty relative to full per-permutation retuning.",
    }


def permute_decoding_targets(
    prepared: PreparedDecodingData,
    *,
    scheme: str,
    rng: np.random.Generator,
    grouping: PermutationGrouping | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Permute targets while respecting within-subject dependence structure."""
    train_values = np.asarray(prepared.train_target_residuals, dtype=float)
    test_values = np.asarray(prepared.test_target_residuals, dtype=float)
    permuted_train = train_values.copy()
    permuted_test = test_values.copy()

    resolved_grouping = grouping if grouping is not None else build_permutation_grouping(prepared, scheme=scheme)
    for group_indices in resolved_grouping.group_indices:
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


def build_permutation_grouping(
    prepared: PreparedDecodingData,
    *,
    scheme: str,
) -> PermutationGrouping:
    """Build ordered within-group index arrays once for repeated permutations."""
    if scheme == "within_subject":
        group_keys = prepared.metadata_df["subject_id"].astype(str)
    elif scheme == "within_subject_within_run":
        if "run" not in prepared.metadata_df.columns:
            raise ValueError("`within_subject_within_run` permutation requires a `run` column in decoding metadata.")
        group_keys = prepared.metadata_df["subject_id"].astype(str) + "::" + prepared.metadata_df["run"].astype(str)
    else:
        raise ValueError(f"Unsupported decoding permutation scheme `{scheme}`.")

    group_key_values = group_keys.to_numpy(dtype=str, copy=False)
    unique_keys = tuple(str(group_key) for group_key in group_keys.drop_duplicates().tolist())
    return PermutationGrouping(
        group_labels=unique_keys,
        group_indices=tuple(np.flatnonzero(group_key_values == group_key) for group_key in unique_keys),
    )


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


def _build_tfce_thresholds(score_matrix: np.ndarray, *, h_start: float, dh: float) -> np.ndarray:
    """Build the deterministic TFCE threshold grid for one TG matrix."""
    max_height = float(np.nanmax(np.maximum(np.asarray(score_matrix, dtype=float), 0.0)))
    if np.isclose(max_height, 0.0):
        return np.asarray([], dtype=float)
    threshold = max(float(h_start), 0.0)
    if np.isclose(threshold, 0.0):
        threshold = float(dh)
    max_threshold = max_height + (0.5 * dh)
    step_count = int(np.floor((max_threshold - threshold) / float(dh))) + 1
    if step_count <= 0:
        return np.asarray([], dtype=float)
    return threshold + (np.arange(step_count, dtype=float) * float(dh))


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
