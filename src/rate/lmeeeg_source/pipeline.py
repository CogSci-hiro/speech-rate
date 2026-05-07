"""Experimental source-space lmeEEG orchestration."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping
from dataclasses import dataclass

import yaml
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from lmeeeg import permute_fixed_effect
from rate.design import get_predictor_column_indices, list_model_names
from rate.lmeeeg.io import build_lmeeeg_model_output_paths
from rate.lmeeeg.mass_model import LMEEEGResult, fit_lmeeeg_backend_result, fit_lmeeeg_mass_model
from rate.lmeeeg.pooling import PooledLMEEEGInputs, fit_pooled_lmeeeg_model, select_lmeeeg_rows
from rate.lmeeeg.trial_data import LMEEEGTrialData, build_lmeeeg_trial_data_from_arrays
from rate.lmeeeg_source.config import DEFAULT_LMEEEG_SOURCE_SECTION, load_lmeeeg_source_settings
from rate.lmeeeg_source.source_space import SourcePreparedPaths, build_source_prepared_paths
from rate.paths import ensure_parent_dir, resolve_out_dir
from rate.util import write_parquet_dataframe

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SourceGroupPaths:
    """Resolved group-map outputs for one source-space predictor."""

    full_stc_stem: Path
    summary_stc_stem: Path
    corrected_p_path: Path
    permutation_observed_path: Path
    cluster_p_values_path: Path
    cluster_masks_path: Path
    summary_json_path: Path


@dataclass(frozen=True)
class SourcePreparedEntry:
    prepared_paths: SourcePreparedPaths
    source_data_path: Path
    metadata_df: pd.DataFrame
    n_trials: int


@dataclass(frozen=True)
class SourcePooledManifest:
    entries: list[SourcePreparedEntry]
    metadata_df: pd.DataFrame
    channel_names: list[str]
    times: np.ndarray
    vertices_payload: dict[str, Any]


def load_pooled_source_inputs(
    config: Mapping[str, Any],
    rows: list[Mapping[str, Any]],
    *,
    config_section: str = DEFAULT_LMEEEG_SOURCE_SECTION,
) -> tuple[PooledLMEEEGInputs, dict[str, Any]]:
    """Load subject-level prepared source arrays into the standard pooled lmeEEG container."""
    settings = load_lmeeeg_source_settings(config, config_section=config_section)
    deduped_paths = _unique_prepared_paths(config, rows, config_section=config_section)

    pooled_data: list[np.ndarray] = []
    pooled_metadata: list[pd.DataFrame] = []
    reference_channel_names: list[str] | None = None
    reference_times: np.ndarray | None = None
    vertices_payload: dict[str, Any] | None = None

    for prepared_paths in tqdm(
        deduped_paths,
        disable=not settings.fitting.progress,
        desc="lmeeeg_source:pool",
    ):
        summary = json.loads(prepared_paths.summary_output_path.read_text(encoding="utf-8"))
        LOGGER.info(
            "Loading prepared source-space data for subject=%s from %s",
            summary.get("subject_id", "<unknown>"),
            prepared_paths.source_data_output_path,
        )
        source_data = np.asarray(np.load(prepared_paths.source_data_output_path), dtype=float)
        metadata_df = pd.read_csv(prepared_paths.metadata_output_path)
        times = np.asarray(np.load(prepared_paths.times_output_path), dtype=float)
        channel_names = json.loads(prepared_paths.channel_names_output_path.read_text(encoding="utf-8"))
        current_vertices = json.loads(prepared_paths.vertices_output_path.read_text(encoding="utf-8"))
        if source_data.shape[0] != len(metadata_df):
            raise ValueError(
                "Prepared source data and metadata rows do not match for "
                f"{prepared_paths.summary_output_path}: {source_data.shape[0]} != {len(metadata_df)}"
            )
        if reference_channel_names is None:
            reference_channel_names = list(channel_names)
            reference_times = times
            vertices_payload = current_vertices
        else:
            if list(channel_names) != reference_channel_names:
                raise ValueError("All prepared source-space inputs must share the same morphed vertex order.")
            if not np.allclose(times, reference_times):
                raise ValueError("All prepared source-space inputs must share the same time axis.")
            if current_vertices.get("vertices") != vertices_payload.get("vertices"):
                raise ValueError("All prepared source-space inputs must share the same fsaverage vertex definition.")
        pooled_data.append(source_data)
        pooled_metadata.append(metadata_df.reset_index(drop=True))

    if reference_channel_names is None or reference_times is None or vertices_payload is None:
        raise ValueError("No prepared source-space inputs were available for pooled fitting.")

    pooled_inputs = PooledLMEEEGInputs(
        eeg_data=np.concatenate(pooled_data, axis=0),
        metadata_df=pd.concat(pooled_metadata, axis=0, ignore_index=True),
        channel_names=reference_channel_names,
        times=reference_times,
        metadata_provenance={"prepared_subjects": [str(path.summary_output_path) for path in deduped_paths]},
    )
    return pooled_inputs, vertices_payload


def load_pooled_source_manifest(
    config: Mapping[str, Any],
    rows: list[Mapping[str, Any]],
    *,
    config_section: str = DEFAULT_LMEEEG_SOURCE_SECTION,
) -> SourcePooledManifest:
    """Read pooled source metadata and validate axes without concatenating all arrays."""
    settings = load_lmeeeg_source_settings(config, config_section=config_section)
    deduped_paths = _unique_prepared_paths(config, rows, config_section=config_section)

    entries: list[SourcePreparedEntry] = []
    pooled_metadata: list[pd.DataFrame] = []
    reference_channel_names: list[str] | None = None
    reference_times: np.ndarray | None = None
    vertices_payload: dict[str, Any] | None = None

    for prepared_paths in tqdm(
        deduped_paths,
        disable=not settings.fitting.progress,
        desc="lmeeeg_source:manifest",
    ):
        summary = json.loads(prepared_paths.summary_output_path.read_text(encoding="utf-8"))
        metadata_df = pd.read_csv(prepared_paths.metadata_output_path).reset_index(drop=True)
        channel_names = json.loads(prepared_paths.channel_names_output_path.read_text(encoding="utf-8"))
        times = np.asarray(np.load(prepared_paths.times_output_path), dtype=float)
        current_vertices = json.loads(prepared_paths.vertices_output_path.read_text(encoding="utf-8"))
        source_data = np.load(prepared_paths.source_data_output_path, mmap_mode="r")
        if int(source_data.shape[0]) != len(metadata_df):
            raise ValueError(
                "Prepared source data and metadata rows do not match for "
                f"{prepared_paths.summary_output_path}: {source_data.shape[0]} != {len(metadata_df)}"
            )
        if reference_channel_names is None:
            reference_channel_names = list(channel_names)
            reference_times = times
            vertices_payload = current_vertices
        else:
            if list(channel_names) != reference_channel_names:
                raise ValueError("All prepared source-space inputs must share the same morphed vertex order.")
            if not np.allclose(times, reference_times):
                raise ValueError("All prepared source-space inputs must share the same time axis.")
            if current_vertices.get("vertices") != vertices_payload.get("vertices"):
                raise ValueError("All prepared source-space inputs must share the same fsaverage vertex definition.")
        pooled_metadata.append(metadata_df)
        entries.append(
            SourcePreparedEntry(
                prepared_paths=prepared_paths,
                source_data_path=prepared_paths.source_data_output_path,
                metadata_df=metadata_df,
                n_trials=int(source_data.shape[0]),
            )
        )
        LOGGER.info(
            "Registered source manifest entry for subject=%s with %d trials from %s",
            summary.get("subject_id", "<unknown>"),
            int(source_data.shape[0]),
            prepared_paths.source_data_output_path,
        )

    if reference_channel_names is None or reference_times is None or vertices_payload is None:
        raise ValueError("No prepared source-space inputs were available for pooled fitting.")

    return SourcePooledManifest(
        entries=entries,
        metadata_df=pd.concat(pooled_metadata, axis=0, ignore_index=True),
        channel_names=reference_channel_names,
        times=np.asarray(reference_times, dtype=float),
        vertices_payload=vertices_payload,
    )


def fit_source_models(
    config: Mapping[str, Any],
    *,
    config_section: str = DEFAULT_LMEEEG_SOURCE_SECTION,
    dataset_index: str | None = None,
    model_name: str | None = None,
) -> list[dict[str, Any]]:
    """Fit pooled source-space lmeEEG models in vertex chunks to reduce RAM pressure."""
    settings = load_lmeeeg_source_settings(config, config_section=config_section)
    if not settings.enabled:
        raise ValueError(f"`{config_section}.enabled` is false. Refusing to run the experimental source-space pipeline.")

    rows = select_lmeeeg_rows(config, dataset_index=dataset_index)
    model_names = [str(model_name)] if model_name else list_model_names(config, section=config_section)
    manifest = load_pooled_source_manifest(config, rows, config_section=config_section)

    summaries: list[dict[str, Any]] = []
    for current_model_name in tqdm(
        model_names,
        disable=not settings.fitting.progress,
        desc="lmeeeg_source:fit",
    ):
        LOGGER.info(
            "Fitting experimental source-space lmeEEG model=%s with %d pooled trials, %d vertices, %d time points in chunks of %d vertices",
            current_model_name,
            len(manifest.metadata_df),
            len(manifest.channel_names),
            int(manifest.times.shape[0]),
            settings.fitting.batch_size_vertices,
        )
        output_paths = build_lmeeeg_model_output_paths(config, config_section=config_section, model_name=current_model_name)
        summary = _fit_chunked_source_model(
            config,
            manifest=manifest,
            output_paths=output_paths,
            config_section=config_section,
            model_name=current_model_name,
            batch_size_vertices=settings.fitting.batch_size_vertices,
        )
        source_metadata_path = output_paths.summary_output_path.parent / "source_vertices.json"
        source_metadata_path.write_text(json.dumps(manifest.vertices_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        summaries.append(
            {
                "model_name": current_model_name,
                "summary_path": str(output_paths.summary_output_path),
                "source_vertices_path": str(source_metadata_path),
                "n_trials_used": int(summary["n_trials_used"]),
            }
        )
    return summaries


def _fit_chunked_source_model(
    config: Mapping[str, Any],
    *,
    manifest: SourcePooledManifest,
    output_paths,
    config_section: str,
    model_name: str,
    batch_size_vertices: int,
) -> dict[str, Any]:
    ensure_parent_dir(output_paths.summary_output_path)
    n_vertices = len(manifest.channel_names)
    n_times = int(manifest.times.shape[0])
    template_result: LMEEEGResult | None = None

    residualised_mm = None
    random_effects_mm = None
    betas: np.ndarray | None = None
    t_values: np.ndarray | None = None
    residual_variance: np.ndarray | None = None
    failed_target_indices: list[int] = []
    failed_targets = 0
    boundary_targets = 0
    converged_targets = 0

    for start in range(0, n_vertices, max(1, int(batch_size_vertices))):
        stop = min(n_vertices, start + max(1, int(batch_size_vertices)))
        chunk_trial_data = _load_source_trial_data_chunk(
            config,
            manifest=manifest,
            channel_slice=slice(start, stop),
            config_section=config_section,
            model_name=model_name,
        )
        LOGGER.info(
            "Fitting source-space chunk model=%s vertices=%d:%d with filtered shape=%s",
            model_name,
            start,
            stop,
            tuple(int(value) for value in chunk_trial_data.eeg_data.shape),
        )
        chunk_result = fit_lmeeeg_mass_model(
            chunk_trial_data,
            config,
            config_section=config_section,
            model_name=model_name,
        )
        if template_result is None:
            template_result = chunk_result
            residualised_mm = np.lib.format.open_memmap(
                output_paths.residualised_trials_output_path,
                mode="w+",
                dtype=np.float32,
                shape=(chunk_result.n_trials_used, n_vertices, n_times),
            )
            random_effects_mm = np.lib.format.open_memmap(
                output_paths.random_effects_output_path,
                mode="w+",
                dtype=np.float32,
                shape=(chunk_result.n_trials_used, n_vertices, n_times),
            )
            betas = np.zeros((chunk_result.betas.shape[0], n_vertices, n_times), dtype=float)
            t_values = np.zeros((chunk_result.t_values.shape[0], n_vertices, n_times), dtype=float)
            residual_variance = np.zeros((n_vertices, n_times), dtype=float)
        else:
            if not np.array_equal(chunk_result.row_index, template_result.row_index):
                raise ValueError("Source-space chunked fitting produced inconsistent row indices across chunks.")
            if chunk_result.column_names != template_result.column_names:
                raise ValueError("Source-space chunked fitting produced inconsistent fixed-effect columns across chunks.")
            if chunk_result.test_predictors != template_result.test_predictors:
                raise ValueError("Source-space chunked fitting produced inconsistent test predictors across chunks.")
        assert residualised_mm is not None
        assert random_effects_mm is not None
        assert betas is not None
        assert t_values is not None
        assert residual_variance is not None
        residualised_mm[:, start:stop, :] = np.asarray(chunk_result.residualised_data, dtype=np.float32)
        random_effects_mm[:, start:stop, :] = np.asarray(chunk_result.random_effects_contribution, dtype=np.float32)
        betas[:, start:stop, :] = np.asarray(chunk_result.betas, dtype=float)
        t_values[:, start:stop, :] = np.asarray(chunk_result.t_values, dtype=float)
        residual_variance[start:stop, :] = np.asarray(chunk_result.residual_variance, dtype=float)
        failed_target_indices.extend([(start * n_times) + int(index) for index in chunk_result.failed_target_indices])
        diagnostics = dict(chunk_result.lmm_diagnostics)
        failed_targets += int(diagnostics.get("failed_targets", 0))
        boundary_targets += int(diagnostics.get("boundary_warning_targets", 0))
        converged_targets += int(diagnostics.get("converged_targets", 0))

    if template_result is None or betas is None or t_values is None or residual_variance is None:
        raise ValueError(f"No source-space chunks were fitted for model `{model_name}`.")

    del residualised_mm
    del random_effects_mm

    np.save(output_paths.betas_output_path, betas)
    np.save(output_paths.row_index_output_path, template_result.row_index)
    np.save(output_paths.times_output_path, template_result.times)
    np.save(output_paths.standard_errors_output_path, np.asarray([], dtype=float))
    np.save(output_paths.t_values_output_path, t_values)
    np.save(output_paths.residual_variance_output_path, residual_variance)

    output_paths.channel_names_output_path.write_text(json.dumps(manifest.channel_names, indent=2) + "\n", encoding="utf-8")
    output_paths.column_names_output_path.write_text(json.dumps(template_result.column_names, indent=2) + "\n", encoding="utf-8")
    output_paths.test_predictors_output_path.write_text(json.dumps(template_result.test_predictors, indent=2) + "\n", encoding="utf-8")
    lmm_diagnostics = {
        "converged_targets": converged_targets,
        "failed_targets": failed_targets,
        "boundary_warning_targets": boundary_targets,
        "valid_targets_for_ols": converged_targets,
    }
    output_paths.lmm_diagnostics_output_path.write_text(json.dumps(lmm_diagnostics, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    selection_df = template_result.design_result.filtered_metadata.copy()
    selection_df.insert(0, "row_index", template_result.row_index.tolist())
    write_parquet_dataframe(selection_df.reset_index(drop=True), output_paths.selection_output_path)

    summary = {
        "status": "ok",
        "backend": template_result.backend,
        "random_effect_structure": template_result.random_effect_structure,
        "group_column": template_result.group_column,
        "model_formula": template_result.design_result.model_spec.formula,
        "predictor_of_interest": (
            template_result.test_predictors[0]
            if len(template_result.test_predictors) == 1
            else list(template_result.test_predictors)
        ),
        "preprocessing_source": template_result.metadata_provenance,
        "n_trials_input": template_result.n_trials_input,
        "n_trials_used": template_result.n_trials_used,
        "n_channels": len(manifest.channel_names),
        "n_times": int(template_result.times.shape[0]),
        "n_targets": len(manifest.channel_names) * int(template_result.times.shape[0]),
        "n_lmm_attempted": len(manifest.channel_names) * int(template_result.times.shape[0]),
        "n_lmm_failed": len(failed_target_indices),
        "failed_target_indices": failed_target_indices,
        "design_matrix_shape": list(template_result.design_result.design_matrix.shape),
        "betas_shape": list(betas.shape),
        "drop_report": dict(template_result.design_result.drop_report),
        "memory_strategy": {
            "mode": "chunked_source_fit",
            "batch_size_vertices": int(batch_size_vertices),
            "residualised_dtype": "float32",
            "random_effects_dtype": "float32",
        },
        "method_note": (
            "This experimental source-space lmeEEG fit streams vertex chunks from disk to reduce RAM pressure. "
            "Residualised trials and fitted random effects are saved incrementally as float32 arrays."
        ),
    }
    output_paths.summary_output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _load_source_trial_data_chunk(
    config: Mapping[str, Any],
    *,
    manifest: SourcePooledManifest,
    channel_slice: slice,
    config_section: str,
    model_name: str,
) -> LMEEEGTrialData:
    chunk_arrays: list[np.ndarray] = []
    for entry in manifest.entries:
        source_data = np.load(entry.source_data_path, mmap_mode="r")
        chunk_arrays.append(np.asarray(source_data[:, channel_slice, :], dtype=float))
    eeg_chunk = np.concatenate(chunk_arrays, axis=0)
    channel_names = manifest.channel_names[channel_slice]
    return build_lmeeeg_trial_data_from_arrays(
        eeg_data=eeg_chunk,
        metadata_df=manifest.metadata_df,
        channel_names=channel_names,
        times=manifest.times,
        config=config,
        config_section=config_section,
        model_name=model_name,
        metadata_provenance={
            "prepared_subjects": [str(entry.prepared_paths.summary_output_path) for entry in manifest.entries],
            "chunk_vertices": [int(channel_slice.start or 0), int(channel_slice.stop or len(manifest.channel_names))],
        },
    )


def build_source_group_paths(
    config: Mapping[str, Any],
    *,
    predictor_name: str,
    measure_name: str,
    config_section: str = DEFAULT_LMEEEG_SOURCE_SECTION,
) -> SourceGroupPaths:
    settings = load_lmeeeg_source_settings(config, config_section=config_section)
    root_dir = resolve_out_dir(config) / settings.outputs.namespace / settings.outputs.group_dirname / predictor_name
    return SourceGroupPaths(
        full_stc_stem=root_dir / f"{measure_name}_full",
        summary_stc_stem=root_dir / f"{measure_name}_summary",
        corrected_p_path=root_dir / "corrected_p_values.npy",
        permutation_observed_path=root_dir / "permutation_observed_statistic.npy",
        cluster_p_values_path=root_dir / "cluster_p_values.npy",
        cluster_masks_path=root_dir / "cluster_masks.npy",
        summary_json_path=root_dir / "summary.json",
    )


def write_source_group_maps(
    config: Mapping[str, Any],
    *,
    config_section: str = DEFAULT_LMEEEG_SOURCE_SECTION,
    stats_cfg: Mapping[str, Any] | None = None,
    dataset_index: str | None = None,
) -> list[dict[str, Any]]:
    """Aggregate pooled fit outputs into source-estimate files for plotting."""
    settings = load_lmeeeg_source_settings(config, config_section=config_section)
    if not settings.enabled:
        raise ValueError(f"`{config_section}.enabled` is false. Refusing to run the experimental source-space pipeline.")

    import mne

    rows = select_lmeeeg_rows(config, dataset_index=dataset_index)
    pooled_inputs, vertices_payload = load_pooled_source_inputs(config, rows, config_section=config_section)
    source_adjacency = _resolve_source_adjacency(config, config_section=config_section) if stats_cfg is not None else None

    summaries: list[dict[str, Any]] = []
    for current_model_name in list_model_names(config, section=config_section):
        output_paths = build_lmeeeg_model_output_paths(config, config_section=config_section, model_name=current_model_name)
        if not output_paths.betas_output_path.exists():
            raise FileNotFoundError(f"Expected pooled source-space fit output is missing: {output_paths.betas_output_path}")
        source_vertices_path = output_paths.summary_output_path.parent / "source_vertices.json"
        if not source_vertices_path.exists():
            raise FileNotFoundError(f"Expected source vertex metadata is missing: {source_vertices_path}")

        betas = np.asarray(np.load(output_paths.betas_output_path), dtype=float)
        t_values = np.asarray(np.load(output_paths.t_values_output_path), dtype=float)
        times = np.asarray(np.load(output_paths.times_output_path), dtype=float)
        column_names = json.loads(output_paths.column_names_output_path.read_text(encoding="utf-8"))
        vertices_payload = json.loads(source_vertices_path.read_text(encoding="utf-8"))
        vertices = [
            np.asarray(vertices_payload["vertices"]["lh"], dtype=int),
            np.asarray(vertices_payload["vertices"]["rh"], dtype=int),
        ]
        inference_by_effect: dict[str, Any] = {}
        if stats_cfg is not None:
            trial_data = build_lmeeeg_trial_data_from_arrays(
                eeg_data=pooled_inputs.eeg_data,
                metadata_df=pooled_inputs.metadata_df,
                channel_names=pooled_inputs.channel_names,
                times=pooled_inputs.times,
                config=config,
                config_section=config_section,
                model_name=current_model_name,
                metadata_provenance=pooled_inputs.metadata_provenance,
            )
            fit_result = fit_lmeeeg_backend_result(
                trial_data,
                config,
                config_section=config_section,
                model_name=current_model_name,
            )
            correction = _resolve_correction_name(stats_cfg)
            tail = int((stats_cfg.get("stats") or {}).get("tail", 0))
            n_permutations = int((stats_cfg.get("stats") or {}).get("n_permutations", 1024))
            threshold = _resolve_threshold(
                stats_cfg,
                n_observations=int(trial_data.eeg_data.shape[0]),
                correction=correction,
                tail=tail,
            )
            for predictor_name, effect_name in _resolve_effect_specs_from_trial_data(trial_data):
                inference_by_effect[effect_name] = (
                    predictor_name,
                    permute_fixed_effect(
                        fit_result=fit_result,
                        effect=effect_name,
                        correction=correction,
                        n_permutations=n_permutations,
                        seed=int((stats_cfg.get("stats") or {}).get("seed", 0)),
                        tail=tail,
                        threshold=threshold,
                        adjacency=source_adjacency,
                    ),
                )
        for predictor_name in _resolve_predictors_for_model(config, config_section=config_section, model_name=current_model_name):
            effect_indices = get_predictor_column_indices(
                predictor_name,
                column_names=column_names,
                term_name_slices={column_name: slice(index, index + 1) for index, column_name in enumerate(column_names)},
            )
            if len(effect_indices) != 1:
                raise ValueError(
                    f"Experimental source-space plotting expects exactly one fixed-effect column for `{predictor_name}`, "
                    f"got {effect_indices!r} in model `{current_model_name}`."
                )
            effect_index = int(effect_indices[0])
            beta_map = np.asarray(betas[effect_index], dtype=float)
            t_value_map = np.asarray(t_values[effect_index], dtype=float)
            beta_group_paths = build_source_group_paths(
                config,
                predictor_name=predictor_name,
                measure_name="beta",
                config_section=config_section,
            )
            t_value_group_paths = build_source_group_paths(
                config,
                predictor_name=predictor_name,
                measure_name="t_value",
                config_section=config_section,
            )
            beta_full_stc = mne.SourceEstimate(beta_map, vertices=vertices, tmin=float(times[0]), tstep=_resolve_tstep(times))
            beta_summary_stc = mne.SourceEstimate(
                _summarise_time(beta_map, times=times, mode=settings.plotting.time_summary, window=settings.plotting.time_window),
                vertices=vertices,
                tmin=0.0,
                tstep=1.0,
            )
            t_value_full_stc = mne.SourceEstimate(t_value_map, vertices=vertices, tmin=float(times[0]), tstep=_resolve_tstep(times))
            t_value_summary_stc = mne.SourceEstimate(
                _summarise_time(t_value_map, times=times, mode=settings.plotting.time_summary, window=settings.plotting.time_window),
                vertices=vertices,
                tmin=0.0,
                tstep=1.0,
            )
            primary_paths = beta_group_paths if settings.plotting.measure == "beta" else t_value_group_paths
            beta_full_stc.save(str(beta_group_paths.full_stc_stem), ftype="stc", overwrite=True)
            beta_summary_stc.save(str(beta_group_paths.summary_stc_stem), ftype="stc", overwrite=True)
            t_value_full_stc.save(str(t_value_group_paths.full_stc_stem), ftype="stc", overwrite=True)
            t_value_summary_stc.save(str(t_value_group_paths.summary_stc_stem), ftype="stc", overwrite=True)
            summary = {
                "status": "ok",
                "predictor": predictor_name,
                "model_name": current_model_name,
                "measure": settings.plotting.measure,
                "full_stc_stem": str(primary_paths.full_stc_stem),
                "summary_stc_stem": str(primary_paths.summary_stc_stem),
                "beta_full_stc_stem": str(beta_group_paths.full_stc_stem),
                "beta_summary_stc_stem": str(beta_group_paths.summary_stc_stem),
                "t_value_full_stc_stem": str(t_value_group_paths.full_stc_stem),
                "t_value_summary_stc_stem": str(t_value_group_paths.summary_stc_stem),
                "vertices_path": str(source_vertices_path),
                "time_summary": settings.plotting.time_summary,
                "time_window": None if settings.plotting.time_window is None else list(settings.plotting.time_window),
            }
            effect_name = str(column_names[effect_index])
            if effect_name in inference_by_effect:
                _predictor_name, inference = inference_by_effect[effect_name]
                np.save(primary_paths.corrected_p_path, np.asarray(inference.corrected_p_values, dtype=float))
                np.save(primary_paths.permutation_observed_path, np.asarray(inference.observed_statistic, dtype=float))
                summary.update(
                    {
                        "correction": str(inference.correction),
                        "backend_metadata": dict(inference.backend_metadata),
                        "corrected_p_values": str(primary_paths.corrected_p_path),
                        "permutation_observed_statistic": str(primary_paths.permutation_observed_path),
                    }
                )
                if inference.cluster_p_values is not None:
                    np.save(primary_paths.cluster_p_values_path, np.asarray(inference.cluster_p_values, dtype=float))
                    summary["cluster_p_values"] = str(primary_paths.cluster_p_values_path)
                cluster_masks = _stack_source_cluster_masks(
                    getattr(inference, "clusters", []) or [],
                    n_vertices=int(beta_map.shape[0]),
                    n_times=int(beta_map.shape[1]),
                )
                if cluster_masks is not None:
                    np.save(primary_paths.cluster_masks_path, cluster_masks)
                    summary["cluster_masks"] = str(primary_paths.cluster_masks_path)
            ensure_parent_dir(primary_paths.summary_json_path)
            primary_paths.summary_json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            summaries.append(summary)
    return summaries


def _resolve_predictors_for_model(
    config: Mapping[str, Any],
    *,
    config_section: str,
    model_name: str,
) -> list[str]:
    model_cfg = dict(((config.get(config_section) or {}).get("models") or {}).get(model_name) or {})
    predictors = [str(value) for value in model_cfg.get("test_predictors", [])]
    if not predictors:
        raise ValueError(f"No test predictors configured for `{config_section}.models.{model_name}`.")
    return predictors


def _resolve_effect_specs_from_trial_data(trial_data) -> list[tuple[str, str]]:
    effects: list[tuple[str, str]] = []
    for predictor_name in trial_data.design_result.test_predictors:
        for column_index in get_predictor_column_indices(
            predictor_name,
            column_names=trial_data.design_result.column_names,
            term_name_slices=trial_data.design_result.term_name_slices,
        ):
            effects.append((predictor_name, str(trial_data.design_result.column_names[column_index])))
    deduped: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in effects:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    if not deduped:
        raise ValueError("No source-space test predictors resolved to concrete fixed-effect columns for inference.")
    return deduped


def _resolve_correction_name(stats_cfg: Mapping[str, Any]) -> str:
    method = str((stats_cfg.get("stats") or {}).get("method", "mne_cluster")).lower()
    if method in {"mne_cluster", "cluster"}:
        return "cluster"
    if method in {"mne_tfce", "tfce"}:
        return "tfce"
    if method == "maxstat":
        return "maxstat"
    raise ValueError(f"Unsupported source-space stats method `{method}`.")


def _resolve_threshold(
    stats_cfg: Mapping[str, Any],
    *,
    n_observations: int,
    correction: str,
    tail: int,
):
    section = stats_cfg.get("stats") or {}
    if correction == "cluster":
        if "cluster_threshold" in section:
            return float(section["cluster_threshold"])
        alpha = section.get("cluster_forming_alpha")
        if alpha is None:
            return None
        from scipy.stats import t

        alpha_value = float(alpha)
        dof = max(1, n_observations - 1)
        if tail == 0:
            return float(t.ppf(1.0 - (alpha_value / 2.0), df=dof))
        return float(t.ppf(1.0 - alpha_value, df=dof))
    if correction == "tfce":
        threshold = section.get("tfce_threshold")
        if threshold is None:
            return None
        if isinstance(threshold, Mapping):
            return {str(key): float(value) for key, value in threshold.items()}
        raise ValueError("TFCE threshold must be a mapping like {start: ..., step: ...}.")
    return None


def _resolve_source_adjacency(
    config: Mapping[str, Any],
    *,
    config_section: str = DEFAULT_LMEEEG_SOURCE_SECTION,
):
    settings = load_lmeeeg_source_settings(config, config_section=config_section)
    _configure_mne_runtime()
    import mne

    fsaverage_src = mne.setup_source_space(
        subject=settings.source_space.fsaverage_subject,
        spacing=settings.source_space.spacing,
        surface=settings.source_space.surface,
        subjects_dir=str(settings.source_space.subjects_dir),
        add_dist=settings.source_space.add_dist,
        verbose="ERROR",
    )
    return mne.spatial_src_adjacency(fsaverage_src)


def _stack_source_cluster_masks(
    clusters: list[Any],
    *,
    n_vertices: int,
    n_times: int,
) -> np.ndarray | None:
    if not clusters:
        return None
    normalized = [_normalize_source_cluster_mask(cluster_mask, n_vertices=n_vertices, n_times=n_times) for cluster_mask in clusters]
    if not normalized:
        return None
    return np.asarray(normalized, dtype=bool)


def _normalize_source_cluster_mask(cluster_mask: Any, *, n_vertices: int, n_times: int) -> np.ndarray:
    mask = np.asarray(cluster_mask, dtype=bool)
    if mask.shape == (n_vertices, n_times):
        return mask
    if mask.shape == (n_times, n_vertices):
        return mask.T
    raise ValueError(
        f"Unexpected source cluster mask shape {mask.shape}; expected {(n_vertices, n_times)} or {(n_times, n_vertices)}."
    )


def _configure_mne_runtime() -> None:
    import os

    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    os.environ.setdefault("MNE_DONTWRITE_HOME", "true")


def _unique_prepared_paths(
    config: Mapping[str, Any],
    rows: list[Mapping[str, Any]],
    *,
    config_section: str,
) -> list[SourcePreparedPaths]:
    deduped: dict[str, SourcePreparedPaths] = {}
    for row in rows:
        prepared_paths = build_source_prepared_paths(config, row, config_section=config_section)
        deduped.setdefault(str(prepared_paths.summary_output_path), prepared_paths)
    missing = [path.summary_output_path for path in deduped.values() if not path.summary_output_path.exists()]
    if missing:
        formatted = ", ".join(str(path) for path in missing[:3])
        suffix = "" if len(missing) <= 3 else f" and {len(missing) - 3} more"
        raise FileNotFoundError(
            "Prepared source-space subject summaries are missing. Run the subject-preparation stage first: "
            f"{formatted}{suffix}"
        )
    return list(deduped.values())


def _resolve_tstep(times: np.ndarray) -> float:
    if times.shape[0] < 2:
        return 1.0
    return float(times[1] - times[0])


def _summarise_time(
    effect_map: np.ndarray,
    *,
    times: np.ndarray,
    mode: str,
    window: tuple[float, float] | None,
) -> np.ndarray:
    if window is None:
        time_mask = np.ones(times.shape, dtype=bool)
    else:
        time_mask = (times >= window[0]) & (times <= window[1])
    if not np.any(time_mask):
        raise ValueError(f"No time points matched the requested summary window: {window}")
    selected = np.asarray(effect_map[:, time_mask], dtype=float)
    mode_key = str(mode).lower()
    if mode_key == "mean":
        summary = np.nanmean(selected, axis=1, keepdims=True)
    elif mode_key == "maxabs":
        max_indices = np.nanargmax(np.abs(selected), axis=1)
        summary = selected[np.arange(selected.shape[0]), max_indices][:, np.newaxis]
    else:
        raise ValueError(f"Unsupported source-space time summary `{mode}`.")
    return np.asarray(summary, dtype=float)
