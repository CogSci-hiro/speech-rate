"""Prepared-data cache helpers for temporal-generalisation decoding."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from rate.decoding.io import build_decoding_model_output_paths
from rate.epochs.io import build_epoch_output_paths
from rate.paths import ensure_parent_dir

PREPARED_DATA_CACHE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PreparedDataCacheKey:
    """Cache-key bundle for prepared decoding artefacts."""

    cache_dir: Path
    cache_key: str
    fingerprint: dict[str, Any]

    @property
    def manifest_path(self) -> Path:
        return self.cache_dir / "manifest.json"

    @property
    def metadata_path(self) -> Path:
        return self.cache_dir / "metadata.parquet"

    @property
    def eeg_residuals_path(self) -> Path:
        return self.cache_dir / "eeg_residuals.npy"

    @property
    def train_target_residuals_path(self) -> Path:
        return self.cache_dir / "train_target_residuals.npy"

    @property
    def test_target_residuals_path(self) -> Path:
        return self.cache_dir / "test_target_residuals.npy"

    @property
    def folds_path(self) -> Path:
        return self.cache_dir / "fold_indices.npz"

    @property
    def times_path(self) -> Path:
        return self.cache_dir / "times.npy"

    @property
    def channel_names_path(self) -> Path:
        return self.cache_dir / "channel_names.json"

    @property
    def summary_path(self) -> Path:
        return self.cache_dir / "summary.json"


def build_prepared_data_cache_key(
    config: Mapping[str, Any],
    decoding_config: Mapping[str, Any],
    *,
    model_cfg: Mapping[str, str],
    dataset_index: str | None,
    rows: list[Mapping[str, Any]],
) -> PreparedDataCacheKey:
    """Build a deterministic key for prepared decoding artefacts."""
    output_paths = build_decoding_model_output_paths(config, decoding_config, model_name=str(model_cfg["name"]))
    source_files = _collect_source_file_signatures(config, decoding_config, dataset_index=dataset_index, rows=rows)
    relevant_config = {
        "analysis_name": decoding_config.get("analysis_name", "decoding"),
        "model": {
            "name": str(model_cfg["name"]),
            "train_target": str(model_cfg["train_target"]),
            "test_target": str(model_cfg["test_target"]),
        },
        "input": dict(decoding_config.get("input") or {}),
        "sfreq": decoding_config.get("sfreq"),
        "channels": dict(decoding_config.get("channels") or {}),
        "time_window": dict(decoding_config.get("time_window") or {}),
        "cv": dict(decoding_config.get("cv") or {}),
        "eeg_residualisation": dict(decoding_config.get("eeg_residualisation") or {}),
        "target_residualisation": dict(decoding_config.get("target_residualisation") or {}),
        "random_state": decoding_config.get("random_state"),
    }
    fingerprint = {
        "schema_version": PREPARED_DATA_CACHE_SCHEMA_VERSION,
        "dataset_index": None if dataset_index is None else str(dataset_index),
        "source_files": source_files,
        "relevant_config": relevant_config,
    }
    cache_key = hashlib.sha256(json.dumps(fingerprint, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return PreparedDataCacheKey(
        cache_dir=output_paths.scores_output_path.parent / "prepared_data_cache" / cache_key,
        cache_key=cache_key,
        fingerprint=fingerprint,
    )


def load_prepared_data_cache(cache_key: PreparedDataCacheKey) -> dict[str, Any] | None:
    """Load cached prepared decoding inputs if the cache is valid."""
    required_paths = (
        cache_key.manifest_path,
        cache_key.metadata_path,
        cache_key.eeg_residuals_path,
        cache_key.train_target_residuals_path,
        cache_key.test_target_residuals_path,
        cache_key.folds_path,
        cache_key.times_path,
        cache_key.channel_names_path,
        cache_key.summary_path,
    )
    if not all(path.exists() for path in required_paths):
        return None

    manifest = json.loads(cache_key.manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != PREPARED_DATA_CACHE_SCHEMA_VERSION:
        return None
    if manifest.get("cache_key") != cache_key.cache_key:
        return None
    if manifest.get("fingerprint") != cache_key.fingerprint:
        return None

    folds_archive = np.load(cache_key.folds_path, allow_pickle=False)
    fold_count = int(manifest["fold_count"])
    fold_indices = [
        (
            np.asarray(folds_archive[f"train_{fold_idx}"]),
            np.asarray(folds_archive[f"test_{fold_idx}"]),
        )
        for fold_idx in range(fold_count)
    ]
    return {
        "model_name": str(manifest["model_name"]),
        "train_target": str(manifest["train_target"]),
        "test_target": str(manifest["test_target"]),
        "metadata_df": pd.read_parquet(cache_key.metadata_path),
        "eeg_residuals": np.load(cache_key.eeg_residuals_path, mmap_mode="r"),
        "train_target_residuals": np.load(cache_key.train_target_residuals_path, mmap_mode="r"),
        "test_target_residuals": np.load(cache_key.test_target_residuals_path, mmap_mode="r"),
        "fold_indices": fold_indices,
        "times": np.load(cache_key.times_path, mmap_mode="r"),
        "channel_names": json.loads(cache_key.channel_names_path.read_text(encoding="utf-8")),
        "summary": json.loads(cache_key.summary_path.read_text(encoding="utf-8")),
    }


def save_prepared_data_cache(
    cache_key: PreparedDataCacheKey,
    *,
    model_name: str,
    train_target: str,
    test_target: str,
    metadata_df: pd.DataFrame,
    eeg_residuals: np.ndarray,
    train_target_residuals: np.ndarray,
    test_target_residuals: np.ndarray,
    fold_indices: list[tuple[np.ndarray, np.ndarray]],
    times: np.ndarray,
    channel_names: list[str],
    summary: Mapping[str, Any],
) -> None:
    """Persist prepared decoding inputs to an on-disk cache."""
    for path in (
        cache_key.manifest_path,
        cache_key.metadata_path,
        cache_key.eeg_residuals_path,
        cache_key.train_target_residuals_path,
        cache_key.test_target_residuals_path,
        cache_key.folds_path,
        cache_key.times_path,
        cache_key.channel_names_path,
        cache_key.summary_path,
    ):
        ensure_parent_dir(path)

    metadata_df.to_parquet(cache_key.metadata_path, index=False)
    np.save(cache_key.eeg_residuals_path, np.asarray(eeg_residuals))
    np.save(cache_key.train_target_residuals_path, np.asarray(train_target_residuals))
    np.save(cache_key.test_target_residuals_path, np.asarray(test_target_residuals))
    np.save(cache_key.times_path, np.asarray(times))
    np.savez_compressed(
        cache_key.folds_path,
        **{
            array_name: array_value
            for fold_idx, (train_indices, test_indices) in enumerate(fold_indices)
            for array_name, array_value in (
                (f"train_{fold_idx}", np.asarray(train_indices)),
                (f"test_{fold_idx}", np.asarray(test_indices)),
            )
        },
    )
    cache_key.channel_names_path.write_text(json.dumps(list(channel_names), indent=2) + "\n", encoding="utf-8")
    cache_key.summary_path.write_text(json.dumps(dict(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": PREPARED_DATA_CACHE_SCHEMA_VERSION,
        "cache_key": cache_key.cache_key,
        "fingerprint": cache_key.fingerprint,
        "model_name": str(model_name),
        "train_target": str(train_target),
        "test_target": str(test_target),
        "fold_count": int(len(fold_indices)),
    }
    cache_key.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _collect_source_file_signatures(
    config: Mapping[str, Any],
    decoding_config: Mapping[str, Any],
    *,
    dataset_index: str | None,
    rows: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    epochs_section = str((decoding_config.get("input") or {}).get("epochs_section", "epochs"))
    source_paths: list[Path] = []
    if dataset_index is not None:
        source_paths.append(Path(dataset_index))
    for row in rows:
        epoch_paths = build_epoch_output_paths(config, row, epochs_section=epochs_section)
        source_paths.extend((epoch_paths.epochs_output_path, epoch_paths.metadata_output_path))
    unique_paths = sorted({path.resolve() for path in source_paths}, key=str)
    return [_path_signature(path) for path in unique_paths]


def _path_signature(path: Path) -> dict[str, Any]:
    stat_result = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat_result.st_size),
        "mtime_ns": int(stat_result.st_mtime_ns),
    }
