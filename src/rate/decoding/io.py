"""IO helpers for decoding outputs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from rate.paths import ensure_parent_dir, resolve_out_dir
from rate.util import write_parquet_dataframe


@dataclass(frozen=True)
class DecodingModelPaths:
    """Resolved output paths for one decoding model."""

    scores_output_path: Path
    fold_scores_output_path: Path
    times_output_path: Path
    channel_names_output_path: Path
    selection_output_path: Path
    summary_output_path: Path
    alpha_scores_output_path: Path
    best_alphas_output_path: Path
    tg_matrix_output_path: Path
    tg_tfce_output_path: Path
    tg_p_fwer_output_path: Path
    tg_null_max_tfce_output_path: Path
    tg_stats_metadata_output_path: Path


def _sanitize_name(value: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value).strip()).strip("._-")
    return token or "model"


def build_decoding_model_output_paths(
    config: Mapping[str, Any],
    decoding_config: Mapping[str, Any],
    *,
    model_name: str,
) -> DecodingModelPaths:
    """Build canonical output paths for one decoding model."""
    out_dir = resolve_out_dir(config)
    model_dir = out_dir / str(decoding_config.get("output_dir", "decoding")) / _sanitize_name(model_name)
    return DecodingModelPaths(
        scores_output_path=model_dir / "scores.npy",
        fold_scores_output_path=model_dir / "fold_scores.npy",
        times_output_path=model_dir / "times.npy",
        channel_names_output_path=model_dir / "channel_names.json",
        selection_output_path=model_dir / "selection.parquet",
        summary_output_path=model_dir / "summary.json",
        alpha_scores_output_path=model_dir / "alpha_scores.npy",
        best_alphas_output_path=model_dir / "best_alphas.npy",
        tg_matrix_output_path=model_dir / "tg_matrix.npy",
        tg_tfce_output_path=model_dir / "tg_tfce.npy",
        tg_p_fwer_output_path=model_dir / "tg_p_fwer.npy",
        tg_null_max_tfce_output_path=model_dir / "tg_null_max_tfce.npy",
        tg_stats_metadata_output_path=model_dir / "tg_stats_metadata.json",
    )


def save_decoding_outputs(
    *,
    output_paths: DecodingModelPaths,
    mean_scores: np.ndarray,
    fold_scores: np.ndarray,
    alpha_scores: np.ndarray | None,
    best_alphas: np.ndarray,
    times: np.ndarray,
    channel_names: list[str],
    selection_df: pd.DataFrame,
    summary: Mapping[str, Any],
) -> None:
    """Write one decoding model result bundle to disk."""
    for output_path in (
        output_paths.scores_output_path,
        output_paths.fold_scores_output_path,
        output_paths.times_output_path,
        output_paths.channel_names_output_path,
        output_paths.selection_output_path,
        output_paths.summary_output_path,
        output_paths.alpha_scores_output_path,
        output_paths.best_alphas_output_path,
    ):
        ensure_parent_dir(output_path)

    np.save(output_paths.scores_output_path, np.asarray(mean_scores, dtype=float))
    np.save(output_paths.fold_scores_output_path, np.asarray(fold_scores, dtype=float))
    if alpha_scores is not None:
        np.save(output_paths.alpha_scores_output_path, np.asarray(alpha_scores, dtype=float))
    np.save(output_paths.best_alphas_output_path, np.asarray(best_alphas, dtype=float))
    np.save(output_paths.times_output_path, np.asarray(times, dtype=float))
    output_paths.channel_names_output_path.write_text(json.dumps(list(channel_names), indent=2) + "\n", encoding="utf-8")
    write_parquet_dataframe(selection_df, output_paths.selection_output_path)
    output_paths.summary_output_path.write_text(json.dumps(dict(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
