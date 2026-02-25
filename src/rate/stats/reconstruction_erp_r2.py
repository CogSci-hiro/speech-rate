from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Sequence

import numpy as np
import pandas as pd

from rate.stats.config import StatsConfig

if TYPE_CHECKING:
    from rate.stats.cluster import ClusterPermutationResult


def _make_eeg_channel_adjacency(*, ch_names: list[str], montage_name: str):
    from rate.stats.adjacency import make_eeg_channel_adjacency

    return make_eeg_channel_adjacency(ch_names=ch_names, montage_name=montage_name)


def _run_cluster_1samp(*, data: np.ndarray, adjacency, config: StatsConfig):
    from rate.stats.cluster import run_cluster_1samp

    return run_cluster_1samp(data=data, adjacency=adjacency, config=config)


def _r2_1d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if yt.shape != yp.shape:
        raise ValueError(f"Shape mismatch for R^2: {yt.shape} vs {yp.shape}")
    if yt.ndim != 1:
        raise ValueError("R^2 inputs must be 1D")
    ss_res = float(np.sum((yt - yp) ** 2))
    yt_mean = float(np.mean(yt))
    ss_tot = float(np.sum((yt - yt_mean) ** 2))
    if not np.isfinite(ss_tot) or ss_tot <= 0.0:
        return float("nan")
    return float(1.0 - (ss_res / ss_tot))


def discover_reconstruction_models(*, out_dir: Path, analysis: str = "evoked_trf") -> list[str]:
    root = out_dir / analysis / "subjects"
    models = {
        str(path.parent.name)
        for path in root.rglob("reconstruction_metrics.parquet")
        if path.parent.name
    }
    return sorted(models)


def _infer_subject_ids_from_metrics_path(metrics_path: Path) -> tuple[str, str, str]:
    model = str(metrics_path.parent.name)
    subject = str(metrics_path.parents[2].name)
    dyad_id = str(metrics_path.parents[3].name)
    return dyad_id, subject, model


def _load_channel_names_for_model_dir(model_dir: Path) -> list[str] | None:
    import mne

    fif_paths = sorted(model_dir.glob("*-ave.fif"))
    if not fif_paths:
        return None
    try:
        evokeds = mne.read_evokeds(fif_paths[0], verbose="ERROR")
    except Exception:
        return None
    if not evokeds:
        return None
    return [str(ch) for ch in evokeds[0].ch_names]


def compute_subject_channel_r2_from_reconstruction_metrics(
    metrics: pd.DataFrame,
    *,
    dyad_id: str,
    subject: str,
    model: str,
    channel_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Compute subject-level per-channel ERP reconstruction R^2 from saved waveform metrics.

    The input is the `reconstruction_metrics.parquet` table produced by evoked TRF jobs,
    containing per-fold ERP waveforms (`orig_amp`, `rec_amp`) over `time_s`.
    """
    required = {"channel", "time_s", "orig_amp", "rec_amp"}
    missing = required.difference(metrics.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    dfx = metrics.copy()
    if "fold_id" not in dfx.columns:
        dfx["fold_id"] = 0
    dfx["channel"] = pd.to_numeric(dfx["channel"], errors="coerce")
    dfx["time_s"] = pd.to_numeric(dfx["time_s"], errors="coerce")
    dfx["orig_amp"] = pd.to_numeric(dfx["orig_amp"], errors="coerce")
    dfx["rec_amp"] = pd.to_numeric(dfx["rec_amp"], errors="coerce")
    dfx["fold_id"] = pd.to_numeric(dfx["fold_id"], errors="coerce")
    dfx = dfx.dropna(subset=["channel", "time_s", "orig_amp", "rec_amp", "fold_id"]).copy()
    if dfx.empty:
        return pd.DataFrame(
            columns=["dyad_id", "subject", "model", "channel", "electrode", "r2", "n_folds", "n_timepoints"]
        )
    dfx["channel"] = dfx["channel"].astype(int)
    dfx["fold_id"] = dfx["fold_id"].astype(int)

    fold_rows: list[dict[str, Any]] = []
    for (fold_id, channel), grp in dfx.groupby(["fold_id", "channel"], sort=True):
        gs = grp.sort_values("time_s", kind="mergesort")
        y_true = gs["orig_amp"].to_numpy(dtype=float)
        y_pred = gs["rec_amp"].to_numpy(dtype=float)
        fold_rows.append(
            {
                "fold_id": int(fold_id),
                "channel": int(channel),
                "r2_fold": _r2_1d(y_true, y_pred),
                "n_timepoints": int(y_true.size),
            }
        )
    folds = pd.DataFrame(fold_rows)
    if folds.empty:
        return pd.DataFrame(
            columns=["dyad_id", "subject", "model", "channel", "electrode", "r2", "n_folds", "n_timepoints"]
        )

    out = (
        folds.groupby("channel", as_index=False)
        .agg(
            r2=("r2_fold", "mean"),
            n_folds=("fold_id", "nunique"),
            n_timepoints=("n_timepoints", "max"),
        )
        .sort_values("channel", kind="mergesort")
        .reset_index(drop=True)
    )
    out["dyad_id"] = str(dyad_id)
    out["subject"] = str(subject)
    out["model"] = str(model)

    if channel_names is not None:
        idx_to_name = {int(i): str(ch) for i, ch in enumerate(channel_names)}
        out["electrode"] = out["channel"].map(idx_to_name)
    else:
        out["electrode"] = pd.NA

    return out[["dyad_id", "subject", "model", "channel", "electrode", "r2", "n_folds", "n_timepoints"]]


def collect_subject_channel_r2(
    *,
    out_dir: Path,
    analysis: str = "evoked_trf",
    models: Iterable[str] | None = None,
) -> pd.DataFrame:
    root = out_dir / analysis / "subjects"
    if not root.exists():
        raise FileNotFoundError(f"Subjects root not found: {root}")

    model_filter = {str(m) for m in models} if models is not None else None
    rows: list[pd.DataFrame] = []
    for metrics_path in sorted(root.rglob("models/*/reconstruction_metrics.parquet")):
        dyad_id, subject, model = _infer_subject_ids_from_metrics_path(metrics_path)
        if model_filter is not None and model not in model_filter:
            continue
        metrics = pd.read_parquet(metrics_path)
        channel_names = _load_channel_names_for_model_dir(metrics_path.parent)
        rows.append(
            compute_subject_channel_r2_from_reconstruction_metrics(
                metrics,
                dyad_id=dyad_id,
                subject=subject,
                model=model,
                channel_names=channel_names,
            )
        )

    if not rows:
        wanted = sorted(model_filter) if model_filter is not None else None
        raise FileNotFoundError(f"No reconstruction metrics found under {root} (models={wanted})")

    out = pd.concat(rows, ignore_index=True)
    out = out.sort_values(["dyad_id", "subject", "model", "channel"], kind="mergesort").reset_index(drop=True)
    return out


@dataclass(frozen=True)
class ReconstructionERPSpatialClusterResult:
    model: str
    baseline_model: str
    ch_names: list[str]
    subjects_used: pd.DataFrame
    subject_channel_r2: pd.DataFrame
    subject_delta_r2: pd.DataFrame
    delta_matrix: np.ndarray
    mean_delta_r2: np.ndarray
    group_cluster: "ClusterPermutationResult"


def run_reconstruction_erp_r2_vs_baseline_cluster(
    *,
    subject_channel_r2: pd.DataFrame,
    model: str,
    baseline_model: str,
    config: StatsConfig,
) -> ReconstructionERPSpatialClusterResult:
    required = {"dyad_id", "subject", "model", "channel", "r2"}
    missing = required.difference(subject_channel_r2.columns)
    if missing:
        raise ValueError(f"subject_channel_r2 missing columns {sorted(missing)}")

    dfx = subject_channel_r2.copy()
    dfx["channel"] = pd.to_numeric(dfx["channel"], errors="coerce")
    dfx["r2"] = pd.to_numeric(dfx["r2"], errors="coerce")
    dfx = dfx.dropna(subset=["channel", "r2"]).copy()
    dfx["channel"] = dfx["channel"].astype(int)

    sel = dfx.loc[dfx["model"].isin([str(model), str(baseline_model)])].copy()
    if sel.empty:
        raise ValueError(f"No rows found for model={model} baseline_model={baseline_model}")

    id_cols = ["dyad_id", "subject", "channel"]
    keep_cols = id_cols + ["model", "r2"] + (["electrode"] if "electrode" in sel.columns else [])
    sel = sel[keep_cols].copy()

    baseline = (
        sel.loc[sel["model"] == str(baseline_model), id_cols + ["r2"]]
        .rename(columns={"r2": "r2_baseline"})
        .drop_duplicates(id_cols, keep="last")
    )
    target = (
        sel.loc[sel["model"] == str(model), id_cols + ["r2"] + (["electrode"] if "electrode" in sel.columns else [])]
        .rename(columns={"r2": "r2_model"})
        .drop_duplicates(id_cols, keep="last")
    )
    merged = target.merge(baseline, on=id_cols, how="inner", validate="one_to_one")
    if merged.empty:
        raise ValueError(f"No overlapping subject/channel rows between model={model} and baseline={baseline_model}")
    merged["delta_r2"] = pd.to_numeric(merged["r2_model"], errors="coerce") - pd.to_numeric(
        merged["r2_baseline"], errors="coerce"
    )
    merged = merged.dropna(subset=["delta_r2"]).copy()
    if merged.empty:
        raise ValueError(f"All model-baseline deltas are NaN for model={model}")

    subject_keys = (
        merged[["dyad_id", "subject"]]
        .drop_duplicates()
        .sort_values(["dyad_id", "subject"], kind="mergesort")
        .reset_index(drop=True)
    )
    pivot = (
        merged.pivot_table(index=["dyad_id", "subject"], columns="channel", values="delta_r2", aggfunc="mean")
        .sort_index(axis=0)
        .sort_index(axis=1)
    )
    complete = pivot.dropna(axis=0, how="any")
    if complete.empty:
        raise ValueError(
            f"No subjects with complete channel coverage for model={model} vs baseline={baseline_model}"
        )
    dropped = int(pivot.shape[0] - complete.shape[0])
    if dropped > 0:
        # Restrict to subjects with complete channels for cluster test; keep full merged table as output.
        pass

    channel_ids = [int(c) for c in complete.columns.tolist()]
    channel_meta = (
        merged.loc[merged["channel"].isin(channel_ids), ["channel"] + (["electrode"] if "electrode" in merged.columns else [])]
        .drop_duplicates("channel", keep="first")
        .sort_values("channel", kind="mergesort")
    )
    if "electrode" in channel_meta.columns:
        ch_names = [
            str(name) if pd.notna(name) and str(name) else f"EEG{int(ch):03d}"
            for ch, name in channel_meta[["channel", "electrode"]].itertuples(index=False)
        ]
    else:
        ch_names = [f"EEG{int(ch):03d}" for ch in channel_meta["channel"].tolist()]

    delta_matrix = complete.to_numpy(dtype=float)
    if delta_matrix.ndim != 2 or delta_matrix.shape[1] != len(ch_names):
        raise ValueError("delta matrix/channel-name shape mismatch")

    adjacency = _make_eeg_channel_adjacency(ch_names=ch_names, montage_name=str(config.montage_name))
    group_cluster = _run_cluster_1samp(data=delta_matrix, adjacency=adjacency, config=config)
    mean_delta_r2 = np.nanmean(delta_matrix, axis=0)

    subjects_used = complete.reset_index()[["dyad_id", "subject"]].copy()
    subjects_used["model"] = str(model)
    subjects_used["baseline_model"] = str(baseline_model)

    subject_delta_r2 = merged.copy()
    return ReconstructionERPSpatialClusterResult(
        model=str(model),
        baseline_model=str(baseline_model),
        ch_names=ch_names,
        subjects_used=subjects_used,
        subject_channel_r2=dfx.copy(),
        subject_delta_r2=subject_delta_r2.sort_values(["dyad_id", "subject", "channel"], kind="mergesort").reset_index(drop=True),
        delta_matrix=np.asarray(delta_matrix, dtype=float),
        mean_delta_r2=np.asarray(mean_delta_r2, dtype=float),
        group_cluster=group_cluster,
    )


def _cluster_rows_and_labels(group_cluster: "ClusterPermutationResult", *, n_tests: int) -> tuple[pd.DataFrame, np.ndarray]:
    rows: list[dict[str, Any]] = []
    labels = np.zeros(int(n_tests), dtype=np.int32)
    for i, (clu, pval) in enumerate(zip(group_cluster.clusters, group_cluster.cluster_p_values)):
        idx = np.asarray(clu[0], dtype=int)
        if idx.size > 0:
            labels[idx] = int(i) + 1
        rows.append(
            {
                "cluster_id": int(i),
                "p_value": float(pval),
                "n_bins": int(idx.size),
            }
        )
    if rows:
        table = pd.DataFrame(rows).sort_values("p_value", ascending=True, kind="mergesort").reset_index(drop=True)
    else:
        table = pd.DataFrame(columns=["cluster_id", "p_value", "n_bins"])
    return table, labels


def write_reconstruction_erp_r2_outputs(
    *,
    out_dir: Path,
    analysis: str,
    all_subject_channel_r2: pd.DataFrame,
    comparison_results: Sequence[ReconstructionERPSpatialClusterResult],
    tables_format: str = "parquet",
) -> list[Path]:
    written: list[Path] = []
    root = out_dir / "stats" / analysis / "reconstruction_erp_r2"
    root.mkdir(parents=True, exist_ok=True)

    all_subject_path = root / "subject_channel_r2.parquet"
    all_subject_channel_r2.to_parquet(all_subject_path, index=False)
    written.append(all_subject_path)

    for res in comparison_results:
        cmp_dir = out_dir / "stats" / analysis / res.model / "reconstruction_erp_r2_vs_baseline"
        cmp_dir.mkdir(parents=True, exist_ok=True)

        res.subject_delta_r2.to_parquet(cmp_dir / "subject_delta_r2.parquet", index=False)
        res.subjects_used.to_parquet(cmp_dir / "subjects_used.parquet", index=False)
        np.save(cmp_dir / "delta_matrix.npy", np.asarray(res.delta_matrix, dtype=float))
        np.save(cmp_dir / "mean_delta_r2.npy", np.asarray(res.mean_delta_r2, dtype=float))
        np.save(cmp_dir / "t_obs.npy", np.asarray(res.group_cluster.t_obs, dtype=float))
        np.save(cmp_dir / "h0.npy", np.asarray(res.group_cluster.h0, dtype=float))
        np.save(cmp_dir / "channel_names.npy", np.asarray(res.ch_names, dtype=str))

        clusters_df, labels = _cluster_rows_and_labels(res.group_cluster, n_tests=len(res.ch_names))
        np.save(cmp_dir / "cluster_labels.npy", labels)
        if str(tables_format).lower() == "csv":
            clusters_df.to_csv(cmp_dir / "clusters.csv", index=False)
            written.append(cmp_dir / "clusters.csv")
        else:
            clusters_df.to_parquet(cmp_dir / "clusters.parquet", index=False)
            written.append(cmp_dir / "clusters.parquet")

        # Lightweight metadata for figure labels and provenance.
        pd.DataFrame(
            [
                {
                    "analysis": str(analysis),
                    "model": str(res.model),
                    "baseline_model": str(res.baseline_model),
                    "n_subjects": int(res.delta_matrix.shape[0]),
                    "n_channels": int(res.delta_matrix.shape[1]),
                }
            ]
        ).to_parquet(cmp_dir / "meta.parquet", index=False)
        written.append(cmp_dir)

    return written
