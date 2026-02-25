"""FOOOF/PSD visualization helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from .base import save_figure

DEFAULT_PSD_SPECTRUM_COLS: dict[str, str] = {
    "raw": "psd_raw",
    "aperiodic": "psd_aperiodic",
    "periodic": "psd_periodic",
}

DEFAULT_PSD_TARGET_ROWS: tuple[tuple[str, str], ...] = (
    ("exponent", "Exponent"),
    ("offset", "Offset"),
    ("theta_power", "Theta Power"),
    ("theta_cf", "Theta Center Frequency"),
)

DEFAULT_PSD_COLUMN_TITLES: dict[str, str] = {
    "raw": "Raw PSD",
    "aperiodic": "Aperiodic-Only PSD",
    "periodic": "Periodic-Only PSD",
}


def plot_psd_quantile_grid(
    *,
    subject_spectra: pd.DataFrame,
    q: int,
    out_dir: Path,
    family: str,
    name: str,
    freq_col: str = "freq_hz",
    subject_col: str = "subject_id",
    trial_col: str = "trial_key",
    spectrum_cols: Mapping[str, str] | None = None,
    target_rows: Sequence[tuple[str, str]] = DEFAULT_PSD_TARGET_ROWS,
    xlabel: str = "Frequency (Hz)",
    ylabel: str = "Power",
    title: str | None = None,
    log_x: bool = False,
    log_y: bool = False,
    xlim_hz: tuple[float, float] | None = None,
) -> None:
    """Plot a 4x3 PSD panel with within-subject trial quantile splits.

    Quantiles are assigned *within each subject* based on trial-level target values.
    Then PSD is averaged per ``subject x quantile x frequency`` and finally summarized
    across subjects (mean +/- SE) for each quantile.
    """
    if int(q) < 2:
        raise ValueError("q must be >= 2")
    if subject_spectra.empty:
        raise ValueError("subject_spectra must not be empty")

    spectrum_map = dict(DEFAULT_PSD_SPECTRUM_COLS if spectrum_cols is None else spectrum_cols)
    expected_spectrum_keys = ("raw", "aperiodic", "periodic")
    missing_keys = [k for k in expected_spectrum_keys if k not in spectrum_map]
    if missing_keys:
        raise ValueError(f"spectrum_cols missing keys: {missing_keys}")

    _validate_psd_table(
        df=subject_spectra,
        freq_col=freq_col,
        subject_col=subject_col,
        trial_col=trial_col,
        spectrum_cols=spectrum_map,
        target_rows=target_rows,
    )

    n_rows = len(target_rows)
    n_cols = len(expected_spectrum_keys)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12.0, 10.5), sharex=True)
    if n_rows == 1:
        axes = np.asarray([axes])

    cmap = plt.get_cmap("inferno")
    quantile_colors = [cmap(v) for v in np.linspace(0.15, 0.95, int(q))]
    panel_mean_values: dict[tuple[int, int], list[np.ndarray]] = {}

    for row_idx, (target_col, row_label) in enumerate(target_rows):
        quant_df = _assign_within_subject_trial_quantiles(
            df=subject_spectra,
            subject_col=subject_col,
            trial_col=trial_col,
            target_col=target_col,
            q=int(q),
        )
        panel_df = subject_spectra.merge(
            quant_df,
            on=[subject_col, trial_col],
            how="inner",
            validate="many_to_one",
        )

        for col_idx, spectrum_key in enumerate(expected_spectrum_keys):
            ax = axes[row_idx, col_idx]
            spectrum_col = spectrum_map[spectrum_key]
            summary = _summarize_quantile_spectra(
                df=panel_df,
                subject_col=subject_col,
                trial_col=trial_col,
                freq_col=freq_col,
                value_col=spectrum_col,
            )
            plotted_y: list[np.ndarray] = []

            for q_idx in range(int(q)):
                q_slice = summary.loc[summary["quantile_index"] == q_idx]
                if q_slice.empty:
                    continue
                x = q_slice[freq_col].to_numpy(dtype=float)
                mean = q_slice["mean"].to_numpy(dtype=float)
                se = q_slice["se"].to_numpy(dtype=float)
                color = quantile_colors[q_idx]
                ax.plot(x, mean, color=color, linewidth=1.8, label=f"Q{q_idx + 1}")
                panel_mean_values.setdefault((row_idx, col_idx), []).append(mean)
                plotted_y.append(mean)
                if np.isfinite(se).any():
                    ax.fill_between(x, mean - se, mean + se, color=color, alpha=0.12, linewidth=0)
                    plotted_y.append(mean - se)
                    plotted_y.append(mean + se)

            if row_idx == 0:
                ax.set_title(DEFAULT_PSD_COLUMN_TITLES.get(spectrum_key, spectrum_key))
            if col_idx == 0:
                ax.set_ylabel(f"{row_label}\n{ylabel}")
            if row_idx == n_rows - 1:
                ax.set_xlabel(xlabel)
            if log_x:
                ax.set_xscale("log")
            if log_y:
                _set_safe_loglike_yscale(ax=ax, plotted_y=plotted_y)
            if xlim_hz is not None:
                ax.set_xlim(float(xlim_hz[0]), float(xlim_hz[1]))

    _apply_robust_panel_ylims(axes=axes, panel_mean_values=panel_mean_values)
    axes[0, -1].legend(title=f"Quantiles (Q={int(q)})", frameon=False, loc="best")
    if title:
        fig.suptitle(title)
        fig.tight_layout(rect=(0, 0, 1, 0.98))
    else:
        fig.tight_layout()

    save_figure(fig, out_dir, family, name)


def _set_safe_loglike_yscale(*, ax: plt.Axes, plotted_y: list[np.ndarray]) -> None:
    vals = []
    for arr in plotted_y:
        a = np.asarray(arr, dtype=float).reshape(-1)
        a = a[np.isfinite(a)]
        if a.size:
            vals.append(a)
    if not vals:
        return
    y = np.concatenate(vals)
    if np.all(y > 0):
        ax.set_yscale("log")
        return

    # FOOOF spectra in this pipeline are typically already in log10-power units and can
    # be <= 0, so a log transform is not valid. Fall back to linear scaling.
    ax.set_yscale("linear")


def _apply_robust_panel_ylims(
    *,
    axes: np.ndarray,
    panel_mean_values: Mapping[tuple[int, int], list[np.ndarray]],
) -> None:
    """Tighten y-limits per subplot using robust quantiles of mean traces."""
    if axes.ndim != 2:
        return
    n_rows, n_cols = axes.shape
    for row_idx in range(n_rows):
        for col_idx in range(n_cols):
            arrays = panel_mean_values.get((row_idx, col_idx), [])
            if not arrays:
                continue
            vals = []
            for arr in arrays:
                a = np.asarray(arr, dtype=float).reshape(-1)
                a = a[np.isfinite(a)]
                if a.size:
                    vals.append(a)
            if not vals:
                continue
            y = np.concatenate(vals)
            if y.size < 2:
                continue
            lo = float(np.quantile(y, 0.01))
            hi = float(np.quantile(y, 0.99))
            if not np.isfinite(lo) or not np.isfinite(hi):
                continue
            if lo >= hi:
                lo = float(np.min(y))
                hi = float(np.max(y))
                if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
                    continue
            pad = 0.05 * (hi - lo)
            axes[row_idx, col_idx].set_ylim(lo - pad, hi + pad)


def _validate_psd_table(
    *,
    df: pd.DataFrame,
    freq_col: str,
    subject_col: str,
    trial_col: str,
    spectrum_cols: Mapping[str, str],
    target_rows: Sequence[tuple[str, str]],
) -> None:
    required = {freq_col, subject_col, trial_col, *spectrum_cols.values(), *(col for col, _ in target_rows)}
    missing = sorted(c for c in required if c not in df.columns)
    if missing:
        raise ValueError(f"subject_spectra missing required columns: {missing}")


def _assign_within_subject_trial_quantiles(
    *,
    df: pd.DataFrame,
    subject_col: str,
    trial_col: str,
    target_col: str,
    q: int,
) -> pd.DataFrame:
    trials = (
        df[[subject_col, trial_col, target_col]]
        .copy()
        .assign(**{target_col: lambda x: pd.to_numeric(x[target_col], errors="coerce")})
        .dropna(subset=[target_col])
        .groupby([subject_col, trial_col], as_index=False)
        .agg(target_value=(target_col, "mean"))
    )
    if trials.empty:
        raise ValueError(f"No finite trial values for target '{target_col}'")

    out_parts: list[pd.DataFrame] = []
    for subject_value, subj_df in trials.groupby(subject_col, sort=False):
        if len(subj_df) < q:
            continue
        ranked = subj_df["target_value"].rank(method="first")
        quant = pd.qcut(ranked, q=q, labels=False, duplicates="drop")
        if quant.isna().any():
            continue
        n_bins = int(pd.Series(quant).nunique())
        if n_bins != int(q):
            continue
        tmp = subj_df[[subject_col, trial_col]].copy()
        tmp["quantile_index"] = pd.Series(quant, index=subj_df.index).to_numpy(dtype=int)
        out_parts.append(tmp)

    if not out_parts:
        raise ValueError(f"Could not assign within-subject quantiles for '{target_col}' (q={q})")
    return pd.concat(out_parts, ignore_index=True)


def _summarize_quantile_spectra(
    *,
    df: pd.DataFrame,
    subject_col: str,
    trial_col: str,
    freq_col: str,
    value_col: str,
) -> pd.DataFrame:
    work = df[[subject_col, trial_col, "quantile_index", freq_col, value_col]].copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work[freq_col] = pd.to_numeric(work[freq_col], errors="coerce")
    work = work.dropna(subset=[subject_col, trial_col, "quantile_index", freq_col, value_col]).copy()
    if work.empty:
        return pd.DataFrame(columns=["quantile_index", freq_col, "mean", "se"])

    # Step 1: average trials within each subject and quantile.
    subj_quant = (
        work.groupby([subject_col, "quantile_index", freq_col], as_index=False)
        .agg(value=(value_col, "mean"))
        .sort_values([subject_col, "quantile_index", freq_col], kind="mergesort")
        .reset_index(drop=True)
    )

    # Step 2: summarize across subjects (SE across subject means).
    grouped = (
        subj_quant.groupby(["quantile_index", freq_col], as_index=False)
        .agg(
            mean=("value", "mean"),
            sd=("value", "std"),
            n_subjects=("value", "count"),
        )
        .sort_values(["quantile_index", freq_col], kind="mergesort")
        .reset_index(drop=True)
    )
    grouped["se"] = grouped["sd"] / np.sqrt(grouped["n_subjects"].clip(lower=1))
    grouped["se"] = grouped["se"].fillna(0.0)
    return grouped[["quantile_index", freq_col, "mean", "se"]]
