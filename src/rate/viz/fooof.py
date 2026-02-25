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
    spectrum_cols: Mapping[str, str] | None = None,
    target_rows: Sequence[tuple[str, str]] = DEFAULT_PSD_TARGET_ROWS,
    xlabel: str = "Frequency (Hz)",
    ylabel: str = "Power",
    title: str | None = None,
    log_x: bool = False,
    log_y: bool = False,
) -> None:
    """Plot a 4x3 PSD panel with subject-level quantile splits for target variables.

    Expected input is a wide table with one row per subject-frequency pair and columns:
    - ``subject_col``, ``freq_col``
    - PSD columns (raw / aperiodic-only / periodic-only)
    - target variables repeated per subject (e.g. exponent, offset, theta_power, theta_cf)
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

    for row_idx, (target_col, row_label) in enumerate(target_rows):
        quant_df = _assign_subject_quantiles(
            df=subject_spectra,
            subject_col=subject_col,
            target_col=target_col,
            q=int(q),
        )
        panel_df = subject_spectra.merge(quant_df, on=subject_col, how="inner", validate="many_to_one")

        for col_idx, spectrum_key in enumerate(expected_spectrum_keys):
            ax = axes[row_idx, col_idx]
            spectrum_col = spectrum_map[spectrum_key]
            summary = _summarize_quantile_spectra(
                df=panel_df,
                freq_col=freq_col,
                value_col=spectrum_col,
            )

            for q_idx in range(int(q)):
                q_slice = summary.loc[summary["quantile_index"] == q_idx]
                if q_slice.empty:
                    continue
                x = q_slice[freq_col].to_numpy(dtype=float)
                mean = q_slice["mean"].to_numpy(dtype=float)
                se = q_slice["se"].to_numpy(dtype=float)
                color = quantile_colors[q_idx]
                ax.plot(x, mean, color=color, linewidth=1.8, label=f"Q{q_idx + 1}")
                if np.isfinite(se).any():
                    ax.fill_between(x, mean - se, mean + se, color=color, alpha=0.12, linewidth=0)

            if row_idx == 0:
                ax.set_title(DEFAULT_PSD_COLUMN_TITLES.get(spectrum_key, spectrum_key))
            if col_idx == 0:
                ax.set_ylabel(f"{row_label}\n{ylabel}")
            if row_idx == n_rows - 1:
                ax.set_xlabel(xlabel)
            if log_x:
                ax.set_xscale("log")
            if log_y:
                ax.set_yscale("log")

    axes[0, -1].legend(title=f"Quantiles (Q={int(q)})", frameon=False, loc="best")
    if title:
        fig.suptitle(title)
        fig.tight_layout(rect=(0, 0, 1, 0.98))
    else:
        fig.tight_layout()

    save_figure(fig, out_dir, family, name)


def _validate_psd_table(
    *,
    df: pd.DataFrame,
    freq_col: str,
    subject_col: str,
    spectrum_cols: Mapping[str, str],
    target_rows: Sequence[tuple[str, str]],
) -> None:
    required = {freq_col, subject_col, *spectrum_cols.values(), *(col for col, _ in target_rows)}
    missing = sorted(c for c in required if c not in df.columns)
    if missing:
        raise ValueError(f"subject_spectra missing required columns: {missing}")


def _assign_subject_quantiles(
    *,
    df: pd.DataFrame,
    subject_col: str,
    target_col: str,
    q: int,
) -> pd.DataFrame:
    subj = (
        df[[subject_col, target_col]]
        .copy()
        .assign(**{target_col: lambda x: pd.to_numeric(x[target_col], errors="coerce")})
        .dropna(subset=[target_col])
        .groupby(subject_col, as_index=False)
        .agg(target_value=(target_col, "mean"))
    )
    if len(subj) < q:
        raise ValueError(f"Not enough subjects with finite '{target_col}' to create q={q} quantiles")

    # Rank first to avoid qcut failures when target values contain ties.
    ranked = subj["target_value"].rank(method="first")
    quant = pd.qcut(ranked, q=q, labels=False, duplicates="drop")
    if quant.isna().any():
        raise ValueError(f"Failed to assign quantiles for target '{target_col}'")
    subj["quantile_index"] = quant.astype(int)

    n_bins = int(subj["quantile_index"].nunique())
    if n_bins != int(q):
        raise ValueError(f"Requested q={q} quantiles for '{target_col}', but only got {n_bins}")
    return subj[[subject_col, "quantile_index"]]


def _summarize_quantile_spectra(
    *,
    df: pd.DataFrame,
    freq_col: str,
    value_col: str,
) -> pd.DataFrame:
    work = df[["quantile_index", freq_col, value_col]].copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work[freq_col] = pd.to_numeric(work[freq_col], errors="coerce")
    work = work.dropna(subset=["quantile_index", freq_col, value_col]).copy()
    if work.empty:
        return pd.DataFrame(columns=["quantile_index", freq_col, "mean", "se"])

    grouped = (
        work.groupby(["quantile_index", freq_col], as_index=False)
        .agg(
            mean=(value_col, "mean"),
            sd=(value_col, "std"),
            n=(value_col, "count"),
        )
        .sort_values(["quantile_index", freq_col], kind="mergesort")
        .reset_index(drop=True)
    )
    grouped["se"] = grouped["sd"] / np.sqrt(grouped["n"].clip(lower=1))
    grouped["se"] = grouped["se"].fillna(0.0)
    return grouped[["quantile_index", freq_col, "mean", "se"]]
