from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResamplePlanRow:
    dyad_id: str
    subject_id: str
    run: int
    eeg_path_in: Path
    eeg_path_out: Path


def _is_macos_dot_underscore(path: Path) -> bool:
    return path.name.startswith("._")


def _safe_str(x: object) -> str:
    return str(x).strip()


def _build_out_path(
    *,
    eeg_cache_dir: Path,
    dyad_id: str,
    subject_id: str,
    run: int,
    downsample_hz: float,
) -> Path:
    hz_tag = f"{int(round(downsample_hz))}Hz"
    return eeg_cache_dir / dyad_id / subject_id / f"run-{int(run)}_raw_{hz_tag}.fif"


def _validate_index_columns(ds_index: pd.DataFrame) -> None:
    required = {"dyad_id", "subject_id", "run", "eeg_path"}
    missing = required - set(ds_index.columns)
    if missing:
        raise ValueError(f"dataset index missing required columns: {sorted(missing)}")


def _make_plan(
    *,
    ds_index: pd.DataFrame,
    eeg_cache_dir: Path,
    downsample_hz: float,
) -> list[ResamplePlanRow]:
    _validate_index_columns(ds_index)

    rows: list[ResamplePlanRow] = []
    for r in ds_index.itertuples(index=False):
        dyad_id = _safe_str(getattr(r, "dyad_id"))
        subject_id = _safe_str(getattr(r, "subject_id"))
        run = int(getattr(r, "run"))
        eeg_str = _safe_str(getattr(r, "eeg_path"))
        eeg_in = Path(eeg_str).expanduser()

        # Ignore macOS sidecar files if they accidentally entered the index
        if _is_macos_dot_underscore(eeg_in):
            continue

        eeg_out = _build_out_path(
            eeg_cache_dir=eeg_cache_dir,
            dyad_id=dyad_id,
            subject_id=subject_id,
            run=run,
            downsample_hz=downsample_hz,
        )
        rows.append(
            ResamplePlanRow(
                dyad_id=dyad_id,
                subject_id=subject_id,
                run=run,
                eeg_path_in=eeg_in,
                eeg_path_out=eeg_out,
            )
        )
    return rows


def _resample_one(plan: ResamplePlanRow, *, downsample_hz: float, overwrite: bool) -> None:
    out_path = plan.eeg_path_out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not overwrite:
        return

    in_path = plan.eeg_path_in

    if not in_path.exists():
        raise FileNotFoundError(f"EEG path does not exist: {in_path}")

    if in_path.is_dir():
        raise ValueError(f"EEG path points to a directory (expected a file): {in_path}")

    suffix = in_path.suffix.lower().strip()

    # Prefer explicit readers for clarity and better error messages
    if suffix == ".fif":
        raw = mne.io.read_raw_fif(in_path, preload=True, verbose="ERROR")
    elif suffix == ".edf":
        raw = mne.io.read_raw_edf(in_path, preload=True, verbose="ERROR")
    elif suffix == ".bdf":
        raw = mne.io.read_raw_bdf(in_path, preload=True, verbose="ERROR")
    elif suffix == ".vhdr":
        raw = mne.io.read_raw_brainvision(in_path, preload=True, verbose="ERROR")
    else:
        raise ValueError(f"Unsupported EEG file extension '{suffix}' for path: {in_path}")

    # Avoid redundant resampling if already at desired sfreq (within tolerance)
    if not np.isclose(raw.info["sfreq"], downsample_hz, rtol=0.0, atol=1e-6):
        raw.resample(downsample_hz, npad="auto", verbose="ERROR")

    raw.save(out_path, overwrite=True, verbose="ERROR")


def _assert_no_missing_eeg_path(ds_index: pd.DataFrame) -> None:
    bad = ds_index.loc[ds_index["eeg_path"].isna(), ["dyad_id", "subject_id", "run", "eeg_path"]].copy()
    if bad.empty:
        return
    preview = bad.head(20).to_string(index=False)
    raise ValueError(
        "dataset index contains missing eeg_path values (NaN). "
        "Fix dataset/index.parquet generation or drop these runs.\n"
        f"First rows:\n{preview}"
    )


def resample_index_table(
    *,
    ds_index: pd.DataFrame,
    eeg_cache_dir: Path,
    downsample_hz: float,
    overwrite: bool,
) -> pd.DataFrame:

    if "status" in ds_index.columns:
        ok_mask = ds_index["status"].astype(str).str.lower() == "ok"
    else:
        # Backward compatibility: assume all rows are OK
        ok_mask = pd.Series(True, index=ds_index.index)

    ok_rows = ds_index.loc[ok_mask].copy()
    skip_rows = ds_index.loc[~ok_mask].copy()

    LOGGER.info("Rows total: %d", len(ds_index))
    LOGGER.info("Rows to resample (status=='ok'): %d", len(ok_rows))
    LOGGER.info("Rows skipped (excluded/missing): %d", len(skip_rows))

    _assert_no_missing_eeg_path(ok_rows)

    plan = _make_plan(
        ds_index=ok_rows,
        eeg_cache_dir=eeg_cache_dir,
        downsample_hz=downsample_hz,
    )

    LOGGER.info("Resample plan: %d files", len(plan))

    missing_keys = set()

    for i, row in enumerate(plan, start=1):
        try:
            _resample_one(row, downsample_hz=downsample_hz, overwrite=overwrite)
        except FileNotFoundError:
            LOGGER.error("Missing EEG file: %s", row.eeg_path_in)
            missing_keys.add((row.dyad_id, row.subject_id, row.run))

    # Build a new table (keep all original columns, overwrite eeg_path)
    out = ds_index.copy()

    out_paths = {
        (p.dyad_id, p.subject_id, p.run): str(p.eeg_path_out)
        for p in plan
        if (p.dyad_id, p.subject_id, p.run) not in missing_keys
    }

    new_status = []
    new_paths = []

    for r in out.itertuples():
        key = (str(r.dyad_id), str(r.subject_id), int(r.run))

        if key in out_paths:
            new_paths.append(out_paths[key])
            new_status.append("ok")
        elif key in missing_keys:
            new_paths.append(r.eeg_path)
            new_status.append("missing")
        else:
            new_paths.append(r.eeg_path)
            new_status.append(getattr(r, "status", "excluded"))

    out["eeg_path"] = new_paths
    out["status"] = new_status
    out["eeg_is_resampled"] = out["status"] == "ok"
    out["eeg_downsample_hz"] = float(downsample_hz)

    return out
