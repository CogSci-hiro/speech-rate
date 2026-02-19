from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


RegressorKind = Literal["impulse", "self_rate", "other_rate", "abs_diff"]
TimeLock = Literal["self_IPU_onset", "other_IPU_offset"]


@dataclass(frozen=True)
class RegressorSpec:
    """
    Specification for a single TRF regressor.

    Parameters
    ----------
    name
        Logical regressor name (used in model definition).
    column
        Column in events table used for amplitudes.
        For impulses, this is ignored (amplitude is 1.0).
    time_lock
        Which event timestamp to lock to.
    kind
        Regressor semantic type.

    Usage example
    -------------
        reg = RegressorSpec(
            name="self_rate",
            column="self_rate",
            time_lock="self_IPU_onset",
            kind="self_rate",
        )
    """

    name: str
    column: str | None
    time_lock: TimeLock
    kind: RegressorKind


@dataclass(frozen=True)
class ModelSpec:
    """
    TRF model specification.

    Parameters
    ----------
    model_name
        Name used in outputs.
    regressors
        Ordered list of regressors included in the model.

    Notes
    -----
    The baseline model defines the reference.
    All other models extend the baseline by exactly one regressor,
    ensuring that delta_r reflects incremental explanatory power.

    Usage example
    -------------
        model = ModelSpec(
            model_name="self_rate",
            regressors=(
                baseline_self_onset_impulse(),
                baseline_other_offset_impulse(),
                self_rate_regressor(),
            ),
        )
    """

    model_name: str
    regressors: tuple[RegressorSpec, ...]


# ---------------------------------------------------------------------
# Baseline regressors
# ---------------------------------------------------------------------


def baseline_self_onset_impulse() -> RegressorSpec:
    return RegressorSpec(
        name="self_onset_impulse",
        column=None,
        time_lock="self_IPU_onset",
        kind="impulse",
    )


def baseline_other_offset_impulse() -> RegressorSpec:
    return RegressorSpec(
        name="other_offset_impulse",
        column=None,
        time_lock="other_IPU_offset",
        kind="impulse",
    )


# ---------------------------------------------------------------------
# Rate-based regressors
# ---------------------------------------------------------------------


def self_rate_regressor() -> RegressorSpec:
    return RegressorSpec(
        name="self_rate",
        column="self_rate",
        time_lock="self_IPU_onset",
        kind="self_rate",
    )


def other_rate_regressor() -> RegressorSpec:
    return RegressorSpec(
        name="other_rate",
        column="other_rate",
        time_lock="other_IPU_offset",
        kind="other_rate",
    )


def abs_diff_regressor() -> RegressorSpec:
    return RegressorSpec(
        name="abs_diff",
        column="abs_diff",
        time_lock="self_IPU_onset",
        kind="abs_diff",
    )


# ---------------------------------------------------------------------
# Model collection
# ---------------------------------------------------------------------


def get_models() -> tuple[ModelSpec, ...]:
    """
    Return all models in fixed order.

    Models
    ------
    baseline
        Event impulses only.
    self_rate
        Baseline + self speech rate.
    other_rate
        Baseline + partner speech rate.
    abs_diff
        Baseline + absolute speech-rate difference.

    Usage example
    -------------
        models = get_models()
        names = [m.model_name for m in models]
    """

    baseline = ModelSpec(
        model_name="baseline",
        regressors=(
            baseline_self_onset_impulse(),
            baseline_other_offset_impulse(),
        ),
    )

    model_self = ModelSpec(
        model_name="self_rate",
        regressors=baseline.regressors + (self_rate_regressor(),),
    )

    model_other = ModelSpec(
        model_name="other_rate",
        regressors=baseline.regressors + (other_rate_regressor(),),
    )

    model_abs = ModelSpec(
        model_name="abs_diff",
        regressors=baseline.regressors + (abs_diff_regressor(),),
    )

    return baseline, model_self, model_other, model_abs


# ---------------------------------------------------------------------
# Event utilities
# ---------------------------------------------------------------------


def add_timelock_columns(events: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure canonical time-lock columns exist.

    Required input columns
    ----------------------
    t_onset : float (seconds from run start)
    t_offset : float (seconds from run start)

    Output columns
    --------------
    self_IPU_onset : float
    other_IPU_offset : float

    Assumption
    ----------
    Events table is already in subject-perspective format.

    Usage example
    -------------
        events = add_timelock_columns(events)
    """

    out = events.copy()

    if "t_onset" not in out.columns or "t_offset" not in out.columns:
        raise ValueError("events must contain 't_onset' and 't_offset' in seconds.")

    out["self_IPU_onset"] = out["t_onset"].astype(float)
    out["other_IPU_offset"] = out["t_offset"].astype(float)

    return out


def get_event_times_and_amps(
    events: pd.DataFrame,
    reg: RegressorSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract (times, amplitudes) for a given regressor.

    Returns
    -------
    times : ndarray, shape (n_events,)
    amps  : ndarray, shape (n_events,)

    Usage example
    -------------
        t, a = get_event_times_and_amps(events, self_rate_regressor())
    """

    if reg.time_lock not in events.columns:
        raise ValueError(f"Missing time-lock column: {reg.time_lock}")

    times = events[reg.time_lock].to_numpy(dtype=float)

    if reg.kind == "impulse":
        amps = np.ones_like(times, dtype=float)
    else:
        if reg.column is None or reg.column not in events.columns:
            raise ValueError(
                f"Missing amplitude column '{reg.column}' for regressor '{reg.name}'."
            )
        amps = events[reg.column].to_numpy(dtype=float)

    return times, amps
