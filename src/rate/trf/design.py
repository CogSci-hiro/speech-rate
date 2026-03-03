from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from rate.evoked_trf.fit import _events_to_continuous_regressor
from rate.evoked_trf.spec import RegressorSpec

AllowedFeature = Literal[
    "self_ipu_onset",
    "other_ipu_offset",
    "self_rate",
    "other_rate",
    "abs_diff",
    "emg",
    "run",
    "time_within_run",
]
AllowedLock = Literal["self_ipu_onset", "other_ipu_offset"]

_FEATURES: set[str] = {
    "self_ipu_onset",
    "other_ipu_offset",
    "self_rate",
    "other_rate",
    "abs_diff",
    "emg",
    "run",
    "time_within_run",
}
_LOCK_ALIASES = {
    "self_ipu_onset": "self_IPU_onset",
    "other_ipu_offset": "other_IPU_offset",
}
_DEFAULT_LOCK_BY_FEATURE = {
    "self_ipu_onset": "self_ipu_onset",
    "other_ipu_offset": "other_ipu_offset",
    "self_rate": "self_ipu_onset",
    "other_rate": "other_ipu_offset",
    "abs_diff": "self_ipu_onset",
}
_IMPULSE_FEATURES = {"self_ipu_onset", "other_ipu_offset"}
_EVENT_SCALAR_FEATURES = {"self_rate", "other_rate", "abs_diff"}
_CONTINUOUS_FEATURES = {"emg", "run", "time_within_run"}


UNIFIED_TRF_MODEL_NAME = "unified"
UNIFIED_TRF_REGRESSOR_EXPRESSIONS: tuple[str, ...] = (
    "self_ipu_onset",
    "other_ipu_offset",
    "emg",
    "self_rate",
    "other_rate",
    "abs_diff",
)

DEFAULT_EVOKED_TRF_MODELS: tuple[dict[str, Any], ...] = (
    {"name": UNIFIED_TRF_MODEL_NAME, "regressors": list(UNIFIED_TRF_REGRESSOR_EXPRESSIONS)},
)


@dataclass(frozen=True)
class DesignFactor:
    feature: str
    lock_event: str | None
    squared: bool = False


@dataclass(frozen=True)
class DesignRegressorSpec:
    name: str
    expression: str
    kind: Literal["term", "interaction"]
    left: DesignFactor
    right: DesignFactor | None = None


@dataclass(frozen=True)
class DesignModelSpec:
    model_name: str
    regressors: tuple[DesignRegressorSpec, ...]


def _normalize_token(value: str) -> str:
    return str(value).strip().lower()


def _parse_factor(expr: str) -> DesignFactor:
    text = str(expr).strip()
    if not text:
        raise ValueError("Empty regressor term is not allowed")
    if "*" in text:
        raise ValueError("Nested interactions are not supported")

    lock_event: str | None = None
    if "@" in text:
        parts = [p.strip() for p in text.split("@")]
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError(f"Invalid lock syntax: '{expr}'")
        text, lock_event = parts[0], _normalize_token(parts[1])
        if lock_event not in _LOCK_ALIASES:
            raise ValueError(f"Unsupported lock_event '{lock_event}' in '{expr}'")

    squared = False
    if text.endswith("^2"):
        squared = True
        text = text[:-2].strip()
    elif "^" in text:
        raise ValueError(f"Only '^2' is supported in '{expr}'")

    feature = _normalize_token(text)
    if feature not in _FEATURES:
        raise ValueError(f"Unsupported feature '{feature}' in '{expr}'")

    if feature in _CONTINUOUS_FEATURES and lock_event is not None:
        raise ValueError(f"Feature '{feature}' does not support '@ lock_event' syntax")
    if feature in _IMPULSE_FEATURES and squared:
        raise ValueError(f"Impulse feature '{feature}' cannot be squared")

    if lock_event is None and feature in _DEFAULT_LOCK_BY_FEATURE:
        lock_event = _DEFAULT_LOCK_BY_FEATURE[feature]
    if lock_event is not None and feature in _IMPULSE_FEATURES and lock_event != feature:
        raise ValueError(f"Impulse feature '{feature}' must be locked to itself")

    return DesignFactor(feature=feature, lock_event=lock_event, squared=squared)


def _canonical_term_name(*, left: DesignFactor, right: DesignFactor | None) -> str:
    if right is None:
        if left.feature == "self_ipu_onset":
            return "self_onset_impulse"
        if left.feature == "other_ipu_offset":
            return "other_offset_impulse"
        if left.squared and left.feature in {"self_rate", "other_rate", "abs_diff"}:
            return f"{left.feature}_sq"
        return left.feature
    left_name = _canonical_term_name(left=left, right=None)
    right_name = _canonical_term_name(left=right, right=None)
    return f"{left_name}__x__{right_name}"


def _parse_regressor_expression(expr: str) -> DesignRegressorSpec:
    text = str(expr).strip()
    if not text:
        raise ValueError("Regressor expression must be a non-empty string")
    if text.count("*") > 1:
        raise ValueError(f"Nested interactions are not supported: '{expr}'")

    if "*" in text:
        left_text, right_text = [part.strip() for part in text.split("*", 1)]
        left = _parse_factor(left_text)
        right = _parse_factor(right_text)
        if left.feature in _IMPULSE_FEATURES and right.feature in _IMPULSE_FEATURES:
            raise ValueError(f"Impulse × impulse is not supported: '{expr}'")
        return DesignRegressorSpec(
            name=_canonical_term_name(left=left, right=right),
            expression=text,
            kind="interaction",
            left=left,
            right=right,
        )

    left = _parse_factor(text)
    return DesignRegressorSpec(
        name=_canonical_term_name(left=left, right=None),
        expression=text,
        kind="term",
        left=left,
        right=None,
    )


def parse_evoked_models_config(models_cfg: Any) -> tuple[DesignModelSpec, ...]:
    if models_cfg is None:
        models_cfg = list(DEFAULT_EVOKED_TRF_MODELS)
    if not isinstance(models_cfg, list):
        raise ValueError("evoked_trf.models must be a list of model mappings")
    if len(models_cfg) != 1:
        raise ValueError("Confirmatory TRF requires exactly one model in evoked_trf.models")

    out: list[DesignModelSpec] = []
    seen_names: set[str] = set()
    for idx, item in enumerate(models_cfg):
        if not isinstance(item, dict):
            raise ValueError(f"evoked_trf.models[{idx}] must be a mapping")
        if set(item.keys()) - {"name", "regressors"}:
            extra = sorted(set(item.keys()) - {"name", "regressors"})
            raise ValueError(f"evoked_trf.models[{idx}] has unsupported keys: {extra}")
        name = str(item.get("name", "")).strip()
        if not name:
            raise ValueError(f"evoked_trf.models[{idx}].name must be non-empty")
        if name in seen_names:
            raise ValueError(f"Duplicate evoked_trf model name: '{name}'")
        regs = item.get("regressors")
        if not isinstance(regs, list) or not regs:
            raise ValueError(f"evoked_trf.models[{idx}].regressors must be a non-empty list")
        parsed_regs: list[DesignRegressorSpec] = []
        for j, reg_item in enumerate(regs):
            if not isinstance(reg_item, str):
                raise ValueError(f"evoked_trf.models[{idx}].regressors[{j}] must be a string")
            parsed_regs.append(_parse_regressor_expression(reg_item))
        canonical_expr = tuple(str(r.expression).strip() for r in parsed_regs)
        if canonical_expr != UNIFIED_TRF_REGRESSOR_EXPRESSIONS:
            raise ValueError(
                "Confirmatory TRF requires regressors exactly "
                f"{list(UNIFIED_TRF_REGRESSOR_EXPRESSIONS)} in that order"
            )
        seen_names.add(name)
        out.append(DesignModelSpec(model_name=name, regressors=tuple(parsed_regs)))
    return tuple(out)


def _factor_to_series(
    *,
    factor: DesignFactor,
    events_df: pd.DataFrame,
    run_id: int | str,
    sfreq: float,
    n_samples: int,
    sample_t0_s: float,
    continuous_regressors: dict[str, np.ndarray] | None,
) -> np.ndarray:
    feature = factor.feature
    if feature == "emg":
        if continuous_regressors is None or "emg" not in continuous_regressors:
            raise ValueError("Missing continuous regressor values for 'emg'")
        x = np.asarray(continuous_regressors["emg"], dtype=float)
    elif feature == "run":
        x = np.full(int(n_samples), float(run_id), dtype=float)
    elif feature == "time_within_run":
        # Deterministic linear time coordinate in seconds relative to sample 0 for this run.
        x = sample_t0_s + (np.arange(int(n_samples), dtype=float) / float(sfreq))
    else:
        if factor.lock_event is None:
            raise ValueError(f"Feature '{feature}' requires a lock_event")
        if feature == "self_ipu_onset":
            reg = RegressorSpec(
                name="self_onset_impulse",
                column=None,
                time_lock="self_IPU_onset",
                kind="impulse",
            )
        elif feature == "other_ipu_offset":
            reg = RegressorSpec(
                name="other_offset_impulse",
                column=None,
                time_lock="other_IPU_offset",
                kind="impulse",
            )
        elif feature == "self_rate":
            reg = RegressorSpec(
                name="self_rate",
                column="self_rate",
                time_lock=_LOCK_ALIASES[factor.lock_event],
                kind="self_rate",
            )
        elif feature == "other_rate":
            reg = RegressorSpec(
                name="other_rate",
                column="other_rate",
                time_lock=_LOCK_ALIASES[factor.lock_event],
                kind="other_rate",
            )
        elif feature == "abs_diff":
            reg = RegressorSpec(
                name="abs_diff",
                column="abs_diff",
                time_lock=_LOCK_ALIASES[factor.lock_event],
                kind="abs_diff",
            )
        else:
            raise ValueError(f"Unsupported feature '{feature}'")
        x = _events_to_continuous_regressor(
            events=events_df,
            reg=reg,
            sfreq_hz=float(sfreq),
            n_samples=int(n_samples),
            sample_t0_s=float(sample_t0_s),
        )

    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.shape[0] != int(n_samples):
        raise ValueError(f"Feature '{feature}' produced shape {x.shape}; expected ({int(n_samples)},)")
    if factor.squared:
        x = np.square(x, dtype=float)
    return np.asarray(x, dtype=float)


def build_design_matrix(
    model_spec: DesignModelSpec,
    events_df: pd.DataFrame,
    run_id: int | str,
    sfreq: float,
    lags: np.ndarray | None,
    *,
    n_samples: int,
    sample_t0_s: float = 0.0,
    continuous_regressors: dict[str, np.ndarray] | None = None,
    return_feature_names: bool = False,
) -> np.ndarray | tuple[np.ndarray, list[str]]:
    """
    Build the non-lagged design matrix from a constrained config grammar.

    `lags` is accepted for interface stability but not used here because lagging is
    handled by the TRF estimator.
    """
    _ = lags

    cols: list[np.ndarray] = []
    names: list[str] = []
    for reg in model_spec.regressors:
        left = _factor_to_series(
            factor=reg.left,
            events_df=events_df,
            run_id=run_id,
            sfreq=float(sfreq),
            n_samples=int(n_samples),
            sample_t0_s=float(sample_t0_s),
            continuous_regressors=continuous_regressors,
        )
        if reg.kind == "interaction":
            if reg.right is None:
                raise ValueError(f"Interaction regressor '{reg.name}' missing right factor")
            right = _factor_to_series(
                factor=reg.right,
                events_df=events_df,
                run_id=run_id,
                sfreq=float(sfreq),
                n_samples=int(n_samples),
                sample_t0_s=float(sample_t0_s),
                continuous_regressors=continuous_regressors,
            )
            x = np.asarray(left * right, dtype=float)
        else:
            x = np.asarray(left, dtype=float)
        cols.append(x)
        names.append(reg.name)

    if not cols:
        raise ValueError("Model has no regressors; cannot build design matrix")
    X = np.column_stack(cols).astype(float)
    if return_feature_names:
        return X, names
    return X
