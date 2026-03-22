import numpy as np
import pandas as pd
import pytest

from rate.evoked_trf.fit import build_design_matrix as build_design_matrix_legacy
from rate.evoked_trf.spec import ModelSpec, RegressorSpec, add_timelock_columns
from rate.trf.design import (
    UNIFIED_TRF_MODEL_NAME,
    UNIFIED_TRF_REGRESSOR_EXPRESSIONS,
    build_design_matrix,
    parse_evoked_models_config,
)


LEGACY_MODELS_CFG = [
    {"name": UNIFIED_TRF_MODEL_NAME, "regressors": list(UNIFIED_TRF_REGRESSOR_EXPRESSIONS)},
]


def _demo_events_df():
    return add_timelock_columns(
        pd.DataFrame(
            {
                "t_onset": [0.5, 1.5, 2.5],
                "t_offset": [0.8, 1.8, 2.8],
                "self_rate": [2.0, 3.0, 4.0],
                "other_rate": [1.5, 2.5, 3.5],
                "alignment": [-0.5, -0.5, -0.5],
                "convergence": [0.0, 1.0, 2.0],
            }
        )
    )


def test_config_design_matrix_matches_legacy_builder_for_existing_models() -> None:
    models_new = {m.model_name: m for m in parse_evoked_models_config(list(LEGACY_MODELS_CFG))}
    model_name = UNIFIED_TRF_MODEL_NAME
    legacy_unified = ModelSpec(
        model_name=model_name,
        regressors=(
            RegressorSpec(name="self_onset_impulse", column=None, time_lock="self_IPU_onset", kind="impulse"),
            RegressorSpec(name="other_offset_impulse", column=None, time_lock="other_IPU_offset", kind="impulse"),
            RegressorSpec(name="emg", column="emg_env", time_lock=None, kind="emg"),
            RegressorSpec(name="self_rate", column="self_rate", time_lock="self_IPU_onset", kind="self_rate"),
            RegressorSpec(name="other_rate", column="other_rate", time_lock="other_IPU_offset", kind="other_rate"),
            RegressorSpec(name="alignment", column="alignment", time_lock="self_IPU_onset", kind="alignment"),
            RegressorSpec(name="convergence", column="convergence", time_lock="self_IPU_onset", kind="convergence"),
        ),
    )

    ev = _demo_events_df()
    emg = np.linspace(0.1, 1.0, 8, dtype=float)
    n_samples = emg.shape[0]
    sfreq = 2.0
    sample_t0_s = 0.0
    lags = np.array([-1.0, 0.0, 1.0], dtype=float)

    X_old, names_old = build_design_matrix_legacy(
        events=ev,
        model=legacy_unified,
        sfreq_hz=sfreq,
        n_samples=n_samples,
        sample_t0_s=sample_t0_s,
        continuous_regressors={"emg": emg},
        return_feature_names=True,
    )
    X_new, names_new = build_design_matrix(
        model_spec=models_new[model_name],
        events_df=ev,
        run_id=1,
        sfreq=sfreq,
        lags=lags,
        n_samples=n_samples,
        sample_t0_s=sample_t0_s,
        continuous_regressors={"emg": emg},
        return_feature_names=True,
    )

    assert names_new == names_old
    assert X_new.dtype == X_old.dtype
    assert X_new.shape == X_old.shape
    assert np.array_equal(X_new, X_old)


def test_parse_evoked_models_config_rejects_unsupported_grammar() -> None:
    with pytest.raises(ValueError, match="at least one model"):
        parse_evoked_models_config([])

    with pytest.raises(ValueError, match="Unsupported feature"):
        parse_evoked_models_config([{"name": "bad", "regressors": ["not_a_feature"]}])


def test_parse_evoked_models_config_accepts_multiple_models_in_order() -> None:
    models = parse_evoked_models_config(
        [
            {"name": "baseline", "regressors": ["self_ipu_onset", "other_ipu_offset", "emg"]},
            {"name": "self_rate", "regressors": ["self_ipu_onset", "other_ipu_offset", "emg", "self_rate"]},
            {"name": "interaction", "regressors": ["self_ipu_onset", "emg", "alignment*emg"]},
        ]
    )

    assert [m.model_name for m in models] == ["baseline", "self_rate", "interaction"]
    assert [r.name for r in models[0].regressors] == ["self_onset_impulse", "other_offset_impulse", "emg"]
    assert [r.name for r in models[1].regressors] == ["self_onset_impulse", "other_offset_impulse", "emg", "self_rate"]
    assert [r.name for r in models[2].regressors] == ["self_onset_impulse", "emg", "alignment__x__emg"]
