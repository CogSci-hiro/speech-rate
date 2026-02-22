import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from rate.config import load_and_validate_config
from rate.evoked_trf.fit import build_design_matrix as build_design_matrix_legacy
from rate.evoked_trf.spec import add_timelock_columns, get_models
from rate.trf.design import build_design_matrix, parse_evoked_models_config


def _demo_events_df():
    return add_timelock_columns(
        pd.DataFrame(
            {
                "t_onset": [0.5, 1.5, 2.5],
                "t_offset": [0.8, 1.8, 2.8],
                "self_rate": [2.0, 3.0, 4.0],
                "other_rate": [1.5, 2.5, 3.5],
                "abs_diff": [0.5, 0.5, 0.5],
            }
        )
    )


@pytest.mark.parametrize("model_name", ["baseline", "self_rate"])
def test_config_design_matrix_matches_legacy_builder_for_existing_models(model_name: str) -> None:
    cfg = load_and_validate_config(Path("config/config.yaml"))
    models_new = {m.model_name: m for m in parse_evoked_models_config(cfg["evoked_trf"].get("models"))}
    models_old = {m.model_name: m for m in get_models()}

    ev = _demo_events_df()
    emg = np.linspace(0.1, 1.0, 8, dtype=float)
    n_samples = emg.shape[0]
    sfreq = 2.0
    sample_t0_s = 0.0
    lags = np.array([-1.0, 0.0, 1.0], dtype=float)

    X_old, names_old = build_design_matrix_legacy(
        events=ev,
        model=models_old[model_name],
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
    with pytest.raises(ValueError, match="Unsupported feature"):
        parse_evoked_models_config([{"name": "bad", "regressors": ["not_a_feature"]}])

    with pytest.raises(ValueError, match="Impulse × impulse"):
        parse_evoked_models_config([{"name": "bad2", "regressors": ["self_ipu_onset * other_ipu_offset"]}])

    with pytest.raises(ValueError, match="Nested interactions"):
        parse_evoked_models_config([{"name": "bad3", "regressors": ["self_rate * other_rate * emg"]}])
