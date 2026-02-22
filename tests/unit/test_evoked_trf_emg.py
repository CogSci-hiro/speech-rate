import numpy as np
import pandas as pd

from rate.evoked_trf.fit import build_design_matrix
from rate.evoked_trf.score import _validate_scores_table
from rate.evoked_trf.spec import get_models


def test_get_models_baseline_contains_emg() -> None:
    models = get_models()
    model = next(m for m in models if m.model_name == "baseline")
    assert model.model_name == "baseline"
    assert [reg.name for reg in model.regressors] == ["self_onset_impulse", "other_offset_impulse", "emg"]


def test_build_design_matrix_includes_emg_continuous_regressor() -> None:
    model = get_models()[0]
    events = pd.DataFrame(
        {
            "t_onset": [1.0],
            "t_offset": [3.0],
            "self_IPU_onset": [1.0],
            "other_IPU_offset": [3.0],
        }
    )
    emg = np.array([0.0, 0.5, 1.0, 1.5, 2.0], dtype=float)

    X, names = build_design_matrix(
        events=events,
        model=model,
        sfreq_hz=1.0,
        n_samples=5,
        sample_t0_s=0.0,
        continuous_regressors={"emg": emg},
        return_feature_names=True,
    )

    assert names == ["self_onset_impulse", "other_offset_impulse", "emg"]
    assert X.shape == (5, 3)
    assert np.isclose(X[1, 0], 1.0)
    assert np.isclose(X[3, 1], 1.0)
    assert np.allclose(X[:, 2], emg)


def test_scores_validator_allows_core_schema_without_delta_columns() -> None:
    scores = pd.DataFrame(
        [
            {
                "subject": "sub-001",
                "dyad_id": "dyad-001",
                "run": 1,
                "model": "baseline",
                "fold_id": 0,
                "r": 0.1,
            }
        ]
    )
    _validate_scores_table(scores=scores)
