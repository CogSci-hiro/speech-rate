import os
from types import SimpleNamespace

import numpy as np
import pandas as pd

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

from rate.stats.config import ClusterMethod, StatsConfig
from rate.stats.reconstruction_erp_r2 import (
    compute_subject_channel_r2_from_reconstruction_metrics,
    run_reconstruction_erp_r2_vs_baseline_cluster,
)


def test_compute_subject_channel_r2_from_reconstruction_metrics_averages_across_folds() -> None:
    metrics = pd.DataFrame(
        {
            "fold_id": [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1],
            "channel": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            "time_s": [0.0, 0.1, 0.2] * 4,
            "orig_amp": [0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            "rec_amp": [0.0, 1.0, 2.0, 0.0, 2.0, 4.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        }
    )
    out = compute_subject_channel_r2_from_reconstruction_metrics(
        metrics,
        dyad_id="dyad-001",
        subject="sub-001",
        model="self_rate",
        channel_names=["Fz", "Pz"],
    )

    assert list(out["channel"]) == [0, 1]
    assert list(out["electrode"]) == ["Fz", "Pz"]
    # channel 0: mean([1.0, -1.5]) = -0.25
    assert np.isclose(float(out.loc[out["channel"] == 0, "r2"].iloc[0]), -0.25)
    assert np.isclose(float(out.loc[out["channel"] == 1, "r2"].iloc[0]), 1.0)
    assert np.all(out["n_folds"].to_numpy(dtype=int) == 2)


def test_run_reconstruction_erp_r2_vs_baseline_cluster_builds_aligned_delta_matrix(monkeypatch) -> None:
    rows = []
    for dyad_id, subject, base_vals, model_vals in [
        ("dyad-001", "sub-001", [0.10, 0.20], [0.30, 0.10]),
        ("dyad-002", "sub-002", [0.00, 0.10], [0.20, 0.40]),
    ]:
        for ch, (rb, rm) in enumerate(zip(base_vals, model_vals)):
            rows.append(
                {
                    "dyad_id": dyad_id,
                    "subject": subject,
                    "model": "baseline",
                    "channel": ch,
                    "electrode": ["Fz", "Pz"][ch],
                    "r2": rb,
                }
            )
            rows.append(
                {
                    "dyad_id": dyad_id,
                    "subject": subject,
                    "model": "self_rate",
                    "channel": ch,
                    "electrode": ["Fz", "Pz"][ch],
                    "r2": rm,
                }
            )
    subject_r2 = pd.DataFrame(rows)

    captured = {}

    def _fake_make_adj(*, ch_names, montage_name):
        captured["adj_ch_names"] = list(ch_names)
        captured["montage_name"] = montage_name
        return np.eye(len(ch_names))

    def _fake_run_cluster_1samp(*, data, adjacency, config):
        captured["data"] = np.asarray(data, dtype=float)
        captured["adjacency"] = adjacency
        captured["config"] = config
        return SimpleNamespace(
            t_obs=np.array([1.0, -2.0], dtype=float),
            clusters=[],
            cluster_p_values=np.array([], dtype=float),
            h0=np.array([], dtype=float),
        )

    monkeypatch.setattr("rate.stats.reconstruction_erp_r2._make_eeg_channel_adjacency", _fake_make_adj)
    monkeypatch.setattr("rate.stats.reconstruction_erp_r2._run_cluster_1samp", _fake_run_cluster_1samp)

    res = run_reconstruction_erp_r2_vs_baseline_cluster(
        subject_channel_r2=subject_r2,
        model="self_rate",
        baseline_model="baseline",
        config=StatsConfig(n_permutations=10, seed=1, method=ClusterMethod.THRESHOLD, threshold_p=0.05, tail=0),
    )

    expected = np.array(
        [
            [0.20, -0.10],  # sub-001
            [0.20, 0.30],   # sub-002
        ],
        dtype=float,
    )
    assert np.allclose(captured["data"], expected)
    assert np.allclose(res.delta_matrix, expected)
    assert np.allclose(res.mean_delta_r2, expected.mean(axis=0))
    assert res.ch_names == ["Fz", "Pz"]
    assert list(res.subjects_used["subject"]) == ["sub-001", "sub-002"]
