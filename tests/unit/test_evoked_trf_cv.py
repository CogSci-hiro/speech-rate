import numpy as np

from rate.evoked_trf.cv import BlockedWithinRunCV, bins_to_sample_mask


def test_blocked_within_run_no_bin_leakage() -> None:
    cv = BlockedWithinRunCV(k_folds=5, time_bin_s=10.0, seed=0)
    splits = cv.make_splits(run_duration_s=120.0, dyad_id="dyad-001", run=1)

    for fold_id in range(5):
        sub = splits[splits["fold_id"] == fold_id]
        test_bins = set(sub[sub["split"] == "test"]["bin_id"].tolist())
        train_bins = set(sub[sub["split"] == "train"]["bin_id"].tolist())

        assert test_bins.isdisjoint(train_bins)
        assert len(test_bins) > 0
        assert len(test_bins | train_bins) == splits["bin_id"].nunique()


def test_blocked_within_run_deterministic() -> None:
    cv1 = BlockedWithinRunCV(k_folds=5, time_bin_s=10.0, seed=123)
    cv2 = BlockedWithinRunCV(k_folds=5, time_bin_s=10.0, seed=123)

    s1 = cv1.make_splits(run_duration_s=120.0, dyad_id="dyad-001", run=1)
    s2 = cv2.make_splits(run_duration_s=120.0, dyad_id="dyad-001", run=1)

    assert s1.equals(s2)


def test_bins_to_sample_mask_no_overlap() -> None:
    cv = BlockedWithinRunCV(k_folds=4, time_bin_s=5.0, seed=0)
    splits = cv.make_splits(run_duration_s=40.0, dyad_id="dyad-X", run=2)

    sfreq_hz = 10.0
    times_s = np.arange(int(40 * sfreq_hz), dtype=float) / sfreq_hz

    for fold_id in range(4):
        train_mask = bins_to_sample_mask(splits=splits, fold_id=fold_id, split="train", times_s=times_s)
        test_mask = bins_to_sample_mask(splits=splits, fold_id=fold_id, split="test", times_s=times_s)

        assert train_mask.shape == times_s.shape
        assert test_mask.shape == times_s.shape
        assert not np.any(train_mask & test_mask)
        assert np.any(train_mask)
        assert np.any(test_mask)
