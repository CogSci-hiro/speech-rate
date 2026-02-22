from dataclasses import dataclass

import numpy as np
import pandas as pd


def make_blocked_within_run_folds(
    *,
    run_duration_s: float,
    dyad_id: str,
    run: int,
    k_folds: int,
    time_bin_s: float,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Functional entrypoint for blocked within-run folds.
    Behavior matches the historical evoked_trf implementation.
    """
    return BlockedWithinRunCV(k_folds=int(k_folds), time_bin_s=float(time_bin_s), seed=int(seed)).make_splits(
        run_duration_s=float(run_duration_s),
        dyad_id=str(dyad_id),
        run=int(run),
    )


@dataclass(frozen=True)
class BlockedWithinRunCV:
    k_folds: int
    time_bin_s: float
    seed: int = 0

    def make_splits(self, *, run_duration_s: float, dyad_id: str, run: int) -> pd.DataFrame:
        if self.k_folds < 2:
            raise ValueError("k_folds must be >= 2")
        if self.time_bin_s <= 0:
            raise ValueError("time_bin_s must be > 0")
        if run_duration_s <= 0:
            raise ValueError("run_duration_s must be > 0")

        n_bins = int(np.ceil(run_duration_s / self.time_bin_s))
        if n_bins < self.k_folds:
            raise ValueError(
                f"Not enough bins ({n_bins}) for k_folds={self.k_folds} "
                f"with time_bin_s={self.time_bin_s} and run_duration_s={run_duration_s}."
            )

        bin_ids = np.arange(n_bins, dtype=int)
        bin_t0 = bin_ids * self.time_bin_s
        bin_t1 = np.minimum((bin_ids + 1) * self.time_bin_s, run_duration_s)

        rng = np.random.default_rng(self._seed_for(dyad_id=dyad_id, run=run))
        perm = rng.permutation(bin_ids)
        fold_bins = np.array_split(perm, self.k_folds)

        rows: list[dict[str, object]] = []
        for fold_id, test_bins in enumerate(fold_bins):
            test_set = set(int(x) for x in test_bins.tolist())
            for b in bin_ids:
                rows.append(
                    {
                        "dyad_id": str(dyad_id),
                        "run": int(run),
                        "fold_id": int(fold_id),
                        "bin_id": int(b),
                        "bin_t0": float(bin_t0[b]),
                        "bin_t1": float(bin_t1[b]),
                        "split": "test" if int(b) in test_set else "train",
                    }
                )

        return pd.DataFrame(rows)

    def _seed_for(self, *, dyad_id: str, run: int) -> int:
        key = f"{self.seed}::{dyad_id}::{run}".encode("utf-8")
        h = np.uint64(1469598103934665603)
        for byte in key:
            h ^= np.uint64(byte)
            h *= np.uint64(1099511628211)
        return int(h % np.uint64(2**32 - 1))


def bins_to_sample_mask(*, splits: pd.DataFrame, fold_id: int, split: str, times_s: np.ndarray) -> np.ndarray:
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")

    sub = splits.loc[(splits["fold_id"] == int(fold_id)) & (splits["split"] == split)].copy()
    if sub.empty:
        raise ValueError(f"No rows for fold_id={fold_id}, split={split}")

    mask = np.zeros(times_s.shape[0], dtype=bool)
    for _, row in sub.iterrows():
        t0 = float(row["bin_t0"])
        t1 = float(row["bin_t1"])
        mask |= (times_s >= t0) & (times_s < t1)
    return mask
