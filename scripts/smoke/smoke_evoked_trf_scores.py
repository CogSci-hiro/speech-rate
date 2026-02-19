from __future__ import annotations

from pathlib import Path

import pandas as pd

from rate.config import load_and_validate_config
from rate.evoked_trf.score import compute_scores_and_splits


def main() -> None:
    cfg = load_and_validate_config(config_path="/Users/hiro/PycharmProjects/speech-rate/config/config.yaml")
    out_dir = Path(cfg["paths"]["out_dir"])

    events_path = out_dir / "events" / "events.parquet"
    ds_path = out_dir / "dataset" / "index.parquet"
    if not events_path.exists():
        raise FileNotFoundError(str(events_path))
    if not ds_path.exists():
        raise FileNotFoundError(str(ds_path))

    events = pd.read_parquet(events_path)
    ds_index = pd.read_parquet(ds_path)

    # Pick a single smallest group to minimize runtime
    group_cols = ["dyad_id", "subject_id", "run"]
    grp = (
        events.groupby(group_cols, sort=True)
        .size()
        .reset_index(name="n_events")
        .sort_values("n_events")
        .head(1)
    )
    dyad_id = str(grp.iloc[0]["dyad_id"])
    subject_id = str(grp.iloc[0]["subject_id"])
    run = int(grp.iloc[0]["run"])

    print(f"Smoke target: dyad_id={dyad_id}, subject_id={subject_id}, run={run}")

    # Filter config runtime by filtering input tables in-place (cheap and safe).
    # This avoids needing a separate "subset" mode in the main pipeline.
    # We temporarily write filtered tables to a temp subdir under out_dir.
    tmp_dir = out_dir / "_smoke_evoked_trf_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    (tmp_dir / "events").mkdir(parents=True, exist_ok=True)
    (tmp_dir / "dataset").mkdir(parents=True, exist_ok=True)

    events_sub = events.loc[
        (events["dyad_id"].astype(str) == dyad_id)
        & (events["subject_id"].astype(str) == subject_id)
        & (events["run"].astype(int) == run)
    ].copy()
    ds_sub = ds_index.loc[
        (ds_index["dyad_id"].astype(str) == dyad_id)
        & (ds_index["subject_id"].astype(str) == subject_id)
        & (ds_index["run"].astype(int) == run)
    ].copy()

    (tmp_dir / "events" / "events.parquet").unlink(missing_ok=True)
    (tmp_dir / "dataset" / "index.parquet").unlink(missing_ok=True)
    events_sub.to_parquet(tmp_dir / "events" / "events.parquet", index=False)
    ds_sub.to_parquet(tmp_dir / "dataset" / "index.parquet", index=False)

    # Repoint out_dir for this run only
    scores, splits = compute_scores_and_splits(cfg=cfg, out_dir=tmp_dir)

    print("Scores head:")
    print(scores.head(10).to_string(index=False))
    print()
    print("Splits head:")
    print(splits.head(10).to_string(index=False))
    print()
    print(f"OK: produced {len(scores)} score rows and {len(splits)} split rows")


if __name__ == "__main__":
    main()
