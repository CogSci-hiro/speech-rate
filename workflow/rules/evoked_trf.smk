import pandas as pd
from pathlib import Path

OUT_DIR = config["paths"]["out_dir"]
DATASET_INDEX = OUT_DIR + "/dataset/index_resampled.parquet"
EVENTS_TABLE = OUT_DIR + "/events/events.parquet"

if Path(DATASET_INDEX).exists() and Path(EVENTS_TABLE).exists():
    df_index = pd.read_parquet(DATASET_INDEX)
    df_events = pd.read_parquet(EVENTS_TABLE)

    index_pairs = {
        (str(row["dyad_id"]).strip(), str(row["subject_id"]).strip())
        for _, row in df_index[["dyad_id", "subject_id"]].drop_duplicates().iterrows()
    }
    event_runs = (
        df_events.assign(
            dyad_id=df_events["dyad_id"].astype(str).str.strip(),
            subject_id=df_events["subject_id"].astype(str).str.strip(),
            run=pd.to_numeric(df_events["run"], errors="coerce"),
        )
        .dropna(subset=["run"])
        .groupby(["dyad_id", "subject_id"], as_index=False)["run"]
        .nunique()
        .rename(columns={"run": "n_runs"})
    )

    SUBJECT_TARGETS = sorted(
        [
            (str(row["dyad_id"]), str(row["subject_id"]))
            for _, row in event_runs.loc[event_runs["n_runs"] >= 2].iterrows()
            if (str(row["dyad_id"]), str(row["subject_id"])) in index_pairs
        ]
    )
else:
    SUBJECT_TARGETS = []

EVOKED_TRF_SUBJECT_OUTPUTS = [
    f"{OUT_DIR}/evoked_trf/subjects/{dyad_id}/{subject}/scores.parquet"
    for dyad_id, subject in SUBJECT_TARGETS
]

rule evoked_trf_subject:
    input:
        events = OUT_DIR + "/events/events.parquet",
        dataset_index = OUT_DIR + "/dataset/index_resampled.parquet",
        validate_config = OUT_DIR + "/_meta/config_validated.json",
        write_manifest = OUT_DIR + "/_meta/run_manifest.json",
    output:
        scores = OUT_DIR + "/evoked_trf/subjects/{dyad_id}/{subject}/scores.parquet"
    log:
        OUT_DIR + "/logs/evoked_trf/{dyad_id}_{subject}.log"
    shell:
        r"""
        set -euo pipefail
        PYTHONPATH=src python -u -m rate.cli.main evoked-trf-subject \
            --dyad-id {wildcards.dyad_id} \
            --subject {wildcards.subject} \
            --config config/config.yaml \
            --schema config/schema.json \
            --out-dir {OUT_DIR} \
            2>&1 | tee {log}
        """

rule evoked_trf_aggregate:
    input:
        EVOKED_TRF_SUBJECT_OUTPUTS
    output:
        OUT_DIR + "/evoked_trf/scores/scores.parquet"
    shell:
        r"""
        set -euo pipefail
        PYTHONPATH=src python -u -m rate.cli.main evoked-trf-aggregate \
            --config config/config.yaml \
            --schema config/schema.json \
            --out-dir {OUT_DIR}
        """


rule all:
    input:
        OUT_DIR + "/evoked_trf/scores/scores.parquet"
