import pandas as pd
from pathlib import Path

OUT_DIR = config["paths"]["out_dir"]
DATASET_INDEX_RESAMPLED = Path(OUT_DIR) / "dataset" / "index_resampled.parquet"
DATASET_INDEX = Path(OUT_DIR) / "dataset" / "index.parquet"

if DATASET_INDEX_RESAMPLED.exists():
    _subject_index_path = DATASET_INDEX_RESAMPLED
elif DATASET_INDEX.exists():
    _subject_index_path = DATASET_INDEX
else:
    _subject_index_path = None

if _subject_index_path is not None:
    df_index = pd.read_parquet(_subject_index_path)
    if "status" in df_index.columns:
        df_index = df_index.loc[df_index["status"].astype(str).str.lower() == "ok"].copy()
    SUBJECT_TARGETS = sorted(
        {
            (str(row["dyad_id"]).strip(), str(row["subject_id"]).strip())
            for _, row in df_index[["dyad_id", "subject_id"]].drop_duplicates().iterrows()
        }
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
