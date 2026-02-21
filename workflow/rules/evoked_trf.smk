import pandas as pd

OUT_DIR = config["paths"]["out_dir"]


def _subject_pairs():
    ckpt = checkpoints.subject_targets.get()
    df = pd.read_csv(ckpt.output[0], sep="\t")
    return [(str(r.dyad_id), str(r.subject_id)) for r in df.itertuples(index=False)]


def _evoked_subject_outputs(_wildcards):
    return [
        f"{OUT_DIR}/evoked_trf/subjects/{dyad_id}/{subject}/scores.parquet"
        for dyad_id, subject in _subject_pairs()
    ]


rule evoked_trf_subject:
    input:
        events=OUT_DIR + "/events/events.parquet",
        dataset_index=OUT_DIR + "/dataset/index_resampled.parquet",
        emg=OUT_DIR + "/features/emg/{dyad_id}/{subject}/emg_env.parquet",
        validate_config=OUT_DIR + "/_meta/config_validated.json",
        write_manifest=OUT_DIR + "/_meta/run_manifest.json",
    output:
        scores=OUT_DIR + "/evoked_trf/subjects/{dyad_id}/{subject}/scores.parquet"
    threads: 1
    log:
        OUT_DIR + "/logs/evoked_trf/{dyad_id}_{subject}.log"
    shell:
        r"""
        set -euo pipefail
        MNE_TQDM=off MNE_LOGGING_LEVEL=WARNING PYTHONPATH=src python -u -m rate.cli.main evoked-trf-subject \
            --dyad-id {wildcards.dyad_id} \
            --subject {wildcards.subject} \
            --config config/config.yaml \
            --schema config/schema.json \
            --out-dir {OUT_DIR} \
            2>&1 | tee {log}
        """


rule evoked_trf_aggregate:
    input:
        _evoked_subject_outputs
    output:
        scores=OUT_DIR + "/evoked_trf/scores/scores.parquet"
    log:
        OUT_DIR + "/logs/evoked_trf/aggregate.log"
    shell:
        r"""
        set -euo pipefail
        MNE_TQDM=off MNE_LOGGING_LEVEL=WARNING PYTHONPATH=src python -u -m rate.cli.main evoked-trf-aggregate \
            --config config/config.yaml \
            --schema config/schema.json \
            --out-dir {OUT_DIR} \
            2>&1 | tee {log}
        """
