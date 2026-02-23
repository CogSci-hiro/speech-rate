import pandas as pd

OUT_DIR = config["paths"]["out_dir"]


def _subject_pairs():
    ckpt = checkpoints.subject_targets.get()
    df = pd.read_csv(ckpt.output[0], sep="\t")
    return [(str(r.dyad_id), str(r.subject_id)) for r in df.itertuples(index=False)]


def _induced_bands():
    ind_cfg = config.get("induced_trf", {}) or {}
    bands = ind_cfg.get("bands")
    if bands:
        return [str(b) for b in bands]
    bands_hz = ind_cfg.get("bands_hz", {}) or {}
    return sorted(str(k) for k in bands_hz.keys())


def _induced_subject_outputs(_wildcards):
    return [
        f"{OUT_DIR}/induced_trf_{band}/subjects/{dyad_id}/{subject}/scores.parquet"
        for band in _induced_bands()
        for dyad_id, subject in _subject_pairs()
    ]


rule induced_trf_subject:
    input:
        events=OUT_DIR + "/events/events.parquet",
        dataset_index=OUT_DIR + "/dataset/index_resampled.parquet",
        emg=OUT_DIR + "/features/emg/{dyad_id}/{subject}/emg_env.parquet",
        validate_config=OUT_DIR + "/_meta/config_validated.json",
        write_manifest=OUT_DIR + "/_meta/run_manifest.json",
    output:
        scores=OUT_DIR + "/induced_trf_{band}/subjects/{dyad_id}/{subject}/scores.parquet"
    threads: 1
    log:
        OUT_DIR + "/logs/induced_trf/{band}_{dyad_id}_{subject}.log"
    shell:
        r"""
        set -euo pipefail
        MNE_TQDM=off MNE_LOGGING_LEVEL=WARNING PYTHONPATH=src python -u -m rate.cli.main induced-trf-subject \
            --dyad-id {wildcards.dyad_id} \
            --subject {wildcards.subject} \
            --band {wildcards.band} \
            --config config/compute.yaml \
            --schema config/schema.json \
            --out-dir {OUT_DIR} \
            2>&1 | tee {log}
        """


rule induced_trf_aggregate:
    input:
        OUT_DIR + "/events/events.parquet",
        OUT_DIR + "/dataset/index_resampled.parquet",
        OUT_DIR + "/_meta/features_emg_done.txt",
        OUT_DIR + "/_meta/config_validated.json",
        OUT_DIR + "/_meta/run_manifest.json",
        _induced_subject_outputs
    output:
        OUT_DIR + "/induced_trf/scores/scores.parquet"
    log:
        OUT_DIR + "/logs/induced_trf/aggregate.log"
    shell:
        r"""
        set -euo pipefail
        MNE_TQDM=off MNE_LOGGING_LEVEL=WARNING PYTHONPATH=src python -u -m rate.cli.main induced-trf-aggregate \
            --config config/compute.yaml \
            --schema config/schema.json \
            --out-dir {OUT_DIR} \
            2>&1 | tee {log}
        """
