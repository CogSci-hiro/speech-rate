import pandas as pd

OUT_DIR = COMPUTE_OUT_DIR


def _subject_pairs():
    ckpt = checkpoints.subject_targets.get()
    df = pd.read_csv(ckpt.output[0], sep="\t")
    return [(str(r.dyad_id), str(r.subject_id)) for r in df.itertuples(index=False)]


def _evoked_subject_outputs(_wildcards):
    return [
        f"{OUT_DIR}/evoked_trf/subjects/{dyad_id}/{subject}/scores.parquet"
        for dyad_id, subject in _subject_pairs()
    ]


def _active_evoked_models():
    block = compute_evoked_trf_config or {}
    specs = block.get("models", []) or []
    out = []
    for spec in specs:
        if not isinstance(spec, dict):
            continue
        name = str(spec.get("name", "")).strip()
        if name:
            out.append(name)
    return out


PRIMARY_EVOKED_MODELS = _active_evoked_models()


def _evoked_reconstruction_subject_sidecars(wildcards):
    return [
        f"{OUT_DIR}/evoked_trf/subjects/{wildcards.dyad_id}/{wildcards.subject}/models/{model}/reconstruction_metrics.parquet"
        for model in PRIMARY_EVOKED_MODELS
    ]


def _evoked_reconstruction_done_outputs(_wildcards):
    return [
        f"{OUT_DIR}/evoked_trf/subjects/{dyad_id}/{subject}/_reconstruction.done"
        for dyad_id, subject in _subject_pairs()
    ]


def _evoked_similarity_done_outputs(_wildcards):
    return [
        f"{OUT_DIR}/evoked_trf/subjects/{dyad_id}/{subject}/_similarity.done"
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


rule evoked_trf_reconstruction_subject:
    input:
        scores=OUT_DIR + "/evoked_trf/subjects/{dyad_id}/{subject}/scores.parquet",
        recon=_evoked_reconstruction_subject_sidecars,
    output:
        done=OUT_DIR + "/evoked_trf/subjects/{dyad_id}/{subject}/_reconstruction.done"
    shell:
        r"""
        set -euo pipefail
        echo ok > {output.done}
        """


rule evoked_trf_similarity_subject:
    input:
        scores=OUT_DIR + "/evoked_trf/subjects/{dyad_id}/{subject}/scores.parquet",
        sim=OUT_DIR + "/evoked_trf/subjects/{dyad_id}/{subject}/models/unified/self_rate-ave.fif",
    output:
        done=OUT_DIR + "/evoked_trf/subjects/{dyad_id}/{subject}/_similarity.done"
    shell:
        r"""
        set -euo pipefail
        echo ok > {output.done}
        """


rule evoked_trf_reconstruction_aggregate:
    input:
        _evoked_reconstruction_done_outputs
    output:
        touch(OUT_DIR + "/evoked_trf/aggregate/reconstruction/_done")
    log:
        OUT_DIR + "/logs/evoked_trf/reconstruction_aggregate.log"
    shell:
        r"""
        set -euo pipefail
        MNE_TQDM=off MNE_LOGGING_LEVEL=WARNING PYTHONPATH=src python -u -m rate.cli.main evoked-trf-reconstruction-aggregate \
            --config config/config.yaml \
            --schema config/schema.json \
            --out-dir {OUT_DIR} \
            2>&1 | tee {log}
        """


rule evoked_trf_similarity_aggregate:
    input:
        _evoked_similarity_done_outputs
    output:
        sim=OUT_DIR + "/evoked_trf/aggregate/similarity/self_vs_other.npy"
    log:
        OUT_DIR + "/logs/evoked_trf/similarity_aggregate.log"
    shell:
        r"""
        set -euo pipefail
        MNE_TQDM=off MNE_LOGGING_LEVEL=WARNING PYTHONPATH=src python -u -m rate.cli.main evoked-trf-similarity-aggregate \
            --config config/config.yaml \
            --schema config/schema.json \
            --out-dir {OUT_DIR} \
            2>&1 | tee {log}
        """
