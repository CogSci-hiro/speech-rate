import pandas as pd

OUT_DIR = COMPUTE_OUT_DIR
SUBJECT_TARGETS_TSV = f"{OUT_DIR}/_meta/subject_targets.tsv"


checkpoint subject_targets:
    input:
        dataset_index=f"{OUT_DIR}/dataset/index.parquet",
        validate_config=f"{OUT_DIR}/_meta/config_validated.json",
        write_manifest=f"{OUT_DIR}/_meta/run_manifest.json",
    output:
        SUBJECT_TARGETS_TSV,
    shell:
        r"""
        set -euo pipefail
        PYTHONPATH=src python -u -m rate.cli.main dataset-subject-targets \
            --config config/config.yaml \
            --schema config/schema.json \
            --out-dir {OUT_DIR} \
            --dataset-index {input.dataset_index} \
            --output {output}
        """


def _subject_pairs():
    ckpt = checkpoints.subject_targets.get()
    df = pd.read_csv(ckpt.output[0], sep="\t")
    return [(str(r.dyad_id), str(r.subject_id)) for r in df.itertuples(index=False)]


def _emg_subject_outputs(_wildcards):
    return [
        f"{OUT_DIR}/features/emg/{dyad_id}/{subject}/emg_env.parquet"
        for dyad_id, subject in _subject_pairs()
    ]


rule emg_subject:
    input:
        dataset_index=f"{OUT_DIR}/dataset/index.parquet",
        validate_config=f"{OUT_DIR}/_meta/config_validated.json",
        write_manifest=f"{OUT_DIR}/_meta/run_manifest.json",
    output:
        parquet=f"{OUT_DIR}/features/emg/{{dyad_id}}/{{subject}}/emg_env.parquet"
    log:
        f"{OUT_DIR}/logs/emg/{{dyad_id}}_{{subject}}.log"
    shell:
        r"""
        set -euo pipefail
        PYTHONPATH=src python -u -m rate.cli.main compute-emg \
            --dyad-id {wildcards.dyad_id} \
            --subject {wildcards.subject} \
            --config config/config.yaml \
            --schema config/schema.json \
            --out-dir {OUT_DIR} \
            --dataset-index {input.dataset_index} \
            --output {output.parquet} \
            2>&1 | tee {log}
        """


rule emg_all:
    input:
        _emg_subject_outputs
    output:
        done=f"{OUT_DIR}/_meta/features_emg_done.txt",
    run:
        with open(output.done, "w", encoding="utf-8") as f:
            f.write("ok\n")
