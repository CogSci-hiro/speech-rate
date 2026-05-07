_PREPROCESSING_OUTPUT_CFG = dict((RESOLVED_CONFIG.get("preprocessing") or {}).get("output") or {})
_PREPROCESSING_EEG_FILENAME = str(_PREPROCESSING_OUTPUT_CFG.get("eeg_filename", "eeg_preprocessed_raw.fif"))
_PREPROCESSING_EMG_FILENAME = str(_PREPROCESSING_OUTPUT_CFG.get("emg_filename", "emg.npz"))
_PREPROCESSING_EVENTS_FILENAME = str(_PREPROCESSING_OUTPUT_CFG.get("events_filename", "events.tsv"))
_PREPROCESSING_SUMMARY_FILENAME = str(_PREPROCESSING_OUTPUT_CFG.get("summary_filename", "preprocessing_summary.json"))

PREPROCESSING_STAGE = f"{OUT_DIR}/_stage/preprocessing_all.done.json"


rule preprocess_subject:
    input:
        raw_eeg=lambda wildcards: str(preprocessing_paths(wildcards).raw_eeg_path),
        channels_tsv=lambda wildcards: str(preprocessing_paths(wildcards).channels_tsv_path),
        dataset_index=ancient(f"{OUT_DIR}/dataset/index.parquet"),
        config_path="config/config.yaml",
        preprocessing_config_path="config/preprocessing.yaml",
    output:
        eeg_raw=f"{OUT_DIR}/preprocessing/{{subject_id}}/task-{{task}}/run-{{run}}/{_PREPROCESSING_EEG_FILENAME}",
        emg=f"{OUT_DIR}/preprocessing/{{subject_id}}/task-{{task}}/run-{{run}}/{_PREPROCESSING_EMG_FILENAME}",
        events=f"{OUT_DIR}/preprocessing/{{subject_id}}/task-{{task}}/run-{{run}}/{_PREPROCESSING_EVENTS_FILENAME}",
        summary=f"{OUT_DIR}/preprocessing/{{subject_id}}/task-{{task}}/run-{{run}}/{_PREPROCESSING_SUMMARY_FILENAME}",
    params:
        dyad_id=lambda wildcards: str(preprocessing_row(wildcards)["dyad_id"]),
        subject_id=lambda wildcards: str(wildcards.subject_id),
        task=lambda wildcards: str(wildcards.task),
        run=lambda wildcards: str(wildcards.run),
        ica_arg=lambda wildcards: (
            f"--ica-path {preprocessing_paths(wildcards).ica_path}"
            if preprocessing_paths(wildcards).ica_path is not None
            else ""
        ),
    shell:
        "PYTHONPATH=src python -m rate.cli.main preprocess-subject "
        "--config {input.config_path} --schema config/schema.json --dataset-index {input.dataset_index} "
        "--dyad-id {params.dyad_id} --subject-id {params.subject_id} --task {params.task} --run {params.run} "
        "--raw-path {input.raw_eeg} --channels-tsv {input.channels_tsv} "
        "{params.ica_arg} "
        "--eeg-output {output.eeg_raw} --emg-output {output.emg} "
        "--events-output {output.events} --summary-output {output.summary}"


rule preprocess_stage:
    input:
        preprocessing_stage_inputs,
    output:
        PREPROCESSING_STAGE,
    shell:
        "python -c \"from pathlib import Path; "
        "path = Path(r'{output}'); "
        "path.parent.mkdir(parents=True, exist_ok=True); "
        "path.write_text('ok\\n', encoding='utf-8')\""
