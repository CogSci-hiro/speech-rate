from pathlib import Path
import sys

sys.path.insert(0, str(Path.cwd() / "src"))

from rate.config import load_and_validate_config
from rate.dataset import build_dataset_index
from rate.preprocessing.io import build_preprocessing_output_paths


_PREPROCESSING_CONFIG = load_and_validate_config(Path("config/config.yaml"), schema_path=Path("config/schema.json"))
_PREPROCESSING_INDEX = build_dataset_index(_PREPROCESSING_CONFIG)
_PREPROCESSING_INDEX = _PREPROCESSING_INDEX.loc[_PREPROCESSING_INDEX["status"].astype(str).str.lower() == "ok"].copy()
_PREPROCESSING_INDEX = _PREPROCESSING_INDEX.dropna(subset=["dyad_id", "subject_id", "task", "run", "eeg_path"]).copy()


def _preprocessing_key(row):
    return (str(row["subject_id"]), str(row["task"]), str(row["run"]))


PREPROCESSING_RUN_MAP = {
    _preprocessing_key(row): row
    for row in _PREPROCESSING_INDEX.to_dict(orient="records")
}

_PREPROCESSING_OUTPUT_CFG = dict((_PREPROCESSING_CONFIG.get("preprocessing") or {}).get("output") or {})
_PREPROCESSING_EEG_FILENAME = str(_PREPROCESSING_OUTPUT_CFG.get("eeg_filename", "eeg_preprocessed_raw.fif"))
_PREPROCESSING_EMG_FILENAME = str(_PREPROCESSING_OUTPUT_CFG.get("emg_filename", "emg.npz"))
_PREPROCESSING_EVENTS_FILENAME = str(_PREPROCESSING_OUTPUT_CFG.get("events_filename", "events.tsv"))
_PREPROCESSING_SUMMARY_FILENAME = str(_PREPROCESSING_OUTPUT_CFG.get("summary_filename", "preprocessing_summary.json"))


def _preprocessing_row(wildcards):
    key = (str(wildcards.subject_id), str(wildcards.task), str(wildcards.run))
    return PREPROCESSING_RUN_MAP[key]


def _preprocessing_paths(wildcards):
    return build_preprocessing_output_paths(_PREPROCESSING_CONFIG, _preprocessing_row(wildcards))


PREPROCESSING_SUMMARIES = [
    str(build_preprocessing_output_paths(_PREPROCESSING_CONFIG, row).summary_output_path)
    for row in PREPROCESSING_RUN_MAP.values()
]

PREPROCESSING_STAGE = f"{OUT_DIR}/_stage/preprocessing_all.done.json"


rule preprocess_subject:
    input:
        raw_eeg=lambda wildcards: str(_preprocessing_paths(wildcards).raw_eeg_path),
        channels_tsv=lambda wildcards: str(_preprocessing_paths(wildcards).channels_tsv_path),
        ica_path=lambda wildcards: str(_preprocessing_paths(wildcards).ica_path),
        dataset_index=f"{OUT_DIR}/dataset/index.parquet",
        config_path="config/config.yaml",
        schema_path="config/schema.json",
    output:
        eeg_raw=f"{OUT_DIR}/preprocessing/{{subject_id}}/task-{{task}}/run-{{run}}/{_PREPROCESSING_EEG_FILENAME}",
        emg=f"{OUT_DIR}/preprocessing/{{subject_id}}/task-{{task}}/run-{{run}}/{_PREPROCESSING_EMG_FILENAME}",
        events=f"{OUT_DIR}/preprocessing/{{subject_id}}/task-{{task}}/run-{{run}}/{_PREPROCESSING_EVENTS_FILENAME}",
        summary=f"{OUT_DIR}/preprocessing/{{subject_id}}/task-{{task}}/run-{{run}}/{_PREPROCESSING_SUMMARY_FILENAME}",
    params:
        dyad_id=lambda wildcards: str(_preprocessing_row(wildcards)["dyad_id"]),
        subject_id=lambda wildcards: str(wildcards.subject_id),
        task=lambda wildcards: str(wildcards.task),
        run=lambda wildcards: str(wildcards.run),
    shell:
        "PYTHONPATH=src python -m rate.cli.main preprocess-subject "
        "--config {input.config_path} --schema {input.schema_path} --dataset-index {input.dataset_index} "
        "--dyad-id {params.dyad_id} --subject-id {params.subject_id} --task {params.task} --run {params.run} "
        "--raw-path {input.raw_eeg} --channels-tsv {input.channels_tsv} --ica-path {input.ica_path} "
        "--eeg-output {output.eeg_raw} --emg-output {output.emg} "
        "--events-output {output.events} --summary-output {output.summary}"


rule preprocess_stage:
    input:
        PREPROCESSING_SUMMARIES,
    output:
        PREPROCESSING_STAGE,
    shell:
        "python -c \"from pathlib import Path; "
        "path = Path(r'{output}'); "
        "path.parent.mkdir(parents=True, exist_ok=True); "
        "path.write_text('ok\\n', encoding='utf-8')\""
