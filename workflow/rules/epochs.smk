import re

def _sanitize_lock_name(value):
    token = re.sub(r"[^a-zA-Z0-9]+", "_", str(value).strip()).strip("_").lower()
    return token or "events"


def _render_filename(pattern, *, lock_name):
    return str(pattern).format(lock_name=_sanitize_lock_name(lock_name))


_EPOCHS_CFG = dict(RESOLVED_CONFIG.get("epochs") or {})
_EPOCHS_OUTPUT_CFG = dict(_EPOCHS_CFG.get("output") or {})
_LOCK_NAME = str(_EPOCHS_CFG.get("event_column", "time_s"))
_EPOCHS_FILENAME = _render_filename(_EPOCHS_OUTPUT_CFG.get("epochs_filename", "epochs-{lock_name}.fif"), lock_name=_LOCK_NAME)
_METADATA_FILENAME = _render_filename(_EPOCHS_OUTPUT_CFG.get("metadata_filename", "metadata-{lock_name}.csv"), lock_name=_LOCK_NAME)
_EVENTS_ARRAY_FILENAME = _render_filename(_EPOCHS_OUTPUT_CFG.get("events_array_filename", "events-{lock_name}.npy"), lock_name=_LOCK_NAME)
_SUMMARY_FILENAME = _render_filename(_EPOCHS_OUTPUT_CFG.get("summary_filename", "epoching_summary-{lock_name}.json"), lock_name=_LOCK_NAME)

EPOCHS_STAGE = f"{OUT_DIR}/_stage/epochs_all.done.json"


rule epoch_subject:
    input:
        preprocessing=epoch_subject_preprocessing_inputs,
        events_table=f"{OUT_DIR}/events/events.csv",
        dataset_index=f"{OUT_DIR}/dataset/index.parquet",
        config_path="config/config.yaml",
        schema_path="config/schema.json",
    output:
        epochs=f"{OUT_DIR}/epochs/{{subject_id}}/{_EPOCHS_FILENAME}",
        metadata=f"{OUT_DIR}/epochs/{{subject_id}}/{_METADATA_FILENAME}",
        events_array=f"{OUT_DIR}/epochs/{{subject_id}}/{_EVENTS_ARRAY_FILENAME}",
        summary=f"{OUT_DIR}/epochs/{{subject_id}}/{_SUMMARY_FILENAME}",
    params:
        dyad_id=lambda wildcards: str(_epochs_subject_map(_dataset_index_checkpoint_path())[str(wildcards.subject_id)]["dyad_id"]),
        subject_id=lambda wildcards: str(wildcards.subject_id),
    shell:
        "PYTHONPATH=src python -m rate.cli.main epoch-subject "
        "--config {input.config_path} --schema {input.schema_path} --dataset-index {input.dataset_index} "
        "--dyad-id {params.dyad_id} --subject-id {params.subject_id} --events-table {input.events_table} "
        "--epochs-output {output.epochs} --metadata-output {output.metadata} "
        "--events-array-output {output.events_array} --summary-output {output.summary}"


rule epochs_stage:
    input:
        epochs_stage_inputs,
    output:
        EPOCHS_STAGE,
    shell:
        "python -c \"from pathlib import Path; "
        "path = Path(r'{output}'); "
        "path.parent.mkdir(parents=True, exist_ok=True); "
        "path.write_text('ok\\n', encoding='utf-8')\""
