from pathlib import Path

EVENTS_DIR = f"{OUT_DIR}/events"
EVENTS_OUTPUT = f"{EVENTS_DIR}/events.csv"
EVENTS_QC_OUTPUT = f"{EVENTS_DIR}/qc_ipu_summary.csv"
EVENTS_VALIDATION_OUTPUT = f"{EVENTS_DIR}/events_validation.json"
EVENTS_STAGE = f"{OUT_DIR}/_stage/events_all.done.json"


rule build_events:
    input:
        dataset_index=f"{OUT_DIR}/dataset/index.parquet",
        preprocessing=PREPROCESSING_STAGE,
        config_path="config/config.yaml",
        schema_path="config/schema.json",
    output:
        events=EVENTS_OUTPUT,
        qc=EVENTS_QC_OUTPUT,
        validation=EVENTS_VALIDATION_OUTPUT,
    shell:
        "PYTHONPATH=src python -m rate.cli.main build-events "
        "--config {input.config_path} --schema {input.schema_path} "
        "--dataset-index {input.dataset_index} "
        "--output {output.events} "
        "--qc-output {output.qc} --validation-output {output.validation}"


rule events_stage:
    input:
        EVENTS_OUTPUT,
        EVENTS_QC_OUTPUT,
        EVENTS_VALIDATION_OUTPUT,
    output:
        EVENTS_STAGE,
    shell:
        "python -c \"from pathlib import Path; "
        "path = Path(r'{output}'); "
        "path.parent.mkdir(parents=True, exist_ok=True); "
        "path.write_text('ok\\n', encoding='utf-8')\""
