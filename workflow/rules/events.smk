OUT_DIR = config["paths"]["out_dir"]
META_DIR = f"{OUT_DIR}/_meta"
DATASET_DIR = f"{OUT_DIR}/dataset"
EVENTS_DIR = f"{OUT_DIR}/events"


rule events_table:
    input:
        config_path="config/compute.yaml",
        schema_path="config/schema.json",
        validated=f"{META_DIR}/config_validated.json",
        manifest=f"{META_DIR}/run_manifest.json",
        dataset_index=f"{DATASET_DIR}/index.parquet",
    output:
        events_table=f"{EVENTS_DIR}/events.parquet",
        qc_summary=f"{EVENTS_DIR}/qc_ipu_summary.csv",
    shell:
        "PYTHONPATH=src python -m rate.cli.main events-table "
        "--config {input.config_path} --schema {input.schema_path} "
        "--dataset-index {input.dataset_index} "
        "--events-output {output.events_table} --qc-output {output.qc_summary}"
