OUT_DIR = COMPUTE_OUT_DIR
DATASET_DIR = f"{OUT_DIR}/dataset"
META_DIR = f"{OUT_DIR}/_meta"


rule dataset_index:
    input:
        config_path="config/config.yaml",
        schema_path="config/schema.json",
        validated=f"{META_DIR}/config_validated.json",
        manifest=f"{META_DIR}/run_manifest.json",
    output:
        index=f"{DATASET_DIR}/index.parquet",
    shell:
        "PYTHONPATH=src python -m rate.cli.main dataset-index "
        "--config {input.config_path} --schema {input.schema_path} --output {output.index}"
