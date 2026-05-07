DATASET_DIR = f"{OUT_DIR}/dataset"
META_DIR = f"{OUT_DIR}/_meta"


checkpoint dataset_index:
    input:
        config_path="config/config.yaml",
        dataset_config_path="config/dataset.yaml",
    output:
        index=f"{DATASET_DIR}/index.parquet",
    shell:
        "PYTHONPATH=src python -m rate.cli.main dataset-index "
        "--config {input.config_path} --schema config/schema.json --output {output.index}"
