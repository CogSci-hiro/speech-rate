OUT_DIR = config["paths"]["out_dir"]
META_DIR = f"{OUT_DIR}/_meta"


rule validate_config:
    input:
        config_path="config/config.yaml",
        schema_path="config/schema.json",
    output:
        validated=f"{META_DIR}/config_validated.json",
    shell:
        "PYTHONPATH=src python -m rate.cli.main validate-config "
        "--config {input.config_path} --schema {input.schema_path} --output {output.validated}"


rule write_manifest:
    input:
        config_path="config/config.yaml",
        schema_path="config/schema.json",
        validated=f"{META_DIR}/config_validated.json",
    output:
        manifest=f"{META_DIR}/run_manifest.json",
    shell:
        "PYTHONPATH=src python -m rate.cli.main write-manifest "
        "--config {input.config_path} --schema {input.schema_path} --output {output.manifest}"
