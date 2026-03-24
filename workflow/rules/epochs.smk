EPOCHS_STAGE = f"{OUT_DIR}/_stage/epochs_all.done.json"


rule epochs_stage:
    input:
        preprocessing=PREPROCESSING_STAGE,
    output:
        EPOCHS_STAGE,
    shell:
        "PYTHONPATH=src python -m rate.cli.main epoch-subject "
        "--config config/config.yaml --schema config/schema.json --all --output {output}"
