LMEEEG_STAGE = f"{OUT_DIR}/_stage/lmeeeg_all.done.json"


rule lmeeeg_stage:
    input:
        epochs=EPOCHS_STAGE,
    output:
        LMEEEG_STAGE,
    shell:
        "PYTHONPATH=src python -m rate.cli.main lmeeeg-prepare-trials "
        "--config config/config.yaml --schema config/schema.json --all --output {output}"
