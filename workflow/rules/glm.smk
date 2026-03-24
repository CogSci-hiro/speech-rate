GLM_STAGE = f"{OUT_DIR}/_stage/glm_all.done.json"


rule glm_stage:
    input:
        epochs=EPOCHS_STAGE,
    output:
        GLM_STAGE,
    shell:
        "PYTHONPATH=src python -m rate.cli.main glm-subject "
        "--config config/config.yaml --schema config/schema.json --all --output {output}"
