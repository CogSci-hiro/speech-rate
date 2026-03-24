FOOOF_MASS_STAGE = f"{OUT_DIR}/_stage/fooof_mass_all.done.json"


rule fooof_mass_stage:
    input:
        epochs=EPOCHS_STAGE,
    output:
        FOOOF_MASS_STAGE,
    shell:
        "PYTHONPATH=src python -m rate.cli.main fooof-mass-subject "
        "--config config/config.yaml --schema config/schema.json --all --output {output}"
