FOOOF_EPOCHS_STAGE = f"{OUT_DIR}/_stage/fooof_epochs_all.done.json"
FOOOF_MASS_STAGE = f"{OUT_DIR}/_stage/fooof_mass_all.done.json"


rule fooof_epochs_subject:
    input:
        epochs=lambda wildcards: str(margin_epoch_subject_paths(wildcards).epochs_output_path),
        metadata=lambda wildcards: str(margin_epoch_subject_paths(wildcards).metadata_output_path),
        dataset_index=f"{OUT_DIR}/dataset/index.parquet",
        config_path="config/config.yaml",
        schema_path="config/schema.json",
    output:
        summary=f"{OUT_DIR}/fooof_epochs/{{subject_id}}/summary.json",
    params:
        dyad_id=lambda wildcards: str(_epochs_subject_map(_dataset_index_checkpoint_path())[str(wildcards.subject_id)]["dyad_id"]),
        subject_id=lambda wildcards: str(wildcards.subject_id),
    shell:
        "PYTHONPATH=src python -m rate.cli.main fooof-epochs-subject "
        "--config {input.config_path} --schema {input.schema_path} --dataset-index {input.dataset_index} "
        "--dyad-id {params.dyad_id} --subject-id {params.subject_id} "
        "--epochs-path {input.epochs} --metadata-path {input.metadata}"


rule fooof_epochs_stage:
    input:
        fooof_epochs_stage_inputs,
    output:
        FOOOF_EPOCHS_STAGE,
    shell:
        "python -c \"from pathlib import Path; "
        "path = Path(r'{output}'); "
        "path.parent.mkdir(parents=True, exist_ok=True); "
        "path.write_text('ok\\n', encoding='utf-8')\""


rule fooof_mass_subject:
    input:
        epochs=lambda wildcards: str(fooof_paths(wildcards).epochs_input_path),
        metadata=lambda wildcards: str(fooof_paths(wildcards).metadata_input_path),
        dataset_index=f"{OUT_DIR}/dataset/index.parquet",
        config_path="config/config.yaml",
        schema_path="config/schema.json",
    output:
        summary=f"{OUT_DIR}/fooof_mass/{{subject_id}}/task-{{task}}/run-{{run}}/{{model_name}}/summary.json",
    params:
        subject_id=lambda wildcards: str(wildcards.subject_id),
        task=lambda wildcards: str(wildcards.task),
        run=lambda wildcards: str(wildcards.run),
        model_name=lambda wildcards: str(wildcards.model_name),
    shell:
        "PYTHONPATH=src python -m rate.cli.main fooof-mass-subject "
        "--config {input.config_path} --schema {input.schema_path} --dataset-index {input.dataset_index} "
        "--subject-id {params.subject_id} --task {params.task} --run {params.run} "
        "--model-name {params.model_name} --epochs-path {input.epochs} --metadata-path {input.metadata}"


rule fooof_mass_stage:
    input:
        fooof_stage_inputs,
    output:
        FOOOF_MASS_STAGE,
    shell:
        "python -c \"from pathlib import Path; "
        "path = Path(r'{output}'); "
        "path.parent.mkdir(parents=True, exist_ok=True); "
        "path.write_text('ok\\n', encoding='utf-8')\""
