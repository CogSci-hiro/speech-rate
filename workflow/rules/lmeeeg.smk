LMEEEG_STAGE = f"{OUT_DIR}/_stage/lmeeeg_all.done.json"


rule lmeeeg_subject:
    input:
        epochs=lambda wildcards: str(lmeeeg_paths(wildcards).epochs_input_path),
        metadata=lambda wildcards: str(lmeeeg_paths(wildcards).metadata_input_path),
        dataset_index=f"{OUT_DIR}/dataset/index.parquet",
        config_path="config/config.yaml",
        schema_path="config/schema.json",
    output:
        residualised_trials=lambda wildcards: str(lmeeeg_paths(wildcards).residualised_trials_output_path),
        random_effects=lambda wildcards: str(lmeeeg_paths(wildcards).random_effects_output_path),
        betas=lambda wildcards: str(lmeeeg_paths(wildcards).betas_output_path),
        standard_errors=lambda wildcards: str(lmeeeg_paths(wildcards).standard_errors_output_path),
        t_values=lambda wildcards: str(lmeeeg_paths(wildcards).t_values_output_path),
        residual_variance=lambda wildcards: str(lmeeeg_paths(wildcards).residual_variance_output_path),
        row_index=lambda wildcards: str(lmeeeg_paths(wildcards).row_index_output_path),
        times=lambda wildcards: str(lmeeeg_paths(wildcards).times_output_path),
        channel_names=lambda wildcards: str(lmeeeg_paths(wildcards).channel_names_output_path),
        column_names=lambda wildcards: str(lmeeeg_paths(wildcards).column_names_output_path),
        test_predictors=lambda wildcards: str(lmeeeg_paths(wildcards).test_predictors_output_path),
        selection=lambda wildcards: str(lmeeeg_paths(wildcards).selection_output_path),
        lmm_diagnostics=lambda wildcards: str(lmeeeg_paths(wildcards).lmm_diagnostics_output_path),
        summary=lambda wildcards: str(lmeeeg_paths(wildcards).summary_output_path),
    params:
        subject_id=lambda wildcards: str(wildcards.subject_id),
        task=lambda wildcards: str(wildcards.task),
        run=lambda wildcards: str(wildcards.run),
        model_name=lambda wildcards: str(wildcards.model_name),
    shell:
        "PYTHONPATH=src python -m rate.cli.main lmeeeg-subject "
        "--config {input.config_path} --schema {input.schema_path} --dataset-index {input.dataset_index} "
        "--subject-id {params.subject_id} --task {params.task} --run {params.run} "
        "--model-name {params.model_name} --epochs-path {input.epochs} --metadata-path {input.metadata}"


rule lmeeeg_stage:
    input:
        lmeeeg_stage_inputs,
    output:
        LMEEEG_STAGE,
    shell:
        "python -c \"from pathlib import Path; "
        "path = Path(r'{output}'); "
        "path.parent.mkdir(parents=True, exist_ok=True); "
        "path.write_text('ok\\n', encoding='utf-8')\""
