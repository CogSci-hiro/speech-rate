FOOOF_MASS_STAGE = f"{OUT_DIR}/_stage/fooof_mass_all.done.json"


rule fooof_mass_subject:
    input:
        epochs=lambda wildcards: str(fooof_paths(wildcards).epochs_input_path),
        metadata=lambda wildcards: str(fooof_paths(wildcards).metadata_input_path),
        dataset_index=f"{OUT_DIR}/dataset/index.parquet",
        config_path="config/config.yaml",
        schema_path="config/schema.json",
    output:
        windows=lambda wildcards: str(fooof_paths(wildcards).windows_output_path),
        frequencies=lambda wildcards: str(fooof_paths(wildcards).frequencies_output_path),
        fit_success=lambda wildcards: str(fooof_paths(wildcards).fit_success_output_path),
        feature_validity=lambda wildcards: str(fooof_paths(wildcards).feature_validity_output_path),
        feature_tensor=lambda wildcards: str(fooof_paths(wildcards).feature_tensor_output_path),
        betas=lambda wildcards: str(fooof_paths(wildcards).betas_output_path),
        standard_errors=lambda wildcards: str(fooof_paths(wildcards).standard_errors_output_path),
        t_values=lambda wildcards: str(fooof_paths(wildcards).t_values_output_path),
        valid_counts=lambda wildcards: str(fooof_paths(wildcards).valid_counts_output_path),
        column_names=lambda wildcards: str(fooof_paths(wildcards).column_names_output_path),
        channel_names=lambda wildcards: str(fooof_paths(wildcards).channel_names_output_path),
        test_predictors=lambda wildcards: str(fooof_paths(wildcards).test_predictors_output_path),
        selection=lambda wildcards: str(fooof_paths(wildcards).selection_output_path),
        summary=lambda wildcards: str(fooof_paths(wildcards).summary_output_path),
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
