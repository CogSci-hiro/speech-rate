# Source-space lmeEEG experimental extension

LMEEEG_SOURCE_STAGE = f"{OUT_DIR}/_stage/lmeeeg_source_all.done.json"
LMEEEG_SOURCE_QC_FIGURES_OUTPUT = f"{OUT_DIR}/figures/lmeeeg_source/figure_manifest.json"


rule lmeeeg_source_prepare_subject:
    input:
        dataset_index=ancient(f"{OUT_DIR}/dataset/index.parquet"),
        config_path="config/config.yaml",
        source_config_path="config/lmeeeg_source.yaml",
        margin_epochs_summary=lambda wildcards: str(margin_epoch_subject_paths(wildcards).summary_output_path),
    output:
        summary=f"{OUT_DIR}/lmeeeg_source/prepared/{{subject_id}}/summary.json",
    shell:
        "NUMBA_DISABLE_JIT=1 MNE_DONTWRITE_HOME=true MPLCONFIGDIR=/tmp/mpl PYTHONPATH=src python -m rate.cli.main lmeeeg-source "
        "--stage prepare-subject "
        "--config {input.config_path} --schema config/schema.json "
        "--dataset-index {input.dataset_index} --subject-id {wildcards.subject_id}"


rule lmeeeg_source_fit:
    input:
        dataset_index=ancient(f"{OUT_DIR}/dataset/index.parquet"),
        config_path="config/config.yaml",
        source_config_path="config/lmeeeg_source.yaml",
        prepared=lmeeeg_source_prepare_inputs,
    output:
        residualised_trials=f"{OUT_DIR}/lmeeeg_source/{{model_name}}/residualised_trials.npy",
        random_effects=f"{OUT_DIR}/lmeeeg_source/{{model_name}}/random_effects.npy",
        betas=f"{OUT_DIR}/lmeeeg_source/{{model_name}}/betas.npy",
        standard_errors=f"{OUT_DIR}/lmeeeg_source/{{model_name}}/standard_errors.npy",
        t_values=f"{OUT_DIR}/lmeeeg_source/{{model_name}}/t_values.npy",
        residual_variance=f"{OUT_DIR}/lmeeeg_source/{{model_name}}/residual_variance.npy",
        row_index=f"{OUT_DIR}/lmeeeg_source/{{model_name}}/row_index.npy",
        times=f"{OUT_DIR}/lmeeeg_source/{{model_name}}/times.npy",
        channel_names=f"{OUT_DIR}/lmeeeg_source/{{model_name}}/channel_names.json",
        column_names=f"{OUT_DIR}/lmeeeg_source/{{model_name}}/column_names.json",
        test_predictors=f"{OUT_DIR}/lmeeeg_source/{{model_name}}/test_predictors.json",
        selection=f"{OUT_DIR}/lmeeeg_source/{{model_name}}/selection.parquet",
        lmm_diagnostics=f"{OUT_DIR}/lmeeeg_source/{{model_name}}/lmm_diagnostics.json",
        source_vertices=f"{OUT_DIR}/lmeeeg_source/{{model_name}}/source_vertices.json",
        summary=f"{OUT_DIR}/lmeeeg_source/{{model_name}}/summary.json",
    params:
        model_name=lambda wildcards: str(wildcards.model_name),
    shell:
        "NUMBA_DISABLE_JIT=1 MNE_DONTWRITE_HOME=true MPLCONFIGDIR=/tmp/mpl PYTHONPATH=src python -m rate.cli.main lmeeeg-source "
        "--stage fit "
        "--config {input.config_path} --schema config/schema.json "
        "--dataset-index {input.dataset_index} --model-name {params.model_name}"


rule lmeeeg_source_group_stats:
    input:
        config_path="config/config.yaml",
        source_config_path="config/lmeeeg_source.yaml",
        fit_outputs=lmeeeg_source_stage_inputs,
    output:
        self_rate=f"{OUT_DIR}/lmeeeg_source/group/self_rate/summary.json",
        other_rate=f"{OUT_DIR}/lmeeeg_source/group/other_rate/summary.json",
    shell:
        "NUMBA_DISABLE_JIT=1 MNE_DONTWRITE_HOME=true MPLCONFIGDIR=/tmp/mpl PYTHONPATH=src python -m rate.cli.main lmeeeg-source "
        "--stage group-stats "
        "--config {input.config_path} --schema config/schema.json"


rule figures_lmeeeg_source_qc:
    input:
        config_path="config/config.yaml",
        source_config_path="config/lmeeeg_source.yaml",
        group_stats=rules.lmeeeg_source_group_stats.output,
    output:
        manifest=LMEEEG_SOURCE_QC_FIGURES_OUTPUT,
        png=f"{OUT_DIR}/figures/qc/lmeeeg_source/self_other_lateral.png",
        pdf=f"{OUT_DIR}/figures/qc/lmeeeg_source/self_other_lateral.pdf",
    shell:
        "NUMBA_DISABLE_JIT=1 MNE_DONTWRITE_HOME=true MPLCONFIGDIR=/tmp/mpl PYTHONPATH=src python -m rate.cli.main lmeeeg-source "
        "--stage plot-qc "
        "--config {input.config_path} --schema config/schema.json "
        "--output {output.manifest}"


rule lmeeeg_source_stage:
    input:
        lmeeeg_source_stage_inputs,
        LMEEEG_SOURCE_QC_FIGURES_OUTPUT,
    output:
        LMEEEG_SOURCE_STAGE,
    shell:
        "python -c \"from pathlib import Path; "
        "path = Path(r'{output}'); "
        "path.parent.mkdir(parents=True, exist_ok=True); "
        "path.write_text('ok\\n', encoding='utf-8')\""
