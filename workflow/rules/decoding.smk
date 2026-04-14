DECODING_STAGE = f"{OUT_DIR}/_stage/decoding_all.done.json"
DECODING_STATS_STAGE = f"{OUT_DIR}/_stage/decoding_stats_all.done.json"
DECODING_OUTPUT_DIR = f"{OUT_DIR}/{RESOLVED_DECODING_CONFIG.get('output_dir', 'decoding')}"
DECODING_THREAD_ENV = (
    "OMP_NUM_THREADS=1 "
    "OPENBLAS_NUM_THREADS=1 "
    "MKL_NUM_THREADS=1 "
    "VECLIB_MAXIMUM_THREADS=1 "
    "NUMEXPR_NUM_THREADS=1"
)


rule decoding_fit:
    input:
        dataset_index=f"{OUT_DIR}/dataset/index.parquet",
        config_path="config/config.yaml",
        schema_path="config/schema.json",
        decoding_config_path="config/decoding.yaml",
        epochs_stage=EPOCHS_STAGE,
    output:
        scores=f"{DECODING_OUTPUT_DIR}/{{model_name}}/scores.npy",
        fold_scores=f"{DECODING_OUTPUT_DIR}/{{model_name}}/fold_scores.npy",
        alpha_scores=f"{DECODING_OUTPUT_DIR}/{{model_name}}/alpha_scores.npy",
        best_alphas=f"{DECODING_OUTPUT_DIR}/{{model_name}}/best_alphas.npy",
        times=f"{DECODING_OUTPUT_DIR}/{{model_name}}/times.npy",
        channel_names=f"{DECODING_OUTPUT_DIR}/{{model_name}}/channel_names.json",
        selection=f"{DECODING_OUTPUT_DIR}/{{model_name}}/selection.parquet",
        summary=f"{DECODING_OUTPUT_DIR}/{{model_name}}/summary.json",
    params:
        model_name=lambda wildcards: str(wildcards.model_name),
    shell:
        f"{DECODING_THREAD_ENV} PYTHONPATH=src python -m rate.cli.main decode-temporal-generalisation "
        "--config {input.config_path} --schema {input.schema_path} "
        "--decoding-config {input.decoding_config_path} "
        "--dataset-index {input.dataset_index} --model-name {params.model_name} --progress"


rule decoding_stage:
    input:
        decoding_stage_inputs,
    output:
        DECODING_STAGE,
    shell:
        "python -c \"from pathlib import Path; "
        "path = Path(r'{output}'); "
        "path.parent.mkdir(parents=True, exist_ok=True); "
        "path.write_text('ok\\n', encoding='utf-8')\""


rule decoding_stats_fit:
    input:
        dataset_index=f"{OUT_DIR}/dataset/index.parquet",
        config_path="config/config.yaml",
        schema_path="config/schema.json",
        decoding_config_path="config/decoding.yaml",
        decoding_summary=f"{DECODING_OUTPUT_DIR}/{{model_name}}/summary.json",
        decoding_scores=f"{DECODING_OUTPUT_DIR}/{{model_name}}/scores.npy",
    output:
        tg_matrix=f"{DECODING_OUTPUT_DIR}/{{model_name}}/tg_matrix.npy",
        tg_tfce=f"{DECODING_OUTPUT_DIR}/{{model_name}}/tg_tfce.npy",
        tg_p_fwer=f"{DECODING_OUTPUT_DIR}/{{model_name}}/tg_p_fwer.npy",
        tg_null_max_tfce=f"{DECODING_OUTPUT_DIR}/{{model_name}}/tg_null_max_tfce.npy",
        metadata=f"{DECODING_OUTPUT_DIR}/{{model_name}}/tg_stats_metadata.json",
    params:
        model_name=lambda wildcards: str(wildcards.model_name),
    shell:
        f"{DECODING_THREAD_ENV} PYTHONPATH=src python -m rate.cli.main decode-temporal-generalisation-stats "
        "--config {input.config_path} --schema {input.schema_path} "
        "--decoding-config {input.decoding_config_path} "
        "--dataset-index {input.dataset_index} --model-name {params.model_name} --progress"


rule decoding_stats_stage:
    input:
        decoding_stats_stage_inputs,
    output:
        DECODING_STATS_STAGE,
    shell:
        "python -c \"from pathlib import Path; "
        "path = Path(r'{output}'); "
        "path.parent.mkdir(parents=True, exist_ok=True); "
        "path.write_text('ok\\n', encoding='utf-8')\""
