OUT_DIR = config["paths"]["out_dir"]

rule evoked_trf_scores:
    input:
        events=f"{OUT_DIR}/events/events.parquet",
        dataset_index=f"{OUT_DIR}/dataset/index_resampled.parquet",
        validate_config=f"{OUT_DIR}/_meta/config_validated.json",
        write_manifest=f"{OUT_DIR}/_meta/run_manifest.json"
    output:
        scores=f"{OUT_DIR}/evoked_trf/scores/scores.parquet",
        cv_splits=f"{OUT_DIR}/evoked_trf/qc/cv_splits.parquet"
    log:
        f"{OUT_DIR}/logs/evoked_trf_scores.log"
    shell:
        r"""
        set -euo pipefail
        python -m rate.cli.main evoked-trf-scores --config config/config.yaml --schema config/schema.json > {log} 2>&1
        """
