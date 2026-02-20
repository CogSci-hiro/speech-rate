OUT_DIR = config["paths"]["out_dir"]

rule eeg_resampled_index:
    """
    Cache resampled EEG FIFs and write dataset/index_resampled.parquet.

    Output index overwrites the eeg_path column to point to resampled FIFs.
    """
    input:
        dataset_index=f"{OUT_DIR}/dataset/index.parquet",
        validate_config=f"{OUT_DIR}/_meta/config_validated.json",
        write_manifest=f"{OUT_DIR}/_meta/run_manifest.json",
    output:
        index_resampled=f"{OUT_DIR}/dataset/index_resampled.parquet",
    log:
        f"{OUT_DIR}/logs/eeg_resample.log",
    shell:
        r"""
        set -euo pipefail
        python -m rate.cli.main resample-eeg --config config/config.yaml --schema config/schema.json 2>&1 | tee {log}
        """