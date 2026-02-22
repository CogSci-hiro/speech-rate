OUT_DIR = config["paths"]["out_dir"]
RESAMPLED_DIR = f"{OUT_DIR}/eeg_resampled"
INDEX_RESAMPLED = f"{OUT_DIR}/dataset/index_resampled.parquet"

rule eeg_resampled_raws:
    """
    Cache resampled EEG FIF raws and write dataset/index_resampled.parquet.

    Output index overwrites the eeg_path column to point to resampled FIF paths.
    """
    input:
        dataset_index=f"{OUT_DIR}/dataset/index.parquet",
        validate_config=f"{OUT_DIR}/_meta/config_validated.json",
        write_manifest=f"{OUT_DIR}/_meta/run_manifest.json",
    output:
        resampled_dir=directory(RESAMPLED_DIR),
        index_resampled=INDEX_RESAMPLED,
    log:
        f"{OUT_DIR}/logs/eeg_resample_raws.log",
    shell:
        r"""
        set -euo pipefail
        PYTHONPATH=src python -m rate.cli.main resample-eeg --config config/compute.yaml --schema config/schema.json 2>&1 | tee {log}
        test -d {output.resampled_dir}
        find {output.resampled_dir} -type f -name "*.fif" -print -quit | grep -q .
        """


rule eeg_resampled_index:
    """
    Backward-compatible alias target for the resampled dataset index.
    """
    input:
        INDEX_RESAMPLED
