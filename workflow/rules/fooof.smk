OUT_DIR = config["paths"]["out_dir"]

FOOOF_TARGETS = [
    f"{OUT_DIR}/fooof/trial_level/fooof_trial_channels.parquet",
    f"{OUT_DIR}/fooof/trial_level/fooof_trial_rois.csv",
    f"{OUT_DIR}/fooof/meta/fooof_settings.json",
    f"{OUT_DIR}/fooof/meta/qc_summary.parquet",
]


rule fooof:
    input:
        dataset_index=f"{OUT_DIR}/dataset/index_resampled.parquet",
        events=f"{OUT_DIR}/events/events.parquet",
        config="config/compute.yaml",
        config_validated=f"{OUT_DIR}/_meta/config_validated.json",
        run_manifest=f"{OUT_DIR}/_meta/run_manifest.json",
    output:
        trial_channels=f"{OUT_DIR}/fooof/trial_level/fooof_trial_channels.parquet",
        trial_rois=f"{OUT_DIR}/fooof/trial_level/fooof_trial_rois.csv",
        settings=f"{OUT_DIR}/fooof/meta/fooof_settings.json",
        qc_summary=f"{OUT_DIR}/fooof/meta/qc_summary.parquet",
    log:
        f"{OUT_DIR}/logs/fooof.log"
    threads: 1
    shell:
        r"""
        set -euo pipefail
        PYTHONPATH=src python -u -m rate.cli.main fooof \
            --config {input.config} \
            --out-dir {OUT_DIR} \
            2>&1 | tee {log}
        """
