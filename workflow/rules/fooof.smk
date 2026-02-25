OUT_DIR = COMPUTE_OUT_DIR

FOOOF_TARGETS = [
    f"{OUT_DIR}/fooof/trial_level/fooof_trial_channels.parquet",
    f"{OUT_DIR}/fooof/trial_level/fooof_trial_rois.csv",
    f"{OUT_DIR}/fooof/aggregated/trial_psd_spectra.parquet",
    f"{OUT_DIR}/fooof/aggregated/subject_psd_spectra.parquet",
    f"{OUT_DIR}/fooof/meta/fooof_settings.json",
    f"{OUT_DIR}/fooof/meta/qc_summary.parquet",
]


def _fooof_threads():
    cfg = fooof_config.get("fooof", {}) or {}
    try:
        n = int(cfg.get("n_jobs", 1))
    except Exception:
        n = 1
    return max(1, n)


rule fooof:
    input:
        dataset_index=f"{OUT_DIR}/dataset/index.parquet",
        events=f"{OUT_DIR}/events/events.parquet",
        config="config/config.yaml",
        config_validated=f"{OUT_DIR}/_meta/config_validated.json",
        run_manifest=f"{OUT_DIR}/_meta/run_manifest.json",
    output:
        trial_channels=f"{OUT_DIR}/fooof/trial_level/fooof_trial_channels.parquet",
        trial_rois=f"{OUT_DIR}/fooof/trial_level/fooof_trial_rois.csv",
        trial_psd_spectra=f"{OUT_DIR}/fooof/aggregated/trial_psd_spectra.parquet",
        subject_psd_spectra=f"{OUT_DIR}/fooof/aggregated/subject_psd_spectra.parquet",
        settings=f"{OUT_DIR}/fooof/meta/fooof_settings.json",
        qc_summary=f"{OUT_DIR}/fooof/meta/qc_summary.parquet",
    log:
        f"{OUT_DIR}/logs/fooof.log"
    threads: _fooof_threads()
    shell:
        r"""
        set -euo pipefail
        PYTHONPATH=src python -u -m rate.cli.main fooof \
            --config {input.config} \
            --out-dir {OUT_DIR} \
            --n-jobs {threads} \
            2>&1 | tee {log}
        """


rule fooof_all:
    input:
        FOOOF_TARGETS
