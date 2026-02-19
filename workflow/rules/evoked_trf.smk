OUT_DIR = config["paths"]["out_dir"]

rule evoked_trf_scores:
    """
    Compute evoked TRF score tables with blocked within-run CV.
    """
    input:
        events=f"{OUT_DIR}/events/events.parquet",
        dataset_index=f"{OUT_DIR}/dataset/index.parquet",
        validate_config=f"{OUT_DIR}/_meta/config_validated.json",
        write_manifest=f"{OUT_DIR}/_meta/run_manifest.json",
    output:
        scores=f"{OUT_DIR}/evoked_trf/scores/scores.parquet",
        cv_splits=f"{OUT_DIR}/evoked_trf/qc/cv_splits.parquet",
    log:
        f"{OUT_DIR}/logs/evoked_trf_scores.log",
    shell:
        r"""
        set -euo pipefail
        python - <<'PY' > {log} 2>&1
        from pathlib import Path
        from rate.config import load_and_validate_config
        from rate.evoked_trf.score import write_outputs
        
        cfg = load_and_validate_config()
        out_dir = Path(cfg["paths"]["out_dir"])
        write_outputs(cfg=cfg, out_dir=out_dir)
        PY
        """
