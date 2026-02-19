
rule induced_features:
    input:
        events="{out_dir}/events/events.parquet",
    output:
        touch("{out_dir}/induced_trf/_features.done"),
    shell:
        "python -m rate.cli.run_induced_features"

rule induced_trf_scores:
    input:
        "{out_dir}/induced_trf/_features.done",
        "{out_dir}/events/events.parquet",
    output:
        "{out_dir}/induced_trf/scores/scores.parquet",
    shell:
        "python -m rate.cli.run_induced_scores"
