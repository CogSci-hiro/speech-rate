
rule features_emg:
    input:
        edf=lambda wildcards: f"/Volumes/work-4T/hyperscanning-bids/{wildcards.subject}/eeg/{wildcards.subject}_task-conversation_run-{wildcards.run}_eeg.edf"
    output:
        parquet=f"{OUT_DIR}/features/emg/{{subject}}/run-{{run}}_emg_env.parquet"
    params:
        sfreq=config["evoked_trf"]["downsample_hz"]
    shell:
        """
        python -m rate.cli.commands.compute_emg \
            --edf {input.edf} \
            --out {output.parquet} \
            --sfreq {params.sfreq}
        """