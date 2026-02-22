from pathlib import Path


def _viz_out_dir() -> str:
    # Read from viz.yaml (loaded manually in Snakefile) so visualization changes do
    # not retrigger compute jobs via the tracked Snakemake `config`.
    return str(viz_config["paths"]["out_dir"])


def _viz_regressors():
    trf_reporting = viz_config.get("reporting", {}).get("trf", {})
    return list(trf_reporting.get("regressors", []))


def _viz_table_ext() -> str:
    tables_format = str(viz_config.get("reporting", {}).get("tables_format", "parquet")).lower()
    return "csv" if tables_format == "csv" else "parquet"


REGRESSORS = _viz_regressors()

if not REGRESSORS:
    raise ValueError("config/viz.yaml reporting.trf.regressors must define at least one TRF regressor")


def _evoked_model_for_regressor(regressor: str) -> str:
    return "baseline" if str(regressor) == "emg" else str(regressor)


def _evoked_kernel_input(wildcards):
    model = _evoked_model_for_regressor(str(wildcards.regressor))
    return f"{_viz_out_dir()}/stats/evoked_trf/{model}/{wildcards.regressor}/grand_average_trfs-ave.fif"


def _evoked_cluster_input(wildcards):
    model = _evoked_model_for_regressor(str(wildcards.regressor))
    return f"{_viz_out_dir()}/stats/evoked_trf/{model}/{wildcards.regressor}/clusters.{_viz_table_ext()}"


def _evoked_timecourse_kernel_inputs(_wildcards):
    return [
        f"{_viz_out_dir()}/stats/evoked_trf/{_evoked_model_for_regressor(reg)}/{reg}/grand_average_trfs-ave.fif"
        for reg in REGRESSORS
    ]


def _available_evoked_regressors():
    out = []
    for reg in REGRESSORS:
        model = _evoked_model_for_regressor(reg)
        kernel = Path(f"{_viz_out_dir()}/stats/evoked_trf/{model}/{reg}/grand_average_trfs-ave.fif")
        clusters = Path(f"{_viz_out_dir()}/stats/evoked_trf/{model}/{reg}/clusters.{_viz_table_ext()}")
        if kernel.exists() and clusters.exists():
            out.append(reg)
    return out


def _available_induced_regressors():
    out = []
    for reg in REGRESSORS:
        kernel = Path(f"{_viz_out_dir()}/induced_trf/group_kernel_{reg}.fif")
        clusters = Path(f"{_viz_out_dir()}/induced_trf/clusters_{reg}.parquet")
        if kernel.exists() and clusters.exists():
            out.append(reg)
    return out


AVAILABLE_EVOKED_REGRESSORS = _available_evoked_regressors()
AVAILABLE_INDUCED_REGRESSORS = _available_induced_regressors()

EVOKED_JOINT_TARGETS = [f"{_viz_out_dir()}/final/figures/evoked_trf/joint_{reg}.pdf" for reg in AVAILABLE_EVOKED_REGRESSORS]
EVOKED_TOPO_TARGETS = [f"{_viz_out_dir()}/final/figures/evoked_trf/topomap_{reg}.pdf" for reg in AVAILABLE_EVOKED_REGRESSORS]
INDUCED_JOINT_TARGETS = [f"{_viz_out_dir()}/final/figures/induced_trf/joint_{reg}.pdf" for reg in AVAILABLE_INDUCED_REGRESSORS]
INDUCED_TOPO_TARGETS = [f"{_viz_out_dir()}/final/figures/induced_trf/topomap_{reg}.pdf" for reg in AVAILABLE_INDUCED_REGRESSORS]


rule figure_evoked_trf_joint_by_regressor:
    input:
        kernel=_evoked_kernel_input,
        clusters=_evoked_cluster_input,
    output:
        f"{_viz_out_dir()}/final/figures/evoked_trf/joint_{{regressor}}.pdf"
    params:
        out_dir=lambda wildcards: _viz_out_dir()
    shell:
        """
        python -m rate.cli.main figures \
            --out-dir {params.out_dir} \
            --family evoked_trf \
            --figure joint \
            --regressor {wildcards.regressor}
        """


rule figure_evoked_trf_joint:
    input:
        EVOKED_JOINT_TARGETS


rule figure_evoked_trf_topo_by_regressor:
    input:
        kernel=_evoked_kernel_input,
        clusters=_evoked_cluster_input,
    output:
        f"{_viz_out_dir()}/final/figures/evoked_trf/topomap_{{regressor}}.pdf"
    params:
        out_dir=lambda wildcards: _viz_out_dir()
    shell:
        """
        python -m rate.cli.main figures \
            --out-dir {params.out_dir} \
            --family evoked_trf \
            --figure topo \
            --regressor {wildcards.regressor}
        """


rule figure_evoked_trf_topo:
    input:
        EVOKED_TOPO_TARGETS


rule figure_induced_trf_joint_by_regressor:
    input:
        kernel=f"{_viz_out_dir()}/induced_trf/group_kernel_{{regressor}}.fif",
        clusters=f"{_viz_out_dir()}/induced_trf/clusters_{{regressor}}.parquet",
    output:
        f"{_viz_out_dir()}/final/figures/induced_trf/joint_{{regressor}}.pdf"
    params:
        out_dir=lambda wildcards: _viz_out_dir()
    shell:
        """
        python -m rate.cli.main figures \
            --out-dir {params.out_dir} \
            --family induced_trf \
            --figure joint \
            --regressor {wildcards.regressor}
        """


rule figure_induced_trf_joint:
    input:
        INDUCED_JOINT_TARGETS


rule figure_induced_trf_topo_by_regressor:
    input:
        kernel=f"{_viz_out_dir()}/induced_trf/group_kernel_{{regressor}}.fif",
        clusters=f"{_viz_out_dir()}/induced_trf/clusters_{{regressor}}.parquet",
    output:
        f"{_viz_out_dir()}/final/figures/induced_trf/topomap_{{regressor}}.pdf"
    params:
        out_dir=lambda wildcards: _viz_out_dir()
    shell:
        """
        python -m rate.cli.main figures \
            --out-dir {params.out_dir} \
            --family induced_trf \
            --figure topo \
            --regressor {wildcards.regressor}
        """


rule figure_induced_trf_topo:
    input:
        INDUCED_TOPO_TARGETS


rule figure_evoked_trf_timecourse:
    input:
        _evoked_timecourse_kernel_inputs
    output:
        f"{_viz_out_dir()}/final/figures/evoked_trf/timecourse_panel.pdf"
    params:
        out_dir=lambda wildcards: _viz_out_dir()
    shell:
        """
        python -m rate.cli.main figures \
            --out-dir {params.out_dir} \
            --family evoked_trf \
            --figure timecourse
        """


rule figure_induced_trf_timecourse:
    input:
        expand(
            f"{_viz_out_dir()}/induced_trf/group_kernel_{{regressor}}.fif",
            regressor=REGRESSORS,
        )
    output:
        f"{_viz_out_dir()}/final/figures/induced_trf/timecourse_panel.pdf"
    params:
        out_dir=lambda wildcards: _viz_out_dir()
    shell:
        """
        python -m rate.cli.main figures \
            --out-dir {params.out_dir} \
            --family induced_trf \
            --figure timecourse
        """


rule figures_all:
    input:
        EVOKED_JOINT_TARGETS
        + EVOKED_TOPO_TARGETS
        # + INDUCED_JOINT_TARGETS
        # + INDUCED_TOPO_TARGETS
        + [
            f"{_viz_out_dir()}/final/figures/evoked_trf/timecourse_panel.pdf",
            # f"{_viz_out_dir()}/final/figures/induced_trf/timecourse_panel.pdf",
        ]
