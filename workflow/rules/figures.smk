OUT_DIR = config["paths"]["out_dir"]
TRF_REPORTING = config.get("reporting", {}).get("trf", {})
REGRESSORS = list(TRF_REPORTING.get("regressors", []))
TABLES_FORMAT = str(config.get("reporting", {}).get("tables_format", "parquet")).lower()
TABLE_EXT = "csv" if TABLES_FORMAT == "csv" else "parquet"

if not REGRESSORS:
    raise ValueError("config.reporting.trf.regressors must define at least one TRF regressor")


def _evoked_model_for_regressor(regressor: str) -> str:
    return "baseline" if str(regressor) == "emg" else str(regressor)


def _evoked_kernel_input(wildcards):
    model = _evoked_model_for_regressor(str(wildcards.regressor))
    return f"{OUT_DIR}/stats/evoked_trf/{model}/{wildcards.regressor}/grand_average_trfs-ave.fif"


def _evoked_cluster_input(wildcards):
    model = _evoked_model_for_regressor(str(wildcards.regressor))
    return f"{OUT_DIR}/stats/evoked_trf/{model}/{wildcards.regressor}/clusters.{TABLE_EXT}"


def _evoked_timecourse_kernel_inputs(_wildcards):
    return [
        f"{OUT_DIR}/stats/evoked_trf/{_evoked_model_for_regressor(reg)}/{reg}/grand_average_trfs-ave.fif"
        for reg in REGRESSORS
    ]


def _available_evoked_regressors():
    out = []
    for reg in REGRESSORS:
        model = _evoked_model_for_regressor(reg)
        kernel = Path(f"{OUT_DIR}/stats/evoked_trf/{model}/{reg}/grand_average_trfs-ave.fif")
        clusters = Path(f"{OUT_DIR}/stats/evoked_trf/{model}/{reg}/clusters.{TABLE_EXT}")
        if kernel.exists() and clusters.exists():
            out.append(reg)
    return out


def _available_induced_regressors():
    out = []
    for reg in REGRESSORS:
        kernel = Path(f"{OUT_DIR}/induced_trf/group_kernel_{reg}.fif")
        clusters = Path(f"{OUT_DIR}/induced_trf/clusters_{reg}.parquet")
        if kernel.exists() and clusters.exists():
            out.append(reg)
    return out


AVAILABLE_EVOKED_REGRESSORS = _available_evoked_regressors()
AVAILABLE_INDUCED_REGRESSORS = _available_induced_regressors()

EVOKED_JOINT_TARGETS = [f"{OUT_DIR}/final/figures/evoked_trf/joint_{reg}.pdf" for reg in AVAILABLE_EVOKED_REGRESSORS]
EVOKED_TOPO_TARGETS = [f"{OUT_DIR}/final/figures/evoked_trf/topomap_{reg}.pdf" for reg in AVAILABLE_EVOKED_REGRESSORS]
INDUCED_JOINT_TARGETS = [f"{OUT_DIR}/final/figures/induced_trf/joint_{reg}.pdf" for reg in AVAILABLE_INDUCED_REGRESSORS]
INDUCED_TOPO_TARGETS = [f"{OUT_DIR}/final/figures/induced_trf/topomap_{reg}.pdf" for reg in AVAILABLE_INDUCED_REGRESSORS]


rule figure_evoked_trf_joint_by_regressor:
    input:
        kernel=_evoked_kernel_input,
        clusters=_evoked_cluster_input,
    output:
        f"{OUT_DIR}/final/figures/evoked_trf/joint_{{regressor}}.pdf"
    shell:
        """
        python -m rate.cli.main figures \
            --out-dir {OUT_DIR} \
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
        f"{OUT_DIR}/final/figures/evoked_trf/topomap_{{regressor}}.pdf"
    shell:
        """
        python -m rate.cli.main figures \
            --out-dir {OUT_DIR} \
            --family evoked_trf \
            --figure topo \
            --regressor {wildcards.regressor}
        """


rule figure_evoked_trf_topo:
    input:
        EVOKED_TOPO_TARGETS


rule figure_induced_trf_joint_by_regressor:
    input:
        kernel=f"{OUT_DIR}/induced_trf/group_kernel_{{regressor}}.fif",
        clusters=f"{OUT_DIR}/induced_trf/clusters_{{regressor}}.parquet",
    output:
        f"{OUT_DIR}/final/figures/induced_trf/joint_{{regressor}}.pdf"
    shell:
        """
        python -m rate.cli.main figures \
            --out-dir {OUT_DIR} \
            --family induced_trf \
            --figure joint \
            --regressor {wildcards.regressor}
        """


rule figure_induced_trf_joint:
    input:
        INDUCED_JOINT_TARGETS


rule figure_induced_trf_topo_by_regressor:
    input:
        kernel=f"{OUT_DIR}/induced_trf/group_kernel_{{regressor}}.fif",
        clusters=f"{OUT_DIR}/induced_trf/clusters_{{regressor}}.parquet",
    output:
        f"{OUT_DIR}/final/figures/induced_trf/topomap_{{regressor}}.pdf"
    shell:
        """
        python -m rate.cli.main figures \
            --out-dir {OUT_DIR} \
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
        f"{OUT_DIR}/final/figures/evoked_trf/timecourse_panel.pdf"
    shell:
        """
        python -m rate.cli.main figures \
            --out-dir {OUT_DIR} \
            --family evoked_trf \
            --figure timecourse
        """


rule figure_induced_trf_timecourse:
    input:
        expand(
            f"{OUT_DIR}/induced_trf/group_kernel_{{regressor}}.fif",
            regressor=REGRESSORS,
        )
    output:
        f"{OUT_DIR}/final/figures/induced_trf/timecourse_panel.pdf"
    shell:
        """
        python -m rate.cli.main figures \
            --out-dir {OUT_DIR} \
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
            f"{OUT_DIR}/final/figures/evoked_trf/timecourse_panel.pdf",
            # f"{OUT_DIR}/final/figures/induced_trf/timecourse_panel.pdf",
        ]
