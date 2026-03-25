"""`figures-glm` CLI subcommand."""

from __future__ import annotations

import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Mapping, Sequence

from rate.cli.commands._workflow_common import add_common_config_arguments, build_stage_payload, write_placeholder_output
from rate.design import list_model_names, load_model_spec
from rate.paths import resolve_out_dir

LOGGER = logging.getLogger(__name__)
REQUIRES_CONFIG = True


def _normalize_regressors(regressors: Sequence[str]) -> list[str]:
    """Normalize regressor labels into filename-safe tokens."""
    normalized: list[str] = []
    for regressor in regressors:
        token = str(regressor).strip().replace(" * ", "__x__").replace("*", "__x__")
        normalized.append(token)
    return normalized


def _resolve_glm_regressors(config: Mapping[str, Any]) -> list[str]:
    glm_config = config.get("glm") or {}
    direct = list(glm_config.get("regressors", []))
    if direct:
        return _normalize_regressors(direct)

    regressors: list[str] = []
    for model_name in list_model_names(config, section="glm"):
        model_spec = load_model_spec(config, section="glm", model_name=model_name)
        for predictor in model_spec.test_predictors:
            if predictor not in regressors:
                regressors.append(predictor)
    return _normalize_regressors(regressors)


def add_subparser(subparsers) -> ArgumentParser:
    parser = subparsers.add_parser("figures-glm", help="Run or scaffold GLM figure generation.")
    add_common_config_arguments(parser)
    parser.add_argument("--viz-config", default="config/viz.yaml", help="Path to the figure config file.")
    parser.add_argument("--output", help="Optional explicit output path.")
    return parser


def run(args: Namespace, config: Mapping[str, Any] | None = None) -> int:
    assert config is not None
    output_path = Path(args.output) if args.output else resolve_out_dir(config) / "figures" / "glm" / "figure_manifest.json"
    regressors = _resolve_glm_regressors(config)
    payload = build_stage_payload(
        stage="figures_glm",
        args=args,
        config=config,
        extra={"viz_config": args.viz_config, "regressors": regressors},
    )
    write_placeholder_output(output_path, payload)
    LOGGER.info("Wrote GLM figure scaffold output: %s", output_path)
    return 0
