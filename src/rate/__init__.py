"""Public package API for SG0 scaffolding."""

from rate.config import ConfigValidationError, load_and_validate_config
from rate.dataset import DatasetIndexError, build_dataset_index, write_dataset_index
from rate.manifest import write_run_manifest

__all__ = [
    "ConfigValidationError",
    "DatasetIndexError",
    "build_dataset_index",
    "load_and_validate_config",
    "write_dataset_index",
    "write_run_manifest",
]
