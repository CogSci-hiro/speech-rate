"""Public package API for SG0 scaffolding."""

from rate.config import ConfigValidationError, load_and_validate_config
from rate.manifest import write_run_manifest

__all__ = ["ConfigValidationError", "load_and_validate_config", "write_run_manifest"]
