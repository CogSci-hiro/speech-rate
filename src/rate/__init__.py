"""Public package API for SG0 scaffolding."""

from rate.config import ConfigValidationError, load_and_validate_config
from rate.dataset import DatasetIndexError, build_dataset_index, write_dataset_index
from rate.events import EventsTableError, build_events_table, write_events_outputs
from rate.manifest import write_run_manifest

__all__ = [
    "ConfigValidationError",
    "DatasetIndexError",
    "EventsTableError",
    "build_dataset_index",
    "build_events_table",
    "load_and_validate_config",
    "write_events_outputs",
    "write_dataset_index",
    "write_run_manifest",
]
