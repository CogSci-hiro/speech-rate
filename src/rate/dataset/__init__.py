"""Dataset indexing package."""

from rate.dataset.index import DatasetIndexError, build_dataset_index, write_dataset_index

__all__ = [
    "DatasetIndexError",
    "build_dataset_index",
    "write_dataset_index",
]
