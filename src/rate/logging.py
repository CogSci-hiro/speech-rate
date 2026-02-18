"""Logging configuration helpers."""

import logging
from pathlib import Path

from rate.paths import ensure_parent_dir

DEFAULT_LOG_FILE_NAME = "rate.log"


def configure_logging(logs_dir: Path | str, level: int = logging.INFO) -> Path:
    """Configure root logging with console + file handlers."""
    logs_path = Path(logs_dir)
    log_file = logs_path / DEFAULT_LOG_FILE_NAME
    ensure_parent_dir(log_file)

    root = logging.getLogger()
    root.setLevel(level)

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    has_stream = any(isinstance(handler, logging.StreamHandler) for handler in root.handlers)
    if not has_stream:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        root.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    return log_file
