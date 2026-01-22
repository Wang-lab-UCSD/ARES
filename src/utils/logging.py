"""Structured logging setup for the pipeline."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add extra fields if present
        if hasattr(record, "extra_data"):
            log_data["data"] = record.extra_data

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


class PipelineLogger:
    """Logger wrapper with structured logging support."""

    def __init__(self, name: str, logger: logging.Logger):
        self.name = name
        self._logger = logger

    def _log(self, level: int, message: str, data: dict[str, Any] | None = None) -> None:
        record = self._logger.makeRecord(
            self.name, level, "", 0, message, (), None
        )
        if data:
            record.extra_data = data
        self._logger.handle(record)

    def debug(self, message: str, data: dict[str, Any] | None = None) -> None:
        self._log(logging.DEBUG, message, data)

    def info(self, message: str, data: dict[str, Any] | None = None) -> None:
        self._log(logging.INFO, message, data)

    def warning(self, message: str, data: dict[str, Any] | None = None) -> None:
        self._log(logging.WARNING, message, data)

    def error(self, message: str, data: dict[str, Any] | None = None) -> None:
        self._log(logging.ERROR, message, data)

    def critical(self, message: str, data: dict[str, Any] | None = None) -> None:
        self._log(logging.CRITICAL, message, data)


_loggers: dict[str, PipelineLogger] = {}
_log_dir: Path | None = None
_initialized: bool = False


def setup_logging(
    log_dir: str | Path = "logs",
    level: int = logging.INFO,
    console_output: bool = True,
) -> None:
    """Initialize logging for the pipeline.

    Args:
        log_dir: Directory to write log files
        level: Logging level
        console_output: Whether to also log to console
    """
    global _log_dir, _initialized

    _log_dir = Path(log_dir)
    _log_dir.mkdir(parents=True, exist_ok=True)

    # Create log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = _log_dir / f"pipeline_{timestamp}.jsonl"

    # Configure root logger
    root_logger = logging.getLogger("pipeline")
    root_logger.setLevel(level)
    root_logger.handlers.clear()

    # File handler (JSON format)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(JSONFormatter())
    root_logger.addHandler(file_handler)

    # Console handler (human-readable)
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
        )
        root_logger.addHandler(console_handler)

    _initialized = True

    # Log initialization
    logger = get_logger("setup")
    logger.info("Logging initialized", {"log_file": str(log_file)})


def get_logger(name: str) -> PipelineLogger:
    """Get a logger instance for the given name."""
    global _loggers, _initialized

    if not _initialized:
        setup_logging()

    if name not in _loggers:
        full_name = f"pipeline.{name}"
        _loggers[name] = PipelineLogger(full_name, logging.getLogger(full_name))

    return _loggers[name]
