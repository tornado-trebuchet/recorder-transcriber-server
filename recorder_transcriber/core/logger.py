import json
import logging
import sys
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

from recorder_transcriber.core.config import LoggingConfig

# Root logger name for the application
ROOT_LOGGER_NAME = "recorder_transcriber"


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging output."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Include exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Include any extra fields
        for key, value in record.__dict__.items():
            if key not in {
                "name",
                "msg",
                "args",
                "created",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "stack_info",
                "exc_info",
                "exc_text",
                "message",
                "taskName",
            }:
                log_data[key] = value

        return json.dumps(log_data, default=str)


def setup_logging(config: LoggingConfig, log_dir: Path) -> None:
    """
    Configure application logging.

    Args:
        config: Logging configuration from app config.
        log_dir: Directory for log files (typically paths.fs_dir from config).
    """
    # Ensure log directory exists
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "app.log"

    # Get root logger for the application
    root_logger = logging.getLogger(ROOT_LOGGER_NAME)
    root_logger.setLevel(config.level)

    # Clear any existing handlers
    root_logger.handlers.clear()

    # Create formatter based on config
    formatter: logging.Formatter
    if config.json_output:
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(config.format)

    # Console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(config.level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Rotating file handler
    file_handler = RotatingFileHandler(
        filename=log_file,
        maxBytes=config.rotate_max_bytes,
        backupCount=config.rotate_backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(config.level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Configure uvicorn loggers to use our handlers
    _configure_uvicorn_loggers(config.level, formatter)

    # Prevent propagation to root logger to avoid duplicate logs
    root_logger.propagate = False

    root_logger.info(
        "Logging configured: level=%s, json=%s, file=%s",
        config.level,
        config.json_output,
        log_file,
    )


def _configure_uvicorn_loggers(level: str, formatter: logging.Formatter) -> None:
    """Configure uvicorn's loggers to match our logging setup."""
    uvicorn_loggers = ["uvicorn", "uvicorn.error", "uvicorn.access"]

    for logger_name in uvicorn_loggers:
        logger = logging.getLogger(logger_name)
        logger.handlers.clear()

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the application namespace.

    Args:
        name: Logger name (will be prefixed with 'recorder_transcriber.').

    Returns:
        A configured logger instance.

    Example:
        >>> logger = get_logger("services.transcription")
        >>> logger.info("Transcription started")
        # Logs as: recorder_transcriber.services.transcription
    """
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")
