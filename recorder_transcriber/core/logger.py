import json
import logging
import sys
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

from recorder_transcriber.core.settings import LoggingConfig

ROOT_LOGGER_NAME = "recorder_transcriber"


class JsonFormatter(logging.Formatter):

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

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

    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "app.log"

    root_logger = logging.getLogger(ROOT_LOGGER_NAME)
    root_logger.setLevel(config.level)

    root_logger.handlers.clear()

    formatter: logging.Formatter
    if config.json_output:
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(config.format)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(config.level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    file_handler = RotatingFileHandler(
        filename=log_file,
        maxBytes=config.rotate_max_bytes,
        backupCount=config.rotate_backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(config.level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    _configure_uvicorn_loggers(config.level, formatter)

    root_logger.propagate = False

    root_logger.info(
        "Logging configured: level=%s, json=%s, file=%s",
        config.level,
        config.json_output,
        log_file,
    )


def _configure_uvicorn_loggers(level: str, formatter: logging.Formatter) -> None:

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
        name: Logger name.

    Returns:
        A configured logger instance.

    Example:
        >>> logger = get_logger("services.example")
        >>> logger.info("Example started")
        # Logs as: Example.services.example
    """
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")
