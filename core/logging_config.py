"""Structured logging setup with rotating file handlers.

Provides JSON-formatted file logs and simple console output.
Three log files:
  - app.log     : all logs
  - trades.log  : paper trading engine + algorithm logs only
  - errors.log  : WARNING and above
"""

import json
import logging
import logging.handlers
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path


class JsonFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects.

    Output keys: timestamp, level, logger, message, plus any extra fields
    attached to the LogRecord. Exception info is included under the
    ``exception`` key when present.
    """

    def format(self, record: logging.LogRecord) -> str:
        entry: dict = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Attach any extra fields that are not part of the standard LogRecord
        standard_attrs = {
            "name", "msg", "args", "created", "relativeCreated", "exc_info",
            "exc_text", "stack_info", "lineno", "funcName", "pathname",
            "filename", "module", "levelno", "levelname", "msecs",
            "processName", "process", "threadName", "thread", "taskName",
            "message",
        }
        for key, value in record.__dict__.items():
            if key not in standard_attrs and not key.startswith("_"):
                try:
                    json.dumps(value)  # ensure serializable
                    entry[key] = value
                except (TypeError, ValueError):
                    entry[key] = str(value)

        if record.exc_info and record.exc_info[0] is not None:
            entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info),
            }

        return json.dumps(entry, default=str)


class _TradeLogFilter(logging.Filter):
    """Passes only records from paper trading engine and algorithm loggers."""

    _prefixes = ("core.paper_trading_engine", "algorithms.")

    def filter(self, record: logging.LogRecord) -> bool:
        return any(record.name.startswith(p) for p in self._prefixes)


def setup(log_dir: str = "data/logs", level: str = "INFO") -> logging.Logger:
    """Configure root logger with rotating file handlers and console output.

    Parameters
    ----------
    log_dir:
        Directory for log files. Created if it does not exist.
    level:
        Root logger level (e.g. ``"DEBUG"``, ``"INFO"``, ``"WARNING"``).

    Returns
    -------
    logging.Logger
        The root logger, fully configured.
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Avoid adding duplicate handlers on repeated calls (e.g. Streamlit reruns)
    if root_logger.handlers:
        return root_logger

    json_fmt = JsonFormatter()
    console_fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    max_bytes = 10 * 1024 * 1024  # 10 MB
    backup_count = 5

    # -- app.log: everything ------------------------------------------------
    app_handler = logging.handlers.RotatingFileHandler(
        filename=os.path.join(log_dir, "app.log"),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    app_handler.setLevel(logging.DEBUG)
    app_handler.setFormatter(json_fmt)

    # -- trades.log: paper trading engine + algorithms only -----------------
    trades_handler = logging.handlers.RotatingFileHandler(
        filename=os.path.join(log_dir, "trades.log"),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    trades_handler.setLevel(logging.DEBUG)
    trades_handler.setFormatter(json_fmt)
    trades_handler.addFilter(_TradeLogFilter())

    # -- errors.log: WARNING and above --------------------------------------
    errors_handler = logging.handlers.RotatingFileHandler(
        filename=os.path.join(log_dir, "errors.log"),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    errors_handler.setLevel(logging.WARNING)
    errors_handler.setFormatter(json_fmt)

    # -- console: simple human-readable output ------------------------------
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(console_fmt)

    root_logger.addHandler(app_handler)
    root_logger.addHandler(trades_handler)
    root_logger.addHandler(errors_handler)
    root_logger.addHandler(console_handler)

    return root_logger
