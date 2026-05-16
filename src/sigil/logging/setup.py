"""Structured logging built on `structlog`.

Goals:
    - JSON output in production (parseable by Loki / Datadog / file scrapers).
    - Pretty colored output when developing in a TTY.
    - Context binding: every log entry from a worker carries its worker name
      and any active gesture-event ID, so timelines are easy to reconstruct.
    - Zero-cost when a log level is disabled — structlog wraps stdlib logging
      and inherits its level checks.

Usage:
    from sigil.logging import setup_logging, get_logger

    setup_logging(level="INFO", json=False)        # once, at process start
    log = get_logger(__name__)
    log.info("perception_started", fps=15)
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Literal

import structlog

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

_configured = False


def setup_logging(
    level: LogLevel = "INFO",
    *,
    json: bool | None = None,
    extra_context: dict[str, Any] | None = None,
) -> None:
    """Configure structlog + stdlib logging for the whole process.

    Idempotent: safe to call multiple times; subsequent calls are no-ops to
    avoid duplicating processors when test fixtures re-setup logging.

    The architecture: structlog wraps stdlib loggers (`LoggerFactory()`), and
    stdlib's StreamHandler uses structlog's `ProcessorFormatter` as its
    formatter. This means *both* structlog calls and foreign (third-party
    library) stdlib log calls go through the same processor chain and
    renderer. One coherent stream, JSON-parseable in prod.

    Args:
        level: Standard log level.
        json: If None, auto-detect — JSON when stderr is not a TTY, pretty
            when it is. Override explicitly in tests or daemon mode.
        extra_context: Process-global context vars bound to every record.
    """
    global _configured  # noqa: PLW0603 — module-level flag is the simplest answer
    if _configured:
        return

    if json is None:
        json = not sys.stderr.isatty()

    numeric_level = getattr(logging, level)

    # Processors applied to every record (both structlog-native and foreign).
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    renderer: structlog.types.Processor = (
        structlog.processors.JSONRenderer()
        if json
        else structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
    )

    # 1. Configure structlog. BoundLogger wraps stdlib loggers, so
    #    `add_logger_name` (which reads `.name` off the wrapped logger) works.
    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        context_class=dict,
        cache_logger_on_first_use=True,
    )

    # 2. Set up stdlib's root logger with a single StreamHandler that uses
    #    structlog's ProcessorFormatter as its formatter. Foreign logs
    #    (mediapipe, openwakeword, anything using stdlib `logging`) get the
    #    same shared_processors via `foreign_pre_chain`.
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    handler.setLevel(numeric_level)

    root_logger = logging.getLogger()
    root_logger.handlers[:] = [handler]  # replace, don't append
    root_logger.setLevel(numeric_level)

    if extra_context:
        structlog.contextvars.bind_contextvars(**extra_context)

    _configured = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Get a structlog logger.

    Pass `__name__` so log records carry the originating module — invaluable
    when grepping JSON output.
    """
    return structlog.get_logger(name) if name else structlog.get_logger()


def bind_context(**kwargs: Any) -> None:
    """Bind context vars visible to every log call on the current task/thread.

    Use sparingly — typically once per worker process or per gesture event.
    """
    structlog.contextvars.bind_contextvars(**kwargs)


def clear_context() -> None:
    """Clear bound context vars on the current task/thread."""
    structlog.contextvars.clear_contextvars()


__all__ = [
    "LogLevel",
    "bind_context",
    "clear_context",
    "get_logger",
    "setup_logging",
]
