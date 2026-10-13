"""
Structured logging configuration using structlog.
"""

import logging
import sys
from typing import Any

import structlog


def suppress_provider_payload_logs() -> None:
    """Override SDK debug flags, including SDKs imported after app startup."""
    prefixes = (
        "openai",
        "anthropic",
        "groq",
        "httpx",
        "httpcore",
        "google",
        "elevenlabs",
    )
    for name in set(prefixes) | set(logging.Logger.manager.loggerDict):
        if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes):
            logging.getLogger(name).setLevel(logging.WARNING)


def configure_logging(debug: bool = False) -> None:
    """
    Configure structured logging for the application.

    Args:
        debug: Enable debug-level logging if True
    """
    # Set log level
    log_level = logging.DEBUG if debug else logging.INFO

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    # SDK request logs may contain prompts, credentials or provider bodies.
    suppress_provider_payload_logs()

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            (
                structlog.dev.ConsoleRenderer()
                if debug
                else structlog.processors.JSONRenderer()
            ),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> Any:
    """
    Get a structured logger instance.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Structured logger instance
    """
    return structlog.get_logger(name)
