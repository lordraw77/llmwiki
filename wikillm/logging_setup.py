"""Logging configuration that keeps ``stdout`` pristine.

The MCP ``stdio`` transport uses ``stdout`` exclusively for newline-delimited
JSON-RPC 2.0 messages. **Any** stray byte written to ``stdout`` corrupts that
framing and breaks the client (e.g. Claude Desktop). Therefore this module
configures the root logger to emit exclusively to ``stderr`` and/or a log file,
and never to ``stdout``.

Call :func:`setup_logging` exactly once, as early as possible, in every entry
point (both the HTTP server and the stdio server).
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

_CONFIGURED = False

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Configure root logging to ``stderr`` and optionally a file.

    This function is idempotent: calling it more than once has no additional
    effect, which is convenient when several entry points import each other.

    Args:
        level: Logging level name (e.g. ``"DEBUG"``, ``"INFO"``). Invalid names
            fall back to ``INFO``.
        log_file: Optional path to a log file. When provided, logs are written
            there *in addition* to ``stderr``.

    Note:
        ``stdout`` is never used as a logging destination — see the module
        docstring for why this matters for the MCP stdio transport.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)

    root = logging.getLogger()
    root.setLevel(numeric_level)

    # Remove pre-existing handlers (e.g. installed by libraries on import) to
    # guarantee nothing slips onto stdout.
    for handler in list(root.handlers):
        root.removeHandler(handler)

    stderr_handler = logging.StreamHandler(stream=sys.stderr)
    stderr_handler.setFormatter(formatter)
    root.addHandler(stderr_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    # Tame noisy third-party loggers so application logs stay readable.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)

    _CONFIGURED = True
    logging.getLogger(__name__).debug("Logging configured (level=%s, file=%s)", level, log_file)


def get_logger(name: str) -> logging.Logger:
    """Return a module logger.

    A thin convenience wrapper around :func:`logging.getLogger` so call sites
    don't import :mod:`logging` directly.

    Args:
        name: Usually ``__name__`` of the calling module.

    Returns:
        A configured :class:`logging.Logger`.
    """
    return logging.getLogger(name)
