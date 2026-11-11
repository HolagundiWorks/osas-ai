"""Logging helpers that redact credential-shaped values."""

from __future__ import annotations

import logging
import os
import re

_SECRET_PATTERNS = (
    re.compile(r"(?i)(hf_token|huggingface_token|api[_-]?key|authorization|bearer)\s*[=:]\s*\S+"),
    re.compile(r"hf_[A-Za-z0-9]{8,}"),
    re.compile(r"sk-[A-Za-z0-9]{8,}"),
)


class _RedactFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact(str(record.msg))
        if record.args:
            record.args = tuple(redact(str(arg)) for arg in record.args)
        return True


def redact(text: str) -> str:
    """Replace credential-shaped substrings and known secret env values."""
    redacted = text
    for key, value in os.environ.items():
        if value and len(value) >= 8 and _looks_secret(key):
            redacted = redacted.replace(value, f"[REDACTED:{key}]")
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def _looks_secret(name: str) -> bool:
    upper = name.upper()
    return any(part in upper for part in ("TOKEN", "SECRET", "PASSWORD", "API_KEY", "CREDENTIAL"))


def configure_logging(level: str = "INFO", name: str = "myna_ai") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level.upper())
    logger.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    handler.addFilter(_RedactFilter())
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def get_logger(child: str) -> logging.Logger:
    return logging.getLogger(f"myna_ai.{child}")
