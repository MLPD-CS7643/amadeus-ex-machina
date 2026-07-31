"""
Core exception types and message sanitization.

These are framework-agnostic: the inference engine and feature extractor raise
:class:`ProcessingError`, and both the API and training code reuse
:func:`sanitize_message` so internal paths / tracebacks never leak to clients
(design Property 18).
"""

import re


class AmadeusError(Exception):
    """Base class for all amadeus-core domain errors."""


class ProcessingError(AmadeusError):
    """Raised when the inference pipeline fails at a specific stage."""

    def __init__(
        self, message: str, stage: str, original_error: Exception | None = None
    ):
        self.message = message
        self.stage = stage  # e.g. 'chromagram_extraction', 'model_inference'
        self.original_error = original_error
        super().__init__(self.message)


# Patterns that indicate an internal detail we never want to expose to a client.
_PATH_PATTERN = re.compile(r"(/[\w.\-]+)+/?|[A-Za-z]:\\[\\\w.\-]+")
_NEWLINE_PATTERN = re.compile(r"\s*\n\s*")


def sanitize_message(message: str) -> str:
    """
    Strip filesystem paths and collapse multi-line/stack-trace text so error
    messages returned to clients do not expose internal details.
    """
    if not message:
        return "An error occurred."
    # Collapse any multi-line content (e.g. tracebacks) to a single line.
    cleaned = _NEWLINE_PATTERN.sub(" ", message).strip()
    # Redact anything that looks like an absolute path (POSIX or Windows).
    cleaned = _PATH_PATTERN.sub("<path>", cleaned)
    return cleaned
