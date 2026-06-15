"""
Central exception types and error-response helpers.

All user-facing error payloads use a single shape (per the design doc, section
"Error Handling") and are run through :func:`sanitize_message` so that internal
filesystem paths, stack traces, or other implementation details never leak to the
client (design Property 18 / Requirements 8.3, 10.6).
"""
import re
from typing import Any, Optional, Tuple


class FileValidationError(Exception):
    """Raised when an uploaded file fails validation (format, size, content)."""

    def __init__(self, message: str, error_code: str):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)


class ProcessingError(Exception):
    """Raised when the inference pipeline fails at a specific stage."""

    def __init__(self, message: str, stage: str, original_error: Optional[Exception] = None):
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


def error_response(
    code: str, message: str, http_status: int, details: Any = None
) -> Tuple[dict, int]:
    """
    Build the standard error body and HTTP status.

    Returns a ``(body, status)`` tuple ready to hand to ``jsonify``/Flask.
    """
    return (
        {
            "error": {
                "code": code,
                "message": sanitize_message(message),
                "details": details,
            }
        },
        http_status,
    )
