"""
Web-layer error types + FastAPI exception handlers.

The sanitized error body shape ``{"error": {"code", "message", "details"}}`` is shared by
every failure path (design Property 18); messages are run through
``amadeus_core.errors.sanitize_message`` so internal paths never leak.
"""

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from amadeus_core.errors import ProcessingError, sanitize_message

logger = logging.getLogger(__name__)

# FileValidationError codes that map to HTTP 413 rather than 400.
OVERSIZE_CODE = "FILE_TOO_LARGE"


class FileValidationError(Exception):
    """Raised when an uploaded file fails validation (format, size, content)."""

    def __init__(self, message: str, error_code: str):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)


def error_body(code: str, message: str, details: Any = None) -> dict:
    """Build the standard sanitized error body."""
    return {
        "error": {
            "code": code,
            "message": sanitize_message(message),
            "details": details,
        }
    }


def error_response(
    code: str, message: str, http_status: int, details: Any = None
) -> JSONResponse:
    return JSONResponse(error_body(code, message, details), status_code=http_status)


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(FileValidationError)
    async def _handle_validation(
        _request: Request, exc: FileValidationError
    ) -> JSONResponse:
        http_status = 413 if exc.error_code == OVERSIZE_CODE else 400
        return error_response(exc.error_code, exc.message, http_status)

    @app.exception_handler(ProcessingError)
    async def _handle_processing(
        _request: Request, exc: ProcessingError
    ) -> JSONResponse:
        return error_response("PROCESSING_ERROR", exc.message, 500)

    @app.exception_handler(StarletteHTTPException)
    async def _handle_http(
        _request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        # Preserve FastAPI's status codes but wrap the body in our error shape.
        if exc.status_code == 404:
            return error_response("NOT_FOUND", "Resource not found.", 404)
        return error_response("HTTP_ERROR", str(exc.detail), exc.status_code)

    @app.exception_handler(Exception)
    async def _handle_unexpected(_request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error")
        return error_response("INTERNAL_ERROR", str(exc), 500)
