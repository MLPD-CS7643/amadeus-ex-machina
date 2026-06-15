"""
REST API endpoints for chord recognition.

Routes (blueprint prefix ``/api``):
    POST /upload            -> queue an audio file for processing
    GET  /status/<job_id>   -> poll job status/progress
    GET  /results/<job_id>  -> chord annotations as JSON
    GET  /download/<job_id> -> chord annotations as a .lab attachment
    GET  /health            -> liveness + engine-loaded flag

Error mapping (design Property 15): invalid input -> 400, oversize -> 413,
missing resource -> 404, server failure -> 500, rate limited -> 429. All error
bodies use the shared sanitized shape (Property 18).
"""
import time
import uuid
from collections import defaultdict, deque
from pathlib import Path
from threading import Lock

from flask import Blueprint, current_app, jsonify, request, send_file
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

from app.errors import FileValidationError, ProcessingError, error_response
from app.job_manager import COMPLETE

api_bp = Blueprint("api", __name__, url_prefix="/api")

# FileValidationError codes that should map to HTTP 413 rather than 400.
_OVERSIZE_CODE = "FILE_TOO_LARGE"


# --------------------------------------------------------------------------- #
# Rate limiting (in-memory sliding window, per client IP)
# --------------------------------------------------------------------------- #
class RateLimiter:
    def __init__(self, max_per_minute: int):
        self.max_per_minute = max_per_minute
        self._hits = defaultdict(deque)
        self._lock = Lock()

    def allow(self, key: str) -> bool:
        if self.max_per_minute <= 0:
            return True
        now = time.time()
        window_start = now - 60.0
        with self._lock:
            hits = self._hits[key]
            while hits and hits[0] < window_start:
                hits.popleft()
            if len(hits) >= self.max_per_minute:
                return False
            hits.append(now)
            return True


def _get_limiter() -> RateLimiter:
    """Per-app rate limiter, created lazily and cached on the app object."""
    limiter = getattr(current_app, "rate_limiter", None)
    if limiter is None:
        limiter = RateLimiter(current_app.config.get("RATE_LIMIT_PER_MIN", 10))
        current_app.rate_limiter = limiter
    return limiter


@api_bp.before_request
def _enforce_rate_limit():
    # Health checks are exempt.
    if request.endpoint == "api.health":
        return None
    client = request.remote_addr or "unknown"
    if not _get_limiter().allow(client):
        body, status = error_response(
            "RATE_LIMITED", "Too many requests. Please slow down.", 429
        )
        return jsonify(body), status
    return None


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _validate_upload(file) -> None:
    """Precise pre-save validation so error codes map to the right HTTP status."""
    handler = current_app.file_handler
    if file is None or not file.filename:
        raise FileValidationError("No file was provided.", "NO_FILE")

    ext = Path(file.filename).suffix.lower()
    if ext not in handler.allowed_extensions:
        allowed = ", ".join(sorted(handler.allowed_extensions))
        raise FileValidationError(
            f"Unsupported file format. Supported formats: {allowed}.", "INVALID_FORMAT"
        )

    file.seek(0, 2)  # SEEK_END
    size = file.tell()
    file.seek(0)
    if size == 0:
        raise FileValidationError("Uploaded file is empty.", "EMPTY_FILE")
    if size > handler.max_file_size:
        raise FileValidationError(
            "File exceeds the maximum allowed size.", _OVERSIZE_CODE
        )


def _parse_lab(path: Path):
    """Read a .lab file into annotation dicts."""
    annotations = []
    with open(path, "r") as fin:
        for line in fin:
            parts = line.strip().split(maxsplit=2)
            if len(parts) < 3:
                continue
            annotations.append(
                {
                    "startTime": float(parts[0]),
                    "endTime": float(parts[1]),
                    "chordLabel": parts[2],
                }
            )
    return annotations


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@api_bp.route("/health", methods=["GET"])
def health():
    return jsonify(
        {"status": "ok", "engineLoaded": bool(current_app.engine.is_loaded)}
    ), 200


@api_bp.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("file")
    _validate_upload(file)

    job_id = str(uuid.uuid4())
    # save_upload re-validates and verifies content (magic bytes) -> INVALID_CONTENT.
    file_path, sanitized_name = current_app.file_handler.save_upload(file, job_id)
    current_app.job_manager.create_job(file_path, sanitized_name, job_id=job_id)
    current_app.worker.submit(job_id)

    return jsonify(
        {"jobId": job_id, "status": "queued", "message": "File uploaded successfully"}
    ), 200


@api_bp.route("/status/<job_id>", methods=["GET"])
def status(job_id):
    job = current_app.job_manager.get_job_status(job_id)
    if job is None:
        body, code = error_response("JOB_NOT_FOUND", "Job not found.", 404)
        return jsonify(body), code

    estimated_total = job.get("estimated_time")
    progress = job.get("progress", 0)
    remaining = None
    if estimated_total is not None:
        remaining = round(estimated_total * max(0.0, 1 - progress / 100.0), 2)

    return jsonify(
        {
            "jobId": job_id,
            "status": job.get("status"),
            "progress": progress,
            "estimatedTimeRemaining": remaining,
            "message": job.get("message", ""),
        }
    ), 200


@api_bp.route("/results/<job_id>", methods=["GET"])
def results(job_id):
    job = current_app.job_manager.get_job_status(job_id)
    if job is None or job.get("status") != COMPLETE:
        body, code = error_response(
            "RESULTS_NOT_READY", "Results not found or not ready.", 404
        )
        return jsonify(body), code

    result_path = job.get("result_path")
    if not result_path or not Path(result_path).exists():
        body, code = error_response("RESULTS_NOT_READY", "Results not available.", 404)
        return jsonify(body), code

    annotations = _parse_lab(Path(result_path))
    return jsonify(
        {
            "jobId": job_id,
            "filename": job.get("filename"),
            "duration": job.get("audio_duration"),
            "annotations": annotations,
        }
    ), 200


@api_bp.route("/download/<job_id>", methods=["GET"])
def download(job_id):
    job = current_app.job_manager.get_job_status(job_id)
    if job is None or job.get("status") != COMPLETE:
        body, code = error_response(
            "RESULTS_NOT_READY", "Results not found or not ready.", 404
        )
        return jsonify(body), code

    result_path = job.get("result_path")
    if not result_path or not Path(result_path).exists():
        body, code = error_response("RESULTS_NOT_READY", "Results not available.", 404)
        return jsonify(body), code

    stem = Path(job.get("filename") or "audio").stem
    download_name = f"{stem}_chords.lab"
    return send_file(
        result_path,
        mimetype="text/plain",
        as_attachment=True,
        download_name=download_name,
    )


# --------------------------------------------------------------------------- #
# Error handlers (registered on the app in main.create_app)
# --------------------------------------------------------------------------- #
def register_error_handlers(app):
    @app.errorhandler(FileValidationError)
    def _handle_validation(err: FileValidationError):
        http_status = 413 if err.error_code == _OVERSIZE_CODE else 400
        body, code = error_response(err.error_code, err.message, http_status)
        return jsonify(body), code

    @app.errorhandler(RequestEntityTooLarge)
    def _handle_too_large(_err):
        body, code = error_response(
            _OVERSIZE_CODE, "File exceeds the maximum allowed size.", 413
        )
        return jsonify(body), code

    @app.errorhandler(ProcessingError)
    def _handle_processing(err: ProcessingError):
        body, code = error_response("PROCESSING_ERROR", err.message, 500)
        return jsonify(body), code

    @app.errorhandler(404)
    def _handle_404(_err):
        body, code = error_response("NOT_FOUND", "Resource not found.", 404)
        return jsonify(body), code

    @app.errorhandler(Exception)
    def _handle_unexpected(err: Exception):
        # Let Flask handle HTTP exceptions that have their own (or default) handling.
        if isinstance(err, HTTPException):
            return err
        app.logger.exception("Unhandled error")
        body, code = error_response(
            "INTERNAL_ERROR", "An internal error occurred.", 500
        )
        return jsonify(body), code
