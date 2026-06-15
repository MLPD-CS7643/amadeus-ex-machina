"""
API endpoint tests.

These build an app with bogus artifact paths (so no real model loads) and a mocked
worker, keeping route behavior fast and deterministic.
"""
from io import BytesIO
from unittest import mock

import pytest

from app.job_manager import COMPLETE, PROCESSING
from app.routes import RateLimiter
from tests.conftest import make_test_config

# Minimal valid WAV header (RIFF....WAVE) so magic-byte verification passes.
VALID_WAV = b"RIFF" + b"\x00\x00\x00\x00" + b"WAVE" + b"\x00" * 64


def build_app(tmp_path, max_file_size=50 * 1024 * 1024):
    from app.main import create_app

    cfg = make_test_config(tmp_path, max_file_size=max_file_size)
    # Point artifacts at nonexistent paths -> engine load is a fast no-op.
    cfg.MODEL_PATH = str(tmp_path / "no_model.pth")
    cfg.SCALER_PATH = str(tmp_path / "no_scaler.pkl")
    cfg.ENCODER_PATH = str(tmp_path / "no_encoder.pkl")
    app = create_app(cfg)
    app.worker = mock.Mock()  # don't actually run inference
    return app


@pytest.fixture
def app(tmp_path):
    return build_app(tmp_path)


@pytest.fixture
def client(app):
    return app.test_client()


def _complete_job(app, filename="song.wav"):
    jm = app.job_manager
    job_id = jm.create_job("/tmp/up.wav", filename)
    jm.update_job_status(job_id, status=PROCESSING)
    result_dir = jm.results_dir / job_id
    result_dir.mkdir(parents=True, exist_ok=True)
    lab = result_dir / "annotations.lab"
    lab.write_text("0.0000 1.0000 C:maj\n1.0000 2.0000 A:min\n")
    jm.update_job_status(
        job_id, status=COMPLETE, result_path=str(lab), audio_duration=2.0
    )
    return job_id


@pytest.mark.unit
class TestHealth:
    def test_health_ok(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        assert resp.get_json()["status"] == "ok"


@pytest.mark.integration
class TestUpload:
    def test_upload_success(self, client, app):
        resp = client.post(
            "/api/upload",
            data={"file": (BytesIO(VALID_WAV), "test.wav")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["status"] == "queued"
        assert body["jobId"]
        app.worker.submit.assert_called_once_with(body["jobId"])

    def test_upload_rejects_bad_format(self, client):
        # Property 15: 400 for invalid input
        resp = client.post(
            "/api/upload",
            data={"file": (BytesIO(b"hello"), "notes.txt")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400
        assert resp.get_json()["error"]["code"] == "INVALID_FORMAT"

    def test_upload_rejects_empty_file(self, client):
        resp = client.post(
            "/api/upload",
            data={"file": (BytesIO(b""), "empty.wav")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400
        assert resp.get_json()["error"]["code"] == "EMPTY_FILE"

    def test_upload_no_file(self, client):
        resp = client.post(
            "/api/upload", data={}, content_type="multipart/form-data"
        )
        assert resp.status_code == 400
        assert resp.get_json()["error"]["code"] == "NO_FILE"

    def test_upload_oversize_returns_413(self, tmp_path):
        # Property 15: 413 for oversized files
        app = build_app(tmp_path)
        app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # let Flask accept body
        app.file_handler.max_file_size = 1024  # but handler rejects > 1KB
        client = app.test_client()
        resp = client.post(
            "/api/upload",
            data={"file": (BytesIO(b"\x00" * 4096), "big.wav")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 413
        assert resp.get_json()["error"]["code"] == "FILE_TOO_LARGE"


@pytest.mark.unit
class TestStatus:
    def test_status_not_found(self, client):
        # Property 15: 404 for missing resource
        resp = client.get("/api/status/unknown-id")
        assert resp.status_code == 404
        assert resp.get_json()["error"]["code"] == "JOB_NOT_FOUND"

    def test_status_reports_progress(self, client, app):
        jm = app.job_manager
        job_id = jm.create_job("/tmp/up.wav", "song.wav")
        jm.update_job_status(job_id, status=PROCESSING, progress=42, message="Working")
        resp = client.get(f"/api/status/{job_id}")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["status"] == "processing"
        assert body["progress"] == 42


@pytest.mark.integration
class TestResultsAndDownload:
    def test_results_returns_annotations(self, client, app):
        job_id = _complete_job(app)
        resp = client.get(f"/api/results/{job_id}")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["duration"] == 2.0
        assert len(body["annotations"]) == 2
        assert body["annotations"][0] == {
            "startTime": 0.0,
            "endTime": 1.0,
            "chordLabel": "C:maj",
        }

    def test_results_not_ready(self, client, app):
        job_id = app.job_manager.create_job("/tmp/up.wav", "song.wav")  # queued
        resp = client.get(f"/api/results/{job_id}")
        assert resp.status_code == 404

    def test_download_filename_pattern(self, client, app):
        # Property 10: Output Filename Pattern
        job_id = _complete_job(app, filename="My_Song.wav")
        resp = client.get(f"/api/download/{job_id}")
        assert resp.status_code == 200
        assert "My_Song_chords.lab" in resp.headers["Content-Disposition"]
        assert b"C:maj" in resp.data

    def test_download_not_found(self, client):
        resp = client.get("/api/download/nope")
        assert resp.status_code == 404


@pytest.mark.security
class TestCorsRateLimitAndSanitization:
    def test_cors_header_present(self, client):
        # Property 17: CORS Header Presence
        resp = client.get(
            "/api/health", headers={"Origin": "http://localhost:3000"}
        )
        assert resp.headers.get("Access-Control-Allow-Origin") == "http://localhost:3000"

    def test_rate_limit_returns_429(self, tmp_path):
        # Requirement 10.3: rate limiting
        app = build_app(tmp_path)
        app.rate_limiter = RateLimiter(2)
        client = app.test_client()
        assert client.get("/api/status/x").status_code == 404
        assert client.get("/api/status/x").status_code == 404
        resp = client.get("/api/status/x")
        assert resp.status_code == 429
        assert resp.get_json()["error"]["code"] == "RATE_LIMITED"

    def test_error_message_sanitized(self, client, app):
        # Property 18: Error Message Sanitization
        def boom(_job_id):
            raise Exception("failed at /Users/secret/path/model.pth boom")

        app.job_manager.get_job_status = boom
        resp = client.get("/api/status/anything")
        assert resp.status_code == 500
        message = resp.get_json()["error"]["message"]
        assert "/Users/secret" not in message
        assert "secret" not in message
