"""API endpoint tests (no real model; mocked worker for determinism)."""

import pytest
from fastapi.testclient import TestClient

from amadeus_api.jobs import COMPLETE, PROCESSING
from amadeus_api.ratelimit import RateLimiter
from tests.conftest import build_app

# Minimal valid WAV header (RIFF....WAVE) so magic-byte verification passes.
VALID_WAV = b"RIFF" + b"\x00\x00\x00\x00" + b"WAVE" + b"\x00" * 64


def _complete_job(app, filename="song.wav"):
    jm = app.state.job_manager
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
        assert resp.json()["status"] == "ok"


@pytest.mark.integration
class TestUpload:
    def test_upload_success(self, client, app):
        resp = client.post(
            "/api/upload", files={"file": ("test.wav", VALID_WAV, "audio/wav")}
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "queued"
        assert body["jobId"]
        app.state.worker.submit.assert_called_once_with(body["jobId"])

    def test_upload_rejects_bad_format(self, client):
        resp = client.post(
            "/api/upload", files={"file": ("notes.txt", b"hello", "text/plain")}
        )
        assert resp.status_code == 400
        assert resp.json()["error"]["code"] == "INVALID_FORMAT"

    def test_upload_rejects_empty_file(self, client):
        resp = client.post(
            "/api/upload", files={"file": ("empty.wav", b"", "audio/wav")}
        )
        assert resp.status_code == 400
        assert resp.json()["error"]["code"] == "EMPTY_FILE"

    def test_upload_no_file(self, client):
        resp = client.post("/api/upload")
        assert resp.status_code == 400
        assert resp.json()["error"]["code"] == "NO_FILE"

    def test_upload_oversize_returns_413(self, tmp_path):
        app = build_app(tmp_path)
        app.state.file_handler.max_file_size = 1024  # reject > 1KB
        client = TestClient(app)
        resp = client.post(
            "/api/upload", files={"file": ("big.wav", b"\x00" * 4096, "audio/wav")}
        )
        assert resp.status_code == 413
        assert resp.json()["error"]["code"] == "FILE_TOO_LARGE"


@pytest.mark.unit
class TestStatus:
    def test_status_not_found(self, client):
        resp = client.get("/api/status/unknown-id")
        assert resp.status_code == 404
        assert resp.json()["error"]["code"] == "JOB_NOT_FOUND"

    def test_status_reports_progress(self, client, app):
        jm = app.state.job_manager
        job_id = jm.create_job("/tmp/up.wav", "song.wav")
        jm.update_job_status(job_id, status=PROCESSING, progress=42, message="Working")
        resp = client.get(f"/api/status/{job_id}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "processing"
        assert body["progress"] == 42


@pytest.mark.integration
class TestResultsAndDownload:
    def test_results_returns_annotations(self, client, app):
        job_id = _complete_job(app)
        resp = client.get(f"/api/results/{job_id}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["duration"] == 2.0
        assert len(body["annotations"]) == 2
        assert body["annotations"][0] == {
            "startTime": 0.0,
            "endTime": 1.0,
            "chordLabel": "C:maj",
        }

    def test_results_not_ready(self, client, app):
        job_id = app.state.job_manager.create_job("/tmp/up.wav", "song.wav")  # queued
        resp = client.get(f"/api/results/{job_id}")
        assert resp.status_code == 404

    def test_download_filename_pattern(self, client, app):
        job_id = _complete_job(app, filename="My_Song.wav")
        resp = client.get(f"/api/download/{job_id}")
        assert resp.status_code == 200
        assert "My_Song_chords.lab" in resp.headers["content-disposition"]
        assert b"C:maj" in resp.content

    def test_download_not_found(self, client):
        resp = client.get("/api/download/nope")
        assert resp.status_code == 404


@pytest.mark.security
class TestCorsRateLimitAndSanitization:
    def test_cors_header_present(self, client):
        resp = client.get("/api/health", headers={"Origin": "http://localhost:3000"})
        assert (
            resp.headers.get("access-control-allow-origin") == "http://localhost:3000"
        )

    def test_rate_limit_returns_429(self, tmp_path):
        app = build_app(tmp_path)
        app.state.rate_limiter = RateLimiter(2)
        client = TestClient(app)
        assert client.get("/api/status/x").status_code == 404
        assert client.get("/api/status/x").status_code == 404
        resp = client.get("/api/status/x")
        assert resp.status_code == 429
        assert resp.json()["error"]["code"] == "RATE_LIMITED"

    def test_error_message_sanitized(self, app):
        def boom(_job_id):
            raise Exception("failed at /Users/secret/path/model.pt boom")

        app.state.job_manager.get_job_status = boom
        # raise_server_exceptions=False so the catch-all handler's 500 is returned.
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/api/status/anything")
        assert resp.status_code == 500
        message = resp.json()["error"]["message"]
        assert "/Users/secret" not in message
        assert "secret" not in message
