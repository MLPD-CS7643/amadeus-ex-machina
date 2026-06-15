"""
End-to-end tests using the REAL trained artifacts.

Skipped automatically when the artifact bundle is not present.
"""

import time

import pytest
from fastapi.testclient import TestClient

from tests.conftest import ARTIFACTS_PRESENT, build_app

pytestmark = pytest.mark.skipif(
    not ARTIFACTS_PRESENT, reason="trained model artifacts not present"
)


@pytest.mark.integration
def test_full_api_flow(tmp_path, wav_factory):
    app = build_app(tmp_path, mock_worker=False, real_artifacts=True)
    assert app.state.engine.is_loaded
    client = TestClient(app)

    wav = wav_factory("clip.wav", seconds=2.0)
    with open(wav, "rb") as fh:
        resp = client.post(
            "/api/upload", files={"file": ("clip.wav", fh.read(), "audio/wav")}
        )
    assert resp.status_code == 200
    job_id = resp.json()["jobId"]

    status = {}
    for _ in range(100):
        status = client.get(f"/api/status/{job_id}").json()
        if status["status"] in ("complete", "failed"):
            break
        time.sleep(0.1)

    assert status["status"] == "complete", status
    results = client.get(f"/api/results/{job_id}").json()
    assert len(results["annotations"]) >= 1

    download = client.get(f"/api/download/{job_id}")
    assert download.status_code == 200
    assert b" " in download.content  # "start end label" lines

    # Uploaded audio should have been cleaned up.
    upload_path = tmp_path / "uploads" / job_id / "clip.wav"
    assert not upload_path.exists()

    app.state.worker.shutdown(wait=True)
