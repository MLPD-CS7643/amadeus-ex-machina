"""
End-to-end tests using the REAL trained artifacts.

Skipped automatically when the checkpoint/scaler/encoder are not present, so CI
without the artifacts still passes.
"""
import time
from io import BytesIO

import pytest

from app.inference import InferenceEngine
from tests.conftest import (
    ARTIFACTS_PRESENT,
    ENCODER_PATH,
    MODEL_PATH,
    SCALER_PATH,
    make_test_config,
)

pytestmark = pytest.mark.skipif(
    not ARTIFACTS_PRESENT, reason="trained model artifacts not present"
)


@pytest.mark.integration
def test_engine_end_to_end(chord_wav):
    engine = InferenceEngine()
    engine.load_artifacts(str(MODEL_PATH), str(SCALER_PATH), str(ENCODER_PATH))
    assert engine.is_loaded

    annotations, duration = engine.process_audio(str(chord_wav))
    assert duration > 0
    assert len(annotations) >= 1

    # Gap-free coverage from 0 to duration (Property 6).
    assert annotations[0][0] == 0.0
    assert annotations[-1][1] == pytest.approx(max(duration, annotations[-1][1]))
    for prev, nxt in zip(annotations, annotations[1:]):
        assert prev[1] == nxt[0]
    for start, end, label in annotations:
        assert start < end
        assert isinstance(label, str) and label


@pytest.mark.integration
def test_full_api_flow(tmp_path, wav_factory):
    from app.main import create_app

    cfg = make_test_config(tmp_path)  # real artifact paths
    app = create_app(cfg)
    assert app.engine.is_loaded
    client = app.test_client()

    wav = wav_factory("clip.wav", seconds=2.0)
    resp = client.post(
        "/api/upload",
        data={"file": (BytesIO(wav.read_bytes()), "clip.wav")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 200
    job_id = resp.get_json()["jobId"]

    status = {}
    for _ in range(100):
        status = client.get(f"/api/status/{job_id}").get_json()
        if status["status"] in ("complete", "failed"):
            break
        time.sleep(0.1)

    assert status["status"] == "complete", status
    results = client.get(f"/api/results/{job_id}").get_json()
    assert len(results["annotations"]) >= 1

    download = client.get(f"/api/download/{job_id}")
    assert download.status_code == 200
    assert b" " in download.data  # "start end label" lines

    # Uploaded audio should have been cleaned up (Property 21).
    upload_path = cfg.UPLOAD_DIR / job_id / "clip.wav"
    assert not upload_path.exists()

    app.worker.shutdown(wait=True)
