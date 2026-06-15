"""
Tests for JobWorker: success path, failure path, and upload cleanup (Property 21).
"""
from pathlib import Path

import pytest

from app.job_manager import COMPLETE, FAILED, JobManager
from app.worker import JobWorker


class FakeEngine:
    def __init__(self, fail: bool = False):
        self.fail = fail

    def process_audio(self, audio_path, progress_cb=None):
        if progress_cb:
            progress_cb(50, "halfway")
        if self.fail:
            raise RuntimeError("boom at /Users/secret/model.pth")
        return [(0.0, 1.0, "C:maj")], 1.0

    def save_annotations(self, annotations, output_path):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("0.0000 1.0000 C:maj\n")


def _setup(tmp_path, fail=False):
    jm = JobManager(tmp_path / "jobs", tmp_path / "results", max_concurrent=2)
    upload = tmp_path / "uploads" / "a.wav"
    upload.parent.mkdir(parents=True, exist_ok=True)
    upload.write_bytes(b"RIFF....WAVE")
    job_id = jm.create_job(str(upload), "a.wav")
    worker = JobWorker(jm, FakeEngine(fail=fail), tmp_path / "results", max_concurrent=1)
    return jm, worker, job_id, upload


@pytest.mark.unit
def test_worker_success_writes_results_and_cleans_upload(tmp_path):
    jm, worker, job_id, upload = _setup(tmp_path)
    worker.submit(job_id).result(timeout=10)

    job = jm.get_job_status(job_id)
    assert job["status"] == COMPLETE
    assert job["progress"] == 100
    assert Path(job["result_path"]).exists()
    assert not upload.exists()  # Property 21
    worker.shutdown(wait=True)


@pytest.mark.unit
def test_worker_failure_marks_failed_sanitized_and_cleans(tmp_path):
    jm, worker, job_id, upload = _setup(tmp_path, fail=True)
    worker.submit(job_id).result(timeout=10)

    job = jm.get_job_status(job_id)
    assert job["status"] == FAILED
    assert "/Users/secret" not in (job["error"] or "")  # Property 18
    assert not upload.exists()  # Property 21 (cleanup even on failure)
    worker.shutdown(wait=True)
