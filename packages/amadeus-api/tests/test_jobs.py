"""Tests for JobManager: state machine, id uniqueness, estimation, cleanup."""

import json
import time

import pytest

from amadeus_api.jobs import COMPLETE, FAILED, PROCESSING, QUEUED, JobManager


@pytest.fixture
def manager(tmp_path):
    return JobManager(
        jobs_dir=tmp_path / "jobs", results_dir=tmp_path / "results", max_concurrent=5
    )


@pytest.mark.unit
class TestLifecycle:
    def test_create_job_is_queued(self, manager):
        job_id = manager.create_job("/tmp/x.wav", "x.wav")
        job = manager.get_job_status(job_id)
        assert job["status"] == QUEUED
        assert job["progress"] == 0
        assert job["filename"] == "x.wav"

    def test_valid_transitions(self, manager):
        job_id = manager.create_job("/tmp/x.wav", "x.wav")
        manager.update_job_status(job_id, status=PROCESSING, progress=50)
        assert manager.get_job_status(job_id)["status"] == PROCESSING
        manager.update_job_status(job_id, status=COMPLETE, progress=100)
        assert manager.get_job_status(job_id)["status"] == COMPLETE

    def test_terminal_state_is_final(self, manager):
        job_id = manager.create_job("/tmp/x.wav", "x.wav")
        manager.update_job_status(job_id, status=PROCESSING)
        manager.update_job_status(job_id, status=COMPLETE)
        with pytest.raises(ValueError):
            manager.update_job_status(job_id, status=PROCESSING)
        with pytest.raises(ValueError):
            manager.update_job_status(job_id, status=FAILED)

    def test_illegal_skip_transition_rejected(self, manager):
        job_id = manager.create_job("/tmp/x.wav", "x.wav")
        with pytest.raises(ValueError):
            manager.update_job_status(job_id, status=COMPLETE)  # queued -> complete

    def test_update_missing_job_raises(self, manager):
        with pytest.raises(KeyError):
            manager.update_job_status("does-not-exist", status=PROCESSING)

    def test_get_missing_job_returns_none(self, manager):
        assert manager.get_job_status("nope") is None


@pytest.mark.unit
class TestUniquenessAndCounts:
    def test_job_ids_are_unique(self, manager):
        ids = {manager.create_job("/tmp/x.wav", "x.wav") for _ in range(500)}
        assert len(ids) == 500

    def test_active_job_count(self, manager):
        a = manager.create_job("/tmp/a.wav", "a.wav")
        b = manager.create_job("/tmp/b.wav", "b.wav")
        manager.create_job("/tmp/c.wav", "c.wav")  # stays queued
        manager.update_job_status(a, status=PROCESSING)
        manager.update_job_status(b, status=PROCESSING)
        assert manager.get_active_job_count() == 2


@pytest.mark.unit
class TestEstimation:
    def test_estimate_is_proportional(self):
        assert JobManager.estimate_processing_time(120) == pytest.approx(
            2 * JobManager.estimate_processing_time(60)
        )
        assert JobManager.estimate_processing_time(0) == 0


@pytest.mark.unit
class TestCleanup:
    def test_cleanup_removes_old_jobs_and_files(self, manager, tmp_path):
        upload = tmp_path / "uploads" / "old.wav"
        upload.parent.mkdir(parents=True, exist_ok=True)
        upload.write_bytes(b"RIFF....WAVE")
        job_id = manager.create_job(str(upload), "old.wav")

        job_file = manager.jobs_dir / f"{job_id}.json"
        data = json.loads(job_file.read_text())
        data["created_at"] = time.time() - 7200  # 2 hours ago
        job_file.write_text(json.dumps(data))

        removed = manager.cleanup_old_jobs(max_age_hours=1)
        assert removed == 1
        assert not job_file.exists()
        assert not upload.exists()

    def test_cleanup_keeps_recent_jobs(self, manager):
        job_id = manager.create_job("/tmp/x.wav", "x.wav")
        assert manager.cleanup_old_jobs(max_age_hours=1) == 0
        assert manager.get_job_status(job_id) is not None
