"""
JobManager: file-based job lifecycle + status tracking.

Each job is a JSON file under ``jobs_dir``. Status follows a strict state machine:
``queued -> processing -> (complete | failed)``; terminal states never transition again.
Job IDs are UUID4. All read-modify-write access is guarded by a single lock.
"""

import json
import logging
import shutil
import time
import uuid
from pathlib import Path
from threading import RLock

logger = logging.getLogger(__name__)

QUEUED = "queued"
PROCESSING = "processing"
COMPLETE = "complete"
FAILED = "failed"
TERMINAL_STATES = {COMPLETE, FAILED}

# Allowed target statuses from each current status (same-state allowed for updates).
_ALLOWED_TRANSITIONS = {
    QUEUED: {QUEUED, PROCESSING, FAILED},
    PROCESSING: {PROCESSING, COMPLETE, FAILED},
    COMPLETE: {COMPLETE},
    FAILED: {FAILED},
}


class JobManager:
    # Estimated processing seconds per second of audio.
    PROCESSING_TIME_FACTOR = 0.25

    def __init__(self, jobs_dir: Path, results_dir: Path, max_concurrent: int = 5):
        self.jobs_dir = Path(jobs_dir)
        self.results_dir = Path(results_dir)
        self.max_concurrent = max_concurrent
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()

    # -- helpers ---------------------------------------------------------

    def _job_path(self, job_id: str) -> Path:
        return self.jobs_dir / f"{job_id}.json"

    def _read(self, job_id: str) -> dict | None:
        path = self._job_path(job_id)
        if not path.exists():
            return None
        try:
            with open(path) as fin:
                return json.load(fin)
        except (json.JSONDecodeError, OSError):
            return None

    def _write(self, job: dict) -> None:
        path = self._job_path(job["job_id"])
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w") as fout:
            json.dump(job, fout, indent=2)
        tmp.replace(path)  # atomic on POSIX

    @classmethod
    def estimate_processing_time(cls, duration_s: float) -> float:
        """Estimated seconds to process, proportional to audio duration."""
        return round(cls.PROCESSING_TIME_FACTOR * max(0.0, float(duration_s)), 2)

    # -- public API ------------------------------------------------------

    def create_job(
        self, file_path: str, filename: str, job_id: str | None = None
    ) -> str:
        """Create a new queued job and return its unique id."""
        job_id = job_id or str(uuid.uuid4())
        now = time.time()
        job = {
            "job_id": job_id,
            "filename": filename,
            "status": QUEUED,
            "created_at": now,
            "updated_at": now,
            "progress": 0,
            "estimated_time": None,
            "message": "Queued",
            "error": None,
            "file_path": file_path,
            "result_path": None,
            "audio_duration": None,
        }
        with self._lock:
            self._write(job)
        return job_id

    def get_job_status(self, job_id: str) -> dict | None:
        """Return the job metadata dict, or None if the job does not exist."""
        with self._lock:
            return self._read(job_id)

    def update_job_status(
        self, job_id: str, status: str | None = None, **fields
    ) -> dict:
        """
        Update a job's status and/or other fields.

        Raises:
            KeyError: if the job does not exist.
            ValueError: if ``status`` is an invalid transition from the current state.
        """
        with self._lock:
            job = self._read(job_id)
            if job is None:
                raise KeyError(f"Job not found: {job_id}")

            if status is not None:
                current = job["status"]
                if status not in _ALLOWED_TRANSITIONS.get(current, set()):
                    raise ValueError(
                        f"Invalid status transition: {current} -> {status}"
                    )
                job["status"] = status

            for key, value in fields.items():
                job[key] = value

            job["updated_at"] = time.time()
            self._write(job)
            return job

    def get_active_job_count(self) -> int:
        """Number of jobs currently in the processing state."""
        with self._lock:
            count = 0
            for path in self.jobs_dir.glob("*.json"):
                try:
                    with open(path) as fin:
                        if json.load(fin).get("status") == PROCESSING:
                            count += 1
                except (json.JSONDecodeError, OSError):
                    continue
            return count

    def cleanup_old_jobs(self, max_age_hours: float) -> int:
        """
        Remove jobs (and their uploaded files + result dirs) older than the threshold.
        Returns the number of jobs removed.
        """
        cutoff = time.time() - max_age_hours * 3600
        removed = 0
        with self._lock:
            for path in list(self.jobs_dir.glob("*.json")):
                try:
                    with open(path) as fin:
                        job = json.load(fin)
                except (json.JSONDecodeError, OSError):
                    continue
                if job.get("created_at", 0) >= cutoff:
                    continue

                # Best-effort removal of associated artifacts.
                file_path = job.get("file_path")
                if file_path:
                    self._safe_remove(Path(file_path))
                    parent = Path(file_path).parent
                    if parent != parent.parent:  # don't touch fs root
                        self._safe_rmdir(parent)
                result_dir = self.results_dir / job["job_id"]
                self._safe_rmtree(result_dir)
                self._safe_remove(path)
                removed += 1
        return removed

    # -- best-effort fs helpers -----------------------------------------

    @staticmethod
    def _safe_remove(path: Path) -> None:
        try:
            if path.exists() and path.is_file():
                path.unlink()
        except OSError:
            pass

    @staticmethod
    def _safe_rmdir(path: Path) -> None:
        try:
            path.rmdir()  # only if empty
        except OSError:
            pass

    @staticmethod
    def _safe_rmtree(path: Path) -> None:
        try:
            if path.exists():
                shutil.rmtree(path)
        except OSError:
            pass
