"""
JobWorker: asynchronous job processing.

A bounded thread pool runs inference jobs off the request path. Submissions beyond
``max_concurrent`` queue naturally in the executor (design Property 20). Each job
updates its status as it progresses, writes its results, and -- win or lose --
cleans up the uploaded audio file in a ``finally`` block (Property 21).
"""
import json
import logging
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from app.errors import sanitize_message
from app.inference import InferenceEngine
from app.job_manager import COMPLETE, FAILED, PROCESSING, JobManager

logger = logging.getLogger(__name__)


class JobWorker:
    def __init__(
        self,
        job_manager: JobManager,
        engine: InferenceEngine,
        results_dir: Path,
        file_handler=None,
        max_concurrent: int = 5,
    ):
        self.job_manager = job_manager
        self.engine = engine
        self.results_dir = Path(results_dir)
        self.file_handler = file_handler
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._executor = ThreadPoolExecutor(
            max_workers=max_concurrent, thread_name_prefix="chord-worker"
        )

    def submit(self, job_id: str) -> Future:
        """Queue a job for asynchronous processing."""
        return self._executor.submit(self._process, job_id)

    def _process(self, job_id: str) -> None:
        job = self.job_manager.get_job_status(job_id)
        if job is None:
            logger.error("[%s] job vanished before processing", job_id)
            return

        audio_path = job.get("file_path")
        try:
            self.job_manager.update_job_status(
                job_id, status=PROCESSING, progress=0, message="Processing audio"
            )

            def progress_cb(pct: int, stage: str) -> None:
                self.job_manager.update_job_status(job_id, progress=pct, message=stage)

            annotations, duration = self.engine.process_audio(audio_path, progress_cb)

            result_dir = self.results_dir / job_id
            result_dir.mkdir(parents=True, exist_ok=True)
            lab_path = result_dir / "annotations.lab"
            self.engine.save_annotations(annotations, str(lab_path))

            metadata = {
                "job_id": job_id,
                "filename": job.get("filename"),
                "audio_duration": duration,
                "annotation_count": len(annotations),
            }
            with open(result_dir / "metadata.json", "w") as fout:
                json.dump(metadata, fout, indent=2)

            self.job_manager.update_job_status(
                job_id,
                status=COMPLETE,
                progress=100,
                message="Complete",
                result_path=str(lab_path),
                audio_duration=duration,
                estimated_time=JobManager.estimate_processing_time(duration),
            )
            logger.info("[%s] complete: %d segments", job_id, len(annotations))

        except Exception as exc:  # noqa: BLE001 - any failure marks the job failed
            logger.exception("[%s] processing failed", job_id)
            try:
                self.job_manager.update_job_status(
                    job_id,
                    status=FAILED,
                    message="Failed",
                    error=sanitize_message(str(exc)),
                )
            except Exception:  # noqa: BLE001
                logger.exception("[%s] could not record failure", job_id)
        finally:
            # Always remove the uploaded audio (Property 21); results are retained.
            self._cleanup_upload(audio_path)

    def _cleanup_upload(self, audio_path: Optional[str]) -> None:
        if not audio_path:
            return
        if self.file_handler is not None:
            self.file_handler.cleanup_file(audio_path)
            return
        try:
            p = Path(audio_path)
            if p.exists():
                p.unlink()
        except OSError:
            pass

    def shutdown(self, wait: bool = False) -> None:
        self._executor.shutdown(wait=wait)
