"""
REST API endpoints for chord recognition (router prefix ``/api``).

    POST /upload            -> queue an audio file for processing
    GET  /status/{job_id}   -> poll job status/progress
    GET  /results/{job_id}  -> chord annotations as JSON
    GET  /chart/{job_id}    -> structured chord chart (shapes + timing) as JSON
    GET  /tab/{job_id}      -> chords as ASCII guitar tablature (inline text)
    GET  /download/{job_id} -> chord annotations as a .lab attachment
                               (?format=tab -> guitar tablature attachment)
    GET  /health            -> liveness + engine-loaded flag

Components (engine, job manager, worker, file handler) live on ``request.app.state``.
"""

import uuid
from pathlib import Path

from fastapi import APIRouter, File, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse

from amadeus_api.errors import error_response
from amadeus_api.jobs import COMPLETE
from amadeus_api.schemas import (
    Annotation,
    ChartChord,
    ChartResponse,
    HealthResponse,
    JobCreated,
    JobStatus,
    ResultsResponse,
)

router = APIRouter(prefix="/api")


def _parse_lab(path: Path) -> list[dict]:
    """Read a .lab file into annotation dicts."""
    annotations = []
    with open(path) as fin:
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


def _completed_job(
    request: Request, job_id: str
) -> tuple[dict, Path] | JSONResponse:
    """Resolve a completed job's result ``.lab`` path, or a 404 error response."""
    job = request.app.state.job_manager.get_job_status(job_id)
    if job is None or job.get("status") != COMPLETE:
        return error_response(
            "RESULTS_NOT_READY", "Results not found or not ready.", 404
        )
    result_path = job.get("result_path")
    if not result_path or not Path(result_path).exists():
        return error_response("RESULTS_NOT_READY", "Results not available.", 404)
    return job, Path(result_path)


def _render_tab(job: dict, result_path: Path) -> str:
    """Render the stored ``.lab`` as ASCII guitar tab (cheap; O(segments))."""
    from amadeus_core.tab import lab_to_tab

    return lab_to_tab(result_path, title=job.get("filename"))


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    return HealthResponse(
        status="ok", engineLoaded=bool(request.app.state.engine.is_loaded)
    )


@router.post("/upload", response_model=JobCreated)
async def upload(request: Request, file: UploadFile | None = File(None)) -> JobCreated:
    from amadeus_api.errors import FileValidationError

    if file is None or not file.filename:
        raise FileValidationError("No file was provided.", "NO_FILE")

    file_handler = request.app.state.file_handler
    job_id = str(uuid.uuid4())
    # save_upload validates (format/size/content) and raises FileValidationError.
    file_path, sanitized_name = file_handler.save_upload(
        file.file, file.filename, job_id
    )
    request.app.state.job_manager.create_job(file_path, sanitized_name, job_id=job_id)
    request.app.state.worker.submit(job_id)

    return JobCreated(
        jobId=job_id, status="queued", message="File uploaded successfully"
    )


@router.get("/status/{job_id}", response_model=JobStatus)
async def status(request: Request, job_id: str) -> JobStatus | JSONResponse:
    job = request.app.state.job_manager.get_job_status(job_id)
    if job is None:
        return error_response("JOB_NOT_FOUND", "Job not found.", 404)

    estimated_total = job.get("estimated_time")
    progress = job.get("progress", 0)
    remaining = None
    if estimated_total is not None:
        remaining = round(estimated_total * max(0.0, 1 - progress / 100.0), 2)

    return JobStatus(
        jobId=job_id,
        status=job.get("status"),
        progress=progress,
        estimatedTimeRemaining=remaining,
        message=job.get("message", ""),
    )


@router.get("/results/{job_id}", response_model=ResultsResponse)
async def results(request: Request, job_id: str) -> ResultsResponse | JSONResponse:
    resolved = _completed_job(request, job_id)
    if isinstance(resolved, JSONResponse):
        return resolved
    job, result_path = resolved

    annotations = [Annotation(**a) for a in _parse_lab(result_path)]
    return ResultsResponse(
        jobId=job_id,
        filename=job.get("filename"),
        duration=job.get("audio_duration"),
        annotations=annotations,
    )


@router.get("/chart/{job_id}", response_model=ChartResponse)
async def chart(request: Request, job_id: str) -> ChartResponse | JSONResponse:
    """Structured chord chart (JSON): shapes + timing for rich front-end rendering."""
    resolved = _completed_job(request, job_id)
    if isinstance(resolved, JSONResponse):
        return resolved
    job, result_path = resolved

    from amadeus_core.tab import ChordShape, lab_to_chart

    def _chord(shape: ChordShape) -> ChartChord:
        return ChartChord(
            name=shape.name,
            label=shape.label,
            frets=list(shape.frets),
            start=shape.start,
            end=shape.end,
            isNoChord=shape.is_no_chord,
        )

    c = lab_to_chart(result_path, title=job.get("filename"))
    return ChartResponse(
        jobId=job_id,
        title=c.title,
        key=c.key,
        capo=c.capo,
        capoShapes=list(c.capo_shapes),
        duration=c.duration,
        chords=[_chord(s) for s in c.chords],
        capoChords=[_chord(s) for s in c.capo_chords],
    )


@router.get("/tab/{job_id}", response_model=None)
async def tab(request: Request, job_id: str) -> PlainTextResponse | JSONResponse:
    """Chords as ASCII guitar tab, inline (no attachment) for display in the UI."""
    resolved = _completed_job(request, job_id)
    if isinstance(resolved, JSONResponse):
        return resolved
    job, result_path = resolved
    return PlainTextResponse(_render_tab(job, result_path))


@router.get("/download/{job_id}", response_model=None)
async def download(
    request: Request, job_id: str, fmt: str = Query("lab", alias="format")
) -> FileResponse | PlainTextResponse | JSONResponse:
    resolved = _completed_job(request, job_id)
    if isinstance(resolved, JSONResponse):
        return resolved
    job, result_path = resolved

    stem = Path(job.get("filename") or "audio").stem
    if fmt == "tab":
        return PlainTextResponse(
            _render_tab(job, result_path),
            headers={
                "Content-Disposition": f'attachment; filename="{stem}_chords.txt"'
            },
        )
    return FileResponse(
        result_path,
        media_type="text/plain",
        filename=f"{stem}_chords.lab",
    )
