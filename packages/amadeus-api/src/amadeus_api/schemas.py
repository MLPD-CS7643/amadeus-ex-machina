"""Pydantic request/response models for the API (drives validation + OpenAPI docs)."""

from typing import Any

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    engineLoaded: bool


class JobCreated(BaseModel):
    jobId: str
    status: str
    message: str


class JobStatus(BaseModel):
    jobId: str
    status: str
    progress: int
    estimatedTimeRemaining: float | None = None
    message: str = ""


class Annotation(BaseModel):
    startTime: float
    endTime: float
    chordLabel: str


class ResultsResponse(BaseModel):
    jobId: str
    filename: str | None = None
    duration: float | None = None
    annotations: list[Annotation]


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: Any = None


class ErrorResponse(BaseModel):
    error: ErrorDetail
