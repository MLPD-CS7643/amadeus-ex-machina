# amadeus-api

FastAPI backend for chord recognition. Upload audio → async per-job inference →
download a `.lab` chord annotation file.

Depends on `amadeus-core` for the model + feature + inference logic; this package only
owns the web concerns (routing, job lifecycle, file handling, config, schemas).

## Run

```bash
just api            # uvicorn on :8000 with --reload
# or
uv run uvicorn amadeus_api.main:app --port 8000
```

Endpoints live under `/api`: `POST /upload`, `GET /status/{job_id}`,
`GET /results/{job_id}`, `GET /download/{job_id}`, `GET /health`. Interactive docs at `/docs`.

## Test

```bash
just test-api
```
