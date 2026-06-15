# amadeus-ex-machina

A deep-learning **Automatic Chord Recognition** system: upload music audio and get a
time-aligned chord annotation (`.lab`) back. Built as a modern, package-based ML web app.

## What's here

A **uv workspace** with three Python packages plus a React frontend:

| Package / dir | What it is |
| --- | --- |
| `packages/amadeus-core` | Shared ML library: feature extraction, model zoo, chord vocabulary, the artifact format, and a framework-agnostic inference engine. |
| `packages/amadeus-api` | **FastAPI** backend: upload → async inference → download `.lab`. Depends on core. |
| `packages/amadeus-train` | Training (GuitarSet), hyperparameter search, and synthetic-audio datagen. Depends on core. |
| `frontend/` | Vite + React + TypeScript SPA (upload, status polling, chord timeline, download). |

The key design rule: everything training and serving must agree on lives in
**`amadeus-core`**; the API and training packages depend on core and never on each other.

## Quick start

Prerequisites: [`uv`](https://docs.astral.sh/uv/) and [`just`](https://github.com/casey/just)
(and Node 20+ for the frontend).

```bash
just sync          # create the venv and install all packages + dev tools (uv)
just check         # ruff lint + mypy + core/api tests

# Run the web app (two terminals):
just api           # FastAPI on http://127.0.0.1:8000  (OpenAPI docs at /docs)
just fe-dev        # frontend on http://localhost:5173  (proxies /api -> :8000)
```

Then open the frontend, drop in a `.wav/.mp3/.flac`, and download the chord `.lab`. Or hit
the API directly:

```bash
curl -F file=@clip.wav localhost:8000/api/upload      # -> {"jobId": "..."}
curl localhost:8000/api/status/<jobId>
curl localhost:8000/api/results/<jobId>
curl -OJ localhost:8000/api/download/<jobId>
```

## Training

The deployed model is a `ChromaTransformerModel` trained on **GuitarSet**, stored in
`artifacts/guitarset_chroma/`. Retrain it with:

```bash
just train --download --no-leadsheet --vocab majmin --arch transformer
# quick smoke run:
just train --max-tracks 4 --epochs 1 --no-leadsheet
```

Training features come from the **same** `chroma24_from_audio` used at inference, so there
is no train/serve feature mismatch. Artifacts are written where the API loads them.

## Common tasks (`just`)

| Recipe | Does |
| --- | --- |
| `just sync` | Install the workspace + dev tooling. |
| `just lint` / `just fmt` | ruff check / format. |
| `just typecheck` | mypy over core + api. |
| `just test` / `test-core` / `test-api` / `test-train` | Run test suites. |
| `just check` | lint + typecheck + tests. |
| `just api` / `just train` | Run the API / training CLI. |
| `just fe-dev` / `fe-build` / `fe-test` | Frontend dev / build / test. |

## Notes

- Synthetic-audio generation (`datagen`) is optional:
  `uv sync --all-packages --extra datagen` and a system `portaudio`.
- The previous single-tree layout (Flask `backend/`, root `main.py`/`solver.py`, conda
  `environment.yaml`, etc.) is preserved under `archive/` for reference.
- See `CLAUDE.md` for a fuller architecture tour.

## License

Distributed under the **MIT License**.
