# CLAUDE.md

Guidance for Claude Code (and humans) working in this repository.

## What this project is

**amadeus-ex-machina** is a deep-learning project for **Automatic Chord Recognition
(ACR)**: given music audio, predict the harmonic chord at each time step (e.g. `C:maj7`,
`D:min`, `G:7`). It began as a Georgia Tech **CS 7643** group project and has been
restructured into a modern, package-based **ML-backed web application**.

There are three threads:
1. **A chord-recognition web app** — upload audio → async inference → download a `.lab`
   chord-annotation file. This is the primary, fully-supported path.
2. **Model training** on real guitar audio (GuitarSet via `mirdata`).
3. **A "timbral / texture bias" research study** + synthetic chord-audio generation
   (the `datagen` code), kept working but not the focus.

## Architecture: a uv workspace with three packages

The repo is a **uv workspace** (`pyproject.toml` at the root is a *virtual* root — not an
installable package). All Python lives in `packages/`:

```
packages/
  amadeus-core/   src/amadeus_core/   The shared ML library. Depended on by both others.
  amadeus-api/    src/amadeus_api/    FastAPI web backend. Depends on core.
  amadeus-train/  src/amadeus_train/  Training, datagen, hyperparameter search. Depends on core.
```

**The cardinal rule of this codebase:** anything both training and serving must agree on
(models, feature extraction, chord vocabulary, the artifact format) lives in
**`amadeus-core`**. `amadeus-api` and `amadeus-train` both depend on core and **never on
each other**. This is the inversion that fixed the old design, where the shared ML code
was trapped inside the Flask backend and the training script reached into it via
`sys.path` hacks.

### `amadeus-core` (`packages/amadeus-core/src/amadeus_core/`)
- `features/chroma.py` — `chroma24_from_audio` / `extract_chroma24`: audio → 24-dim
  (bass+treble) chroma. **Single source of truth** for the feature representation. Has a
  `backend` selector: `"librosa"` (default; band-limited CQT approximation) or `"nnls"`.
- `features/nnls.py` — `nnls_bothchroma`: **true** McGill NNLS `bothchroma` via the Queen
  Mary NNLS Chroma Vamp plugin (parity-by-construction with the Billboard training
  features). Optional native backend; set it up with `just nnls-setup`. `nnls_backend_available()`
  lets the API fall back to librosa when the plugin isn't installed.
- `models/` — the model zoo. Serving models (`chroma_transformer.py` →
  `ChromaTransformerModel`, `chroma_mlp.py` → `ChromaMLP`, `crnn_legacy.py` →
  `CRNNChordModel`) plus `registry.py` (`build_model`, `SEQUENCE_MODELS`). Research nets
  (`CNN.py`, `RNN.py`, `CRNN.py`, `FourierCNN.py`, `FourierS2S.py`, `StridedFourier.py`,
  `AudioDataset.py`, `mlp_chord_classifier.py`) live alongside, imported by notebooks.
- `chords/` — `vocab.py` (`CHORDS`/`INVERSIONS`/`NOTES`/`FLAT_TO_SHARP`, the single vocab
  source), `remap.py` (`remap_chord_label`, `CullMode`), `majmin.py` (`to_majmin`,
  `transpose_label`), `eval.py` (`mir_eval` scoring; keeps the numpy-alias shim).
- `artifacts.py` — the **artifact bundle** contract: `load_bundle` / `save_bundle` over
  `model.pt` + `scaler.pkl` + `label_encoder.pkl` + `model_config.json`. `model_config.json`
  names the architecture so a bundle fully determines the model + feature params.
- `inference.py` — framework-agnostic `InferenceEngine` (audio → `.lab`).
- `device.py` — `get_device` (MPS→CUDA→CPU) + `set_seed`. `errors.py` — `ProcessingError`,
  `sanitize_message`.

### `amadeus-api` (`packages/amadeus-api/`)
FastAPI app. `main.py::create_app` wires `FileHandler`, `JobManager`, `JobWorker`, and a
core `InferenceEngine` onto `app.state`, with CORS + a rate-limit middleware and a
`lifespan` for worker shutdown. Routes (`/api`): `POST /upload`, `GET /status/{id}`,
`GET /results/{id}`, `GET /download/{id}`, `GET /health`; OpenAPI docs at `/docs`.
`config.py` is typed `pydantic-settings`. Inference runs off the request path in a
`ThreadPoolExecutor` (CPU/GPU-bound, not async). 52 tests (`TestClient` + Hypothesis).

### `amadeus-train` (`packages/amadeus-train/`)
- `cli.py` (console script **`amadeus-train`**) + `guitarset.py` — the live GuitarSet
  training pipeline; features come from `amadeus_core.features`, artifacts saved via
  `amadeus_core.artifacts.save_bundle` to the repo-level `artifacts/guitarset_chroma/`.
- `datasets.py` — `MirDataProcessor` / `ChordDataProcessor` (Billboard + audio).
- `solver.py`, `griddy/` — training `Solver` + Optuna hyperparameter search.
- `datagen/` — synthetic chord-audio + FX (texture-bias study). Needs the optional
  `datagen` extra (`tinysoundfont` → `pyaudio` → system `portaudio`).
  `youtube_download.py` — Billboard audio scraper.

## Environment & tooling

- **uv** manages everything (no conda). `just sync` == `uv sync --all-packages`. Python
  3.11 (`.python-version`), pinned in `uv.lock` (committed). `torch`/`librosa`/etc. are
  core deps; the API and training inherit them transitively through `amadeus-core`.
- **just** is the task runner (`justfile`). Common recipes:
  - `just sync` — install the workspace + dev tools.
  - `just lint` / `just fmt` — ruff check / format. `just typecheck` — mypy (core + api).
  - `just test` (core + api), `just test-core` / `test-api` / `test-train`. `just check`
    = lint + typecheck + test.
  - `just api` — run the FastAPI dev server (uvicorn on :8000).
  - `just train *ARGS` — run the GuitarSet trainer (e.g. `just train --max-tracks 4 --epochs 1`).
  - `just fe-dev` / `fe-build` / `fe-test` — frontend (wraps npm in `frontend/`).
  - `just nnls-setup` — one-time (macOS): build + install the NNLS Chroma Vamp plugin and
    the `vamp` Python host, so the API can serve the Billboard model on true bothchroma.
    Not part of `uv sync` (native, not pip-installable); see `scripts/setup_nnls_backend.sh`.
- **ruff** (lint + format, replaces black/flake8) and **mypy** are configured once in the
  root `pyproject.toml`. Relocated research modules have targeted `per-file-ignores` (they
  work but aren't modernized). The frontend (`frontend/`) keeps its Vite/Vitest tooling and
  proxies `/api` → `127.0.0.1:8000` in dev.

## Core concepts

- **Chord labels** are Billboard notation `root:class` (e.g. `C:maj7`); `N` = no chord.
- **Features**: 24-dim chroma (12 bass + 12 treble). The *same* extractor is used at train
  and inference time, so there is **no train/serve feature mismatch** — the crux is picking
  the matching backend for the served model (librosa CQT vs. true NNLS bothchroma).
- **The deployed (primary) model** is a `ChromaTransformerModel` trained on **McGill-Billboard**
  with a **sevenths** vocab (119 classes), in `artifacts/billboard_sevenths/`. It serves on
  **true NNLS `bothchroma`** (`feature_backend: "nnls"` in its config), so it needs the NNLS
  Vamp plugin — run `just nnls-setup` once (macOS). If that backend isn't installed, the API
  **falls back at startup** to the librosa **GuitarSet** maj/min model in
  `artifacts/guitarset_chroma/` (24 classes) so the app still works out of the box.
- The binaries (`model.pt`, `*.pkl`) are gitignored; `model_config.json` is tracked (and now
  carries `feature_backend`, so a bundle self-selects its extractor). Inference slides a
  window over chroma frames, predicts per window, votes per frame, smooths, merges to `.lab`.

## How to run things

```bash
just sync                       # one-time: install everything
just check                      # lint + typecheck + core/api tests
just api                        # FastAPI on http://127.0.0.1:8000 (docs at /docs)
just fe-dev                     # frontend on http://localhost:5173 (proxies to :8000)
just train --download --no-leadsheet --vocab majmin --arch transformer   # retrain
```

To exercise the full web flow: `just api`, then upload via the frontend or
`curl -F file=@clip.wav localhost:8000/api/upload`, poll `/api/status/{id}`, and fetch
`/api/results/{id}` + `/api/download/{id}`.

## Conventions

- Import from package roots (`from amadeus_core.features import chroma24_from_audio`).
  Notebooks no longer need `sys.path` hacks (the packages are installed by `uv sync`).
- Saved model artifacts → an artifact-bundle directory (default
  `artifacts/guitarset_chroma/`). Build models from a config via
  `amadeus_core.models.build_model` / `amadeus_core.artifacts.load_bundle`.
- Large/binary assets (`*.wav *.mp3 *.mid *.sf2 *.pth *.pt *.pkl *.csv`, `data/raw`,
  `data/processed`) are gitignored. Secrets (`secrets/`) too.

## Notes / gotchas

- The old single-tree layout (root `main.py`, `solver.py`, `models/`, `data/`, `utils/`,
  the Flask `backend/`, conda `environment.yaml`) is **archived under `archive/`** for
  reference; it is not on the import path. Don't resurrect it — use the packages.
- `notebooks/` were swept to the new import paths. One pre-existing file with invalid JSON,
  `notebooks/chordgen_example/chordgen_example.ipynb`, was left untouched (fix its JSON
  before relying on it).
- Loading the shipped `scaler.pkl` may emit a benign scikit-learn version warning
  (artifacts were fit with an older sklearn). Inference is unaffected; retraining clears it.
- `datagen` requires `uv sync --all-packages --extra datagen` and a system `portaudio`
  (`brew install portaudio`); it is excluded from the default install.
- Infra (Docker, deployment) is intentionally out of scope right now.

When in doubt about how something runs end-to-end, start from `just <recipe>` and the
package the recipe targets.
