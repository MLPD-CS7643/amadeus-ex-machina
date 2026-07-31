# amadeus-ex-machina task runner.
# Run `just` with no arguments to list available recipes.

set shell := ["bash", "-uc"]

# Default: show the recipe list.
default:
    @just --list

# Sync all workspace packages + dev tooling into the uv-managed venv.
sync:
    uv sync --all-packages

# Format all Python in the workspace packages.
fmt:
    uv run ruff format packages

# Lint all Python in the workspace packages.
lint:
    uv run ruff check packages

# Auto-fix lint issues where ruff can.
lint-fix:
    uv run ruff check --fix packages

# Static type-check all packages (core + api held to a stricter bar; see pyproject).
typecheck:
    uv run mypy packages/amadeus-core/src packages/amadeus-api/src packages/amadeus-train/src

# Run the full Python test suite. Separate invocations avoid the duplicate-`tests`-
# package collision between packages.
test: test-core test-api test-train

test-core:
    uv run pytest packages/amadeus-core

test-api:
    uv run pytest packages/amadeus-api

test-train:
    uv run pytest packages/amadeus-train

# Coverage reports (per package, for the same reason as `test`).
cov:
    uv run pytest --cov=amadeus_core --cov-report=term-missing packages/amadeus-core
    uv run pytest --cov=amadeus_api --cov-report=term-missing packages/amadeus-api

# Train the GuitarSet chroma model (forwards extra args, e.g. `just train --max-tracks 4`).
train *ARGS:
    uv run amadeus-train {{ARGS}}

# Run the FastAPI dev server (override the port with `just api 5000`).
api port="8000":
    uv run uvicorn amadeus_api.main:app --reload --port {{port}}

# Frontend helpers (wrap npm in frontend/).
fe-install:
    cd frontend && npm install

fe-dev:
    cd frontend && npm run dev

fe-build:
    cd frontend && npm run build

fe-test:
    cd frontend && npm run test

# Set up the optional NNLS feature backend (true McGill bothchroma at serve time):
# builds + installs the nnls-chroma Vamp plugin and the vamp Python host. macOS only.
# Needed to serve the Billboard model; the librosa backend needs none of this.
nnls-setup:
    bash scripts/setup_nnls_backend.sh

# Install the git pre-commit hooks (run `just sync` first).
hooks:
    uv run pre-commit install

# Run the pre-commit hooks against all tracked files.
pre-commit:
    uv run pre-commit run --all-files

# Aggregate gates.
check: lint typecheck test

ci: sync check fe-build
