"""Shared fixtures: synthetic audio generation and a test-configured app/client."""

import types
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from amadeus_api.main import create_app

# packages/amadeus-api/tests/conftest.py -> repo root is parents[3].
REPO_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "guitarset_chroma"
MODEL_PATH = ARTIFACT_DIR / "model.pt"
SCALER_PATH = ARTIFACT_DIR / "scaler.pkl"
ENCODER_PATH = ARTIFACT_DIR / "label_encoder.pkl"
CONFIG_PATH = ARTIFACT_DIR / "model_config.json"
ARTIFACTS_PRESENT = (
    MODEL_PATH.exists() and SCALER_PATH.exists() and ENCODER_PATH.exists()
)


def write_chord_wav(path: Path, seconds: float = 2.0, sr: int = 22050) -> Path:
    """Write a short C-major-triad WAV file for testing."""
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    freqs = (261.63, 329.63, 392.00)  # C4, E4, G4
    wave = sum(np.sin(2 * np.pi * f * t) for f in freqs) / len(freqs)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), wave.astype(np.float32), sr, subtype="PCM_16")
    return path


@pytest.fixture
def wav_factory(tmp_path):
    def _make(name: str = "sample.wav", seconds: float = 2.0, sr: int = 22050) -> Path:
        return write_chord_wav(tmp_path / name, seconds=seconds, sr=sr)

    return _make


@pytest.fixture
def chord_wav(wav_factory):
    return wav_factory()


def make_test_config(
    tmp_path: Path, max_file_size: int = 50 * 1024 * 1024, real_artifacts: bool = True
) -> types.SimpleNamespace:
    """Build a settings-like object pointing at temp directories."""
    return types.SimpleNamespace(
        MODEL_PATH=str(MODEL_PATH),
        SCALER_PATH=str(SCALER_PATH),
        ENCODER_PATH=str(ENCODER_PATH),
        MODEL_CONFIG_PATH=str(CONFIG_PATH)
        if real_artifacts
        else str(tmp_path / "none.json"),
        MAX_FILE_SIZE=max_file_size,
        UPLOAD_DIR=tmp_path / "uploads",
        JOBS_DIR=tmp_path / "jobs",
        RESULTS_DIR=tmp_path / "results",
        MAX_CONCURRENT_JOBS=2,
        JOB_TIMEOUT_HOURS=1,
        CORS_ORIGINS=["http://localhost:3000", "http://localhost:5173"],
        RATE_LIMIT_PER_MIN=1000,  # effectively off unless a test overrides it
        ALLOWED_EXTENSIONS={".wav", ".mp3", ".flac"},
    )


def build_app(
    tmp_path, max_file_size=50 * 1024 * 1024, mock_worker=True, real_artifacts=False
):
    """Create an app with temp dirs; by default no real model + a mocked worker."""
    cfg = make_test_config(
        tmp_path, max_file_size=max_file_size, real_artifacts=real_artifacts
    )
    if not real_artifacts:
        cfg.MODEL_PATH = str(tmp_path / "no_model.pt")
        cfg.SCALER_PATH = str(tmp_path / "no_scaler.pkl")
        cfg.ENCODER_PATH = str(tmp_path / "no_encoder.pkl")
        cfg.MODEL_CONFIG_PATH = str(tmp_path / "no_config.json")
    app = create_app(cfg)
    if mock_worker:
        app.state.worker = mock.Mock()
    return app


@pytest.fixture
def app(tmp_path):
    return build_app(tmp_path)


@pytest.fixture
def client(app):
    return TestClient(app)
