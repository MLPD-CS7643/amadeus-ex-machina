"""
Shared pytest fixtures: synthetic audio generation and a test-configured app.
"""
import types
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

# Resolve real artifact paths (used by integration tests; skipped if absent).
_BACKEND_DIR = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _BACKEND_DIR.parent
MODEL_PATH = _PROJECT_ROOT / "models" / "checkpoints" / "CRNNModel_best_model.pth"
SCALER_PATH = _PROJECT_ROOT / "data" / "processed" / "scaler.pkl"
ENCODER_PATH = _PROJECT_ROOT / "data" / "processed" / "label_encoder.pkl"
ARTIFACTS_PRESENT = MODEL_PATH.exists() and SCALER_PATH.exists() and ENCODER_PATH.exists()


def write_chord_wav(path: Path, seconds: float = 2.0, sr: int = 22050) -> Path:
    """Write a short C-major-triad WAV file for testing."""
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    freqs = (261.63, 329.63, 392.00)  # C4, E4, G4
    wave = sum(np.sin(2 * np.pi * f * t) for f in freqs) / len(freqs)
    audio = np.int16(wave / np.max(np.abs(wave)) * 32767)
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(path), sr, audio)
    return path


@pytest.fixture
def wav_factory(tmp_path):
    """Factory: wav_factory(name='a.wav', seconds=2.0) -> Path to a generated WAV."""
    def _make(name: str = "sample.wav", seconds: float = 2.0, sr: int = 22050) -> Path:
        return write_chord_wav(tmp_path / name, seconds=seconds, sr=sr)
    return _make


@pytest.fixture
def chord_wav(wav_factory):
    return wav_factory()


def make_test_config(tmp_path: Path, max_file_size: int = 50 * 1024 * 1024) -> types.SimpleNamespace:
    """Build a settings-like object pointing at temp directories."""
    return types.SimpleNamespace(
        MODEL_PATH=str(MODEL_PATH),
        SCALER_PATH=str(SCALER_PATH),
        ENCODER_PATH=str(ENCODER_PATH),
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
