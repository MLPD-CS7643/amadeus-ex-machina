"""
Typed application settings (pydantic-settings).

Values come from environment variables / a local ``.env`` file, with sensible defaults.
The default model artifacts are the repo-level ``artifacts/guitarset_chroma`` bundle the
training package produces. Directory creation happens in the app factory, not at import.
"""

from pathlib import Path
from typing import Annotated

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

# packages/amadeus-api/src/amadeus_api/config.py -> repo root is parents[4].
REPO_ROOT = Path(__file__).resolve().parents[4]
# The primary/deployed model: Billboard-trained, NNLS bothchroma features, 7th-chord
# vocab. Serving it needs the NNLS feature backend (`just nnls-setup`). If that backend
# isn't installed, the app falls back to FALLBACK_ARTIFACT_DIR at startup (see main.py).
DEFAULT_ARTIFACT_DIR = REPO_ROOT / "artifacts" / "billboard_sevenths"
FALLBACK_ARTIFACT_DIR = REPO_ROOT / "artifacts" / "guitarset_chroma"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Model artifacts (primary).
    MODEL_PATH: str = str(DEFAULT_ARTIFACT_DIR / "model.pt")
    SCALER_PATH: str = str(DEFAULT_ARTIFACT_DIR / "scaler.pkl")
    ENCODER_PATH: str = str(DEFAULT_ARTIFACT_DIR / "label_encoder.pkl")
    MODEL_CONFIG_PATH: str = str(DEFAULT_ARTIFACT_DIR / "model_config.json")

    # Fallback model, served if the primary model needs the NNLS backend but it isn't
    # installed. Librosa-based GuitarSet maj/min model that works with no native setup.
    FALLBACK_MODEL_PATH: str = str(FALLBACK_ARTIFACT_DIR / "model.pt")
    FALLBACK_SCALER_PATH: str = str(FALLBACK_ARTIFACT_DIR / "scaler.pkl")
    FALLBACK_ENCODER_PATH: str = str(FALLBACK_ARTIFACT_DIR / "label_encoder.pkl")
    FALLBACK_MODEL_CONFIG_PATH: str = str(FALLBACK_ARTIFACT_DIR / "model_config.json")

    # Upload + processing.
    MAX_FILE_SIZE: int = 52_428_800  # 50 MB
    UPLOAD_DIR: Path = Path("/tmp/chord_recognition/uploads")
    JOBS_DIR: Path = Path("/tmp/chord_recognition/jobs")
    RESULTS_DIR: Path = Path("/tmp/chord_recognition/results")
    MAX_CONCURRENT_JOBS: int = 5
    JOB_TIMEOUT_HOURS: int = 1

    # Harmonic source separation before chroma: "none" | "hpss" | "demucs".
    # "hpss" is cheap and near-identity on clean audio while stripping drum transients
    # from full mixes; "demucs" (optional dep) also removes vocals for the best result.
    # Ignored when the served model uses the "nnls" feature backend (McGill bothchroma is
    # raw-mix, so the engine forces separation off there to preserve train/serve parity).
    SEPARATION_METHOD: str = "hpss"

    # Feature backend override: "librosa" (CQT approximation) | "nnls" (true McGill
    # bothchroma via the NNLS Chroma Vamp plugin). None -> defer to the served bundle's
    # model_config.json (self-describing), which is the intended default. Set explicitly
    # only to force a backend regardless of what the bundle declares.
    FEATURE_BACKEND: str | None = None

    # Server.
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Security. NoDecode: take the env value as a raw string (comma-separated, handled by
    # _split_csv below) instead of pydantic-settings' default JSON decoding, so
    # `CORS_ORIGINS=https://app.example` works as an env var, not just a JSON array.
    CORS_ORIGINS: Annotated[list[str], NoDecode] = [
        "http://localhost:3000",
        "http://localhost:5173",
    ]
    # Uploads per minute per client (only POST /upload is throttled; reads are free).
    RATE_LIMIT_PER_MIN: int = 30
    ALLOWED_EXTENSIONS: set[str] = {".wav", ".mp3", ".flac"}

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def _split_csv(cls, value):
        """Allow CORS_ORIGINS to be given as a comma-separated env string."""
        if isinstance(value, str):
            return [v.strip() for v in value.split(",") if v.strip()]
        return value


settings = Settings()
