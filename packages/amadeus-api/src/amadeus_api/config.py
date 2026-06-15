"""
Typed application settings (pydantic-settings).

Values come from environment variables / a local ``.env`` file, with sensible defaults.
The default model artifacts are the repo-level ``artifacts/guitarset_chroma`` bundle the
training package produces. Directory creation happens in the app factory, not at import.
"""

from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# packages/amadeus-api/src/amadeus_api/config.py -> repo root is parents[4].
REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ARTIFACT_DIR = REPO_ROOT / "artifacts" / "guitarset_chroma"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Model artifacts.
    MODEL_PATH: str = str(DEFAULT_ARTIFACT_DIR / "model.pt")
    SCALER_PATH: str = str(DEFAULT_ARTIFACT_DIR / "scaler.pkl")
    ENCODER_PATH: str = str(DEFAULT_ARTIFACT_DIR / "label_encoder.pkl")
    MODEL_CONFIG_PATH: str = str(DEFAULT_ARTIFACT_DIR / "model_config.json")

    # Upload + processing.
    MAX_FILE_SIZE: int = 52_428_800  # 50 MB
    UPLOAD_DIR: Path = Path("/tmp/chord_recognition/uploads")
    JOBS_DIR: Path = Path("/tmp/chord_recognition/jobs")
    RESULTS_DIR: Path = Path("/tmp/chord_recognition/results")
    MAX_CONCURRENT_JOBS: int = 5
    JOB_TIMEOUT_HOURS: int = 1

    # Server.
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Security.
    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:5173"]
    RATE_LIMIT_PER_MIN: int = 10
    ALLOWED_EXTENSIONS: set[str] = {".wav", ".mp3", ".flac"}

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def _split_csv(cls, value):
        """Allow CORS_ORIGINS to be given as a comma-separated env string."""
        if isinstance(value, str):
            return [v.strip() for v in value.split(",") if v.strip()]
        return value


settings = Settings()
