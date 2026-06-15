"""
Application configuration settings loaded from environment variables.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent

# Model Configuration
# Default to the retrained, self-consistent GuitarSet chroma model. The legacy
# Billboard CRNN artifacts remain available via env overrides (and are still used by
# the legacy integration test). If model_config.json is present the architecture is
# rebuilt from it; otherwise the loader falls back to the legacy CRNN.
ARTIFACT_DIR = BASE_DIR / 'artifacts' / 'guitarset_chroma'
MODEL_PATH = os.getenv('MODEL_PATH', str(ARTIFACT_DIR / 'model.pt'))
SCALER_PATH = os.getenv('SCALER_PATH', str(ARTIFACT_DIR / 'scaler.pkl'))
ENCODER_PATH = os.getenv('ENCODER_PATH', str(ARTIFACT_DIR / 'label_encoder.pkl'))
MODEL_CONFIG_PATH = os.getenv('MODEL_CONFIG_PATH', str(ARTIFACT_DIR / 'model_config.json'))

# File Upload Configuration
MAX_FILE_SIZE = int(os.getenv('MAX_FILE_SIZE', 52428800))  # 50MB in bytes
UPLOAD_DIR = Path(os.getenv('UPLOAD_DIR', '/tmp/chord_recognition/uploads'))

# Processing Configuration
MAX_CONCURRENT_JOBS = int(os.getenv('MAX_CONCURRENT_JOBS', 5))
JOB_TIMEOUT_HOURS = int(os.getenv('JOB_TIMEOUT_HOURS', 1))

# Server Configuration
FLASK_ENV = os.getenv('FLASK_ENV', 'development')
FLASK_DEBUG = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
HOST = os.getenv('HOST', '0.0.0.0')
PORT = int(os.getenv('PORT', 5000))

# CORS Configuration
CORS_ORIGINS = os.getenv('CORS_ORIGINS', 'http://localhost:3000,http://localhost:5173').split(',')

# Rate limiting (requests per minute per client IP)
RATE_LIMIT_PER_MIN = int(os.getenv('RATE_LIMIT_PER_MIN', 10))

# Allowed file extensions
ALLOWED_EXTENSIONS = {'.wav', '.mp3', '.flac'}

# Job storage paths
JOBS_DIR = Path('/tmp/chord_recognition/jobs')
RESULTS_DIR = Path('/tmp/chord_recognition/results')

# Ensure directories exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
JOBS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
