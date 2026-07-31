"""
Flask application factory.

Wires together the FileHandler, JobManager, InferenceEngine, and JobWorker, enables
CORS, registers routes + error handlers, and eagerly loads the model artifacts. If
artifacts are missing the app still boots (graceful degradation) -- uploads then
surface a clean config error instead of crashing the server.

Run:  flask --app app.main run --debug
      gunicorn -w 1 -b 0.0.0.0:5000 app.main:app
"""
import logging

from flask import Flask, jsonify
from flask_cors import CORS

from config import settings
from app.errors import error_response
from app.file_handler import FileHandler
from app.inference import InferenceEngine
from app.job_manager import JobManager
from app.routes import RateLimiter, api_bp, register_error_handlers
from app.worker import JobWorker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


def create_app(config=settings) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = config.MAX_FILE_SIZE
    app.config["MAX_FILE_SIZE"] = config.MAX_FILE_SIZE
    app.config["ALLOWED_EXTENSIONS"] = config.ALLOWED_EXTENSIONS
    app.config["RATE_LIMIT_PER_MIN"] = config.RATE_LIMIT_PER_MIN
    app.config["JOB_TIMEOUT_HOURS"] = config.JOB_TIMEOUT_HOURS

    CORS(
        app,
        resources={r"/api/*": {"origins": config.CORS_ORIGINS}},
        supports_credentials=True,
    )

    # Shared components, attached to the app for access via current_app.
    app.file_handler = FileHandler(
        upload_dir=config.UPLOAD_DIR,
        max_file_size=config.MAX_FILE_SIZE,
        allowed_extensions=config.ALLOWED_EXTENSIONS,
    )
    app.job_manager = JobManager(
        jobs_dir=config.JOBS_DIR,
        results_dir=config.RESULTS_DIR,
        max_concurrent=config.MAX_CONCURRENT_JOBS,
    )
    app.rate_limiter = RateLimiter(config.RATE_LIMIT_PER_MIN)
    app.engine = InferenceEngine()
    try:
        app.engine.load_artifacts(
            config.MODEL_PATH,
            config.SCALER_PATH,
            config.ENCODER_PATH,
            config_path=getattr(config, "MODEL_CONFIG_PATH", None),
        )
    except Exception:  # noqa: BLE001 - boot anyway; uploads will report misconfig
        logger.exception(
            "Inference engine failed to load artifacts; uploads will fail until fixed."
        )
    app.worker = JobWorker(
        job_manager=app.job_manager,
        engine=app.engine,
        results_dir=config.RESULTS_DIR,
        file_handler=app.file_handler,
        max_concurrent=config.MAX_CONCURRENT_JOBS,
    )

    register_error_handlers(app)
    app.register_blueprint(api_bp)

    @app.route("/")
    def index():
        return jsonify(
            {"service": "chord-recognition", "version": app.config.get("VERSION", "0.1.0")}
        ), 200

    return app


# Module-level app for `flask --app app.main` and gunicorn.
app = create_app()
