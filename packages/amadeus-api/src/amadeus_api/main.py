"""
FastAPI application factory.

Wires the FileHandler, JobManager, InferenceEngine, and JobWorker onto ``app.state``,
enables CORS, installs the rate-limit middleware, registers exception handlers, and
eagerly loads the model artifacts. If artifacts are missing the app still boots (graceful
degradation) -- uploads then surface a clean config error instead of crashing.

Run:  uvicorn amadeus_api.main:app --port 8000
"""

import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from amadeus_api.config import Settings
from amadeus_api.config import settings as default_settings
from amadeus_api.errors import error_body, register_exception_handlers
from amadeus_api.files import FileHandler
from amadeus_api.jobs import JobManager
from amadeus_api.ratelimit import RateLimiter
from amadeus_api.routes import router
from amadeus_api.worker import JobWorker
from amadeus_core.inference import InferenceEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


def _load_engine(config: Settings) -> InferenceEngine:
    """
    Build the inference engine and load the primary model artifacts.

    The primary (deployed) model uses the NNLS feature backend. If a bundle selects
    ``nnls`` but that backend isn't installed on this host (``just nnls-setup`` not run),
    fall back to the librosa-based fallback bundle so the app still serves chords instead
    of failing every upload. If loading fails entirely, the app still boots and uploads
    surface a clean config error.
    """
    from amadeus_core.features.nnls import nnls_backend_available

    engine = InferenceEngine(
        separation=getattr(config, "SEPARATION_METHOD", "none"),
        feature_backend=getattr(config, "FEATURE_BACKEND", None),
    )
    try:
        engine.load_artifacts(
            config.MODEL_PATH,
            config.SCALER_PATH,
            config.ENCODER_PATH,
            config_path=getattr(config, "MODEL_CONFIG_PATH", None),
        )
        fallback_model = getattr(config, "FALLBACK_MODEL_PATH", None)
        if (
            engine.feature_backend == "nnls"
            and not nnls_backend_available()
            and fallback_model
        ):
            logger.warning(
                "Primary model needs the NNLS feature backend, which is not installed "
                "(run `just nnls-setup`). Falling back to the librosa model at %s.",
                fallback_model,
            )
            engine = InferenceEngine(
                separation=getattr(config, "SEPARATION_METHOD", "none")
            )
            engine.load_artifacts(
                fallback_model,
                config.FALLBACK_SCALER_PATH,
                config.FALLBACK_ENCODER_PATH,
                config_path=config.FALLBACK_MODEL_CONFIG_PATH,
            )
    except Exception:  # noqa: BLE001 - boot anyway; uploads will report misconfig
        logger.exception(
            "Inference engine failed to load artifacts; uploads will fail until fixed."
        )
    return engine


def create_app(config: Settings = default_settings) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield
        # Drain the worker pool on shutdown.
        worker = getattr(app.state, "worker", None)
        if worker is not None:
            worker.shutdown(wait=False)

    app = FastAPI(
        title="amadeus chord-recognition API", version="0.1.0", lifespan=lifespan
    )
    app.state.settings = config

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(config.CORS_ORIGINS),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def _rate_limit(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        limiter = getattr(request.app.state, "rate_limiter", None)
        # Only throttle the expensive operation (upload -> inference). Cheap reads the
        # UI polls constantly (status/results/chart/tab/download) are never limited, so
        # normal use can't trip the limiter.
        if (
            limiter is not None
            and request.method == "POST"
            and request.url.path.startswith("/api")
        ):
            client = request.client.host if request.client else "unknown"
            if not limiter.allow(client):
                return JSONResponse(
                    error_body("RATE_LIMITED", "Too many requests. Please slow down."),
                    status_code=429,
                )
        return await call_next(request)

    # Shared components on app.state (their __init__ creates the needed directories).
    app.state.file_handler = FileHandler(
        upload_dir=config.UPLOAD_DIR,
        max_file_size=config.MAX_FILE_SIZE,
        allowed_extensions=config.ALLOWED_EXTENSIONS,
    )
    app.state.job_manager = JobManager(
        jobs_dir=config.JOBS_DIR,
        results_dir=config.RESULTS_DIR,
        max_concurrent=config.MAX_CONCURRENT_JOBS,
    )
    app.state.rate_limiter = RateLimiter(config.RATE_LIMIT_PER_MIN)
    app.state.engine = _load_engine(config)
    app.state.worker = JobWorker(
        job_manager=app.state.job_manager,
        engine=app.state.engine,
        results_dir=config.RESULTS_DIR,
        file_handler=app.state.file_handler,
        max_concurrent=config.MAX_CONCURRENT_JOBS,
    )

    register_exception_handlers(app)
    app.include_router(router)

    @app.get("/")
    async def index():
        return {"service": "chord-recognition", "version": app.version}

    return app


app = create_app()
