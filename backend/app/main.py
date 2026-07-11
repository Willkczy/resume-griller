"""
Resume Griller - FastAPI Backend
Main application entry point.
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes import resume, session, voice, websocket
from backend.app.config import settings
from backend.app.core.logging_config import configure_logging, get_logger
from backend.app.middleware.rate_limit import setup_rate_limiting
from backend.app.models.schemas import HealthCheck

# Configure logging
configure_logging(debug=settings.DEBUG)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info(
        "starting_application",
        app_name=settings.APP_NAME,
        llm_mode=settings.LLM_MODE,
        llm_provider=settings.LLM_PROVIDER if settings.LLM_MODE == "api" else None,
    )

    # Ensure directories exist
    upload_dir = Path(settings.UPLOAD_DIR)
    chroma_dir = Path(settings.CHROMA_PERSIST_DIR)
    parsed_resume_dir = Path(settings.PARSED_RESUME_DIR)
    checkpoint_dir = Path(settings.CHECKPOINT_DB_PATH).parent

    upload_dir.mkdir(parents=True, exist_ok=True)
    chroma_dir.mkdir(parents=True, exist_ok=True)
    parsed_resume_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "data_directories_initialized",
        upload_dir=str(upload_dir),
        chroma_dir=str(chroma_dir),
    )

    yield

    # Shutdown
    logger.info("shutting_down_application", app_name=settings.APP_NAME)


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description="AI-powered interview simulator that grills candidates with resume-specific questions.",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup rate limiting
setup_rate_limiting(app)


# ============== Routes ==============

# Include API routers
app.include_router(resume.router, prefix=settings.API_V1_PREFIX)
app.include_router(session.router, prefix=settings.API_V1_PREFIX)
app.include_router(voice.router, prefix=settings.API_V1_PREFIX)
app.include_router(websocket.router)


# ============== Health Check ==============


@app.get("/health", response_model=HealthCheck, tags=["health"])
async def health_check():
    """
    Health check endpoint.

    Checks:
    - Application status
    - LLM configuration
    - Custom model availability (if enabled)
    - Voice services (if enabled)
    """
    dependencies = {}

    # Check custom model availability
    custom_model_available = False
    if settings.CUSTOM_MODEL_ENABLED:
        try:
            import httpx

            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(
                    f"{settings.CUSTOM_MODEL_URL.rstrip('/')}/models"
                )
                custom_model_available = response.status_code == 200
        except Exception:
            custom_model_available = False

    dependencies["custom_model"] = custom_model_available
    dependencies["voice_services"] = settings.VOICE_ENABLED

    return HealthCheck(
        status="healthy",
        version="1.0.0",
        llm_mode=settings.LLM_MODE,
        llm_provider=settings.LLM_PROVIDER if settings.LLM_MODE == "api" else None,
        voice_enabled=settings.VOICE_ENABLED,
        custom_model_available=custom_model_available,
        dependencies=dependencies,
    )


@app.get("/", tags=["root"])
async def root():
    """Root endpoint."""
    return {
        "name": settings.APP_NAME,
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


# ============== Run directly ==============

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
    )
