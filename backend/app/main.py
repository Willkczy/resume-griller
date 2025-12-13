"""
Resume Griller - FastAPI Backend
Main application entry point.
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.config import settings
from backend.app.models.schemas import HealthCheck
from backend.app.api.routes import resume

from backend.app.api.routes import resume, session


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    print(f"Starting {settings.APP_NAME}...")
    print(f"LLM Mode: {settings.LLM_MODE}")
    if settings.LLM_MODE == "api":
        print(f"LLM Provider: {settings.LLM_PROVIDER}")
    
    # Ensure directories exist
    from pathlib import Path
    Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
    Path(settings.CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
    
    yield
    
    # Shutdown
    print(f"Shutting down {settings.APP_NAME}...")


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


# ============== Routes ==============

# Include API routers
app.include_router(resume.router, prefix=settings.API_V1_PREFIX)
app.include_router(session.router, prefix=settings.API_V1_PREFIX) 


# ============== Health Check ==============

@app.get("/health", response_model=HealthCheck, tags=["health"])
async def health_check():
    """Health check endpoint."""
    return HealthCheck(
        status="healthy",
        version="1.0.0",
        llm_mode=settings.LLM_MODE,
        llm_provider=settings.LLM_PROVIDER if settings.LLM_MODE == "api" else None,
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