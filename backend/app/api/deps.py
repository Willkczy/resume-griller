"""
Dependency injection for FastAPI routes.
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from typing import Generator, Optional
from functools import lru_cache

from backend.app.config import settings
from backend.app.services.llm_service import BaseLLMService, get_llm_service

# RAG components - use try/except for graceful handling
try:
    from rag.resume_parser import ResumeParser
    from rag.chunker import ResumeChunker
    from rag.embedder import ResumeEmbedder
    from rag.retriever import InterviewRetriever
except ImportError as e:
    print(f"Warning: Could not import RAG modules: {e}")
    ResumeParser = None
    ResumeChunker = None
    ResumeEmbedder = None
    InterviewRetriever = None


# ============== RAG Components ==============

@lru_cache()
def get_resume_parser() -> ResumeParser:
    """Get cached ResumeParser instance."""
    if ResumeParser is None:
        raise ImportError("RAG modules not available")
    return ResumeParser()


@lru_cache()
def get_chunker() -> ResumeChunker:
    """Get cached ResumeChunker instance."""
    if ResumeChunker is None:
        raise ImportError("RAG modules not available")
    return ResumeChunker()


@lru_cache()
def get_embedder() -> ResumeEmbedder:
    """Get cached ResumeEmbedder instance."""
    if ResumeEmbedder is None:
        raise ImportError("RAG modules not available")
    return ResumeEmbedder(
        model_name=settings.EMBEDDING_MODEL,
        persist_dir=settings.CHROMA_PERSIST_DIR,
    )


@lru_cache()
def get_retriever() -> InterviewRetriever:
    """Get cached InterviewRetriever instance."""
    if InterviewRetriever is None:
        raise ImportError("RAG modules not available")
    embedder = get_embedder()
    return InterviewRetriever(embedder=embedder)


# ============== LLM Service ==============

def get_llm() -> BaseLLMService:
    """Get LLM service instance."""
    return get_llm_service()


# ============== File Upload Helpers ==============

import os


def ensure_upload_dir() -> Path:
    """Ensure upload directory exists and return path."""
    upload_path = Path(settings.UPLOAD_DIR)
    upload_path.mkdir(parents=True, exist_ok=True)
    return upload_path


def validate_file_extension(filename: str) -> bool:
    """Validate file extension is allowed."""
    if not filename:
        return False
    ext = filename.rsplit(".", 1)[-1].lower()
    return ext in settings.ALLOWED_EXTENSIONS


def generate_resume_id(filename: str) -> str:
    """Generate a unique resume ID from filename."""
    import uuid
    from datetime import datetime
    
    # Clean filename
    base_name = Path(filename).stem
    clean_name = "".join(c if c.isalnum() else "_" for c in base_name)
    
    # Add timestamp and short UUID for uniqueness
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uuid = str(uuid.uuid4())[:8]
    
    return f"{clean_name}_{timestamp}_{short_uuid}"