"""
Configuration settings for Resume Griller Backend.
Supports both API-based and Local LLM inference.
"""

from typing import Literal, Optional
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # App settings
    APP_NAME: str = "Resume Griller"
    DEBUG: bool = False
    API_V1_PREFIX: str = "/api/v1"
    
    # CORS
    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    
    # LLM Configuration
    LLM_MODE: Literal["api", "local"] = "api"  # "api" or "local"
    
    # API Mode settings (Claude / OpenAI)
    LLM_PROVIDER: Literal["anthropic", "openai", "gemini", "groq"] = "groq"
    ANTHROPIC_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    GOOGLE_API_KEY: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None
    
    # Model names
    ANTHROPIC_MODEL: str = "claude-sonnet-4-20250514"
    OPENAI_MODEL: str = "gpt-4o"
    GEMINI_MODEL: str = "gemini-2.5-flash"
    GROQ_MODEL: str = "llama-3.3-70b-versatile"

    # Custom Model settings (GCP VM with vLLM)
    CUSTOM_MODEL_ENABLED: bool = False
    CUSTOM_MODEL_URL: str = "http://localhost:8001/v1"  # IAP tunnel endpoint
    CUSTOM_MODEL_NAME: str = "mistralai/Mistral-7B-Instruct-v0.2"
    CUSTOM_MODEL_TIMEOUT: int = 120  #
    
    # Local Mode settings (LoRA model)
    LOCAL_MODEL_BASE: str = "mistralai/Mistral-7B-Instruct-v0.2"
    LOCAL_MODEL_LORA: str = "shubhampareek/interview-coach-lora"
    LOCAL_MODEL_DEVICE: Optional[str] = None  # auto-detect if None

    # ============== Voice Services (NEW) ==============
    VOICE_ENABLED: bool = True
    
    # STT (Speech-to-Text) - Deepgram
    STT_PROVIDER: Literal["deepgram"] = "deepgram"
    DEEPGRAM_API_KEY: Optional[str] = None
    DEEPGRAM_MODEL: str = "nova-2"  # nova-2 is the latest model
    DEEPGRAM_LANGUAGE: str = "en"
    
    # TTS (Text-to-Speech) - ElevenLabs
    TTS_PROVIDER: Literal["elevenlabs"] = "elevenlabs"
    ELEVENLABS_API_KEY: Optional[str] = None
    ELEVENLABS_VOICE_ID: str = "21m00Tcm4TlvDq8ikWAM"  # Rachel voice
    ELEVENLABS_MODEL: str = "eleven_flash_v2"
    
    # Voice Settings
    VOICE_SAMPLE_RATE: int = 16000  # 16kHz for STT
    VOICE_CHANNELS: int = 1  # Mono

    
    # RAG settings
    CHROMA_PERSIST_DIR: str = "./data/chromadb"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    
    # File upload settings
    UPLOAD_DIR: str = "./data/uploads"
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: set[str] = {"pdf", "txt", "docx"}
    
    # Interview settings
    DEFAULT_QUESTIONS_PER_SESSION: int = 5
    MAX_FOLLOW_UP_QUESTIONS: int = 3
    
    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./data/resume_griller.db"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Convenience instance
settings = get_settings()