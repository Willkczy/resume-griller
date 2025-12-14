"""
Speech-to-Text (STT) Service using Deepgram.
Converts audio to text for interview responses.
"""

import asyncio
import httpx
from typing import Optional
from dataclasses import dataclass
import base64

from backend.app.config import settings


@dataclass
class TranscriptionResult:
    """Result of speech-to-text transcription."""
    text: str
    confidence: float
    is_final: bool
    duration_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "is_final": self.is_final,
            "duration_seconds": self.duration_seconds,
        }


class DeepgramSTTService:
    """
    Speech-to-Text service using Deepgram API.
    Uses direct HTTP API for simplicity and reliability.
    """
    
    BASE_URL = "https://api.deepgram.com/v1/listen"
    
    def __init__(self):
        if not settings.DEEPGRAM_API_KEY:
            raise ValueError("DEEPGRAM_API_KEY not set in environment")
        
        self.api_key = settings.DEEPGRAM_API_KEY
        self.model = settings.DEEPGRAM_MODEL
        self.language = settings.DEEPGRAM_LANGUAGE
    
    async def transcribe_audio(
        self,
        audio_data: bytes,
        mime_type: str = "audio/webm",
    ) -> TranscriptionResult:
        """
        Transcribe pre-recorded audio.
        
        Args:
            audio_data: Raw audio bytes
            mime_type: Audio MIME type (audio/webm, audio/wav, audio/mp3, etc.)
        
        Returns:
            TranscriptionResult with transcribed text
        """
        try:
            params = {
                "model": self.model,
                "language": self.language,
                "smart_format": "true",
                "punctuate": "true",
            }
            
            headers = {
                "Authorization": f"Token {self.api_key}",
                "Content-Type": mime_type,
            }
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.BASE_URL,
                    params=params,
                    headers=headers,
                    content=audio_data,
                )
                
                if response.status_code != 200:
                    print(f"Deepgram API error: {response.status_code} - {response.text}")
                    return TranscriptionResult(
                        text="",
                        confidence=0.0,
                        is_final=True,
                        duration_seconds=0.0,
                    )
                
                result = response.json()
                
                # Extract transcript from response
                channels = result.get("results", {}).get("channels", [])
                if channels and channels[0].get("alternatives"):
                    alternative = channels[0]["alternatives"][0]
                    transcript = alternative.get("transcript", "")
                    confidence = alternative.get("confidence", 0.0)
                else:
                    transcript = ""
                    confidence = 0.0
                
                duration = result.get("metadata", {}).get("duration", 0.0)
                
                return TranscriptionResult(
                    text=transcript,
                    confidence=confidence,
                    is_final=True,
                    duration_seconds=duration,
                )
            
        except Exception as e:
            print(f"Deepgram transcription error: {e}")
            return TranscriptionResult(
                text="",
                confidence=0.0,
                is_final=True,
                duration_seconds=0.0,
            )
    
    async def transcribe_base64(
        self,
        audio_base64: str,
        mime_type: str = "audio/webm",
    ) -> TranscriptionResult:
        """
        Transcribe base64-encoded audio.
        
        Args:
            audio_base64: Base64 encoded audio string
            mime_type: Audio MIME type
        
        Returns:
            TranscriptionResult with transcribed text
        """
        audio_data = base64.b64decode(audio_base64)
        return await self.transcribe_audio(audio_data, mime_type)


class STTServiceFactory:
    """Factory to create STT service based on configuration."""
    
    _instance: Optional[DeepgramSTTService] = None
    
    @classmethod
    def get_service(cls) -> DeepgramSTTService:
        if cls._instance is None:
            cls._instance = cls._create_service()
        return cls._instance
    
    @classmethod
    def _create_service(cls) -> DeepgramSTTService:
        if settings.STT_PROVIDER == "deepgram":
            print("Initializing Deepgram STT Service")
            return DeepgramSTTService()
        else:
            raise ValueError(f"Unknown STT provider: {settings.STT_PROVIDER}")
    
    @classmethod
    def reset(cls):
        cls._instance = None


def get_stt_service() -> DeepgramSTTService:
    """Get the configured STT service."""
    return STTServiceFactory.get_service()