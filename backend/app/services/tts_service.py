"""
Text-to-Speech (TTS) Service using ElevenLabs.
Converts text responses to natural speech for the interviewer.
"""

import asyncio
from typing import Optional, AsyncIterator
from dataclasses import dataclass
import base64

from backend.app.config import settings


@dataclass
class SpeechResult:
    """Result of text-to-speech synthesis."""
    audio_data: bytes
    audio_base64: str
    content_type: str
    duration_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "audio_base64": self.audio_base64,
            "content_type": self.content_type,
            "duration_seconds": self.duration_seconds,
        }


class ElevenLabsTTSService:
    """
    Text-to-Speech service using ElevenLabs API.
    
    Produces natural-sounding speech for the AI interviewer.
    """
    
    def __init__(self):
        if not settings.ELEVENLABS_API_KEY:
            raise ValueError("ELEVENLABS_API_KEY not set in environment")
        
        try:
            from elevenlabs.client import ElevenLabs
            from elevenlabs import VoiceSettings
            
            self.client = ElevenLabs(api_key=settings.ELEVENLABS_API_KEY)
            self.voice_id = settings.ELEVENLABS_VOICE_ID
            self.model = settings.ELEVENLABS_MODEL
            self.VoiceSettings = VoiceSettings
            
        except ImportError:
            raise ImportError(
                "elevenlabs not installed. Run: pip install elevenlabs"
            )
    
    async def synthesize(
        self,
        text: str,
        voice_id: Optional[str] = None,
        stability: float = 0.5,
        similarity_boost: float = 0.75,
    ) -> SpeechResult:
        """
        Convert text to speech.
        
        Args:
            text: Text to convert to speech
            voice_id: Optional voice ID (uses default if not provided)
            stability: Voice stability (0-1)
            similarity_boost: Voice similarity boost (0-1)
        
        Returns:
            SpeechResult with audio data
        """
        if not text or not text.strip():
            return SpeechResult(
                audio_data=b"",
                audio_base64="",
                content_type="audio/mpeg",
            )
        
        try:
            voice_id = voice_id or self.voice_id
            
            # Run in thread pool (SDK may be synchronous)
            loop = asyncio.get_event_loop()
            
            def _synthesize():
                voice_settings = self.VoiceSettings(
                    stability=stability,
                    similarity_boost=similarity_boost,
                )
                
                audio_generator = self.client.text_to_speech.convert(
                    voice_id=voice_id,
                    text=text,
                    model_id=self.model,
                    voice_settings=voice_settings,
                )
                
                # Collect all audio chunks
                audio_chunks = []
                for chunk in audio_generator:
                    audio_chunks.append(chunk)
                
                return b"".join(audio_chunks)
            
            audio_data = await loop.run_in_executor(None, _synthesize)
            audio_base64 = base64.b64encode(audio_data).decode("utf-8")
            
            return SpeechResult(
                audio_data=audio_data,
                audio_base64=audio_base64,
                content_type="audio/mpeg",
            )
            
        except Exception as e:
            print(f"ElevenLabs TTS error: {e}")
            return SpeechResult(
                audio_data=b"",
                audio_base64="",
                content_type="audio/mpeg",
            )
    
    async def synthesize_stream(
        self,
        text: str,
        voice_id: Optional[str] = None,
    ) -> AsyncIterator[bytes]:
        """
        Stream synthesized audio chunks.
        
        Useful for real-time playback while synthesis is ongoing.
        
        Args:
            text: Text to convert to speech
            voice_id: Optional voice ID
        
        Yields:
            Audio data chunks
        """
        if not text or not text.strip():
            return
        
        try:
            voice_id = voice_id or self.voice_id
            
            loop = asyncio.get_event_loop()
            
            def _get_stream():
                return self.client.text_to_speech.convert(
                    voice_id=voice_id,
                    text=text,
                    model_id=self.model,
                )
            
            audio_generator = await loop.run_in_executor(None, _get_stream)
            
            for chunk in audio_generator:
                yield chunk
                
        except Exception as e:
            print(f"ElevenLabs TTS stream error: {e}")
    
    async def get_available_voices(self) -> list:
        """Get list of available voices."""
        try:
            loop = asyncio.get_event_loop()
            
            def _get_voices():
                response = self.client.voices.get_all()
                return [
                    {
                        "voice_id": v.voice_id,
                        "name": v.name,
                        "category": v.category,
                    }
                    for v in response.voices
                ]
            
            return await loop.run_in_executor(None, _get_voices)
            
        except Exception as e:
            print(f"Error getting voices: {e}")
            return []


class TTSServiceFactory:
    """Factory to create TTS service based on configuration."""
    
    _instance: Optional[ElevenLabsTTSService] = None
    
    @classmethod
    def get_service(cls) -> ElevenLabsTTSService:
        if cls._instance is None:
            cls._instance = cls._create_service()
        return cls._instance
    
    @classmethod
    def _create_service(cls) -> ElevenLabsTTSService:
        if settings.TTS_PROVIDER == "elevenlabs":
            print("Initializing ElevenLabs TTS Service")
            return ElevenLabsTTSService()
        else:
            raise ValueError(f"Unknown TTS provider: {settings.TTS_PROVIDER}")
    
    @classmethod
    def reset(cls):
        cls._instance = None


def get_tts_service() -> ElevenLabsTTSService:
    """Get the configured TTS service."""
    return TTSServiceFactory.get_service()