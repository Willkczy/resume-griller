"""
Voice API routes for speech-to-text and text-to-speech.
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Depends, status
from fastapi.responses import Response
from pydantic import BaseModel, Field

from backend.app.config import settings
from backend.app.services.stt_service import get_stt_service, DeepgramSTTService
from backend.app.services.tts_service import get_tts_service, ElevenLabsTTSService


router = APIRouter(prefix="/voice", tags=["voice"])


# ============== Request/Response Schemas ==============

class TranscribeRequest(BaseModel):
    """Request for audio transcription."""
    audio_base64: str = Field(..., description="Base64 encoded audio")
    mime_type: str = Field(default="audio/webm", description="Audio MIME type")


class TranscribeResponse(BaseModel):
    """Response from transcription."""
    text: str
    confidence: float
    is_final: bool
    duration_seconds: float


class SynthesizeRequest(BaseModel):
    """Request for text-to-speech synthesis."""
    text: str = Field(..., min_length=1, max_length=5000)
    voice_id: Optional[str] = None
    stability: float = Field(default=0.5, ge=0, le=1)
    similarity_boost: float = Field(default=0.75, ge=0, le=1)


class SynthesizeResponse(BaseModel):
    """Response from synthesis."""
    audio_base64: str
    content_type: str
    duration_seconds: float


class VoiceInfo(BaseModel):
    """Voice information."""
    voice_id: str
    name: str
    category: str


class VoiceStatusResponse(BaseModel):
    """Voice service status."""
    enabled: bool
    stt_provider: str
    tts_provider: str
    stt_available: bool
    tts_available: bool


# ============== Endpoints ==============

@router.get("/status", response_model=VoiceStatusResponse)
async def get_voice_status():
    """Check voice services status."""
    stt_available = False
    tts_available = False
    
    if settings.VOICE_ENABLED:
        try:
            get_stt_service()
            stt_available = True
        except Exception as e:
            print(f"STT not available: {e}")
        
        try:
            get_tts_service()
            tts_available = True
        except Exception as e:
            print(f"TTS not available: {e}")
    
    return VoiceStatusResponse(
        enabled=settings.VOICE_ENABLED,
        stt_provider=settings.STT_PROVIDER,
        tts_provider=settings.TTS_PROVIDER,
        stt_available=stt_available,
        tts_available=tts_available,
    )


@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_audio(
    request: TranscribeRequest,
):
    """
    Transcribe audio to text using Deepgram.
    
    Send base64-encoded audio and receive transcribed text.
    """
    if not settings.VOICE_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Voice services are disabled",
        )
    
    try:
        stt = get_stt_service()
        result = await stt.transcribe_base64(
            audio_base64=request.audio_base64,
            mime_type=request.mime_type,
        )
        
        return TranscribeResponse(
            text=result.text,
            confidence=result.confidence,
            is_final=result.is_final,
            duration_seconds=result.duration_seconds,
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Transcription failed: {str(e)}",
        )


@router.post("/transcribe/file", response_model=TranscribeResponse)
async def transcribe_audio_file(
    file: UploadFile = File(...),
):
    """
    Transcribe an uploaded audio file.
    
    Accepts audio files (webm, wav, mp3, ogg, etc.)
    """
    if not settings.VOICE_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Voice services are disabled",
        )
    
    # Validate file type
    allowed_types = ["audio/webm", "audio/wav", "audio/mp3", "audio/mpeg", "audio/ogg"]
    content_type = file.content_type or "audio/webm"
    
    if content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid audio type. Allowed: {allowed_types}",
        )
    
    try:
        stt = get_stt_service()
        audio_data = await file.read()
        
        result = await stt.transcribe_audio(
            audio_data=audio_data,
            mime_type=content_type,
        )
        
        return TranscribeResponse(
            text=result.text,
            confidence=result.confidence,
            is_final=result.is_final,
            duration_seconds=result.duration_seconds,
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Transcription failed: {str(e)}",
        )


@router.post("/synthesize", response_model=SynthesizeResponse)
async def synthesize_speech(
    request: SynthesizeRequest,
):
    """
    Convert text to speech using ElevenLabs.
    
    Returns base64-encoded audio.
    """
    if not settings.VOICE_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Voice services are disabled",
        )
    
    try:
        tts = get_tts_service()
        result = await tts.synthesize(
            text=request.text,
            voice_id=request.voice_id,
            stability=request.stability,
            similarity_boost=request.similarity_boost,
        )
        
        return SynthesizeResponse(
            audio_base64=result.audio_base64,
            content_type=result.content_type,
            duration_seconds=result.duration_seconds,
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Synthesis failed: {str(e)}",
        )


@router.post("/synthesize/audio")
async def synthesize_speech_audio(
    request: SynthesizeRequest,
):
    """
    Convert text to speech and return raw audio.
    
    Returns audio/mpeg directly for playback.
    """
    if not settings.VOICE_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Voice services are disabled",
        )
    
    try:
        tts = get_tts_service()
        result = await tts.synthesize(
            text=request.text,
            voice_id=request.voice_id,
            stability=request.stability,
            similarity_boost=request.similarity_boost,
        )
        
        return Response(
            content=result.audio_data,
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": "inline; filename=speech.mp3"
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Synthesis failed: {str(e)}",
        )


@router.get("/voices", response_model=list[VoiceInfo])
async def get_available_voices():
    """Get list of available TTS voices."""
    if not settings.VOICE_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Voice services are disabled",
        )
    
    try:
        tts = get_tts_service()
        voices = await tts.get_available_voices()
        return [VoiceInfo(**v) for v in voices]
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get voices: {str(e)}",
        )