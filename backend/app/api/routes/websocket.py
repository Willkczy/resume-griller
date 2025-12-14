"""
WebSocket endpoint for real-time interview communication.
"""

import json
import asyncio
from typing import Optional
from datetime import datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, status
from fastapi.websockets import WebSocketState

from backend.app.config import settings

from backend.app.db.session_store import (
    InterviewSession,
    SessionStatus,
    MessageRole,
    SessionStore,
    get_session_store,
)
from backend.app.core.interview_agent import InterviewAgent, ResponseType
from backend.app.api.deps import get_retriever, get_llm
from backend.app.services.llm_service import BaseLLMService
from backend.app.services.stt_service import get_stt_service
from backend.app.services.tts_service import get_tts_service
from rag.retriever import InterviewRetriever


router = APIRouter(tags=["websocket"])


# ============== WebSocket Message Types ==============

class WSMessageType:
    """WebSocket message types."""
    # Client -> Server
    START = "start"           # Start interview
    ANSWER = "answer"         # Submit text answer
    ANSWER_AUDIO = "answer_audio" # Submit audio answer (Base64)
    SKIP = "skip"             # Skip question
    END = "end"               # End interview
    PING = "ping"             # Keep-alive ping
    
    # Server -> Client
    QUESTION = "question"     # New question
    FOLLOW_UP = "follow_up"   # Follow-up question
    EVALUATION = "evaluation" # Answer evaluation
    COMPLETE = "complete"     # Interview complete
    ERROR = "error"           # Error message
    PONG = "pong"             # Keep-alive pong
    CONNECTED = "connected"   # Connection established
    TRANSCRIPT = "transcript" # STT result (what the AI heard)


class ConnectionManager:
    """Manages WebSocket connections."""
    
    def __init__(self):
        # session_id -> WebSocket
        self.active_connections: dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str):
        """Accept connection and register it."""
        await websocket.accept()
        self.active_connections[session_id] = websocket
        print(f"WebSocket connected: {session_id}")
    
    def disconnect(self, session_id: str):
        """Remove connection from registry."""
        if session_id in self.active_connections:
            del self.active_connections[session_id]
            print(f"WebSocket disconnected: {session_id}")
    
    async def send_message(self, session_id: str, message: dict):
        """Send message to a specific session."""
        websocket = self.active_connections.get(session_id)
        if websocket and websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_json(message)
    
    async def broadcast(self, message: dict):
        """Broadcast message to all connections."""
        for session_id, websocket in self.active_connections.items():
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_json(message)


# Global connection manager
manager = ConnectionManager()


def create_ws_message(
    msg_type: str,
    content: str = "",
    data: dict = None,
    error: str = None,
) -> dict:
    """Create a standardized WebSocket message."""
    message = {
        "type": msg_type,
        "content": content,
        "timestamp": datetime.utcnow().isoformat(),
    }
    
    if data:
        message["data"] = data
    
    if error:
        message["error"] = error
    
    return message

# ============== Voice Helper Functions ==============

async def generate_voice_response(text: str) -> Optional[str]:
    """
    Generate TTS audio for the response text.
    Returns Base64 audio string or None if failed/disabled.
    """
    if not settings.VOICE_ENABLED:
        return None
        
    try:
        tts = get_tts_service()
        # Ensure we don't try to speak empty text
        if not text or not text.strip():
            return None
            
        result = await tts.synthesize(text)
        return result.audio_base64
    except Exception as e:
        print(f"TTS Generation failed: {e}")
        return None


# ============== WebSocket Endpoint ==============

@router.websocket("/ws/interview/{session_id}")
async def interview_websocket(
    websocket: WebSocket,
    session_id: str,
):
    """
    WebSocket endpoint for real-time interview.
    
    Message format (Client -> Server):
    {
        "type": "start" | "answer" | "skip" | "end" | "ping",
        "content": "answer text" (for answer type),
        "data": {} (optional additional data)
    }
    
    Message format (Server -> Client):
    {
        "type": "question" | "follow_up" | "evaluation" | "complete" | "error" | "pong" | "connected",
        "content": "message content",
        "timestamp": "ISO timestamp",
        "data": {
            "question_number": 1,
            "total_questions": 5,
            "evaluation": {...},
            ...
        }
    }
    """
    # Get dependencies
    session_store = get_session_store()
    
    # Verify session exists
    session = session_store.get(session_id)
    if not session:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    
    # Accept connection
    await manager.connect(websocket, session_id)
    
    # Send connected message
    await websocket.send_json(create_ws_message(
        msg_type=WSMessageType.CONNECTED,
        content="Connected to interview session",
        data={
            "session_id": session_id,
            "resume_id": session.resume_id,
            "mode": session.mode,
            "status": session.status.value,
            "current_question": session.current_question,
            "question_number": session.current_question_index + 1,
            "total_questions": len(session.questions),
        }
    ))
    
    # Create interview agent
    from backend.app.services.llm_service import get_llm_service
    from backend.app.api.deps import get_retriever
    
    llm = get_llm_service()
    retriever = get_retriever()
    agent = InterviewAgent(llm_service=llm, retriever=retriever)
    
    try:
        while True:
            # Receive message
            try:
                raw_message = await websocket.receive_text()
                message = json.loads(raw_message)
            except json.JSONDecodeError:
                await websocket.send_json(create_ws_message(
                    msg_type=WSMessageType.ERROR,
                    error="Invalid JSON format"
                ))
                continue
            
            msg_type = message.get("type", "")
            content = message.get("content", "")
            
            # Refresh session from store
            session = session_store.get(session_id)
            if not session:
                await websocket.send_json(create_ws_message(
                    msg_type=WSMessageType.ERROR,
                    error="Session not found"
                ))
                break

            # --- Handle Ping ---
            if msg_type == WSMessageType.PING:
                await websocket.send_json(create_ws_message(
                    msg_type=WSMessageType.PONG,
                    content="pong"
                ))
                continue

            # --- Handle Audio Input (STT) ---
            if msg_type == WSMessageType.ANSWER_AUDIO:
                if not settings.VOICE_ENABLED:
                    await websocket.send_json(create_ws_message(
                        msg_type=WSMessageType.ERROR,
                        error="Voice services disabled"
                    ))
                    continue
                
                try:
                    # 1. Transcribe Audio
                    stt = get_stt_service()
                    transcription = await stt.transcribe_base64(content)
                    text_answer = transcription.text
                    
                    # 2. Send Transcript back to client (User Experience)
                    await websocket.send_json(create_ws_message(
                        msg_type=WSMessageType.TRANSCRIPT,
                        content=text_answer,
                        data={"confidence": transcription.confidence}
                    ))
                    
                    # 3. Treat as normal answer
                    msg_type = WSMessageType.ANSWER
                    content = text_answer
                    
                except Exception as e:
                    print(f"STT Error: {e}")
                    await websocket.send_json(create_ws_message(
                        msg_type=WSMessageType.ERROR,
                        error=f"Transcription failed: {str(e)}"
                    ))
                    continue

            # --- Handle Text Logic ---
            
            if msg_type == WSMessageType.START:
                response_msg = await handle_start(agent, session, message)
                # Attach Audio
                audio = await generate_voice_response(response_msg.get("content"))
                if audio:
                    response_msg["audio_base64"] = audio
                await websocket.send_json(response_msg)
            
            elif msg_type == WSMessageType.ANSWER:
                response_msg = await handle_answer(agent, session, content)
                # Attach Audio
                audio = await generate_voice_response(response_msg.get("content"))
                if audio:
                    response_msg["audio_base64"] = audio
                await websocket.send_json(response_msg)
            
            elif msg_type == WSMessageType.SKIP:
                response_msg = await handle_skip(agent, session)
                # Attach Audio
                audio = await generate_voice_response(response_msg.get("content"))
                if audio:
                    response_msg["audio_base64"] = audio
                await websocket.send_json(response_msg)
            
            elif msg_type == WSMessageType.END:
                response_msg = await handle_end(agent, session)
                # Usually no audio for end summary unless desired
                await websocket.send_json(response_msg)
                break
            
            # If msg_type was unknown (and not handled by audio logic)
            elif msg_type not in [WSMessageType.START, WSMessageType.ANSWER, WSMessageType.SKIP, WSMessageType.END]:
                await websocket.send_json(create_ws_message(
                    msg_type=WSMessageType.ERROR,
                    error=f"Unknown message type: {msg_type}"
                ))
    
    except WebSocketDisconnect:
        print(f"Client disconnected: {session_id}")
    
    except Exception as e:
        print(f"WebSocket error: {e}")
        try:
            await websocket.send_json(create_ws_message(
                msg_type=WSMessageType.ERROR,
                error=str(e)
            ))
        except:
            pass
    
    finally:
        manager.disconnect(session_id)


# ============== Message Handlers ==============

async def handle_start(
    agent: InterviewAgent,
    session: InterviewSession,
    message: dict,
) -> dict:
    """Handle start interview message."""
    
    # Check if already started
    if session.status == SessionStatus.IN_PROGRESS and session.questions:
        # Already started, return current question
        return create_ws_message(
            msg_type=WSMessageType.QUESTION,
            content=session.current_question or "No question available",
            data={
                "question_number": session.current_question_index + 1,
                "total_questions": len(session.questions),
                "status": "resumed",
            }
        )
    
    # Start interview
    num_questions = message.get("data", {}).get("num_questions", 5)
    
    response = await agent.start_interview(
        session=session,
        num_questions=num_questions,
    )
    
    if response.type == ResponseType.ERROR:
        return create_ws_message(
            msg_type=WSMessageType.ERROR,
            error=response.content,
        )
    
    return create_ws_message(
        msg_type=WSMessageType.QUESTION,
        content=response.content,
        data={
            "question_number": response.question_number,
            "total_questions": response.total_questions,
        }
    )


async def handle_answer(
    agent: InterviewAgent,
    session: InterviewSession,
    answer: str,
) -> dict:
    """Handle answer submission."""
    
    if not answer or not answer.strip():
        return create_ws_message(
            msg_type=WSMessageType.ERROR,
            error="Answer cannot be empty",
        )
    
    if session.status != SessionStatus.IN_PROGRESS:
        return create_ws_message(
            msg_type=WSMessageType.ERROR,
            error=f"Interview is not in progress. Status: {session.status.value}",
        )
    
    # Process answer
    response = await agent.process_answer(
        session=session,
        answer=answer,
    )
    
    # Map response type to WebSocket message type
    if response.type == ResponseType.FOLLOW_UP:
        return create_ws_message(
            msg_type=WSMessageType.FOLLOW_UP,
            content=response.content,
            data={
                "question_number": response.question_number,
                "total_questions": response.total_questions,
                "evaluation": response.evaluation,
            }
        )
    
    elif response.type == ResponseType.QUESTION:
        return create_ws_message(
            msg_type=WSMessageType.QUESTION,
            content=response.content,
            data={
                "question_number": response.question_number,
                "total_questions": response.total_questions,
                "previous_evaluation": response.evaluation,
            }
        )
    
    elif response.type == ResponseType.COMPLETE:
        return create_ws_message(
            msg_type=WSMessageType.COMPLETE,
            content=response.content,
            data={
                "final_evaluation": response.evaluation,
            }
        )
    
    else:
        return create_ws_message(
            msg_type=WSMessageType.ERROR,
            error=response.content,
        )


async def handle_skip(
    agent: InterviewAgent,
    session: InterviewSession,
) -> dict:
    """Handle skip question."""
    
    if session.status != SessionStatus.IN_PROGRESS:
        return create_ws_message(
            msg_type=WSMessageType.ERROR,
            error=f"Interview is not in progress. Status: {session.status.value}",
        )
    
    response = await agent.skip_question(session=session)
    
    if response.type == ResponseType.COMPLETE:
        return create_ws_message(
            msg_type=WSMessageType.COMPLETE,
            content=response.content,
        )
    
    return create_ws_message(
        msg_type=WSMessageType.QUESTION,
        content=response.content,
        data={
            "question_number": response.question_number,
            "total_questions": response.total_questions,
            "skipped": True,
        }
    )


async def handle_end(
    agent: InterviewAgent,
    session: InterviewSession,
) -> dict:
    """Handle end interview."""
    
    response = await agent.end_interview(
        session=session,
        reason="ended by user",
    )
    
    # Get summary
    summary = await agent.get_interview_summary(session)
    
    return create_ws_message(
        msg_type=WSMessageType.COMPLETE,
        content=response.content,
        data={
            "summary": summary,
        }
    )