"""
WebSocket endpoint for real-time interview communication.

Supports both API mode and Custom/Hybrid mode.
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
from backend.app.core.grilling_engine import GrillingEngine
from backend.app.api.deps import get_retriever, get_llm
from backend.app.services.llm_service import (
    BaseLLMService,
    LLMServiceFactory,
    HybridModelService,
)
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


# ============== Helper Functions for Model Type ==============

def get_hybrid_service() -> HybridModelService:
    """Get the hybrid model service instance."""
    return LLMServiceFactory.get_hybrid_service()


def create_interview_agent_for_session(session: InterviewSession) -> Optional[InterviewAgent]:
    """
    Create InterviewAgent based on session's model_type.
    
    Returns None for custom mode (handled separately).
    """
    if session.model_type == "custom":
        # Custom mode uses _process_answer_hybrid, not InterviewAgent
        return None
    
    # API mode: create standard InterviewAgent
    from backend.app.services.llm_service import get_llm_service
    from backend.app.api.deps import get_retriever
    
    llm = get_llm_service()
    retriever = get_retriever()
    return InterviewAgent(llm_service=llm, retriever=retriever)


async def process_answer_hybrid(
    session: InterviewSession,
    answer: str,
    session_store: SessionStore,
) -> dict:
    """
    Process answer using Hybrid approach (Custom Model execution).
    
    Uses GrillingEngine with Custom Model for evaluation and follow-up generation.
    Returns a WebSocket-formatted message dict.
    """
    current_question = session.current_question
    if not current_question:
        return create_ws_message(
            msg_type=WSMessageType.COMPLETE,
            content="Interview complete! Thank you for your responses.",
        )
    
    # Add answer to conversation
    session.add_message(role=MessageRole.CANDIDATE, content=answer)
    
    # Get hybrid service
    hybrid_service = get_hybrid_service()
    
    # Restore prepared context if needed
    prepared_context = {}
    if hasattr(session, 'prepared_context') and session.prepared_context:
        prepared_context = session.prepared_context
        hybrid_service.set_prepared_context(session.session_id, prepared_context)
    
    # Create GrillingEngine with Custom Model and prepared context
    grilling_engine = GrillingEngine(
        llm_service=hybrid_service.interviewer,  # Use Custom Model
        model_type="custom",  # Enables compact prompts
        prepared_context=prepared_context,
    )
    
    # Build conversation history
    conversation_history = [
        {"role": m.role.value, "content": m.content, "is_follow_up": m.is_follow_up}
        for m in session.conversation[-10:]
    ]
    
    # Evaluate answer using GrillingEngine (with Custom Model)
    evaluation = await grilling_engine.evaluate_answer(
        question=current_question,
        answer=answer,
        resume_context="",  # Not needed - using prepared_context
        question_type=session.mode,
        conversation_history=conversation_history,
        follow_up_count=session.current_follow_up_count,
        question_index=session.current_question_index,
    )
    
    # Use GrillingEngine's decision logic
    should_followup = grilling_engine.should_grill(
        evaluation=evaluation,
        follow_up_count=session.current_follow_up_count,
        max_follow_ups=session.max_follow_ups,
    )
    
    if should_followup:
        # Generate follow-up using GrillingEngine (with Custom Model)
        followup = await grilling_engine.generate_follow_up(
            question=current_question,
            answer=answer,
            evaluation=evaluation,
            conversation_history=conversation_history,
            question_type=session.mode,
        )
        
        session.increment_follow_up()
        session.add_message(
            role=MessageRole.INTERVIEWER,
            content=followup,
            is_follow_up=True,
            metadata={
                "gap": evaluation.gap_analysis.priority_gap.value if evaluation.gap_analysis.priority_gap else None,
                "score": evaluation.score,
                "detected_gaps": [g.value for g in evaluation.gap_analysis.detected_gaps],
            },
        )
        
        session_store.update(session)
        
        return create_ws_message(
            msg_type=WSMessageType.FOLLOW_UP,
            content=followup,
            data={
                "question_number": session.current_question_index + 1,
                "total_questions": len(session.questions),
                "evaluation": evaluation.to_dict(),
                "follow_up_count": session.current_follow_up_count,
                "priority_gap": evaluation.gap_analysis.priority_gap.value if evaluation.gap_analysis.priority_gap else None,
            }
        )
    
    # Move to next question
    next_question = session.next_question()
    
    if next_question is None:
        session.status = SessionStatus.COMPLETED
        session.add_message(role=MessageRole.SYSTEM, content="Interview completed.")
        session_store.update(session)
        
        return create_ws_message(
            msg_type=WSMessageType.COMPLETE,
            content="Excellent! That concludes our interview. Thank you for your thoughtful responses.",
            data={
                "final_evaluation": evaluation.to_dict(),
            }
        )
    
    # Ask next question
    session.add_message(
        role=MessageRole.INTERVIEWER,
        content=next_question,
        metadata={"question_number": session.current_question_index + 1},
    )
    
    session_store.update(session)
    
    return create_ws_message(
        msg_type=WSMessageType.QUESTION,
        content=next_question,
        data={
            "question_number": session.current_question_index + 1,
            "total_questions": len(session.questions),
            "previous_evaluation": evaluation.to_dict(),
        }
    )


# ============== WebSocket Endpoint ==============

@router.websocket("/ws/interview/{session_id}")
async def interview_websocket(
    websocket: WebSocket,
    session_id: str,
):
    """
    WebSocket endpoint for real-time interview.
    
    Supports both API mode (InterviewAgent) and Custom mode (Hybrid).
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
            "model_type": session.model_type,
            "status": session.status.value,
            "current_question": session.current_question,
            "question_number": session.current_question_index + 1,
            "total_questions": len(session.questions),
        }
    ))
    
    # Create interview agent based on model_type
    # For custom mode, agent will be None and we use process_answer_hybrid
    agent = create_interview_agent_for_session(session)
    
    if agent:
        print(f"[WebSocket] Using InterviewAgent for API mode")
    else:
        print(f"[WebSocket] Using Hybrid mode for Custom model")
    
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
                # For custom mode with pre-generated questions, just return current question
                if session.model_type == "custom" and session.questions:
                    response_msg = create_ws_message(
                        msg_type=WSMessageType.QUESTION,
                        content=session.current_question or "No question available",
                        data={
                            "question_number": session.current_question_index + 1,
                            "total_questions": len(session.questions),
                            "status": "resumed",
                        }
                    )
                else:
                    response_msg = await handle_start(agent, session, message)
                
                # Attach Audio
                audio = await generate_voice_response(response_msg.get("content"))
                if audio:
                    response_msg["audio_base64"] = audio
                await websocket.send_json(response_msg)
            
            elif msg_type == WSMessageType.ANSWER:
                # Route to appropriate handler based on model_type
                if session.model_type == "custom":
                    response_msg = await handle_answer_hybrid(session, content, session_store)
                else:
                    response_msg = await handle_answer(agent, session, content)
                
                # Attach Audio
                audio = await generate_voice_response(response_msg.get("content"))
                if audio:
                    response_msg["audio_base64"] = audio
                await websocket.send_json(response_msg)
            
            elif msg_type == WSMessageType.SKIP:
                # Route to appropriate handler based on model_type
                if session.model_type == "custom":
                    response_msg = await handle_skip_hybrid(session, session_store)
                else:
                    response_msg = await handle_skip(agent, session)
                
                # Attach Audio
                audio = await generate_voice_response(response_msg.get("content"))
                if audio:
                    response_msg["audio_base64"] = audio
                await websocket.send_json(response_msg)
            
            elif msg_type == WSMessageType.END:
                if session.model_type == "custom":
                    response_msg = await handle_end_hybrid(session, session_store)
                else:
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
        import traceback
        traceback.print_exc()
        try:
            await websocket.send_json(create_ws_message(
                msg_type=WSMessageType.ERROR,
                error=str(e)
            ))
        except:
            pass
    
    finally:
        manager.disconnect(session_id)


# ============== Message Handlers (API Mode) ==============

async def handle_start(
    agent: InterviewAgent,
    session: InterviewSession,
    message: dict,
) -> dict:
    """Handle start interview message (API mode)."""
    
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
    """Handle answer submission (API mode)."""
    
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
    """Handle skip question (API mode)."""
    
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
    """Handle end interview (API mode)."""
    
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


# ============== Message Handlers (Hybrid/Custom Mode) ==============

async def handle_answer_hybrid(
    session: InterviewSession,
    answer: str,
    session_store: SessionStore,
) -> dict:
    """Handle answer submission (Hybrid/Custom mode)."""
    
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
    
    return await process_answer_hybrid(session, answer, session_store)


async def handle_skip_hybrid(
    session: InterviewSession,
    session_store: SessionStore,
) -> dict:
    """Handle skip question (Hybrid/Custom mode)."""
    
    if session.status != SessionStatus.IN_PROGRESS:
        return create_ws_message(
            msg_type=WSMessageType.ERROR,
            error=f"Interview is not in progress. Status: {session.status.value}",
        )
    
    session.add_message(
        role=MessageRole.CANDIDATE,
        content="[Skipped]",
        metadata={"skipped": True},
    )
    
    next_question = session.next_question()
    
    if next_question is None:
        session.status = SessionStatus.COMPLETED
        session_store.update(session)
        
        return create_ws_message(
            msg_type=WSMessageType.COMPLETE,
            content="Interview complete! Thank you for your responses.",
        )
    
    session.add_message(
        role=MessageRole.INTERVIEWER,
        content=next_question,
        metadata={"question_number": session.current_question_index + 1},
    )
    
    session_store.update(session)
    
    return create_ws_message(
        msg_type=WSMessageType.QUESTION,
        content=next_question,
        data={
            "question_number": session.current_question_index + 1,
            "total_questions": len(session.questions),
            "skipped": True,
        }
    )


async def handle_end_hybrid(
    session: InterviewSession,
    session_store: SessionStore,
) -> dict:
    """Handle end interview (Hybrid/Custom mode)."""
    
    session.status = SessionStatus.CANCELLED
    session.add_message(
        role=MessageRole.SYSTEM,
        content="Interview ended by user.",
    )
    
    session_store.update(session)
    
    # Build summary
    candidate_messages = [
        m for m in session.conversation
        if m.role == MessageRole.CANDIDATE and "[Skipped]" not in m.content
    ]
    follow_ups = [m for m in session.conversation if m.is_follow_up]
    
    summary = {
        "session_id": session.session_id,
        "resume_id": session.resume_id,
        "mode": session.mode,
        "model_type": session.model_type,
        "status": session.status.value,
        "questions_asked": session.current_question_index,
        "total_questions": len(session.questions),
        "answers_given": len(candidate_messages),
        "follow_ups_asked": len(follow_ups),
    }
    
    return create_ws_message(
        msg_type=WSMessageType.COMPLETE,
        content="Interview ended. Thank you for your time.",
        data={
            "summary": summary,
        }
    )