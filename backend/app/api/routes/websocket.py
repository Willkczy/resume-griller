"""
WebSocket endpoint for real-time interview communication.

Simplified to use the LangGraph interview graph for all orchestration.
Both "api" and "custom" model types flow through the same graph.

Voice services (STT/TTS) are still handled here since they're
transport-level concerns, not interview logic.
"""

import json
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status
from fastapi.websockets import WebSocketState

from backend.app.config import settings
from backend.app.graph import get_compiled_graph, create_initial_state, GraphServices
from backend.app.api.deps import get_retriever
from backend.app.services.stt_service import get_stt_service
from backend.app.services.tts_service import get_tts_service


router = APIRouter(tags=["websocket"])


# ============== WebSocket Message Types ==============

class WSMessageType:
    """WebSocket message types."""
    # Client -> Server
    START = "start"
    ANSWER = "answer"
    ANSWER_AUDIO = "answer_audio"
    SKIP = "skip"
    END = "end"
    PING = "ping"

    # Server -> Client
    QUESTION = "question"
    FOLLOW_UP = "follow_up"
    EVALUATION = "evaluation"
    COMPLETE = "complete"
    ERROR = "error"
    PONG = "pong"
    CONNECTED = "connected"
    TRANSCRIPT = "transcript"


class ConnectionManager:
    """Manages WebSocket connections."""

    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket

    def disconnect(self, session_id: str):
        self.active_connections.pop(session_id, None)

    async def send_message(self, session_id: str, message: dict):
        websocket = self.active_connections.get(session_id)
        if websocket and websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_json(message)


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


# ============== Voice Helper ==============

async def generate_voice_response(text: str) -> Optional[str]:
    """Generate TTS audio. Returns Base64 string or None."""
    if not settings.VOICE_ENABLED or not text or not text.strip():
        return None
    try:
        tts = get_tts_service()
        result = await tts.synthesize(text)
        return result.audio_base64
    except Exception as e:
        print(f"TTS Generation failed: {e}")
        return None


# ============== Graph Invocation Helper ==============

async def invoke_graph_for_ws(
    session_id: str,
    action: str,
    model_type: str = "api",
    initial_state: dict | None = None,
    current_answer: str | None = None,
    prepared_context: dict | None = None,
) -> dict:
    """
    Invoke the interview graph and convert result to a WS message.

    This is the single point where all WS message types call the graph.
    """
    retriever = get_retriever()
    services = GraphServices.create(model_type, retriever, prepared_context)
    graph = await get_compiled_graph()

    graph_input: dict = {"action": action}
    if initial_state:
        graph_input.update(initial_state)
    if current_answer:
        graph_input["current_answer"] = current_answer

    result = await graph.ainvoke(
        graph_input,
        config={
            "configurable": {
                "thread_id": session_id,
                "services": services,
            }
        },
    )

    # Map graph response_type to WS message type
    response_type = result.get("response_type", "error")
    ws_type_map = {
        "question": WSMessageType.QUESTION,
        "follow_up": WSMessageType.FOLLOW_UP,
        "complete": WSMessageType.COMPLETE,
        "error": WSMessageType.ERROR,
    }

    return create_ws_message(
        msg_type=ws_type_map.get(response_type, WSMessageType.ERROR),
        content=result.get("response_content", ""),
        data=result.get("response_data"),
        error=result.get("error"),
    )


# ============== WebSocket Endpoint ==============

@router.websocket("/ws/interview/{session_id}")
async def interview_websocket(
    websocket: WebSocket,
    session_id: str,
):
    """
    WebSocket endpoint for real-time interview.

    All interview logic flows through the LangGraph graph.
    The WS handler only manages: connection lifecycle, voice (STT/TTS),
    and translating between WS messages and graph invocations.
    """
    # Load session state from checkpoint
    graph = await get_compiled_graph()
    config = {"configurable": {"thread_id": session_id}}
    state_snapshot = await graph.aget_state(config)

    if not state_snapshot or not state_snapshot.values:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    session_state = state_snapshot.values

    # Accept connection
    await manager.connect(websocket, session_id)

    # Send connected message with current session info
    questions = session_state.get("questions", [])
    idx = session_state.get("current_question_index", 0)
    await websocket.send_json(create_ws_message(
        msg_type=WSMessageType.CONNECTED,
        content="Connected to interview session",
        data={
            "session_id": session_id,
            "resume_id": session_state.get("resume_id", ""),
            "mode": session_state.get("mode", "mixed"),
            "model_type": session_state.get("model_type", "api"),
            "status": session_state.get("status", "pending"),
            "current_question": questions[idx] if idx < len(questions) else None,
            "question_number": idx + 1,
            "total_questions": len(questions),
        }
    ))

    model_type = session_state.get("model_type", "api")
    prepared_context = session_state.get("prepared_context")

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

            # --- Ping ---
            if msg_type == WSMessageType.PING:
                await websocket.send_json(create_ws_message(
                    msg_type=WSMessageType.PONG, content="pong"
                ))
                continue

            # --- Audio Input (STT) ---
            if msg_type == WSMessageType.ANSWER_AUDIO:
                if not settings.VOICE_ENABLED:
                    await websocket.send_json(create_ws_message(
                        msg_type=WSMessageType.ERROR,
                        error="Voice services disabled"
                    ))
                    continue

                try:
                    stt = get_stt_service()
                    transcription = await stt.transcribe_base64(content)
                    text_answer = transcription.text

                    await websocket.send_json(create_ws_message(
                        msg_type=WSMessageType.TRANSCRIPT,
                        content=text_answer,
                        data={"confidence": transcription.confidence}
                    ))

                    # Treat as normal text answer
                    msg_type = WSMessageType.ANSWER
                    content = text_answer

                except Exception as e:
                    print(f"STT Error: {e}")
                    await websocket.send_json(create_ws_message(
                        msg_type=WSMessageType.ERROR,
                        error=f"Transcription failed: {str(e)}"
                    ))
                    continue

            # --- Start ---
            if msg_type == WSMessageType.START:
                # Check if already started (resume current question)
                current_state = await graph.aget_state(config)
                s = current_state.values if current_state else {}
                if s.get("status") == "asking" and s.get("questions"):
                    q = s["questions"]
                    qi = s.get("current_question_index", 0)
                    response_msg = create_ws_message(
                        msg_type=WSMessageType.QUESTION,
                        content=q[qi] if qi < len(q) else "No question available",
                        data={
                            "question_number": qi + 1,
                            "total_questions": len(q),
                            "status": "resumed",
                        }
                    )
                else:
                    response_msg = await invoke_graph_for_ws(
                        session_id=session_id,
                        action="start",
                        model_type=model_type,
                        prepared_context=prepared_context,
                    )

            # --- Answer ---
            elif msg_type == WSMessageType.ANSWER:
                if not content or not content.strip():
                    await websocket.send_json(create_ws_message(
                        msg_type=WSMessageType.ERROR,
                        error="Answer cannot be empty",
                    ))
                    continue

                response_msg = await invoke_graph_for_ws(
                    session_id=session_id,
                    action="answer",
                    model_type=model_type,
                    current_answer=content,
                    prepared_context=prepared_context,
                )

            # --- Skip ---
            elif msg_type == WSMessageType.SKIP:
                response_msg = await invoke_graph_for_ws(
                    session_id=session_id,
                    action="skip",
                    model_type=model_type,
                    prepared_context=prepared_context,
                )

            # --- End ---
            elif msg_type == WSMessageType.END:
                response_msg = await invoke_graph_for_ws(
                    session_id=session_id,
                    action="end",
                    model_type=model_type,
                    prepared_context=prepared_context,
                )
                # Attach audio and send before breaking
                audio = await generate_voice_response(response_msg.get("content"))
                if audio:
                    response_msg["audio_base64"] = audio
                await websocket.send_json(response_msg)
                break

            else:
                await websocket.send_json(create_ws_message(
                    msg_type=WSMessageType.ERROR,
                    error=f"Unknown message type: {msg_type}"
                ))
                continue

            # Attach TTS audio and send
            audio = await generate_voice_response(response_msg.get("content"))
            if audio:
                response_msg["audio_base64"] = audio
            await websocket.send_json(response_msg)

    except WebSocketDisconnect:
        print(f"Client disconnected: {session_id}")

    except Exception as e:
        print(f"WebSocket error: {e}")
        import traceback
        traceback.print_exc()
        try:
            await websocket.send_json(create_ws_message(
                msg_type=WSMessageType.ERROR, error=str(e)
            ))
        except Exception:
            pass

    finally:
        manager.disconnect(session_id)
