"""
WebSocket endpoint for real-time interview communication.

Simplified to use the LangGraph interview graph for all orchestration.
Both "api" and "custom" model types flow through the same graph.

Voice services (STT/TTS) are still handled here since they're
transport-level concerns, not interview logic.
"""

import json
from datetime import UTC, datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status
from fastapi.websockets import WebSocketState

from backend.app.config import settings
from backend.app.graph import get_compiled_graph
from backend.app.graph.commands import execute_command, load_session_state
from backend.app.graph.state import (
    get_active_question,
    normalize_public_status,
)
from backend.app.graph.transitions import CommandError, command_error
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
        "timestamp": datetime.now(UTC).isoformat(),
    }
    if data:
        message["data"] = data
    if error:
        message["error"] = error
    return message


# ============== Voice Helper ==============


async def generate_voice_response(text: str, voice_enabled: bool = True) -> str | None:
    """Generate TTS audio. Returns Base64 string or None."""
    if not voice_enabled or not settings.VOICE_ENABLED or not text or not text.strip():
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
    initial_state: dict | None = None,
    current_answer: str | None = None,
) -> dict:
    """
    Invoke the interview graph and convert result to a WS message.

    This is the single point where all WS message types call the graph.
    """
    graph = await get_compiled_graph()
    try:
        result = await execute_command(
            graph,
            session_id,
            action,
            initial_state=initial_state,
            current_answer=current_answer,
        )
    except CommandError as exc:
        return create_ws_message(
            msg_type=WSMessageType.ERROR,
            content=exc.detail,
            error=exc.detail,
            data={"error_code": exc.code, "status_code": exc.status_code},
        )

    # Map graph response_type to WS message type
    response_type = result.get("response_type") or "error"
    ws_type_map = {
        "question": WSMessageType.QUESTION,
        "follow_up": WSMessageType.FOLLOW_UP,
        "complete": WSMessageType.COMPLETE,
        "error": WSMessageType.ERROR,
    }

    return create_ws_message(
        msg_type=ws_type_map.get(response_type, WSMessageType.ERROR),
        content=result.get("response_content") or "",
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
    active = get_active_question(session_state)
    await websocket.send_json(
        create_ws_message(
            msg_type=WSMessageType.CONNECTED,
            content="Connected to interview session",
            data={
                "session_id": session_id,
                "resume_id": session_state.get("resume_id", ""),
                "mode": session_state.get("mode", "mixed"),
                "model_type": session_state.get("model_type", "api"),
                "status": normalize_public_status(session_state.get("status")),
                "current_question": active["prompt"] if active else None,
                "current_question_id": active["question_id"] if active else None,
                "parent_question_id": active["parent_question_id"] if active else None,
                "is_follow_up": active["is_follow_up"] if active else False,
                "question_number": idx + 1,
                "total_questions": len(questions),
            },
        )
    )

    try:
        while True:
            # Receive message
            try:
                raw_message = await websocket.receive_text()
                message = json.loads(raw_message)
                if not isinstance(message, dict):
                    raise ValueError("Expected an object")
            except (json.JSONDecodeError, ValueError):
                await websocket.send_json(
                    create_ws_message(
                        msg_type=WSMessageType.ERROR, error="Invalid JSON format"
                    )
                )
                continue

            msg_type = message.get("type", "")
            content = message.get("content", "")
            message_data = message.get("data")
            if message_data is None:
                message_data = {}
            if not isinstance(message_data, dict):
                await websocket.send_json(
                    create_ws_message(
                        msg_type=WSMessageType.ERROR,
                        error="Message data must be an object",
                    )
                )
                continue
            voice_enabled = bool(message_data.get("voice_enabled", True))

            # --- Ping ---
            if msg_type == WSMessageType.PING:
                await websocket.send_json(
                    create_ws_message(msg_type=WSMessageType.PONG, content="pong")
                )
                continue

            # --- Audio Input (STT) ---
            if msg_type == WSMessageType.ANSWER_AUDIO:
                if not settings.VOICE_ENABLED:
                    await websocket.send_json(
                        create_ws_message(
                            msg_type=WSMessageType.ERROR,
                            error="Voice services disabled",
                        )
                    )
                    continue

                try:
                    # Reject inactive/invalid input before sending any audio externally.
                    current_state = await load_session_state(graph, session_id)
                    rejection = command_error(current_state, "answer", content)
                    if rejection:
                        raise rejection
                except CommandError as exc:
                    await websocket.send_json(
                        create_ws_message(
                            msg_type=WSMessageType.ERROR,
                            error=exc.detail,
                            data={
                                "error_code": exc.code,
                                "status_code": exc.status_code,
                            },
                        )
                    )
                    continue

                try:
                    stt = get_stt_service()
                    transcription = await stt.transcribe_base64(content)
                    text_answer = transcription.text

                    await websocket.send_json(
                        create_ws_message(
                            msg_type=WSMessageType.TRANSCRIPT,
                            content=text_answer,
                            data={"confidence": transcription.confidence},
                        )
                    )

                    # Treat as normal text answer
                    msg_type = WSMessageType.ANSWER
                    content = text_answer

                except Exception as e:
                    print(f"STT Error: {e}")
                    await websocket.send_json(
                        create_ws_message(
                            msg_type=WSMessageType.ERROR,
                            error=f"Transcription failed: {str(e)}",
                        )
                    )
                    continue

            # The shared boundary owns lifecycle validation and start/resume behavior.
            response_msg = await invoke_graph_for_ws(
                session_id=session_id,
                action=msg_type,
                current_answer=content if msg_type == WSMessageType.ANSWER else None,
            )

            # Attach TTS audio and send
            audio = None
            if response_msg["type"] != WSMessageType.ERROR:
                audio = await generate_voice_response(
                    response_msg.get("content"), voice_enabled
                )
            if audio:
                response_msg["audio_base64"] = audio
            await websocket.send_json(response_msg)
            if (
                msg_type == WSMessageType.END
                and response_msg["type"] == WSMessageType.COMPLETE
            ):
                break

    except WebSocketDisconnect:
        print(f"Client disconnected: {session_id}")

    except Exception as e:
        print(f"WebSocket error: {e}")
        import traceback

        traceback.print_exc()
        try:
            await websocket.send_json(
                create_ws_message(msg_type=WSMessageType.ERROR, error=str(e))
            )
        except Exception:
            pass

    finally:
        manager.disconnect(session_id)
