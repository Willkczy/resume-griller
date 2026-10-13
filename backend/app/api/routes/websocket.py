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
from backend.app.core.logging_config import get_logger
from backend.app.graph import get_compiled_graph
from backend.app.graph.commands import execute_command, load_session_state
from backend.app.graph.public import websocket_data
from backend.app.graph.transitions import CommandError, command_error
from backend.app.services.stt_service import get_stt_service
from backend.app.services.tts_service import get_tts_service

logger = get_logger(__name__)

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
    except Exception:
        logger.warning("speech_synthesis_unavailable")
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
        try:
            current_state = await load_session_state(graph, session_id)
        except CommandError:
            data = {"error_code": exc.code, "status_code": exc.status_code}
        else:
            data = websocket_data(
                current_state,
                session_id,
                error_code=exc.code,
                status_code=exc.status_code,
            )
        return create_ws_message(
            msg_type=WSMessageType.ERROR,
            content=exc.detail,
            error=exc.detail,
            data=data,
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
        data=websocket_data(result, session_id),
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
    idx = session_state.get("current_question_index", 0)
    await websocket.send_json(
        create_ws_message(
            msg_type=WSMessageType.CONNECTED,
            content="Connected to interview session",
            data=websocket_data(session_state, session_id, question_number=idx + 1),
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
                current_state = await load_session_state(graph, session_id)
                await websocket.send_json(
                    create_ws_message(
                        msg_type=WSMessageType.ERROR,
                        error="Invalid JSON format",
                        data=websocket_data(
                            current_state, session_id, error_code="invalid_message"
                        ),
                    )
                )
                continue

            msg_type = message.get("type", "")
            content = message.get("content", "")
            message_data = message.get("data")
            if message_data is None:
                message_data = {}
            if not isinstance(message_data, dict):
                current_state = await load_session_state(graph, session_id)
                await websocket.send_json(
                    create_ws_message(
                        msg_type=WSMessageType.ERROR,
                        error="Message data must be an object",
                        data=websocket_data(
                            current_state, session_id, error_code="invalid_message"
                        ),
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
                    current_state = await load_session_state(graph, session_id)
                    await websocket.send_json(
                        create_ws_message(
                            msg_type=WSMessageType.ERROR,
                            error="Voice services disabled",
                            data=websocket_data(
                                current_state, session_id, error_code="voice_disabled"
                            ),
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
                            data=websocket_data(
                                current_state,
                                session_id,
                                error_code=exc.code,
                                status_code=exc.status_code,
                            ),
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

                except Exception:
                    logger.warning("transcription_unavailable")
                    current_state = await load_session_state(graph, session_id)
                    await websocket.send_json(
                        create_ws_message(
                            msg_type=WSMessageType.ERROR,
                            error="Transcription failed",
                            data=websocket_data(
                                current_state,
                                session_id,
                                error_code="transcription_failed",
                            ),
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
        logger.info("websocket_disconnected")

    except Exception:
        logger.warning("websocket_command_failed")
        try:
            current_state = await load_session_state(graph, session_id)
            await websocket.send_json(
                create_ws_message(
                    msg_type=WSMessageType.ERROR,
                    error="Interview command could not be completed",
                    data=websocket_data(
                        current_state,
                        session_id,
                        error_code="command_failed",
                        status_code=503,
                    ),
                )
            )
        except Exception:
            pass

    finally:
        manager.disconnect(session_id)
