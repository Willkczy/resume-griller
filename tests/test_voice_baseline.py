"""Text-only baseline: no voice credentials, SDK imports, or provider calls."""

from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.api.routes import voice, websocket
from backend.app.config import Settings


def test_voice_defaults_off_without_environment(monkeypatch):
    monkeypatch.delenv("VOICE_ENABLED", raising=False)
    enabled = Settings(_env_file=None).VOICE_ENABLED
    assert enabled is False


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(voice.settings, "VOICE_ENABLED", False)
    forbidden = Mock(side_effect=AssertionError("Voice provider must not initialize"))
    monkeypatch.setattr(voice, "get_stt_service", forbidden)
    monkeypatch.setattr(voice, "get_tts_service", forbidden)
    app = FastAPI()
    app.include_router(voice.router)
    with TestClient(app) as client:
        yield client
    forbidden.assert_not_called()


def test_disabled_status(client):
    assert client.get("/voice/status").json() == {
        "enabled": False,
        "stt_provider": "deepgram",
        "tts_provider": "elevenlabs",
        "stt_available": False,
        "tts_available": False,
    }


@pytest.mark.parametrize(
    "path,payload",
    [
        ("/transcribe", {"audio_base64": "c3ludGhldGlj"}),
        ("/synthesize", {"text": "Synthetic question"}),
        ("/synthesize/audio", {"text": "Synthetic question"}),
    ],
)
def test_disabled_voice_posts(client, path, payload):
    response = client.post(f"/voice{path}", json=payload)
    assert response.status_code == 503
    assert response.json()["detail"] == "Voice services are disabled"


def test_disabled_file_and_voices(client):
    response = client.post(
        "/voice/transcribe/file",
        files={"file": ("synthetic.webm", b"synthetic", "audio/webm")},
    )
    assert response.status_code == 503
    assert client.get("/voice/voices").status_code == 503


@pytest.mark.asyncio
async def test_disabled_websocket_tts(monkeypatch):
    monkeypatch.setattr(websocket.settings, "VOICE_ENABLED", False)
    forbidden = Mock(side_effect=AssertionError("TTS must not initialize"))
    monkeypatch.setattr(websocket, "get_tts_service", forbidden)
    assert await websocket.generate_voice_response("Synthetic question") is None
    forbidden.assert_not_called()


@pytest.mark.asyncio
async def test_disabled_audio_rejected_without_changing_checkpoint(monkeypatch):
    from langgraph.checkpoint.memory import MemorySaver

    from backend.app.graph.builder import build_interview_graph
    from backend.app.graph.state import create_initial_state

    graph = build_interview_graph().compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "synthetic-voice-session"}}
    initial = create_initial_state("synthetic-voice-session", "synthetic-resume")
    await graph.aupdate_state(config, initial)
    before = (await graph.aget_state(config)).values

    async def get_graph():
        return graph

    monkeypatch.setattr(websocket, "get_compiled_graph", get_graph)
    monkeypatch.setattr(websocket.settings, "VOICE_ENABLED", False)
    forbidden = Mock(side_effect=AssertionError("STT must not initialize"))
    monkeypatch.setattr(websocket, "get_stt_service", forbidden)
    app = FastAPI()
    app.include_router(websocket.router)
    with TestClient(app) as client:
        with client.websocket_connect("/ws/interview/synthetic-voice-session") as ws:
            assert ws.receive_json()["type"] == "connected"
            ws.send_json({"type": "answer_audio", "content": "c3ludGhldGlj"})
            assert ws.receive_json()["type"] == "error"
            ws.send_json({"type": "ping"})
            assert ws.receive_json()["type"] == "pong"
    assert (await graph.aget_state(config)).values == before
    forbidden.assert_not_called()
