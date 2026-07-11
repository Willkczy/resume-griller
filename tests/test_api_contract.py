from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from backend.app.api.routes import session, websocket
from backend.app.graph.state import create_initial_state
from backend.app.main import app


class FakeGraph:
    def __init__(self, values: dict, result: dict | None = None):
        self.values = values
        self.result = result or values
        self.inputs = []

    async def aget_state(self, config):
        return SimpleNamespace(values=self.values)

    async def ainvoke(self, graph_input, config):
        self.inputs.append(graph_input)
        return self.result


def session_values(**updates) -> dict:
    values = create_initial_state(
        "session-1",
        "resume-1",
        full_resume_text="resume",
        num_questions=2,
    )
    values.update(
        {
            "status": "asking",
            "questions": ["Question one?", "Question two?"],
            "conversation": [
                {
                    "role": "interviewer",
                    "content": "Question one?",
                    "timestamp": values["created_at"],
                    "is_follow_up": False,
                    "metadata": {},
                }
            ],
        }
    )
    values.update(updates)
    return values


@pytest.mark.asyncio
async def test_session_detail_normalizes_internal_status(monkeypatch):
    fake = FakeGraph(session_values())
    monkeypatch.setattr(session, "get_compiled_graph", AsyncMock(return_value=fake))

    detail = await session.get_session("session-1")
    assert detail.status == "in_progress"
    assert detail.current_question == "Question one?"
    assert detail.created_at
    assert detail.updated_at


@pytest.mark.asyncio
async def test_summary_contract_counts_questions_and_duration(monkeypatch):
    values = session_values(
        status="completed",
        current_question_index=1,
        conversation=[
            {"role": "interviewer", "content": "Q1", "is_follow_up": False},
            {"role": "candidate", "content": "A1", "is_follow_up": False},
            {"role": "interviewer", "content": "FU", "is_follow_up": True},
            {"role": "candidate", "content": "A2", "is_follow_up": False},
            {"role": "interviewer", "content": "Q2", "is_follow_up": False},
        ],
    )
    fake = FakeGraph(values)
    monkeypatch.setattr(session, "get_compiled_graph", AsyncMock(return_value=fake))

    summary = await session.get_session_summary("session-1")
    assert summary.status == "completed"
    assert summary.questions_asked == 2
    assert summary.answers_given == 2
    assert summary.follow_ups_asked == 1
    assert summary.duration_seconds >= 0


@pytest.mark.asyncio
async def test_invoke_graph_adds_updated_at_and_maps_response(monkeypatch):
    result = {
        "response_type": "question",
        "response_content": "Next question?",
        "response_data": {"question_number": 2, "total_questions": 3},
    }
    fake = FakeGraph(session_values(), result=result)
    monkeypatch.setattr(session, "get_compiled_graph", AsyncMock(return_value=fake))
    monkeypatch.setattr(session.GraphServices, "create", Mock(return_value=Mock()))

    response = await session._invoke_graph("session-1", "skip")
    assert response.type == "question"
    assert response.question_number == 2
    assert "updated_at" in fake.inputs[0]


@pytest.mark.asyncio
async def test_missing_and_inactive_sessions_are_rejected(monkeypatch):
    missing = FakeGraph({})
    monkeypatch.setattr(session, "get_compiled_graph", AsyncMock(return_value=missing))
    with pytest.raises(Exception) as missing_error:
        await session.get_session("missing")
    assert missing_error.value.status_code == 404

    completed = FakeGraph(session_values(status="completed"))
    monkeypatch.setattr(
        session, "get_compiled_graph", AsyncMock(return_value=completed)
    )
    with pytest.raises(Exception) as inactive_error:
        await session.submit_answer.__wrapped__(
            Mock(), "session-1", session.AnswerRequest(answer="x")
        )
    assert inactive_error.value.status_code == 400


def test_websocket_resume_ping_answer_and_voice_toggle(monkeypatch):
    result = {
        "response_type": "follow_up",
        "response_content": "Can you quantify that?",
        "response_data": {"question_number": 1, "total_questions": 2},
    }
    fake = FakeGraph(session_values(), result=result)
    monkeypatch.setattr(websocket, "get_compiled_graph", AsyncMock(return_value=fake))
    monkeypatch.setattr(websocket.GraphServices, "create", Mock(return_value=Mock()))
    voice = AsyncMock(return_value=None)
    monkeypatch.setattr(websocket, "generate_voice_response", voice)

    with TestClient(app) as client:
        with client.websocket_connect("/ws/interview/session-1") as ws:
            connected = ws.receive_json()
            assert connected["type"] == "connected"
            assert connected["data"]["status"] == "in_progress"
            assert connected["data"]["current_question"] == "Question one?"

            ws.send_json({"type": "ping"})
            assert ws.receive_json()["type"] == "pong"

            ws.send_json(
                {
                    "type": "answer",
                    "content": "My answer",
                    "data": {"voice_enabled": False},
                }
            )
            follow_up = ws.receive_json()
            assert follow_up["type"] == "follow_up"
            assert "audio_base64" not in follow_up
            voice.assert_awaited_with("Can you quantify that?", False)


def test_websocket_rejects_missing_session(monkeypatch):
    monkeypatch.setattr(
        websocket,
        "get_compiled_graph",
        AsyncMock(return_value=FakeGraph({})),
    )
    with TestClient(app) as client:
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/ws/interview/missing"):
                pass
