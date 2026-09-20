"""Synthetic HTTP/WS integration cases with real graph and evaluation components."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import MemorySaver

from backend.app.api.deps import get_retriever
from backend.app.api.routes import session, websocket
from backend.app.config import settings
from backend.app.core.grilling_engine import GrillingEngine
from backend.app.graph.builder import build_interview_graph
from backend.app.graph.services import GraphServices
from backend.app.graph.state import create_initial_state
from rag.retriever import InterviewRetriever


@pytest.fixture
def interview(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "PARSED_RESUME_DIR", str(tmp_path))
    monkeypatch.setattr(settings, "VOICE_ENABLED", False)
    monkeypatch.setattr(session.limiter, "enabled", False)
    (tmp_path / "synthetic.md").write_text(
        "# CANDIDATE: Synthetic Candidate\nBuilt a synthetic queue simulator."
    )
    retriever = InterviewRetriever()
    llm = AsyncMock()
    llm.generate.return_value = "1. How did you design the synthetic queue simulator?"
    services = GraphServices(llm=llm, grilling_engine=GrillingEngine(llm))
    graph = build_interview_graph().compile(checkpointer=MemorySaver())

    async def get_graph():
        return graph

    monkeypatch.setattr(session, "get_compiled_graph", get_graph)
    monkeypatch.setattr(websocket, "get_compiled_graph", get_graph)
    monkeypatch.setattr(GraphServices, "create", lambda *args, **kwargs: services)
    app = FastAPI()
    app.include_router(session.router, prefix="/api/v1")
    app.include_router(websocket.router)
    app.dependency_overrides[get_retriever] = lambda: retriever
    with TestClient(app) as client:
        yield SimpleNamespace(client=client, graph=graph, llm=llm, services=services)


def test_http_session_start_runs_real_graph(interview):
    response = interview.client.post(
        "/api/v1/sessions", json={"resume_id": "synthetic", "num_questions": 1}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["type"] == "question"
    assert body["content"] == "How did you design the synthetic queue simulator?"
    interview.llm.generate.assert_awaited_once()
    detail = interview.client.get(
        f"/api/v1/sessions/{body['metadata']['session_id']}"
    ).json()
    assert detail["status"] == "in_progress"
    assert detail["current_question"] == body["content"]
    assert len(detail["conversation"]) == 1


@pytest.mark.asyncio
async def test_invalid_graph_action_with_null_error_returns_text():
    graph = build_interview_graph().compile(checkpointer=MemorySaver())
    state = create_initial_state("synthetic-invalid", "synthetic")
    result = await graph.ainvoke(
        state, {"configurable": {"thread_id": "synthetic-invalid"}}
    )
    assert result["response_type"] == "error"
    assert isinstance(result["response_content"], str)
    assert result["response_content"]


def model_evaluation(*, sufficient=True, follow_up=None):
    import json

    return json.dumps(
        {
            "scores": dict.fromkeys(
                [
                    "relevancy",
                    "clarity",
                    "informativeness",
                    "specificity",
                    "quantification",
                    "depth",
                    "completeness",
                ],
                0.9 if sufficient else 0.5,
            ),
            "detected_gaps": [] if sufficient else ["too_generic"],
            "is_sufficient": sufficient,
            "suggested_follow_up": follow_up,
            "reasoning": "Synthetic deterministic response",
        }
    )


def start_interview(interview, **options):
    response = interview.client.post(
        "/api/v1/sessions",
        json={"resume_id": "synthetic", "num_questions": 1, **options},
    )
    assert response.status_code == 200
    return response.json()


ANSWER = (
    "I implemented a bounded synthetic queue and measured its latency under fixed load."
)
QUESTION = "How did you design the synthetic queue simulator?"
FOLLOW_UP_1 = "Which queue capacity did you choose, and why?"
FOLLOW_UP_2 = "How did you test the chosen capacity under load?"


def test_sufficient_first_answer_does_not_force_follow_up(interview):
    interview.llm.generate.side_effect = [
        f"1. {QUESTION}",
        model_evaluation(),
        '{"is_consistent": true}',
    ]
    started = start_interview(interview)
    session_id = started["metadata"]["session_id"]
    answer = interview.client.post(
        f"/api/v1/sessions/{session_id}/answer", json={"answer": ANSWER}
    ).json()
    assert answer["type"] == "complete"


def test_repeated_follow_up_answers_target_presented_prompt(interview):
    interview.llm.generate.side_effect = [
        f"1. {QUESTION}",
        model_evaluation(sufficient=False, follow_up=FOLLOW_UP_1),
        '{"is_consistent": true}',
        model_evaluation(sufficient=False, follow_up=FOLLOW_UP_2),
        model_evaluation(),
    ]
    started = start_interview(interview, max_follow_ups=2)
    session_id = started["metadata"]["session_id"]
    responses = [
        interview.client.post(
            f"/api/v1/sessions/{session_id}/answer", json={"answer": ANSWER}
        ).json()
        for _ in range(3)
    ]
    assert [r["type"] for r in responses] == ["follow_up", "follow_up", "complete"]
    evaluated = [
        call.kwargs["prompt"].split("QUESTION: ", 1)[1].split("\n\nANSWER:", 1)[0]
        for call in interview.llm.generate.await_args_list
        if call.kwargs["prompt"].startswith("Evaluate this interview answer:")
    ]
    assert evaluated == [QUESTION, FOLLOW_UP_1, FOLLOW_UP_2]
    detail = interview.client.get(f"/api/v1/sessions/{session_id}").json()
    assert detail["current_question"] is None
    questions = [m for m in detail["conversation"] if m["role"] == "interviewer"]
    answers = [m for m in detail["conversation"] if m["role"] == "candidate"]
    assert len({m["question_id"] for m in questions}) == 3
    assert len({m["answer_id"] for m in answers}) == 3
    assert [m["question_id"] for m in answers] == [m["question_id"] for m in questions]
    assert all(
        m["parent_question_id"] == questions[0]["question_id"] for m in questions[1:]
    )


def test_rest_and_websocket_restore_active_follow_up(interview):
    interview.llm.generate.side_effect = [
        f"1. {QUESTION}",
        model_evaluation(sufficient=False, follow_up=FOLLOW_UP_1),
        '{"is_consistent": true}',
    ]
    session_id = start_interview(interview)["metadata"]["session_id"]
    answer = interview.client.post(
        f"/api/v1/sessions/{session_id}/answer", json={"answer": ANSWER}
    ).json()
    detail = interview.client.get(f"/api/v1/sessions/{session_id}").json()
    assert detail["current_question"] == answer["content"] == FOLLOW_UP_1
    assert detail["is_follow_up"] is True
    calls = interview.llm.generate.await_count
    with interview.client.websocket_connect(f"/ws/interview/{session_id}") as ws:
        connected = ws.receive_json()
        assert connected["data"]["current_question"] == FOLLOW_UP_1
        assert connected["data"]["is_follow_up"] is True
        ws.send_json({"type": "start"})
        resumed = ws.receive_json()
        assert resumed["type"] == "follow_up"
        assert resumed["content"] == FOLLOW_UP_1
        assert resumed["data"]["question_id"] == detail["current_question_id"]
    assert interview.llm.generate.await_count == calls


def snapshot(interview, session_id):
    return interview.client.portal.call(
        interview.graph.aget_state, {"configurable": {"thread_id": session_id}}
    ).values


def seed_session(interview, status="pending", **updates):
    session_id = "synthetic-seeded"
    state = create_initial_state(session_id, "synthetic")
    state.update(status=status, questions=[QUESTION])
    state.update(updates)
    interview.client.portal.call(
        interview.graph.aupdate_state,
        {"configurable": {"thread_id": session_id}},
        state,
    )
    return session_id


@pytest.mark.parametrize(
    "status", ["pending", "generating", "evaluating", "completed", "cancelled"]
)
@pytest.mark.parametrize("action", ["answer", "skip", "end"])
def test_rest_ws_reject_same_lifecycle_commands(interview, status, action):
    if status == "pending" and action == "end":
        session_id = seed_session(interview, status)
        response = interview.client.post(f"/api/v1/sessions/{session_id}/end")
        assert response.status_code == 200
        assert snapshot(interview, session_id)["status"] == "cancelled"
        interview.llm.generate.assert_not_awaited()
        return
    session_id = seed_session(interview, status)
    before = snapshot(interview, session_id)
    response = interview.client.post(
        f"/api/v1/sessions/{session_id}/{action}",
        json={"answer": ANSWER} if action == "answer" else None,
    )
    assert response.status_code == (
        409 if status in {"generating", "evaluating"} else 400
    )
    assert snapshot(interview, session_id) == before
    with interview.client.websocket_connect(f"/ws/interview/{session_id}") as ws:
        ws.receive_json()
        ws.send_json({"type": action, "content": ANSWER})
        error = ws.receive_json()
        assert error["type"] == "error"
        assert error["error"] == response.json()["detail"]
        ws.send_json({"type": "ping"})
        assert ws.receive_json()["type"] == "pong"
    assert snapshot(interview, session_id) == before
    interview.llm.generate.assert_not_awaited()


def test_completed_session_cannot_be_changed_to_cancelled(interview):
    session_id = start_interview(interview)["metadata"]["session_id"]
    assert (
        interview.client.post(f"/api/v1/sessions/{session_id}/skip").json()["type"]
        == "complete"
    )
    before = snapshot(interview, session_id)
    response = interview.client.post(f"/api/v1/sessions/{session_id}/end")
    assert response.status_code == 400
    assert snapshot(interview, session_id) == before


def test_blank_answer_rejected_by_both_transports_then_skip_works(interview):
    session_id = start_interview(interview)["metadata"]["session_id"]
    before = snapshot(interview, session_id)
    response = interview.client.post(
        f"/api/v1/sessions/{session_id}/answer", json={"answer": " \n "}
    )
    assert response.status_code == 400
    with interview.client.websocket_connect(f"/ws/interview/{session_id}") as ws:
        ws.receive_json()
        ws.send_json({"type": "answer", "content": " \n "})
        assert ws.receive_json()["error"] == response.json()["detail"]
        assert snapshot(interview, session_id) == before
        ws.send_json({"type": "skip"})
        assert ws.receive_json()["type"] == "complete"
    interview.llm.generate.assert_awaited_once()


@pytest.mark.parametrize(
    "status", ["generating", "evaluating", "completed", "cancelled"]
)
def test_ws_start_rejected_without_mutation(interview, status):
    session_id = seed_session(interview, status)
    before = snapshot(interview, session_id)
    with interview.client.websocket_connect(f"/ws/interview/{session_id}") as ws:
        ws.receive_json()
        ws.send_json({"type": "start"})
        assert ws.receive_json()["type"] == "error"
    assert snapshot(interview, session_id) == before
    interview.llm.generate.assert_not_awaited()


def test_pending_ws_start_and_end_without_provider(interview, monkeypatch):
    session_id = seed_session(interview, num_questions=1)
    with interview.client.websocket_connect(f"/ws/interview/{session_id}") as ws:
        assert ws.receive_json()["data"]["status"] == "pending"
        ws.send_json({"type": "start"})
        assert ws.receive_json()["content"] == QUESTION
        before = snapshot(interview, session_id)
        ws.send_json({"type": "start"})
        assert ws.receive_json()["content"] == QUESTION
        assert snapshot(interview, session_id) == before

        def unavailable(*args, **kwargs):
            raise RuntimeError("Synthetic provider unavailable")

        monkeypatch.setattr(GraphServices, "create", unavailable)
        ws.send_json({"type": "end"})
        assert ws.receive_json()["data"]["summary"]["status"] == "cancelled"
    assert snapshot(interview, session_id)["active_question"] is None
    interview.llm.generate.assert_awaited_once()


def test_malformed_ws_commands_allow_later_skip(interview):
    session_id = start_interview(interview)["metadata"]["session_id"]
    before = snapshot(interview, session_id)
    with interview.client.websocket_connect(f"/ws/interview/{session_id}") as ws:
        ws.receive_json()
        for message in [
            "{",
            "[]",
            '{"type": "answer", "content": 42}',
            '{"type": "skip", "data": []}',
            '{"type": ["skip"]}',
        ]:
            ws.send_text(message)
            assert ws.receive_json()["type"] == "error"
            assert snapshot(interview, session_id) == before
        ws.send_json({"type": "skip"})
        assert ws.receive_json()["type"] == "complete"


def test_provider_setup_failure_is_retryable_before_graph_write(interview, monkeypatch):
    session_id = start_interview(interview)["metadata"]["session_id"]
    before = snapshot(interview, session_id)

    def unavailable(*args, **kwargs):
        raise RuntimeError("Synthetic private provider details")

    monkeypatch.setattr(GraphServices, "create", unavailable)
    response = interview.client.post(
        f"/api/v1/sessions/{session_id}/answer", json={"answer": ANSWER}
    )
    assert response.status_code == 503
    assert "private" not in response.text
    assert snapshot(interview, session_id) == before
    with interview.client.websocket_connect(f"/ws/interview/{session_id}") as ws:
        ws.receive_json()
        ws.send_json({"type": "answer", "content": ANSWER})
        assert ws.receive_json()["data"]["status_code"] == 503
        assert snapshot(interview, session_id) == before
        # Skip has no model dependency, even while provider setup is unavailable.
        ws.send_json({"type": "skip"})
        assert ws.receive_json()["type"] == "complete"


def test_ws_multiple_questions_advance_and_skip_follow_up(interview):
    second = "How did you measure the simulator's throughput?"
    interview.llm.generate.side_effect = [
        f"1. {QUESTION}\n2. {second}",
        model_evaluation(),
        '{"is_consistent": true}',
        model_evaluation(sufficient=False, follow_up=FOLLOW_UP_1),
        '{"is_consistent": true}',
    ]
    session_id = start_interview(interview, num_questions=2)["metadata"]["session_id"]
    with interview.client.websocket_connect(f"/ws/interview/{session_id}") as ws:
        ws.receive_json()
        ws.send_json({"type": "answer", "content": ANSWER})
        advanced = ws.receive_json()
        assert advanced["content"] == second
        assert advanced["data"]["question_id"].endswith(":q2")
        ws.send_json({"type": "answer", "content": ANSWER})
        follow_up = ws.receive_json()
        assert follow_up["type"] == "follow_up"
        ws.send_json({"type": "skip"})
        assert ws.receive_json()["type"] == "complete"
    state = snapshot(interview, session_id)
    assert state["status"] == "completed"
    skipped = [m for m in state["conversation"] if m["content"] == "[Skipped]"][0]
    assert skipped["question_id"] == follow_up["data"]["question_id"]
    assert skipped["answer_id"] is None


@pytest.mark.parametrize("recoverable", [True, False])
def test_legacy_follow_up_recovery_never_guesses_main_prompt(interview, recoverable):
    messages = [
        {
            "role": "interviewer",
            "content": FOLLOW_UP_1,
            "is_follow_up": True,
            "timestamp": "2026-09-20T00:00:00+00:00",
        }
    ]
    session_id = seed_session(
        interview,
        "asking",
        current_follow_up_count=1,
        conversation=messages if recoverable else [],
    )
    before = snapshot(interview, session_id)
    detail = interview.client.get(f"/api/v1/sessions/{session_id}").json()
    assert detail["current_question"] == (FOLLOW_UP_1 if recoverable else None)
    with interview.client.websocket_connect(f"/ws/interview/{session_id}") as ws:
        ws.receive_json()
        ws.send_json({"type": "start"})
        response = ws.receive_json()
        assert response["type"] == ("follow_up" if recoverable else "error")
        assert snapshot(interview, session_id) == before
        if recoverable:
            interview.llm.generate.return_value = model_evaluation()
            ws.send_json({"type": "answer", "content": ANSWER})
            assert ws.receive_json()["type"] == "complete"
            assert (
                f"QUESTION: {FOLLOW_UP_1}"
                in interview.llm.generate.await_args.kwargs["prompt"]
            )
        else:
            assert response["data"]["status_code"] == 409
            interview.llm.generate.assert_not_awaited()


def test_custom_compact_evaluator_receives_active_follow_up(interview):
    prepared = {"questions": [QUESTION], "resume_summary": "Synthetic queue project"}
    interview.services.grilling_engine = GrillingEngine(
        interview.llm, model_type="custom", prepared_context=prepared
    )
    session_id = seed_session(
        interview, model_type="custom", prepared_context=prepared, num_questions=1
    )
    interview.llm.generate.side_effect = [
        '{"score": 0.5, "gap": "too_generic", "sufficient": false, "followup": "'
        + FOLLOW_UP_1
        + '"}',
        '{"score": 0.9, "gap": "too_generic", "sufficient": true}',
    ]
    with interview.client.websocket_connect(f"/ws/interview/{session_id}") as ws:
        ws.receive_json()
        ws.send_json({"type": "start"})
        assert ws.receive_json()["content"] == QUESTION
        ws.send_json({"type": "answer", "content": ANSWER})
        assert ws.receive_json()["content"] == FOLLOW_UP_1
        ws.send_json({"type": "answer", "content": ANSWER})
        assert ws.receive_json()["type"] == "complete"
    prompts = [call.kwargs["prompt"] for call in interview.llm.generate.await_args_list]
    assert len(prompts) == 2
    assert f"Q: {QUESTION}\n" in prompts[0]
    assert f"Q: {FOLLOW_UP_1}\n" in prompts[1]


def test_pending_ws_end_needs_no_provider(interview, monkeypatch):
    session_id = seed_session(interview)

    def unavailable(*args, **kwargs):
        raise AssertionError("End must not initialize a provider")

    monkeypatch.setattr(GraphServices, "create", unavailable)
    with interview.client.websocket_connect(f"/ws/interview/{session_id}") as ws:
        ws.receive_json()
        ws.send_json({"type": "end"})
        assert ws.receive_json()["data"]["summary"]["status"] == "cancelled"
    interview.llm.generate.assert_not_awaited()
