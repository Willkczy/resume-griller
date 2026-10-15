"""Synthetic outcome/UI contract fixtures; no model-quality benchmark claim.

Fixture revision: stage3-outcomes-v1. All provider calls use the real graph and
grilling engine with a mocked LLM; checkpoint reads must not trigger retries.
"""

import copy
import json

import pytest
from pydantic import TypeAdapter, ValidationError
from test_interview_flow import (
    ANSWER,
    FOLLOW_UP_1,
    QUESTION,
    model_evaluation,
    seed_session,
    snapshot,
    start_interview,
)
from test_interview_flow import (
    interview as shared_interview,
)

from backend.app.api.routes.session import InterviewResponseModel
from backend.app.core.grilling_engine import GrillingEngine
from backend.app.graph.public import interview_snapshot, project_evaluation
from backend.app.graph.state import create_initial_state, make_question_turn
from backend.app.graph.transitions import command_error
from backend.app.models.interview import PublicEvaluation

interview = shared_interview

SECOND_QUESTION = "What did you learn from the synthetic queue experiment?"
SNAPSHOT_FIELDS = (
    "model_executions",
    "evaluation",
    "generation_failure",
    "draft_answer",
    "answer_status",
    "allowed_actions",
    "conversation",
    "current_question",
    "current_question_id",
    "parent_question_id",
    "status",
)


def submit(interview, session_id, transport, answer=ANSWER):
    if transport == "rest":
        return interview.client.post(
            f"/api/v1/sessions/{session_id}/answer", json={"answer": answer}
        ).json()
    with interview.client.websocket_connect(f"/ws/interview/{session_id}") as ws:
        ws.receive_json()
        ws.send_json({"type": "answer", "content": answer})
        return ws.receive_json()["data"]


def assert_restored(interview, session_id, original):
    calls = interview.llm.generate.await_count
    checkpoint = snapshot(interview, session_id)
    detail = interview.client.get(f"/api/v1/sessions/{session_id}").json()
    with interview.client.websocket_connect(f"/ws/interview/{session_id}") as ws:
        connected = ws.receive_json()["data"]
        for field in SNAPSHOT_FIELDS:
            assert connected[field] == detail[field] == original[field]
        if detail["status"] == "in_progress":
            ws.send_json({"type": "start"})
            resumed = ws.receive_json()["data"]
            for field in SNAPSHOT_FIELDS:
                assert resumed[field] == original[field]
    assert interview.llm.generate.await_count == calls
    assert snapshot(interview, session_id) == checkpoint
    return detail


@pytest.mark.parametrize("transport", ["rest", "ws"])
@pytest.mark.parametrize(
    "output,outcome,code",
    [
        ("{}", "invalid_output", "schema_violation"),
        ('{"outcome":', "invalid_output", "invalid_json"),
        (TimeoutError("private synthetic sentinel"), "provider_failure", "timeout"),
        (
            json.dumps(
                {
                    "outcome": "insufficient_evidence",
                    "reasoning": "Project identity is missing",
                    "missing_elements": ["Project identity"],
                    "suggested_follow_up": "Which project are you describing?",
                }
            ),
            "insufficient_evidence",
            None,
        ),
    ],
)
def test_non_scored_outcome_restores_draft_and_safe_actions(
    interview, transport, output, outcome, code
):
    started = start_interview(interview, max_follow_ups=0)
    session_id = started["session_id"]
    interview.llm.generate.side_effect = [output]
    result = submit(interview, session_id, transport)
    evaluation = result["evaluation"]
    assert evaluation["outcome"] == outcome
    assert evaluation.get("code") == code
    assert "score" not in evaluation and "detailed_scores" not in evaluation
    assert evaluation["question_id"] == started["current_question_id"]
    assert evaluation["answer_id"] == f"{started['current_question_id']}:answer"
    assert result["draft_answer"] == ANSWER
    assert result["answer_status"] == "unscored"
    assert result["generation_failure"] is None
    assert result["allowed_actions"] == ["start", "answer", "skip", "end"]
    assert result["conversation"] == started["conversation"]
    assert "sentinel" not in json.dumps(result)
    assert_restored(interview, session_id, result)


@pytest.mark.parametrize("action", ["skip", "end"])
@pytest.mark.parametrize("transport", ["rest", "ws"])
def test_skip_end_clear_unscored_draft_and_outcome(interview, action, transport):
    interview.llm.generate.side_effect = [
        json.dumps({"questions": [QUESTION, SECOND_QUESTION]}),
        "{}",
    ]
    session_id = start_interview(interview, num_questions=2)["session_id"]
    submit(interview, session_id, transport)
    calls = interview.llm.generate.await_count
    result = interview.client.post(f"/api/v1/sessions/{session_id}/{action}").json()
    assert result["draft_answer"] is None
    assert result["evaluation"] is None
    assert result["generation_failure"] is None
    assert result["answer_status"] == "none"
    assert result["current_question"] == (SECOND_QUESTION if action == "skip" else None)
    assert result["allowed_actions"] == (
        ["start", "answer", "skip", "end"] if action == "skip" else []
    )
    assert interview.llm.generate.await_count == calls
    assert_restored(interview, session_id, result)


@pytest.mark.parametrize("transport", ["rest", "ws"])
def test_new_explicit_answer_accepts_once_and_preserves_feedback(interview, transport):
    interview.llm.generate.side_effect = [
        json.dumps({"questions": [QUESTION, SECOND_QUESTION]}),
        "{}",
        model_evaluation(),
    ]
    started = start_interview(interview, num_questions=2)
    session_id = started["session_id"]
    submit(interview, session_id, transport)
    result = submit(
        interview, session_id, transport, answer="FIFO with bounded capacity."
    )
    assert result["current_question"] == SECOND_QUESTION
    assert result["evaluation"]["outcome"] == "valid"
    assert result["evaluation"]["question_id"] == started["current_question_id"]
    assert result["evaluation"]["question_id"] != result["current_question_id"]
    assert result["answer_status"] == "none"
    assert result["draft_answer"] is None
    assert len([m for m in result["conversation"] if m["role"] == "candidate"]) == 1
    assert_restored(interview, session_id, result)
    skipped = interview.client.post(f"/api/v1/sessions/{session_id}/skip").json()
    assert skipped["evaluation"] is None


@pytest.mark.parametrize("transport", ["rest", "ws"])
def test_valid_zero_is_scored_and_survives_completed_read(interview, transport):
    zero = json.loads(model_evaluation(sufficient=False))
    zero["scores"] = dict.fromkeys(zero["scores"], 0.0)
    interview.llm.generate.side_effect = [
        json.dumps({"questions": [QUESTION]}),
        json.dumps(zero),
    ]
    session_id = start_interview(interview, max_follow_ups=0)["session_id"]
    result = submit(interview, session_id, transport)
    assert result["evaluation"]["outcome"] == "valid"
    assert result["evaluation"]["score"] == 0.0
    assert result["evaluation"]["detailed_scores"]["overall"] == 0.0
    assert result["status"] == "completed"
    assert result["allowed_actions"] == []
    assert result["draft_answer"] is None
    assert_restored(interview, session_id, result)


@pytest.mark.parametrize("transport", ["rest", "ws"])
@pytest.mark.parametrize("failure", ["{}", TimeoutError("private synthetic sentinel")])
def test_follow_up_failure_retains_accepted_score_and_blocks_answer(
    interview, transport, failure
):
    interview.llm.generate.side_effect = [
        json.dumps({"questions": [QUESTION, SECOND_QUESTION]}),
        model_evaluation(sufficient=False),
        failure,
    ]
    started = start_interview(interview, num_questions=2)
    session_id = started["session_id"]
    result = submit(interview, session_id, transport)
    assert result["evaluation"]["outcome"] == "valid"
    assert result["evaluation"]["score"] == pytest.approx(0.5)
    assert result["generation_failure"]["operation"] == "follow_up_generation"
    assert result["generation_failure"]["outcome"]["outcome"] in {
        "invalid_output",
        "provider_failure",
    }
    assert result["answer_status"] == "accepted"
    assert result["draft_answer"] is None
    assert result["allowed_actions"] == ["start", "skip", "end"]
    assert len([m for m in result["conversation"] if m["role"] == "candidate"]) == 1
    assert_restored(interview, session_id, result)
    before = snapshot(interview, session_id)
    calls = interview.llm.generate.await_count
    assert (
        interview.client.post(
            f"/api/v1/sessions/{session_id}/answer", json={"answer": ANSWER}
        ).status_code
        == 409
    )
    with interview.client.websocket_connect(f"/ws/interview/{session_id}") as ws:
        ws.receive_json()
        ws.send_json({"type": "answer", "content": ANSWER})
        rejected = ws.receive_json()
        assert rejected["data"]["status_code"] == 409
        for field in SNAPSHOT_FIELDS:
            assert rejected["data"][field] == result[field]
    assert snapshot(interview, session_id) == before
    assert interview.llm.generate.await_count == calls
    skipped = interview.client.post(f"/api/v1/sessions/{session_id}/skip").json()
    assert skipped["generation_failure"] is None
    assert skipped["evaluation"] is None
    assert skipped["answer_status"] == "none"


@pytest.mark.parametrize("failure", ["{}", TimeoutError("private synthetic sentinel")])
def test_startup_failure_restores_without_model_retry_then_manual_start_succeeds(
    interview, failure
):
    interview.llm.generate.side_effect = [
        failure,
        json.dumps({"questions": [QUESTION]}),
    ]
    result = start_interview(interview)
    session_id = result["session_id"]
    assert result["status"] == "pending"
    assert result["generation_failure"]["operation"] == "question_generation"
    assert result["allowed_actions"] == ["start", "end"]
    assert result["evaluation"] is None
    assert_restored(interview, session_id, result)
    with interview.client.websocket_connect(f"/ws/interview/{session_id}") as ws:
        ws.receive_json()
        ws.send_json({"type": "start"})
        started = ws.receive_json()["data"]
    assert started["generation_failure"] is None
    assert started["current_question"] == QUESTION
    assert interview.llm.generate.await_count == 2


def test_pending_failure_end_calls_no_model(interview):
    interview.llm.generate.side_effect = ["{}"]
    session_id = start_interview(interview)["session_id"]
    result = interview.client.post(f"/api/v1/sessions/{session_id}/end").json()
    assert result["status"] == "cancelled"
    assert result["generation_failure"] is None
    assert result["allowed_actions"] == []
    interview.llm.generate.assert_awaited_once()


def test_unsupported_evaluation_blocks_second_submission_without_invoking_provider(
    interview,
):
    prepared = {"questions": [QUESTION], "resume_summary": "Synthetic queue project"}
    interview.services.grilling_engine = GrillingEngine(
        interview.llm, model_type="custom", prepared_context=prepared
    )
    session_id = seed_session(
        interview, model_type="custom", prepared_context=prepared, num_questions=1
    )
    with interview.client.websocket_connect(f"/ws/interview/{session_id}") as ws:
        ws.receive_json()
        ws.send_json({"type": "start"})
        ws.receive_json()
        ws.send_json({"type": "answer", "content": ANSWER})
        result = ws.receive_json()["data"]
        assert result["evaluation"]["outcome"] == "unsupported_mode"
        assert result["allowed_actions"] == ["start", "skip", "end"]
        assert result["draft_answer"] == ANSWER
        ws.send_json({"type": "answer", "content": "New answer"})
        rejected = ws.receive_json()["data"]
        assert rejected["status_code"] == 409
        assert rejected["error_code"] == "evaluation_unsupported_mode"
        assert rejected["draft_answer"] == ANSWER
    assert_restored(interview, session_id, result)
    interview.llm.generate.assert_not_awaited()


def test_follow_up_failure_outcome_targets_answered_follow_up(interview):
    interview.llm.generate.side_effect = [
        json.dumps({"questions": [QUESTION]}),
        model_evaluation(sufficient=False, follow_up=FOLLOW_UP_1),
        "{}",
    ]
    session_id = start_interview(interview)["session_id"]
    accepted = submit(interview, session_id, "rest")
    assert accepted["draft_answer"] is None
    assert accepted["answer_status"] == "none"
    result = submit(interview, session_id, "rest", answer="My follow-up answer")
    assert result["current_question"] == FOLLOW_UP_1
    assert result["evaluation"]["question_id"] == accepted["current_question_id"]
    assert result["evaluation"]["parent_question_id"] == f"{session_id}:q1"
    assert result["draft_answer"] == "My follow-up answer"
    assert_restored(interview, session_id, result)


def valid_public_evaluation():
    service = GrillingEngine(None)
    return service._parse_llm_evaluation(model_evaluation(), 0).to_dict()


def test_public_schema_discriminates_every_outcome_and_forbids_unscored_scores():
    schema = InterviewResponseModel.model_json_schema()
    assert set(
        schema["properties"]["evaluation"]["anyOf"][0]["discriminator"]["mapping"]
    ) == {
        "valid",
        "insufficient_evidence",
        "invalid_output",
        "provider_failure",
        "unsupported_mode",
    }
    adapter = TypeAdapter(PublicEvaluation)
    with pytest.raises(ValidationError):
        adapter.validate_json(
            json.dumps(
                {"outcome": "invalid_output", "code": "invalid_json", "score": 0}
            )
        )
    with pytest.raises(ValidationError):
        adapter.validate_json(json.dumps({"score": 0}))
    assert (
        adapter.validate_json(json.dumps(valid_public_evaluation())).outcome == "valid"
    )


def test_legacy_valid_tag_projection_is_read_only_and_keeps_zero():
    legacy = valid_public_evaluation()
    legacy.pop("outcome")
    legacy["score"] = 0.0
    before = copy.deepcopy(legacy)
    result = project_evaluation(legacy)
    assert result.outcome == "valid"
    assert result.score == 0.0
    assert legacy == before


@pytest.mark.parametrize(
    "mutation", [{"score": 9}, {"is_sufficient": "false"}, {"reasoning": ""}]
)
def test_corrupt_checkpoint_evaluation_projects_without_score(mutation):
    legacy = valid_public_evaluation()
    legacy.pop("outcome")
    legacy.update(mutation)
    result = project_evaluation(legacy)
    assert result.outcome == "invalid_output"
    assert "score" not in result.model_dump()


@pytest.mark.parametrize("stored", [7, ["synthetic private sentinel"], "sentinel"])
def test_non_dict_checkpoint_evaluation_is_safe(stored):
    state = create_initial_state("corrupt-shape", "synthetic")
    state.update(status="asking", questions=[QUESTION], current_evaluation=stored)
    result = interview_snapshot(state, "corrupt-shape")
    assert result.evaluation.outcome == "invalid_output"
    assert "score" not in result.evaluation.model_dump()
    assert "sentinel" not in result.model_dump_json()


@pytest.mark.parametrize("kind", ["invalid_output", "provider_failure"])
def test_stage2_pending_failure_is_restored_read_only(kind):
    state = create_initial_state("legacy", "synthetic")
    state.update(
        response_type="error",
        response_data={"error_code": f"question_generation_{kind}"},
    )
    before = copy.deepcopy(state)
    result = interview_snapshot(state, "legacy")
    assert result.generation_failure.operation == "question_generation"
    assert result.generation_failure.outcome.outcome == kind
    assert result.allowed_actions == ["start", "end"]
    assert state == before


def test_corrupt_follow_up_failure_keeps_accepted_answer_guard():
    state = create_initial_state("corrupt", "synthetic")
    state.update(status="asking", questions=[QUESTION])
    state["active_question"] = make_question_turn(state, QUESTION)
    state["generation_failure"] = {
        "operation": "follow_up_generation",
        "outcome": {"private": "sentinel"},
    }
    state["current_evaluation"] = {"score": 0.9}
    state["conversation"] = [
        {
            "role": "candidate",
            "content": ANSWER,
            "question_id": state["active_question"]["question_id"],
        }
    ]
    result = interview_snapshot(state, "corrupt")
    assert result.generation_failure.outcome.outcome == "invalid_output"
    assert result.evaluation.outcome == "invalid_output"
    assert result.answer_status == "accepted"
    assert result.draft_answer is None
    assert result.allowed_actions == ["start", "skip", "end"]
    assert command_error(state, "answer", ANSWER).status_code == 409
    assert "sentinel" not in result.model_dump_json()


def test_unknown_ws_completion_error_restores_snapshot_and_never_retries(
    interview, monkeypatch
):
    session_id = start_interview(interview)["session_id"]
    calls = interview.llm.generate.await_count
    from backend.app.graph.services import GraphServices

    def broken_service(*args, **kwargs):
        raise RuntimeError("private sentinel")

    monkeypatch.setattr(GraphServices, "create", broken_service)
    with interview.client.websocket_connect(f"/ws/interview/{session_id}") as ws:
        ws.receive_json()
        ws.send_json({"type": "answer", "content": ANSWER})
        result = ws.receive_json()
        assert result["data"]["error_code"] == "command_failed"
        assert result["data"]["current_question"] == QUESTION
        assert "sentinel" not in json.dumps(result)
    assert interview.llm.generate.await_count == calls
