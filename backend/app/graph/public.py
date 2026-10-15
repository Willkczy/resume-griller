"""Read-only checkpoint projection shared by all public interview transports."""

import json

from pydantic import TypeAdapter, ValidationError

from backend.app.graph.state import (
    InterviewState,
    get_active_question,
    normalize_public_status,
)
from backend.app.graph.transitions import command_error
from backend.app.models.interview import (
    GenerationFailure,
    InterviewSnapshot,
    PublicEvaluation,
    WebSocketInterviewData,
)

_EVALUATION = TypeAdapter(PublicEvaluation)


def project_generation_failure(value, state: InterviewState):
    if value:
        try:
            return GenerationFailure.model_validate_json(json.dumps(value))
        except (ValidationError, ValueError, TypeError):
            return GenerationFailure(
                operation=(
                    "question_generation"
                    if state.get("status") == "pending"
                    else "follow_up_generation"
                ),
                outcome={"outcome": "invalid_output", "code": "schema_violation"},
            )
    # Stage 2 checkpoints stored only safe error codes for startup failures.
    code = (state.get("response_data") or {}).get("error_code")
    if state.get("status") == "pending" and code in {
        "question_generation_invalid_output",
        "question_generation_provider_failure",
        "question_generation_unsupported_mode",
    }:
        outcome = code.removeprefix("question_generation_")
        return GenerationFailure(
            operation="question_generation",
            outcome={
                "outcome": outcome,
                "code": {
                    "invalid_output": "schema_violation",
                    "provider_failure": "unknown",
                    "unsupported_mode": "evaluation_contract_unavailable",
                }[outcome],
            },
        )
    return None


def project_evaluation(value: dict | None):
    """Normalize older valid tags in memory; reject corrupt stored scores safely."""
    if not value:
        return None
    try:
        normalized = dict(value)
        if "outcome" not in normalized and "score" in normalized:
            normalized["outcome"] = "valid"
        return _EVALUATION.validate_json(json.dumps(normalized))
    except (ValidationError, ValueError, TypeError):
        return _EVALUATION.validate_python(
            {"outcome": "invalid_output", "code": "schema_violation"}
        )


def interview_snapshot(state: InterviewState, session_id: str) -> InterviewSnapshot:
    """Expose only safe outcome fields; no provider call, mutation, or recovery retry."""
    active = get_active_question(state)
    finished = state.get("status") in {"completed", "cancelled"}
    raw_evaluation = state.get("current_evaluation") or state.get("response_evaluation")
    evaluation = project_evaluation(raw_evaluation)
    if finished and (
        state.get("status") == "cancelled"
        or not evaluation
        or evaluation.outcome != "valid"
    ):
        evaluation = None
    failure = None if finished else state.get("generation_failure")
    # Older follow-up failure checkpoints already contain this safe failure taxonomy.
    if not failure and evaluation and getattr(evaluation, "follow_up_failure", None):
        failure = {
            "operation": "follow_up_generation",
            "outcome": evaluation.follow_up_failure.model_dump(mode="json"),
        }
    generation_failure = (
        None if finished else project_generation_failure(failure, state)
    )
    targets_active = bool(
        evaluation
        and active
        and evaluation.question_id in {None, active["question_id"]}
    )
    recorded_answer = bool(
        active
        and any(
            message.get("role") == "candidate"
            and message.get("content") != "[Skipped]"
            and message.get("question_id") == active["question_id"]
            for message in state.get("conversation", [])
        )
    )
    accepted = recorded_answer or (targets_active and evaluation.outcome == "valid")
    unscored = targets_active and evaluation.outcome != "valid" and not accepted
    return InterviewSnapshot(
        session_id=state.get("session_id", session_id),
        resume_id=state.get("resume_id", ""),
        mode=state.get("mode", "mixed"),
        model_type=state.get("model_type", "api"),
        status=normalize_public_status(state.get("status")),
        current_question=active["prompt"] if active else None,
        current_question_id=active["question_id"] if active else None,
        parent_question_id=active["parent_question_id"] if active else None,
        is_follow_up=active["is_follow_up"] if active else False,
        current_question_index=state.get("current_question_index", 0),
        total_questions=len(state.get("questions", [])),
        follow_up_count=state.get("current_follow_up_count", 0),
        max_follow_ups=state.get("max_follow_ups", 3),
        conversation=[
            {
                "role": message.get("role", "system"),
                "content": message.get("content", ""),
                "timestamp": message.get("timestamp", ""),
                "is_follow_up": message.get("is_follow_up", False),
                "question_id": message.get("question_id"),
                "parent_question_id": message.get("parent_question_id"),
                "answer_id": message.get("answer_id"),
            }
            for message in state.get("conversation", [])
        ],
        created_at=state.get("created_at", ""),
        updated_at=state.get("updated_at", ""),
        model_executions=state.get("model_executions", []),
        evaluation=evaluation,
        generation_failure=generation_failure,
        draft_answer=state.get("current_answer") if unscored else None,
        answer_status="unscored" if unscored else "accepted" if accepted else "none",
        allowed_actions=[
            action
            for action in ["start", "answer", "skip", "end"]
            if command_error(state, action, "Explicit answer") is None
        ],
    )


def websocket_data(state: InterviewState, session_id: str, **extra) -> dict:
    """Validate outcome data for connected, command success, and command rejection."""
    data = {
        **(state.get("response_data") or {}),
        **interview_snapshot(state, session_id).model_dump(mode="json"),
        **extra,
    }
    return WebSocketInterviewData.model_validate_json(json.dumps(data)).model_dump(
        mode="json"
    )
