"""Transport-independent interview command preconditions.

These checks do not serialize concurrent requests or provide command idempotency (M5).
"""

from backend.app.graph.state import InterviewState, get_active_question


class CommandError(Exception):
    """A safe command rejection that adapters map to their own error envelopes."""

    def __init__(self, detail: str, code: str, status_code: int = 400):
        super().__init__(detail)
        self.detail = detail
        self.code = code
        self.status_code = status_code


def command_error(
    state: InterviewState, action: str | None, answer: str | None = None
) -> CommandError | None:
    """Return a rejection without mutating state or calling services."""
    if not isinstance(action, str) or action not in {"start", "answer", "skip", "end"}:
        return CommandError("Unsupported interview command", "invalid_command")
    phase = state.get("status")
    if phase in {"completed", "cancelled"}:
        return CommandError("Session has already ended", "session_ended")
    if phase in {"generating", "evaluating"}:
        return CommandError("Session is busy processing a command", "session_busy", 409)
    if phase not in {"pending", "asking"}:
        return CommandError("Session has an unsupported state", "invalid_state", 409)
    if action in {"answer", "skip"} and phase != "asking":
        return CommandError("Session is not accepting answers or skips", "not_asking")
    if phase == "asking" and action != "end" and get_active_question(state) is None:
        return CommandError("No active question is available", "missing_question", 409)
    stored_evaluation = state.get("current_evaluation")
    evaluation = stored_evaluation if isinstance(stored_evaluation, dict) else {}
    stored_failure = state.get("generation_failure")
    failure = stored_failure if isinstance(stored_failure, dict) else {}
    if action == "answer" and (
        evaluation.get("follow_up_failure")
        or failure.get("operation") == "follow_up_generation"
    ):
        return CommandError(
            "This answer is already evaluated, but its follow-up is unavailable. Skip or end.",
            "follow_up_unavailable",
            409,
        )
    active = get_active_question(state)
    if (
        action == "answer"
        and active
        and any(
            message.get("role") == "candidate"
            and message.get("content") != "[Skipped]"
            and message.get("question_id") == active["question_id"]
            for message in state.get("conversation", [])
        )
    ):
        return CommandError(
            "This answer is already recorded. Skip or end to continue.",
            "answer_already_accepted",
            409,
        )
    if action == "answer" and evaluation.get("outcome") == "unsupported_mode":
        return CommandError(
            "Evaluation is unavailable in this mode. Skip or end this session.",
            "evaluation_unsupported_mode",
            409,
        )
    if action == "answer" and (not isinstance(answer, str) or not answer.strip()):
        return CommandError("Answer cannot be empty", "invalid_answer")
    return None
