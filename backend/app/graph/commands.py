"""Shared command boundary for REST and WS; LangGraph still owns all progression."""

from datetime import UTC, datetime

from backend.app.graph.services import GraphServices
from backend.app.graph.state import InterviewState, active_question_response
from backend.app.graph.transitions import CommandError, command_error


async def load_session_state(graph, session_id: str) -> InterviewState:
    snapshot = await graph.aget_state({"configurable": {"thread_id": session_id}})
    if not snapshot or not snapshot.values:
        raise CommandError("Session not found", "session_not_found", 404)
    return snapshot.values


async def execute_command(
    graph,
    session_id: str,
    action: str,
    *,
    initial_state: InterviewState | None = None,
    current_answer: str | None = None,
) -> dict:
    """Validate fresh checkpoint state, invoke once, or project an active start/resume.

    There is no independent session store here. Validation precedes writes/provider
    creation. Atomic concurrency and partial-provider-failure recovery are M5 work.
    """
    if initial_state is not None:
        existing = await graph.aget_state({"configurable": {"thread_id": session_id}})
        if existing and existing.values:
            raise CommandError("Session already exists", "session_exists")
        if action != "start":
            raise CommandError("A new session must start first", "invalid_command")
        state = initial_state
    else:
        state = await load_session_state(graph, session_id)

    rejection = command_error(state, action, current_answer)
    if rejection:
        raise rejection
    if action == "start" and state["status"] == "asking":
        return {
            **state,
            **active_question_response(state),
            "model_type": state.get("model_type", "api"),
        }

    # Defaults first, then this command. Never let action=None overwrite start.
    graph_input = dict(initial_state or {})
    graph_input.update(
        action=action,
        current_answer=current_answer if action == "answer" else None,
        error=None,
        response_evaluation=None,
        updated_at=datetime.now(UTC).isoformat(),
    )
    config = {"configurable": {"thread_id": session_id}}
    try:
        if action in {"start", "answer"}:
            config["configurable"]["services"] = GraphServices.create(
                state.get("model_type", "api"), state.get("prepared_context")
            )
        return await graph.ainvoke(graph_input, config=config)
    except Exception as exc:
        # Do not leak provider payloads or pretend a failed invocation completed.
        # The checkpoint may contain partial node progress; don't auto-retry here.
        raise CommandError(
            "Interview command could not be completed", "command_failed", 503
        ) from exc
