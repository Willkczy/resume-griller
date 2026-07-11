"""
Interview Graph State — the data schema that flows through every node.

=== LangGraph Concept: State ===

In LangGraph, a "state" is a TypedDict that acts as the shared memory for your graph.
Every node receives the current state, does work, and returns a PARTIAL dict of
fields it wants to update. LangGraph merges those updates into the full state.

Think of it like a shared whiteboard:
- Each node reads what it needs from the whiteboard
- Each node writes back only the fields it changed
- LangGraph handles the merge

Key rules:
1. State is a TypedDict (not a dataclass) — LangGraph requires this for serialization
2. Nodes return PARTIAL state updates — you don't return the whole state, just changed fields
3. Use Annotated[list, operator.add] for append-only fields (like conversation history)
   This tells LangGraph: "when a node returns this field, APPEND to the list, don't replace it"

Example flow:
  generate_questions node receives: {"session_id": "abc", "questions": [], ...}
  generate_questions node returns:  {"questions": ["Q1", "Q2"], "status": "asking"}
  LangGraph merges → full state now has questions populated and status updated
"""

from __future__ import annotations

import operator
from datetime import UTC, datetime
from typing import Annotated, Any, Literal, TypedDict

# === The Interview State ===
#
# All interview state lives here — checkpointed to SQLite by LangGraph.
# No separate session store needed.


class InterviewState(TypedDict, total=False):
    """
    State schema for the interview graph.

    `total=False` means all fields are optional — nodes only need to return
    the fields they change. LangGraph fills in the rest from the checkpoint.
    """

    # ─── Identity (set once at creation, never changes) ───
    session_id: str
    resume_id: str
    full_resume_text: str  # LLM-parsed resume, passed to all nodes
    mode: Literal["hr", "tech", "mixed"]
    model_type: Literal["api", "custom"]
    created_at: str
    updated_at: str

    # ─── Config (set once at creation) ───
    num_questions: int
    max_follow_ups: int
    focus_areas: list[str]

    # ─── Flow State (updated by nodes as interview progresses) ───
    #
    # status: tracks where we are in the interview lifecycle
    status: Literal[
        "pending",  # Session created, not started
        "generating",  # Generating questions from resume
        "asking",  # Waiting for candidate's answer
        "evaluating",  # Evaluating an answer
        "completed",  # All questions done
        "cancelled",  # User ended early
    ]
    questions: list[str]  # Generated interview questions
    current_question_index: int  # Which question we're on (0-based)
    current_follow_up_count: int  # How many follow-ups for current question
    resume_context: str  # RAG-retrieved context for current question

    # ─── Current Interaction (set/cleared each graph invocation) ───
    current_answer: str | None  # The answer being evaluated right now
    current_evaluation: dict | None  # Result from GrillingEngine.evaluate_answer()

    # ─── Conversation History (append-only) ───
    #
    # Annotated[list, operator.add] tells LangGraph:
    #   "When a node returns {'conversation': [new_msg]}, APPEND it to the list"
    #   Without this, returning [new_msg] would REPLACE the entire list.
    #
    # Each message is a dict: {"role": "interviewer"|"candidate"|"system",
    #                           "content": "...", "is_follow_up": bool, "metadata": {}}
    conversation: Annotated[list[dict], operator.add]

    # ─── Prepared Context (Hybrid mode only) ───
    #
    # When model_type="custom", Groq preprocesses the resume into a compact format.
    # This dict contains: resume_summary, questions, question_contexts
    # Stored here so it flows through the graph and gets checkpointed.
    prepared_context: dict[str, Any] | None

    # ─── Output (read by the route handler after graph returns) ───
    #
    # These fields tell the HTTP/WS handler what to send back to the client.
    # The graph sets these; the route reads them and formats the response.
    response_type: Literal["question", "follow_up", "complete", "error"] | None
    response_content: str | None
    response_data: dict | None

    # ─── Input Signal (set by route handler BEFORE invoking graph) ───
    #
    # This is how HTTP/WS routes tell the graph what to do.
    # Each API call sets action + optionally current_answer, then invokes the graph.
    #
    # "start"  → generate questions and ask the first one
    # "answer" → evaluate current_answer, decide follow-up or next question
    # "skip"   → skip current question, move to next
    # "end"    → cancel interview, generate summary
    action: Literal["start", "answer", "skip", "end"] | None

    # ─── Error ───
    error: str | None


# === Helper: Create initial state for a new interview ===
#
# This is called when POST /sessions creates a new interview.
# It sets up the state that will be passed to the first graph invocation.


def create_initial_state(
    session_id: str,
    resume_id: str,
    mode: str = "mixed",
    model_type: str = "api",
    num_questions: int = 5,
    max_follow_ups: int = 3,
    focus_areas: list[str] | None = None,
    prepared_context: dict | None = None,
    full_resume_text: str = "",
) -> InterviewState:
    """Create the initial state for a new interview session."""
    now = datetime.now(UTC).isoformat()
    return InterviewState(
        # Identity
        session_id=session_id,
        resume_id=resume_id,
        full_resume_text=full_resume_text,
        mode=mode,
        model_type=model_type,
        created_at=now,
        updated_at=now,
        # Config
        num_questions=num_questions,
        max_follow_ups=max_follow_ups,
        focus_areas=focus_areas or [],
        # Flow — starts pending, waiting for "start" action
        status="pending",
        questions=[],
        current_question_index=0,
        current_follow_up_count=0,
        resume_context="",
        # Interaction — empty until first invocation
        current_answer=None,
        current_evaluation=None,
        # Conversation — empty, will be appended to
        conversation=[],
        # Hybrid mode
        prepared_context=prepared_context,
        # Output — empty until graph runs
        response_type=None,
        response_content=None,
        response_data=None,
        # Input — will be set by route handler before each invocation
        action=None,
        error=None,
    )


def normalize_public_status(status: str | None) -> str:
    """Map graph execution phases to the stable public lifecycle contract."""
    if status in {"asking", "evaluating", "generating"}:
        return "in_progress"
    if status in {"pending", "completed", "cancelled"}:
        return status
    return "pending"


def calculate_questions_asked(state: InterviewState) -> int:
    """Count distinct main questions presented, excluding follow-ups."""
    presented = sum(
        1
        for message in state.get("conversation", [])
        if message.get("role") == "interviewer"
        and not message.get("is_follow_up", False)
    )
    return min(presented, len(state.get("questions", [])))


def calculate_duration_seconds(state: InterviewState) -> float:
    """Return elapsed session time; legacy checkpoints without timestamps use zero."""
    try:
        created = datetime.fromisoformat(state["created_at"])
        updated = datetime.fromisoformat(state["updated_at"])
        return max(0.0, (updated - created).total_seconds())
    except (KeyError, TypeError, ValueError):
        return 0.0
