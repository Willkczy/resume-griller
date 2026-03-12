"""
Interview Session API routes.

Simplified to use the LangGraph interview graph for all orchestration.
Both "api" and "custom" model types flow through the same graph —
the difference is in which GraphServices are injected.

The graph handles: question generation, answer evaluation, follow-up
grilling, advancing, skipping, and interview completion.
"""

from typing import List, Optional

from fastapi import APIRouter, HTTPException, Depends, status, Request
from pydantic import BaseModel, Field

from backend.app.api.deps import get_retriever
from backend.app.graph import get_compiled_graph, create_initial_state, GraphServices
from backend.app.middleware.rate_limit import limiter
from rag.retriever import InterviewRetriever


# ============== Request/Response Schemas ==============
# (unchanged — same API contract for the frontend)

class SessionCreateRequest(BaseModel):
    """Request to create a new interview session."""
    resume_id: str
    mode: str = Field(default="mixed", pattern="^(hr|tech|mixed)$")
    model_type: str = Field(default="api", pattern="^(api|custom)$")
    focus_areas: List[str] = Field(default_factory=list)
    num_questions: int = Field(default=5, ge=1, le=15)
    max_follow_ups: int = Field(default=3, ge=0, le=5)


class SessionResponse(BaseModel):
    """Session information response."""
    session_id: str
    resume_id: str
    mode: str
    model_type: str
    status: str
    current_question_index: int
    total_questions: int
    questions_asked: int
    created_at: str
    updated_at: str

    class Config:
        from_attributes = True


class AnswerRequest(BaseModel):
    """Request to submit an answer."""
    answer: str = Field(..., min_length=1)


class InterviewResponseModel(BaseModel):
    """Response from the interviewer."""
    type: str  # "question", "follow_up", "complete", "error"
    content: str
    question_number: Optional[int] = None
    total_questions: Optional[int] = None
    evaluation: Optional[dict] = None
    metadata: Optional[dict] = None


class ConversationMessage(BaseModel):
    """A message in the conversation."""
    role: str
    content: str
    timestamp: str
    is_follow_up: bool = False


class SessionDetailResponse(BaseModel):
    """Detailed session response with conversation."""
    session_id: str
    resume_id: str
    mode: str
    model_type: str
    status: str
    current_question: Optional[str]
    current_question_index: int
    total_questions: int
    follow_up_count: int
    max_follow_ups: int
    conversation: List[ConversationMessage]
    created_at: str
    updated_at: str


class SessionSummaryResponse(BaseModel):
    """Interview summary response."""
    session_id: str
    resume_id: str
    mode: str
    status: str
    questions_asked: int
    total_questions: int
    answers_given: int
    follow_ups_asked: int
    conversation_length: int


# ============== Router ==============

router = APIRouter(prefix="/sessions", tags=["interview sessions"])


# ============== Helper: invoke graph and build response ==============

async def _invoke_graph(
    session_id: str,
    action: str,
    retriever: InterviewRetriever,
    model_type: str = "api",
    initial_state: dict | None = None,
    current_answer: str | None = None,
    prepared_context: dict | None = None,
) -> InterviewResponseModel:
    """
    Invoke the interview graph and convert the result to an API response.

    This is the single point where all session endpoints call the graph.
    The graph handles all orchestration — this function just translates
    between HTTP request/response and graph state.
    """
    # Create services based on model type (API vs Custom/Hybrid)
    services = GraphServices.create(model_type, retriever, prepared_context)

    # Get the compiled graph (cached singleton with SQLite checkpointer)
    graph = await get_compiled_graph()

    # Build the input for this invocation.
    # For the first invocation ("start"), we pass the full initial state.
    # For subsequent invocations, we only pass the action + answer.
    # The checkpointer restores the rest from the previous checkpoint.
    graph_input: dict = {"action": action}
    if initial_state:
        graph_input.update(initial_state)
    if current_answer:
        graph_input["current_answer"] = current_answer

    # Invoke the graph with thread_id = session_id for checkpointing
    result = await graph.ainvoke(
        graph_input,
        config={
            "configurable": {
                "thread_id": session_id,
                "services": services,
            }
        },
    )

    # Convert graph output to API response
    response_data = result.get("response_data", {})
    return InterviewResponseModel(
        type=result.get("response_type", "error"),
        content=result.get("response_content", ""),
        question_number=response_data.get("question_number"),
        total_questions=response_data.get("total_questions"),
        evaluation=response_data.get("evaluation"),
        metadata={
            "session_id": session_id,
            "model_type": model_type,
            **{k: v for k, v in response_data.items()
               if k not in ("question_number", "total_questions", "evaluation", "summary")},
        },
    )


# ============== Endpoints ==============

@router.post("", response_model=InterviewResponseModel)
@limiter.limit("5/minute")
async def create_session(
    http_request: Request,
    request: SessionCreateRequest,
    retriever: InterviewRetriever = Depends(get_retriever),
):
    """
    Create a new interview session and start the interview.

    The graph generates questions from the resume and returns the first one.
    For "custom" model_type, Groq preprocesses the resume first, then the
    Custom Model is used for evaluation during the interview.
    """
    # Verify resume exists
    try:
        summary = retriever.get_resume_summary(request.resume_id)
        if summary.get("total_chunks", 0) == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Resume not found: {request.resume_id}",
            )
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Resume not found: {request.resume_id}",
        )

    # For custom model, prepare context via Groq first
    prepared_context = None
    if request.model_type == "custom":
        try:
            from backend.app.services.llm_service import LLMServiceFactory
            resume_text = retriever.get_full_resume_text(request.resume_id)
            hybrid_service = LLMServiceFactory.get_hybrid_service()
            prepared_context = await hybrid_service.prepare_interview_context(
                session_id=request.resume_id,  # temporary, real session_id set below
                resume_text=resume_text,
                mode=request.mode,
                num_questions=request.num_questions,
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Custom model preparation failed: {e}. Try 'api' model_type.",
            )

    # Generate a session ID
    import uuid
    session_id = f"sess_{uuid.uuid4().hex[:12]}"

    # Create initial state for the graph
    initial = create_initial_state(
        session_id=session_id,
        resume_id=request.resume_id,
        mode=request.mode,
        model_type=request.model_type,
        num_questions=request.num_questions,
        max_follow_ups=request.max_follow_ups,
        focus_areas=request.focus_areas,
        prepared_context=prepared_context,
    )

    return await _invoke_graph(
        session_id=session_id,
        action="start",
        retriever=retriever,
        model_type=request.model_type,
        initial_state=initial,
        prepared_context=prepared_context,
    )


@router.get("/{session_id}", response_model=SessionDetailResponse)
async def get_session(session_id: str):
    """Get detailed session information including conversation history from checkpoint."""
    graph = await get_compiled_graph()

    # Load the latest checkpoint for this session
    config = {"configurable": {"thread_id": session_id}}
    state = await graph.aget_state(config)

    if not state or not state.values:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}",
        )

    s = state.values
    questions = s.get("questions", [])
    idx = s.get("current_question_index", 0)

    return SessionDetailResponse(
        session_id=s.get("session_id", session_id),
        resume_id=s.get("resume_id", ""),
        mode=s.get("mode", "mixed"),
        model_type=s.get("model_type", "api"),
        status=s.get("status", "pending"),
        current_question=questions[idx] if idx < len(questions) else None,
        current_question_index=idx,
        total_questions=len(questions),
        follow_up_count=s.get("current_follow_up_count", 0),
        max_follow_ups=s.get("max_follow_ups", 3),
        conversation=[
            ConversationMessage(
                role=m.get("role", "system"),
                content=m.get("content", ""),
                timestamp=m.get("timestamp", ""),
                is_follow_up=m.get("is_follow_up", False),
            )
            for m in s.get("conversation", [])
        ],
        created_at="",  # Not tracked in graph state (could add if needed)
        updated_at="",
    )


@router.post("/{session_id}/answer", response_model=InterviewResponseModel)
@limiter.limit("10/minute")
async def submit_answer(
    http_request: Request,
    session_id: str,
    request: AnswerRequest,
    retriever: InterviewRetriever = Depends(get_retriever),
):
    """Submit an answer to the current question."""
    # Load current state to get model_type and prepared_context
    graph = await get_compiled_graph()
    config = {"configurable": {"thread_id": session_id}}
    state = await graph.aget_state(config)

    if not state or not state.values:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}",
        )

    s = state.values
    if s.get("status") not in ("asking",):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Session is not accepting answers. Status: {s.get('status')}",
        )

    return await _invoke_graph(
        session_id=session_id,
        action="answer",
        retriever=retriever,
        model_type=s.get("model_type", "api"),
        current_answer=request.answer,
        prepared_context=s.get("prepared_context"),
    )


@router.post("/{session_id}/skip", response_model=InterviewResponseModel)
async def skip_question(
    session_id: str,
    retriever: InterviewRetriever = Depends(get_retriever),
):
    """Skip the current question and move to the next one."""
    graph = await get_compiled_graph()
    config = {"configurable": {"thread_id": session_id}}
    state = await graph.aget_state(config)

    if not state or not state.values:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}",
        )

    s = state.values
    if s.get("status") not in ("asking",):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Session is not in progress. Status: {s.get('status')}",
        )

    return await _invoke_graph(
        session_id=session_id,
        action="skip",
        retriever=retriever,
        model_type=s.get("model_type", "api"),
        prepared_context=s.get("prepared_context"),
    )


@router.post("/{session_id}/end", response_model=InterviewResponseModel)
async def end_session(
    session_id: str,
    retriever: InterviewRetriever = Depends(get_retriever),
):
    """End the interview session early."""
    graph = await get_compiled_graph()
    config = {"configurable": {"thread_id": session_id}}
    state = await graph.aget_state(config)

    if not state or not state.values:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}",
        )

    return await _invoke_graph(
        session_id=session_id,
        action="end",
        retriever=retriever,
        model_type=state.values.get("model_type", "api"),
        prepared_context=state.values.get("prepared_context"),
    )


@router.get("/{session_id}/summary", response_model=SessionSummaryResponse)
async def get_session_summary(session_id: str):
    """Get a summary of the interview session."""
    graph = await get_compiled_graph()
    config = {"configurable": {"thread_id": session_id}}
    state = await graph.aget_state(config)

    if not state or not state.values:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}",
        )

    s = state.values
    conversation = s.get("conversation", [])
    candidate_msgs = [
        m for m in conversation
        if m.get("role") == "candidate" and m.get("content") != "[Skipped]"
    ]
    follow_ups = [m for m in conversation if m.get("is_follow_up")]

    return SessionSummaryResponse(
        session_id=s.get("session_id", session_id),
        resume_id=s.get("resume_id", ""),
        mode=s.get("mode", "mixed"),
        status=s.get("status", "pending"),
        questions_asked=s.get("current_question_index", 0),
        total_questions=len(s.get("questions", [])),
        answers_given=len(candidate_msgs),
        follow_ups_asked=len(follow_ups),
        conversation_length=len(conversation),
    )


@router.delete("/{session_id}")
async def delete_session(session_id: str):
    """Delete a session (removes its checkpoint)."""
    graph = await get_compiled_graph()
    config = {"configurable": {"thread_id": session_id}}
    state = await graph.aget_state(config)

    if not state or not state.values:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}",
        )

    # Delete the checkpoint thread
    from backend.app.graph.checkpointer import get_checkpointer
    checkpointer = await get_checkpointer()
    await checkpointer.adelete_thread(session_id)

    return {"message": f"Session {session_id} deleted successfully"}
