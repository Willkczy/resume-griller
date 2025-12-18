"""
Interview Session API routes.

Supports:
- "api" model_type: Uses configured LLM (Groq/Gemini/etc)
- "custom" model_type: Uses Hybrid approach (Groq preprocessing + Custom Model execution)
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends, status

from backend.app.db.session_store import (
    InterviewSession,
    SessionStatus,
    SessionStore,
    get_session_store,
)
from backend.app.core.interview_agent import InterviewAgent, InterviewerResponse
from backend.app.api.deps import get_retriever, get_llm
from backend.app.services.llm_service import (
    BaseLLMService,
    LLMServiceFactory,
    HybridModelService,
)
from rag.retriever import InterviewRetriever

from pydantic import BaseModel, Field


# ============== Request/Response Schemas ==============

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
    duration_seconds: float
    conversation_length: int


# ============== Router ==============

router = APIRouter(prefix="/sessions", tags=["interview sessions"])


# ============== Helper Functions ==============

def get_llm_service_for_model_type(model_type: str) -> BaseLLMService:
    """
    Get LLM service based on model_type.
    
    Args:
        model_type: "api" for cloud APIs (Groq/Gemini), "custom" for Hybrid mode
    
    Returns:
        Appropriate LLM service instance
    """
    if model_type == "custom":
        # Use Hybrid service for custom model
        try:
            return LLMServiceFactory.get_hybrid_service()
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Hybrid model service not available: {str(e)}",
            )
    else:
        # Use default API provider (Groq/Gemini/etc based on .env)
        return LLMServiceFactory.get_service()


def get_hybrid_service() -> HybridModelService:
    """Get the hybrid model service instance."""
    return LLMServiceFactory.get_hybrid_service()


def create_interview_agent(
    model_type: str,
    retriever: InterviewRetriever,
) -> InterviewAgent:
    """Create InterviewAgent with the appropriate LLM service.
    
    Note: InterviewAgent is only used for API mode.
          Custom mode is handled directly by _process_answer_hybrid().
    """
    llm_service = get_llm_service_for_model_type(model_type)
    return InterviewAgent(
        llm_service=llm_service,
        retriever=retriever,
    )


# ============== Endpoints ==============

@router.post("", response_model=InterviewResponseModel)
async def create_session(
    request: SessionCreateRequest,
    session_store: SessionStore = Depends(get_session_store),
    retriever: InterviewRetriever = Depends(get_retriever),
):
    """
    Create a new interview session and start the interview.
    
    For "custom" model_type:
    - Phase 1: Uses Groq to prepare interview context (summary, questions)
    - Phase 2: Uses Custom Model for interview execution
    
    Returns the first question.
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
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Resume not found: {request.resume_id}",
        )
    
    # Create session
    session = session_store.create(
        resume_id=request.resume_id,
        mode=request.mode,
        model_type=request.model_type,
        focus_areas=request.focus_areas,
        max_follow_ups=request.max_follow_ups,
    )
    
    # For "custom" model_type, use Hybrid approach
    if request.model_type == "custom":
        try:
            # Get full resume text for preprocessing
            resume_text = retriever.get_full_resume_text(request.resume_id)
            
            # Get hybrid service and prepare context
            hybrid_service = get_hybrid_service()
            
            print(f"[Session] Preparing hybrid interview context for {session.session_id}")
            
            # Phase 1: Groq prepares everything
            prepared_context = await hybrid_service.prepare_interview_context(
                session_id=session.session_id,
                resume_text=resume_text,
                mode=request.mode,
                num_questions=request.num_questions,
            )
            
            # Store prepared questions in session
            session.questions = prepared_context["questions"]
            session.status = SessionStatus.IN_PROGRESS
            session.current_question_index = 0
            
            # Store context in session metadata for persistence
            session.prepared_context = prepared_context
            
            # Add first question to conversation
            if session.questions:
                from backend.app.db.session_store import MessageRole
                first_question = session.questions[0]
                session.add_message(
                    role=MessageRole.INTERVIEWER,
                    content=first_question,
                    metadata={"question_number": 1},
                )
            
            session_store.update(session)
            
            return InterviewResponseModel(
                type="question",
                content=session.questions[0] if session.questions else "No questions generated",
                question_number=1,
                total_questions=len(session.questions),
                evaluation=None,
                metadata={
                    "session_id": session.session_id,
                    "model_type": "custom",
                    "mode": request.mode,
                    "preprocessing": "groq",
                    "execution": "custom_model",
                },
            )
            
        except Exception as e:
            print(f"[Session] Hybrid preparation failed: {e}")
            # Fallback to API mode
            session.model_type = "api"
            session_store.update(session)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Custom model preparation failed: {str(e)}. Try using 'api' model_type instead.",
            )
    
    # Standard API mode
    agent = create_interview_agent(request.model_type, retriever)
    
    response = await agent.start_interview(
        session=session,
        num_questions=request.num_questions,
    )
    
    return InterviewResponseModel(
        type=response.type.value,
        content=response.content,
        question_number=response.question_number,
        total_questions=response.total_questions,
        evaluation=response.evaluation,
        metadata={
            "session_id": session.session_id,
            "model_type": session.model_type,
        },
    )


@router.get("", response_model=List[SessionResponse])
async def list_sessions(
    resume_id: Optional[str] = None,
    session_store: SessionStore = Depends(get_session_store),
):
    """List all sessions, optionally filtered by resume_id."""
    sessions = session_store.list_all(resume_id=resume_id)
    
    return [
        SessionResponse(
            session_id=s.session_id,
            resume_id=s.resume_id,
            mode=s.mode,
            model_type=s.model_type,
            status=s.status.value,
            current_question_index=s.current_question_index,
            total_questions=len(s.questions),
            questions_asked=s.current_question_index,
            created_at=s.created_at.isoformat(),
            updated_at=s.updated_at.isoformat(),
        )
        for s in sessions
    ]


@router.get("/{session_id}", response_model=SessionDetailResponse)
async def get_session(
    session_id: str,
    session_store: SessionStore = Depends(get_session_store),
):
    """Get detailed session information including conversation history."""
    session = session_store.get(session_id)
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}",
        )
    
    return SessionDetailResponse(
        session_id=session.session_id,
        resume_id=session.resume_id,
        mode=session.mode,
        model_type=session.model_type,
        status=session.status.value,
        current_question=session.current_question,
        current_question_index=session.current_question_index,
        total_questions=len(session.questions),
        follow_up_count=session.current_follow_up_count,
        max_follow_ups=session.max_follow_ups,
        conversation=[
            ConversationMessage(
                role=m.role.value,
                content=m.content,
                timestamp=m.timestamp.isoformat(),
                is_follow_up=m.is_follow_up,
            )
            for m in session.conversation
        ],
        created_at=session.created_at.isoformat(),
        updated_at=session.updated_at.isoformat(),
    )


@router.post("/{session_id}/answer", response_model=InterviewResponseModel)
async def submit_answer(
    session_id: str,
    request: AnswerRequest,
    session_store: SessionStore = Depends(get_session_store),
    retriever: InterviewRetriever = Depends(get_retriever),
):
    """
    Submit an answer to the current question.
    
    For "custom" model_type:
    - Uses Custom Model with pre-prepared context for evaluation
    - Generates follow-ups using Custom Model
    """
    session = session_store.get(session_id)
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}",
        )
    
    if session.status != SessionStatus.IN_PROGRESS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Session is not in progress. Status: {session.status.value}",
        )
    
    # For custom model, use hybrid evaluation
    if session.model_type == "custom":
        return await _process_answer_hybrid(session, request.answer, session_store)
    
    # Standard API mode
    agent = create_interview_agent(session.model_type, retriever)
    
    response = await agent.process_answer(
        session=session,
        answer=request.answer,
    )
    
    return InterviewResponseModel(
        type=response.type.value,
        content=response.content,
        question_number=response.question_number,
        total_questions=response.total_questions,
        evaluation=response.evaluation,
        metadata=response.metadata,
    )


async def _process_answer_hybrid(
    session: InterviewSession,
    answer: str,
    session_store: SessionStore,
) -> InterviewResponseModel:
    """
    Process answer using Hybrid approach.
    
    Uses GrillingEngine with Custom Model for evaluation and follow-up generation.
    The Custom Model receives compact prompts with pre-prepared context from Groq.
    """
    from backend.app.db.session_store import MessageRole
    from backend.app.core.grilling_engine import GrillingEngine
    
    current_question = session.current_question
    if not current_question:
        return InterviewResponseModel(
            type="complete",
            content="Interview complete! Thank you for your responses.",
            metadata={"session_id": session.session_id},
        )
    
    # Add answer to conversation
    session.add_message(role=MessageRole.CANDIDATE, content=answer)
    
    # Get hybrid service and create GrillingEngine with Custom Model
    hybrid_service = get_hybrid_service()
    
    # Restore prepared context if needed
    prepared_context = {}
    if hasattr(session, 'prepared_context') and session.prepared_context:
        prepared_context = session.prepared_context
        hybrid_service.set_prepared_context(session.session_id, prepared_context)
    
    # Create GrillingEngine with Custom Model (interviewer) and prepared context
    grilling_engine = GrillingEngine(
        llm_service=hybrid_service.interviewer,  # Use Custom Model
        model_type="custom",  # Enables compact prompts
        prepared_context=prepared_context,  # Pass pre-processed context
    )
    
    # Build conversation history
    conversation_history = [
        {"role": m.role.value, "content": m.content, "is_follow_up": m.is_follow_up}
        for m in session.conversation[-10:]
    ]
    
    # Evaluate answer using GrillingEngine (with Custom Model)
    evaluation = await grilling_engine.evaluate_answer(
        question=current_question,
        answer=answer,
        resume_context="",  # Not needed - using prepared_context
        question_type=session.mode,
        conversation_history=conversation_history,
        follow_up_count=session.current_follow_up_count,
        question_index=session.current_question_index,
    )
    
    # Use GrillingEngine's decision logic
    should_followup = grilling_engine.should_grill(
        evaluation=evaluation,
        follow_up_count=session.current_follow_up_count,
        max_follow_ups=session.max_follow_ups,
    )
    
    if should_followup:
        # Generate follow-up using GrillingEngine (with Custom Model)
        followup = await grilling_engine.generate_follow_up(
            question=current_question,
            answer=answer,
            evaluation=evaluation,
            conversation_history=conversation_history,
            question_type=session.mode,
        )
        
        session.increment_follow_up()
        session.add_message(
            role=MessageRole.INTERVIEWER,
            content=followup,
            is_follow_up=True,
            metadata={
                "gap": evaluation.gap_analysis.priority_gap.value if evaluation.gap_analysis.priority_gap else None,
                "score": evaluation.score,
                "detected_gaps": [g.value for g in evaluation.gap_analysis.detected_gaps],
            },
        )
        
        session_store.update(session)
        
        return InterviewResponseModel(
            type="follow_up",
            content=followup,
            question_number=session.current_question_index + 1,
            total_questions=len(session.questions),
            evaluation=evaluation.to_dict(),
            metadata={
                "follow_up_count": session.current_follow_up_count,
                "priority_gap": evaluation.gap_analysis.priority_gap.value if evaluation.gap_analysis.priority_gap else None,
            },
        )
    
    # Move to next question
    next_question = session.next_question()
    
    if next_question is None:
        session.status = SessionStatus.COMPLETED
        session.add_message(role=MessageRole.SYSTEM, content="Interview completed.")
        session_store.update(session)
        
        return InterviewResponseModel(
            type="complete",
            content="Excellent! That concludes our interview. Thank you for your thoughtful responses.",
            evaluation=evaluation.to_dict(),
            metadata={"session_id": session.session_id},
        )
    
    # Ask next question
    session.add_message(
        role=MessageRole.INTERVIEWER,
        content=next_question,
        metadata={"question_number": session.current_question_index + 1},
    )
    
    session_store.update(session)
    
    return InterviewResponseModel(
        type="question",
        content=next_question,
        question_number=session.current_question_index + 1,
        total_questions=len(session.questions),
        evaluation=evaluation.to_dict(),
        metadata={"session_id": session.session_id},
    )


@router.post("/{session_id}/skip", response_model=InterviewResponseModel)
async def skip_question(
    session_id: str,
    session_store: SessionStore = Depends(get_session_store),
    retriever: InterviewRetriever = Depends(get_retriever),
):
    """Skip the current question and move to the next one."""
    session = session_store.get(session_id)
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}",
        )
    
    if session.status != SessionStatus.IN_PROGRESS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Session is not in progress. Status: {session.status.value}",
        )
    
    from backend.app.db.session_store import MessageRole
    
    session.add_message(
        role=MessageRole.CANDIDATE,
        content="[Skipped]",
        metadata={"skipped": True},
    )
    
    next_question = session.next_question()
    
    if next_question is None:
        session.status = SessionStatus.COMPLETED
        session_store.update(session)
        
        return InterviewResponseModel(
            type="complete",
            content="Interview complete! Thank you for your responses.",
            metadata={"session_id": session.session_id},
        )
    
    session.add_message(
        role=MessageRole.INTERVIEWER,
        content=next_question,
        metadata={"question_number": session.current_question_index + 1},
    )
    
    session_store.update(session)
    
    return InterviewResponseModel(
        type="question",
        content=next_question,
        question_number=session.current_question_index + 1,
        total_questions=len(session.questions),
        metadata={"session_id": session.session_id},
    )


@router.post("/{session_id}/end", response_model=InterviewResponseModel)
async def end_session(
    session_id: str,
    session_store: SessionStore = Depends(get_session_store),
):
    """End the interview session early."""
    session = session_store.get(session_id)
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}",
        )
    
    from backend.app.db.session_store import MessageRole
    
    session.status = SessionStatus.CANCELLED
    session.add_message(
        role=MessageRole.SYSTEM,
        content="Interview ended by user.",
    )
    
    session_store.update(session)
    
    return InterviewResponseModel(
        type="complete",
        content="Interview ended. Thank you for your time.",
        metadata={"reason": "ended by user", "session_id": session.session_id},
    )


@router.get("/{session_id}/summary", response_model=SessionSummaryResponse)
async def get_session_summary(
    session_id: str,
    session_store: SessionStore = Depends(get_session_store),
):
    """Get a summary of the interview session."""
    session = session_store.get(session_id)
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}",
        )
    
    from backend.app.db.session_store import MessageRole
    
    candidate_messages = [
        m for m in session.conversation
        if m.role == MessageRole.CANDIDATE and "[Skipped]" not in m.content
    ]
    follow_ups = [m for m in session.conversation if m.is_follow_up]
    
    return SessionSummaryResponse(
        session_id=session.session_id,
        resume_id=session.resume_id,
        mode=session.mode,
        status=session.status.value,
        questions_asked=session.current_question_index,
        total_questions=len(session.questions),
        answers_given=len(candidate_messages),
        follow_ups_asked=len(follow_ups),
        duration_seconds=(session.updated_at - session.created_at).total_seconds(),
        conversation_length=len(session.conversation),
    )


@router.delete("/{session_id}")
async def delete_session(
    session_id: str,
    session_store: SessionStore = Depends(get_session_store),
):
    """Delete a session."""
    success = session_store.delete(session_id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}",
        )
    
    return {"message": f"Session {session_id} deleted successfully"}