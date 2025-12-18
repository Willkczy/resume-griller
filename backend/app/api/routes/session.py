"""
Interview Session API routes.
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
from backend.app.services.llm_service import BaseLLMService, LLMServiceFactory
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
        model_type: "api" for cloud APIs (Groq/Gemini), "custom" for GCP VM model
    
    Returns:
        Appropriate LLM service instance
    """
    if model_type == "custom":
        # Import here to avoid circular imports
        from backend.app.services.llm_service import CustomModelService
        try:
            return CustomModelService()
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Custom model not available: {str(e)}. Make sure IAP tunnel is running.",
            )
    else:
        # Use default API provider (Groq/Gemini/etc based on .env)
        return LLMServiceFactory.get_service()


def create_interview_agent(
    model_type: str,
    retriever: InterviewRetriever,
) -> InterviewAgent:
    """Create InterviewAgent with the appropriate LLM service."""
    llm_service = get_llm_service_for_model_type(model_type)
    return InterviewAgent(llm_service=llm_service, retriever=retriever)


# ============== Endpoints ==============

@router.post("", response_model=InterviewResponseModel)
async def create_session(
    request: SessionCreateRequest,
    session_store: SessionStore = Depends(get_session_store),
    retriever: InterviewRetriever = Depends(get_retriever),
):
    """
    Create a new interview session and start the interview.
    
    Args:
        request.model_type: "api" for cloud LLM, "custom" for fine-tuned model
    
    Returns the first question.
    """
    # Verify resume exists (check if we have chunks for it)
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
    
    # Create session with model_type
    session = session_store.create(
        resume_id=request.resume_id,
        mode=request.mode,
        model_type=request.model_type,
        focus_areas=request.focus_areas,
        max_follow_ups=request.max_follow_ups,
    )
    
    # Create agent with appropriate LLM based on model_type
    agent = create_interview_agent(request.model_type, retriever)
    
    # Start interview (generates questions and returns first one)
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
    
    The agent will evaluate the answer and either:
    - Ask a follow-up question (grilling)
    - Move to the next question
    - Complete the interview
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
    
    # Create agent with the same model_type as the session
    agent = create_interview_agent(session.model_type, retriever)
    
    # Process the answer
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
    
    # Create agent with the same model_type as the session
    agent = create_interview_agent(session.model_type, retriever)
    
    response = await agent.skip_question(session=session)
    
    return InterviewResponseModel(
        type=response.type.value,
        content=response.content,
        question_number=response.question_number,
        total_questions=response.total_questions,
        metadata=response.metadata,
    )


@router.post("/{session_id}/end", response_model=InterviewResponseModel)
async def end_session(
    session_id: str,
    session_store: SessionStore = Depends(get_session_store),
    retriever: InterviewRetriever = Depends(get_retriever),
):
    """End the interview session early."""
    session = session_store.get(session_id)
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}",
        )
    
    # Create agent with the same model_type as the session
    agent = create_interview_agent(session.model_type, retriever)
    
    response = await agent.end_interview(
        session=session,
        reason="ended by user",
    )
    
    return InterviewResponseModel(
        type=response.type.value,
        content=response.content,
        metadata=response.metadata,
    )


@router.get("/{session_id}/summary", response_model=SessionSummaryResponse)
async def get_session_summary(
    session_id: str,
    session_store: SessionStore = Depends(get_session_store),
    retriever: InterviewRetriever = Depends(get_retriever),
):
    """Get a summary of the interview session."""
    session = session_store.get(session_id)
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}",
        )
    
    # Create agent with the same model_type as the session
    agent = create_interview_agent(session.model_type, retriever)
    
    summary = await agent.get_interview_summary(session)
    
    return SessionSummaryResponse(**summary)


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