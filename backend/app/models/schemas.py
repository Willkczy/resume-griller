"""
Pydantic schemas for Resume Griller API.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

# ============== Enums ==============


class InterviewMode(str, Enum):
    """Interview mode types."""

    HR = "hr"
    TECH = "tech"
    MIXED = "mixed"


class QuestionType(str, Enum):
    """Question types."""

    TECHNICAL = "technical"
    BEHAVIORAL = "behavioral"
    MIXED = "mixed"


class SessionStatus(str, Enum):
    """Interview session status."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class MessageRole(str, Enum):
    """Message roles in conversation."""

    SYSTEM = "system"
    INTERVIEWER = "interviewer"
    CANDIDATE = "candidate"


# ============== Resume Schemas ==============


class ResumeUploadResponse(BaseModel):
    """Response after resume upload."""

    resume_id: str
    filename: str
    chunks_created: int
    sections: list[str]
    message: str = "Resume processed successfully"


class ResumeSummary(BaseModel):
    """Resume summary information."""

    resume_id: str
    name: str | None = None
    total_chunks: int
    sections: list[str]
    skills: list[str] = []
    experience_count: int = 0
    education_count: int = 0


# ============== Interview Session Schemas ==============


class SessionCreate(BaseModel):
    """Request to create a new interview session."""

    resume_id: str
    mode: InterviewMode = InterviewMode.MIXED
    focus_areas: list[str] = Field(default_factory=list)
    num_questions: int = Field(default=5, ge=1, le=20)


class SessionResponse(BaseModel):
    """Interview session response."""

    session_id: str
    resume_id: str
    mode: InterviewMode
    status: SessionStatus
    created_at: datetime
    focus_areas: list[str] = []
    total_questions: int = 0
    questions_asked: int = 0


class SessionDetail(SessionResponse):
    """Detailed session information including conversation history."""

    conversation: list["ConversationMessage"] = []
    current_question: str | None = None


# ============== Conversation Schemas ==============


class ConversationMessage(BaseModel):
    """A single message in the conversation."""

    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CandidateAnswer(BaseModel):
    """Candidate's answer to a question."""

    content: str
    audio_base64: str | None = None  # For voice input


class InterviewerResponse(BaseModel):
    """Interviewer's response (question or follow-up)."""

    content: str
    is_follow_up: bool = False
    question_number: int
    total_questions: int
    feedback: str | None = None  # Optional feedback on previous answer


# ============== Legacy, Unwired WebSocket Schemas ==============
#
# The active WebSocket route currently validates its content/data/error mapping in
# backend/app/api/routes/websocket.py and does not instantiate these models. Keep
# these definitions out of public contract documentation until they are either
# removed or deliberately wired to the runtime route.


class WSMessage(BaseModel):
    """Legacy WebSocket envelope; not the active route contract."""

    type: str  # "answer", "control", "status"
    payload: dict[str, Any]


class WSAnswerPayload(BaseModel):
    """Legacy answer payload; not instantiated by the active route."""

    text: str
    audio_base64: str | None = None


class WSControlPayload(BaseModel):
    """Legacy control payload; pause/resume are not active message types."""

    action: str  # "start", "pause", "resume", "end", "skip"


# ============== Question Generation Schemas ==============


class GenerateQuestionsRequest(BaseModel):
    """Request to generate questions for a resume."""

    resume_id: str
    question_type: QuestionType = QuestionType.MIXED
    focus_area: str | None = None
    num_questions: int = Field(default=5, ge=1, le=10)


class GeneratedQuestion(BaseModel):
    """A generated interview question."""

    question: str
    type: QuestionType
    focus_area: str | None = None
    difficulty: str | None = None  # "easy", "medium", "hard"


class GenerateQuestionsResponse(BaseModel):
    """Response with generated questions."""

    resume_id: str
    questions: list[GeneratedQuestion]


# ============== Grilling Schemas ==============


class AnswerEvaluation(BaseModel):
    """Evaluation of candidate's answer."""

    is_sufficient: bool
    score: float = Field(ge=0, le=1)  # 0-1 score
    missing_elements: list[str] = []
    strengths: list[str] = []
    suggested_follow_up: str | None = None


# ============== Health Check ==============


class HealthCheck(BaseModel):
    """Health check response."""

    status: str = "healthy"
    version: str = "1.0.0"
    llm_mode: str
    llm_provider: str | None = None
    voice_enabled: bool = False
    custom_model_available: bool = False
    dependencies: dict[str, bool] = Field(default_factory=dict)


# Update forward references
SessionDetail.model_rebuild()
