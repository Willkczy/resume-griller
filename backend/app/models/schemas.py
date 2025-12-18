"""
Pydantic schemas for Resume Griller API.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
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
    sections: List[str]
    message: str = "Resume processed successfully"


class ResumeSummary(BaseModel):
    """Resume summary information."""
    resume_id: str
    name: Optional[str] = None
    total_chunks: int
    sections: List[str]
    skills: List[str] = []
    experience_count: int = 0
    education_count: int = 0


# ============== Interview Session Schemas ==============

class SessionCreate(BaseModel):
    """Request to create a new interview session."""
    resume_id: str
    mode: InterviewMode = InterviewMode.MIXED
    focus_areas: List[str] = Field(default_factory=list)
    num_questions: int = Field(default=5, ge=1, le=20)


class SessionResponse(BaseModel):
    """Interview session response."""
    session_id: str
    resume_id: str
    mode: InterviewMode
    status: SessionStatus
    created_at: datetime
    focus_areas: List[str] = []
    total_questions: int = 0
    questions_asked: int = 0


class SessionDetail(SessionResponse):
    """Detailed session information including conversation history."""
    conversation: List["ConversationMessage"] = []
    current_question: Optional[str] = None


# ============== Conversation Schemas ==============

class ConversationMessage(BaseModel):
    """A single message in the conversation."""
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CandidateAnswer(BaseModel):
    """Candidate's answer to a question."""
    content: str
    audio_base64: Optional[str] = None  # For voice input


class InterviewerResponse(BaseModel):
    """Interviewer's response (question or follow-up)."""
    content: str
    is_follow_up: bool = False
    question_number: int
    total_questions: int
    feedback: Optional[str] = None  # Optional feedback on previous answer


# ============== WebSocket Schemas ==============

class WSMessage(BaseModel):
    """WebSocket message format."""
    type: str  # "answer", "control", "status"
    payload: Dict[str, Any]


class WSAnswerPayload(BaseModel):
    """Payload for candidate answer via WebSocket."""
    text: str
    audio_base64: Optional[str] = None


class WSControlPayload(BaseModel):
    """Payload for control messages."""
    action: str  # "start", "pause", "resume", "end", "skip"


# ============== Question Generation Schemas ==============

class GenerateQuestionsRequest(BaseModel):
    """Request to generate questions for a resume."""
    resume_id: str
    question_type: QuestionType = QuestionType.MIXED
    focus_area: Optional[str] = None
    num_questions: int = Field(default=5, ge=1, le=10)


class GeneratedQuestion(BaseModel):
    """A generated interview question."""
    question: str
    type: QuestionType
    focus_area: Optional[str] = None
    difficulty: Optional[str] = None  # "easy", "medium", "hard"


class GenerateQuestionsResponse(BaseModel):
    """Response with generated questions."""
    resume_id: str
    questions: List[GeneratedQuestion]


# ============== Grilling Schemas ==============

class AnswerEvaluation(BaseModel):
    """Evaluation of candidate's answer."""
    is_sufficient: bool
    score: float = Field(ge=0, le=1)  # 0-1 score
    missing_elements: List[str] = []
    strengths: List[str] = []
    suggested_follow_up: Optional[str] = None


# ============== Health Check ==============

class HealthCheck(BaseModel):
    """Health check response."""
    status: str = "healthy"
    version: str = "1.0.0"
    llm_mode: str
    llm_provider: Optional[str] = None


# Update forward references
SessionDetail.model_rebuild()