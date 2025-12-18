"""
Session Store for Interview Sessions.
Currently uses in-memory storage. Can be replaced with Redis/PostgreSQL later.

Supports Hybrid model by storing prepared_context from preprocessing phase.
"""

from typing import Dict, Optional, List, Any
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import uuid


class SessionStatus(str, Enum):
    """Interview session status."""
    PENDING = "pending"           
    IN_PROGRESS = "in_progress"   
    PAUSED = "paused"             
    COMPLETED = "completed"       
    CANCELLED = "cancelled"


class MessageRole(str, Enum):
    """Message roles in conversation."""
    INTERVIEWER = "interviewer"
    CANDIDATE = "candidate"
    SYSTEM = "system"


@dataclass
class Message:
    """A single message in the conversation."""
    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    is_follow_up: bool = False
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "is_follow_up": self.is_follow_up,
            "metadata": self.metadata,
        }


@dataclass
class InterviewSession:
    """Interview session data structure."""
    session_id: str
    resume_id: str
    mode: str = "mixed"  # "hr", "tech", "mixed"
    model_type: str = "api"  # "api" or "custom"
    status: SessionStatus = SessionStatus.PENDING
    
    # Questions
    questions: List[str] = field(default_factory=list)
    current_question_index: int = 0
    
    # Conversation
    conversation: List[Message] = field(default_factory=list)
    
    # Follow-up tracking
    current_follow_up_count: int = 0
    max_follow_ups: int = 3
    
    # Metadata
    focus_areas: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    # Hybrid model support: stores preprocessed context
    # Contains: resume_summary, questions, question_contexts
    prepared_context: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict:
        return {
            "session_id": self.session_id,
            "resume_id": self.resume_id,
            "mode": self.mode,
            "model_type": self.model_type, 
            "status": self.status.value,
            "questions": self.questions,
            "current_question_index": self.current_question_index,
            "conversation": [m.to_dict() for m in self.conversation],
            "current_follow_up_count": self.current_follow_up_count,
            "max_follow_ups": self.max_follow_ups,
            "focus_areas": self.focus_areas,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "total_questions": len(self.questions),
            "questions_asked": self.current_question_index,
            "has_prepared_context": self.prepared_context is not None,
        }
    
    @property
    def current_question(self) -> Optional[str]:
        """Get current question."""
        if 0 <= self.current_question_index < len(self.questions):
            return self.questions[self.current_question_index]
        return None
    
    @property
    def is_complete(self) -> bool:
        """Check if all questions have been asked."""
        return self.current_question_index >= len(self.questions)
    
    @property
    def current_question_context(self) -> Optional[str]:
        """Get the prepared context for the current question (Hybrid mode)."""
        if self.prepared_context and "question_contexts" in self.prepared_context:
            contexts = self.prepared_context["question_contexts"]
            if 0 <= self.current_question_index < len(contexts):
                return contexts[self.current_question_index]
        return None
    
    @property
    def resume_summary(self) -> Optional[str]:
        """Get the prepared resume summary (Hybrid mode)."""
        if self.prepared_context:
            return self.prepared_context.get("resume_summary")
        return None
    
    def add_message(self, role: MessageRole, content: str, is_follow_up: bool = False, metadata: Dict = None):
        """Add a message to conversation."""
        self.conversation.append(Message(
            role=role,
            content=content,
            is_follow_up=is_follow_up,
            metadata=metadata or {},
        ))
        self.updated_at = datetime.utcnow()
    
    def next_question(self) -> Optional[str]:
        """Move to next question and return it."""
        self.current_question_index += 1
        self.current_follow_up_count = 0  # Reset follow-up count
        self.updated_at = datetime.utcnow()
        return self.current_question
    
    def can_follow_up(self) -> bool:
        """Check if we can ask more follow-ups."""
        return self.current_follow_up_count < self.max_follow_ups
    
    def increment_follow_up(self):
        """Increment follow-up counter."""
        self.current_follow_up_count += 1
        self.updated_at = datetime.utcnow()


class SessionStore:
    """
    In-memory session storage.
    
    For production, replace with:
    - Redis for fast access + expiration
    - PostgreSQL for persistence
    """
    
    def __init__(self):
        self._sessions: Dict[str, InterviewSession] = {}
    
    def create(
        self,
        resume_id: str,
        mode: str = "mixed",
        model_type: str = "api", 
        focus_areas: List[str] = None,
        max_follow_ups: int = 3,
    ) -> InterviewSession:
        """Create a new session."""
        session_id = f"sess_{uuid.uuid4().hex[:12]}"
        
        session = InterviewSession(
            session_id=session_id,
            resume_id=resume_id,
            mode=mode,
            model_type=model_type,
            focus_areas=focus_areas or [],
            max_follow_ups=max_follow_ups,
        )
        
        self._sessions[session_id] = session
        return session
    
    def get(self, session_id: str) -> Optional[InterviewSession]:
        """Get a session by ID."""
        return self._sessions.get(session_id)
    
    def update(self, session: InterviewSession) -> InterviewSession:
        """Update a session."""
        session.updated_at = datetime.utcnow()
        self._sessions[session.session_id] = session
        return session
    
    def delete(self, session_id: str) -> bool:
        """Delete a session."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False
    
    def list_all(self, resume_id: str = None) -> List[InterviewSession]:
        """List all sessions, optionally filtered by resume_id."""
        sessions = list(self._sessions.values())
        
        if resume_id:
            sessions = [s for s in sessions if s.resume_id == resume_id]
        
        return sorted(sessions, key=lambda s: s.created_at, reverse=True)
    
    def get_active_sessions(self) -> List[InterviewSession]:
        """Get all active (in-progress) sessions."""
        return [
            s for s in self._sessions.values()
            if s.status == SessionStatus.IN_PROGRESS
        ]
    
    def get_sessions_by_model_type(self, model_type: str) -> List[InterviewSession]:
        """Get sessions by model type."""
        return [
            s for s in self._sessions.values()
            if s.model_type == model_type
        ]


# Global session store instance
session_store = SessionStore()


def get_session_store() -> SessionStore:
    """Get the session store instance."""
    return session_store