"""Public interview outcomes shared by HTTP and WebSocket adapters."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from backend.app.models.gap_types import GapType
from backend.app.models.model_execution import ModelExecution
from backend.app.models.model_outputs import (
    DimensionScores,
    InsufficientEvidence,
    InvalidOutput,
    ProviderFailure,
    Score,
    Text,
    UnsupportedMode,
)

PUBLIC_INTERVIEW_SCHEMA_VERSION = "interview-outcome-v2"


class EvaluationIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    question_id: str | None = None
    parent_question_id: str | None = None
    answer_id: str | None = None


GenerationOutcome = Annotated[
    InvalidOutput | ProviderFailure | UnsupportedMode, Field(discriminator="outcome")
]


class PublicDetailedScores(DimensionScores):
    overall: Score


class PublicGapAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    detected_gaps: list[GapType]
    gap_details: dict[GapType, Text]
    severity: Score
    priority_gap: GapType | None


class ValidEvaluation(EvaluationIdentity):
    outcome: Literal["valid"]
    score: Score
    detailed_scores: PublicDetailedScores
    gap_analysis: PublicGapAnalysis
    is_sufficient: bool
    missing_elements: list[Text]
    strengths: list[Text]
    suggested_follow_up: Text | None
    reasoning: Text
    follow_up_policy_version: str | None = None
    follow_up_failure: GenerationOutcome | None = None


class PublicInsufficientEvidence(InsufficientEvidence, EvaluationIdentity):
    pass


class PublicInvalidOutput(InvalidOutput, EvaluationIdentity):
    pass


class PublicProviderFailure(ProviderFailure, EvaluationIdentity):
    pass


class PublicUnsupportedMode(UnsupportedMode, EvaluationIdentity):
    pass


PublicEvaluation = Annotated[
    ValidEvaluation
    | PublicInsufficientEvidence
    | PublicInvalidOutput
    | PublicProviderFailure
    | PublicUnsupportedMode,
    Field(discriminator="outcome"),
]


class GenerationFailure(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation: Literal["question_generation", "follow_up_generation"]
    outcome: GenerationOutcome


class ConversationMessage(BaseModel):
    role: str
    content: str
    timestamp: str
    is_follow_up: bool = False
    question_id: str | None = None
    parent_question_id: str | None = None
    answer_id: str | None = None


class InterviewSnapshot(BaseModel):
    """A projection of the latest checkpoint, never a second state store."""

    session_id: str
    resume_id: str
    mode: str
    model_type: str
    status: Literal["pending", "in_progress", "completed", "cancelled"]
    current_question: str | None
    current_question_id: str | None
    parent_question_id: str | None
    is_follow_up: bool
    current_question_index: int
    total_questions: int
    follow_up_count: int
    max_follow_ups: int
    conversation: list[ConversationMessage]
    created_at: str
    updated_at: str
    model_executions: list[ModelExecution] = Field(default_factory=list)
    evaluation: PublicEvaluation | None = None
    generation_failure: GenerationFailure | None = None
    draft_answer: str | None = None
    answer_status: Literal["none", "unscored", "accepted"] = "none"
    allowed_actions: list[Literal["start", "answer", "skip", "end"]] = Field(
        default_factory=list
    )


class WebSocketInterviewData(InterviewSnapshot):
    """Validated checkpoint data plus the graph's existing command envelope."""

    question_number: int | None = None
    question_id: str | None = None
    priority_gap: GapType | None = None
    follow_up_policy_version: str | None = None
    summary: dict | None = None
    error_code: str | None = None
    status_code: int | None = None
    error: str | None = None
