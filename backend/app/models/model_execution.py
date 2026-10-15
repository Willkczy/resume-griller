"""Application-owned model execution metadata, separate from generated JSON."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

Provider = Literal["anthropic", "openai", "gemini", "groq", "custom"]
Operation = Literal[
    "question_generation",
    "evaluation",
    "follow_up_generation",
    "resume_parsing",
    "resume_summary",
    "question_context",
]
ExecutionOutcome = Literal[
    "valid", "insufficient_evidence", "invalid_output", "provider_failure"
]
TokenCount = Annotated[int, Field(strict=True, ge=0)]


class TokenUsage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_tokens: TokenCount | None = None
    output_tokens: TokenCount | None = None
    total_tokens: TokenCount | None = None
    cache_read_input_tokens: TokenCount | None = None
    cache_creation_input_tokens: TokenCount | None = None


class ModelExecution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["model-execution-v1"] = "model-execution-v1"
    call_id: str
    operation: Operation
    provider: Provider | None = None
    configured_model: str | None = None
    model: str | None = None  # Provider-reported identity; not a guessed revision.
    provider_request_id: str | None = None
    trace_id: str | None = None  # No external trace integration is configured.
    usage: TokenUsage = Field(default_factory=TokenUsage)
    cost_usd: float | None = None  # No provider-reported cost or pricing table yet.
    finish_reason: str | None = None
    latency_ms: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    prompt_version: str
    rubric_version: str | None = None
    output_schema_version: str | None = None
    outcome: ExecutionOutcome
    error_code: str | None = None
