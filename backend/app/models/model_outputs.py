"""M2 model-boundary contracts; runtime wiring follows in stage 2.

These are internal contracts, not the active HTTP/WS response schemas.
Only the application may classify provider failures and unsupported modes.
"""

import json
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    TypeAdapter,
    ValidationError,
    model_validator,
)

from backend.app.models.gap_types import GapType

MODEL_OUTPUT_SCHEMA_VERSION = "model-output-v1"
Text = Annotated[
    str, StringConstraints(strict=True, strip_whitespace=True, min_length=1)
]
Score = Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class DimensionScores(Contract):
    """Existing seven dimensions, explicitly supplied; no defaults or calibration claim."""

    relevancy: Score
    clarity: Score
    informativeness: Score
    specificity: Score
    quantification: Score
    depth: Score
    completeness: Score


class ScoredEvaluation(Contract):
    outcome: Literal["valid"]
    scores: DimensionScores
    detected_gaps: list[GapType]
    gap_details: dict[GapType, Text]
    priority_gap: GapType | None
    missing_elements: list[Text]
    strengths: list[Text]
    is_sufficient: bool
    suggested_follow_up: Text | None
    reasoning: Text

    @model_validator(mode="after")
    def coherent_gaps(self):
        gaps = set(self.detected_gaps)
        if len(gaps) != len(self.detected_gaps):
            raise ValueError("duplicate gap labels")
        if self.priority_gap is not None and self.priority_gap not in gaps:
            raise ValueError("priority gap must be detected")
        if set(self.gap_details) != gaps:
            raise ValueError("every detected gap needs exactly one explanation")
        return self


class InsufficientEvidence(Contract):
    """Not enough evidence to judge; never a fabricated zero score."""

    outcome: Literal["insufficient_evidence"]
    reasoning: Text
    missing_elements: Annotated[list[Text], Field(min_length=1)]
    suggested_follow_up: Text | None


class InvalidOutput(Contract):
    outcome: Literal["invalid_output"] = "invalid_output"
    code: Literal["invalid_json", "schema_violation"]


class ProviderFailure(Contract):
    outcome: Literal["provider_failure"] = "provider_failure"
    code: Literal["timeout", "rate_limited", "authentication", "unavailable", "unknown"]


class UnsupportedMode(Contract):
    outcome: Literal["unsupported_mode"] = "unsupported_mode"
    code: Literal["evaluation_contract_unavailable"] = "evaluation_contract_unavailable"


ModelEvaluation = Annotated[
    ScoredEvaluation | InsufficientEvidence, Field(discriminator="outcome")
]
EvaluationOutcome = Annotated[
    ScoredEvaluation
    | InsufficientEvidence
    | InvalidOutput
    | ProviderFailure
    | UnsupportedMode,
    Field(discriminator="outcome"),
]
_EVALUATION = TypeAdapter(ModelEvaluation)


class QuestionBatch(Contract):
    questions: Annotated[list[Text], Field(min_length=1, max_length=20)]

    @model_validator(mode="after")
    def distinct_questions(self):
        if len({q.casefold() for q in self.questions}) != len(self.questions):
            raise ValueError("duplicate questions")
        return self


class FollowUpQuestion(Contract):
    question: Text


def _reject_constant(value):
    raise ValueError("non-finite JSON number")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _checked_json(response: str) -> str:
    # No substring extraction, markdown repair, duplicate-key overwrite or NaN.
    json.loads(
        response, parse_constant=_reject_constant, object_pairs_hook=_unique_object
    )
    return response


def parse_evaluation(
    response: str,
) -> ScoredEvaluation | InsufficientEvidence | InvalidOutput:
    try:
        checked = _checked_json(response)
    except (ValueError, TypeError):
        return InvalidOutput(code="invalid_json")
    try:
        return _EVALUATION.validate_json(checked)
    except ValidationError:
        # Never retain exception text: it can contain the model's full payload.
        return InvalidOutput(code="schema_violation")


def parse_questions(
    response: str, *, expected_count: int
) -> QuestionBatch | InvalidOutput:
    if type(expected_count) is not int or not 1 <= expected_count <= 20:
        raise ValueError("expected_count must be an integer from 1 to 20")
    try:
        checked = _checked_json(response)
    except (ValueError, TypeError):
        return InvalidOutput(code="invalid_json")
    try:
        batch = QuestionBatch.model_validate_json(checked)
    except ValidationError:
        return InvalidOutput(code="schema_violation")
    if len(batch.questions) != expected_count:
        return InvalidOutput(code="schema_violation")
    return batch


def parse_follow_up(response: str) -> FollowUpQuestion | InvalidOutput:
    try:
        checked = _checked_json(response)
    except (ValueError, TypeError):
        return InvalidOutput(code="invalid_json")
    try:
        return FollowUpQuestion.model_validate_json(checked)
    except ValidationError:
        return InvalidOutput(code="schema_violation")
