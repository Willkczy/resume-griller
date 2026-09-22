"""Shared answer-gap vocabulary for evaluation contracts and the engine."""

from enum import Enum


class GapType(str, Enum):
    """
    Gap types based on the paper's 14 interviewer mistake types.
    Converted from "mistakes to avoid" to "gaps in answer".
    """

    # From "Follow-up questions" category
    NO_SPECIFIC_EXAMPLE = "no_specific_example"
    NO_ALTERNATIVES_CONSIDERED = "no_alternatives"
    UNCLEAR_STATEMENT = "unclear_statement"
    CONTRADICTORY_STATEMENT = "contradictory"
    MISSING_TACIT_KNOWLEDGE = "missing_tacit"

    # From "Question framing" category
    TOO_GENERIC = "too_generic"
    TOO_LONG_UNFOCUSED = "too_long_unfocused"
    USES_JARGON_WITHOUT_EXPLANATION = "unexplained_jargon"
    MISSING_TECHNICAL_DEPTH = "no_tech_depth"
    INAPPROPRIATE_SCOPE = "wrong_scope"
    PROVIDES_SOLUTION_NOT_EXPERIENCE = "solution_not_exp"
    MIXES_MULTIPLE_TOPICS = "mixed_topics"
    VAGUE_MULTIPLE_INTERPRETATIONS = "vague_ambiguous"
    NO_CONCRETE_MEANING = "no_concrete_meaning"

    # Additional gaps for interview context
    NO_METRICS = "no_metrics"
    UNCLEAR_PERSONAL_ROLE = "unclear_role"
    NO_RESULT_OUTCOME = "no_outcome"
    RESUME_INCONSISTENT = "resume_inconsistent"
