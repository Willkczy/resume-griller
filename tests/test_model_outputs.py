"""Synthetic contract examples: model-output-cases-v2; no external calls."""

import json
from unittest.mock import AsyncMock

import pytest
from pydantic import TypeAdapter, ValidationError

from backend.app.core.grilling_engine import GrillingEngine
from backend.app.models.model_outputs import (
    EvaluationOutcome,
    InsufficientEvidence,
    InvalidOutput,
    ProviderFailure,
    ScoredEvaluation,
    parse_evaluation,
    parse_follow_up,
    parse_questions,
)


@pytest.fixture
def valid():
    return {
        "outcome": "valid",
        "scores": dict.fromkeys(
            [
                "relevancy",
                "clarity",
                "informativeness",
                "specificity",
                "quantification",
                "depth",
                "completeness",
            ],
            0.2,
        ),
        "detected_gaps": ["too_generic"],
        "gap_details": {"too_generic": "No implementation detail supplied."},
        "priority_gap": "too_generic",
        "missing_elements": ["Implementation detail"],
        "strengths": ["Addresses the topic"],
        "is_sufficient": False,
        "suggested_follow_up": "Which implementation did you choose?",
        "reasoning": "Relevant but lacks requested implementation detail.",
    }


def test_valid_low_score_is_not_failure(valid):
    result = parse_evaluation(json.dumps(valid))
    assert isinstance(result, ScoredEvaluation)
    assert result.scores.relevancy == 0.2
    assert result.is_sufficient is False


@pytest.mark.parametrize("score", [0, 1, 0.0, 1.0])
def test_score_boundaries(valid, score):
    valid["scores"]["depth"] = score
    assert isinstance(parse_evaluation(json.dumps(valid)), ScoredEvaluation)


@pytest.mark.parametrize(
    "score", [-0.01, 1.01, "0.5", True, None, float("nan"), float("inf"), -float("inf")]
)
def test_invalid_scores_are_never_clamped_or_coerced(valid, score):
    valid["scores"]["depth"] = score
    assert isinstance(parse_evaluation(json.dumps(valid)), InvalidOutput)


@pytest.mark.parametrize(
    "field",
    [
        "scores",
        "detected_gaps",
        "gap_details",
        "priority_gap",
        "missing_elements",
        "strengths",
        "is_sufficient",
        "suggested_follow_up",
        "reasoning",
        "outcome",
    ],
)
def test_required_fields_have_no_fabricated_defaults(valid, field):
    del valid[field]
    assert isinstance(parse_evaluation(json.dumps(valid)), InvalidOutput)


@pytest.mark.parametrize(
    "field,value",
    [
        ("outcome", "provider_failure"),
        ("outcome", "excellent"),
        ("detected_gaps", ["invented_gap"]),
        ("detected_gaps", ["too_generic", "too_generic"]),
        ("gap_details", {}),
        ("priority_gap", "no_metrics"),
        ("is_sufficient", "false"),
        ("is_sufficient", 0),
        ("strengths", "good"),
        ("reasoning", "  "),
        ("extra", "ignored?"),
    ],
)
def test_wrong_types_enums_and_incoherent_gaps(valid, field, value):
    valid[field] = value
    assert isinstance(parse_evaluation(json.dumps(valid)), InvalidOutput)


@pytest.mark.parametrize(
    "response",
    [
        "{}",
        "[]",
        "null",
        '{"outcome":',
        "```json\n{}\n```",
        "Here is the result: {}",
        '{"outcome":"valid","outcome":"valid"}',
    ],
)
def test_malformed_or_wrong_shape_never_becomes_score(response):
    result = parse_evaluation(response)
    assert isinstance(result, InvalidOutput)
    assert not hasattr(result, "score")


def test_insufficient_evidence_has_no_score():
    payload = {
        "outcome": "insufficient_evidence",
        "reasoning": "The answer does not identify the referenced project.",
        "missing_elements": ["Project identity"],
        "suggested_follow_up": "Which project are you referring to?",
    }
    result = parse_evaluation(json.dumps(payload))
    assert isinstance(result, InsufficientEvidence)
    payload["score"] = 0
    assert isinstance(parse_evaluation(json.dumps(payload)), InvalidOutput)


def test_failure_is_application_owned_and_payload_free():
    failure = ProviderFailure(code="timeout")
    adapter = TypeAdapter(EvaluationOutcome)
    assert adapter.validate_json(failure.model_dump_json()) == failure
    assert isinstance(parse_evaluation(failure.model_dump_json()), InvalidOutput)
    with pytest.raises(ValidationError):
        ProviderFailure(code="timeout", score=0)
    result = parse_evaluation('{"private_payload": "synthetic sentinel"}')
    assert "sentinel" not in result.model_dump_json()


@pytest.mark.parametrize(
    "questions", [[], [""], ["  "], [5], ["One?", "one?"], ["One?", "Two?"]]
)
def test_invalid_question_batches(questions):
    assert isinstance(
        parse_questions(json.dumps({"questions": questions}), expected_count=1),
        InvalidOutput,
    )


def test_question_and_followup_success():
    assert parse_questions(
        '{"questions":[" Why a queue? "]}', expected_count=1
    ).questions == ["Why a queue?"]
    assert (
        parse_follow_up('{"question":" Which capacity? "}').question
        == "Which capacity?"
    )


@pytest.mark.parametrize(
    "response",
    [
        '{"question":" "}',
        '{"question":7}',
        "{}",
        "Which capacity?",
        '{"question":"Why?","score":1}',
    ],
)
def test_invalid_followups(response):
    assert isinstance(parse_follow_up(response), InvalidOutput)


# These production defects were strict expected failures in stage 1.
def test_engine_rejects_empty_evaluation():
    result = GrillingEngine(AsyncMock())._parse_llm_evaluation("{}", 0)
    assert isinstance(result, InvalidOutput)


def test_engine_rejects_out_of_range_evaluation(valid):
    valid["scores"] = dict.fromkeys(valid["scores"], 100)
    result = GrillingEngine(AsyncMock())._parse_llm_evaluation(json.dumps(valid), 0)
    assert isinstance(result, InvalidOutput)


async def test_engine_preserves_provider_failure(capsys):
    llm = AsyncMock()
    llm.generate.side_effect = TimeoutError("synthetic timeout")
    result = await GrillingEngine(llm).evaluate_answer(
        "Why a queue?",
        "I built a synthetic bounded queue and measured latency under fixed load.",
    )
    assert isinstance(result, ProviderFailure)
    assert "synthetic timeout" not in capsys.readouterr().out
