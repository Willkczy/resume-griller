"""Provider errors and experimental helpers never synthesize a judgment."""

from unittest.mock import AsyncMock

import httpx
import pytest

from backend.app.services.llm_service import HybridModelService
from backend.app.services.model_failures import provider_failure


@pytest.mark.parametrize(
    "exc,code",
    [
        (TimeoutError("synthetic sentinel"), "timeout"),
        (httpx.ReadTimeout("synthetic sentinel"), "timeout"),
        (httpx.ConnectError("synthetic sentinel"), "unavailable"),
        (ValueError("synthetic sentinel"), "unknown"),
    ],
)
def test_provider_failure_classification_is_payload_free(exc, code):
    outcome = provider_failure(exc)
    assert outcome.code == code
    assert "sentinel" not in outcome.model_dump_json()
    assert "score" not in outcome.model_dump()


@pytest.mark.parametrize(
    "status,code",
    [
        (401, "authentication"),
        (403, "authentication"),
        (429, "rate_limited"),
        (503, "unavailable"),
    ],
)
def test_sdk_status_error_classification(status, code):
    exc = RuntimeError("synthetic sentinel")
    exc.status_code = status
    assert provider_failure(exc).code == code


def hybrid_without_clients():
    # Bypass constructor only to avoid credentials/SDK clients; methods are real.
    service = HybridModelService.__new__(HybridModelService)
    service.preprocessor = AsyncMock()
    service.interviewer = AsyncMock()
    service._prepared_contexts = {}
    return service


async def test_legacy_compact_helpers_report_unsupported_without_provider_calls():
    service = hybrid_without_clients()
    evaluation = await service.evaluate_answer("synthetic", 0, "Why?", "Because.")
    follow_up = await service.generate_followup(
        "synthetic", "Why?", "Because.", "too_generic"
    )
    assert (
        evaluation
        == follow_up
        == {"outcome": "unsupported_mode", "code": "evaluation_contract_unavailable"}
    )
    service.interviewer.generate.assert_not_awaited()


@pytest.mark.parametrize(
    "output", ['{"questions": []}', "1. Why a queue?", '{"questions":["Why?","Why?"]}']
)
async def test_hybrid_preparation_rejects_invalid_questions_before_context_calls(
    output,
):
    service = hybrid_without_clients()
    service.preprocessor.generate.side_effect = ["Synthetic summary", output]
    with pytest.raises(ValueError, match="Invalid prepared question output"):
        await service.prepare_interview_context(
            "synthetic", "Synthetic resume", "tech", 1
        )
    assert service.preprocessor.generate.await_count == 2
    assert service._prepared_contexts == {}


async def test_hybrid_preparation_accepts_exact_valid_question_batch():
    service = hybrid_without_clients()
    service.preprocessor.generate.side_effect = [
        "Synthetic summary",
        '{"questions":["Why a queue?"]}',
        "Synthetic context",
    ]
    result = await service.prepare_interview_context(
        "synthetic", "Synthetic resume", "tech", 1
    )
    assert result["questions"] == ["Why a queue?"]
    assert result["question_contexts"] == ["Synthetic context"]
