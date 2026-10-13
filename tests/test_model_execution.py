"""Synthetic provider-metadata-v1 cases; no live SDK calls or quality claims."""

import asyncio
import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from structlog.testing import capture_logs
from test_interview_flow import interview as shared_interview
from test_interview_flow import model_evaluation, start_interview
from test_stage3_outcomes import assert_restored, submit

from backend.app.config import settings
from backend.app.core.grilling_engine import GrillingEngine
from backend.app.core.logging_config import configure_logging
from backend.app.graph.public import interview_snapshot
from backend.app.graph.state import create_initial_state
from backend.app.models.model_execution import TokenUsage
from backend.app.services.llm_service import (
    AnthropicService,
    CustomModelService,
    GeminiService,
    GroqService,
    HybridModelService,
    OpenAIService,
)
from backend.app.services.model_execution import ModelResult, call_model

interview = shared_interview


def completion(service_type, *, usage=None, reason="stop", text="synthetic"):
    """Actual SDK response classes, mocked transport only."""
    if service_type is OpenAIService:
        from openai.types.chat import ChatCompletion
    else:
        from groq.types.chat import ChatCompletion
    response = ChatCompletion(
        id="synthetic-completion",
        created=1,
        model="reported-model-revision",
        object="chat.completion",
        usage=usage,
        choices=[
            {
                "index": 0,
                "finish_reason": reason,
                "message": {"role": "assistant", "content": text},
            }
        ],
    )
    if service_type is OpenAIService:
        response._request_id = "request-synthetic"
    else:
        from groq.types.chat.chat_completion import XGroq

        response.x_groq = XGroq(id="request-synthetic")
    service = service_type.__new__(service_type)
    service.model = "configured-alias"
    service.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=AsyncMock(return_value=response))
        )
    )
    return service


async def invoke(service):
    return await call_model(
        service,
        operation="question_generation",
        prompt_version="synthetic-v1",
        prompt="private-prompt-sentinel",
    )


@pytest.mark.parametrize("service_type", [OpenAIService, GroqService])
@pytest.mark.parametrize(
    "usage",
    [
        None,
        {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15},
    ],
)
async def test_chat_adapters_preserve_missing_zero_and_reported_identity(
    service_type, usage
):
    call = await invoke(completion(service_type, usage=usage))
    assert call.text == "synthetic" and call.failure is None
    record = call.execution
    assert record.provider == service_type.provider
    assert record.configured_model == "configured-alias"
    assert record.model == "reported-model-revision"
    assert record.provider_request_id == "request-synthetic"
    assert record.finish_reason == "stop"
    assert record.usage.input_tokens == (usage["prompt_tokens"] if usage else None)
    assert record.usage.total_tokens == (usage["total_tokens"] if usage else None)
    assert record.cost_usd is None and record.trace_id is None
    assert record.latency_ms >= 0


async def test_anthropic_preserves_separate_cache_counts_without_guessing_total():
    from anthropic.types import Message

    response = Message(
        id="msg-synthetic",
        model="reported-claude",
        role="assistant",
        type="message",
        content=[{"type": "text", "text": "synthetic"}],
        stop_reason="end_turn",
        usage={
            "input_tokens": 5,
            "output_tokens": 2,
            "cache_read_input_tokens": 9,
            "cache_creation_input_tokens": 0,
        },
    )
    response._request_id = "request-synthetic"
    service = AnthropicService.__new__(AnthropicService)
    service.model = "configured-claude"
    service.client = SimpleNamespace(
        messages=SimpleNamespace(create=AsyncMock(return_value=response))
    )
    call = await invoke(service)
    assert call.failure is None
    assert call.execution.usage == TokenUsage(
        input_tokens=5,
        output_tokens=2,
        cache_read_input_tokens=9,
        cache_creation_input_tokens=0,
    )
    assert call.execution.model == "reported-claude"
    assert call.execution.provider_request_id == "request-synthetic"
    assert call.execution.finish_reason == "end_turn"


@pytest.mark.parametrize(
    "usage",
    [
        None,
        {},
        {
            "prompt_token_count": 10,
            "candidates_token_count": 2,
            "total_token_count": 12,
        },
    ],
)
async def test_gemini_protobuf_presence_distinguishes_missing_usage(usage):
    from google.ai.generativelanguage_v1beta.types import GenerateContentResponse
    from google.generativeai.types import generation_types

    raw = {
        "model_version": "reported-gemini",
        "candidates": [
            {"finish_reason": "STOP", "content": {"parts": [{"text": "synthetic"}]}}
        ],
    }
    if usage is not None:
        raw["usage_metadata"] = usage
    response = generation_types.GenerateContentResponse.from_response(
        GenerateContentResponse(**raw)
    )
    service = GeminiService.__new__(GeminiService)
    service.model = SimpleNamespace(
        model_name="configured-gemini", generate_content=Mock(return_value=response)
    )
    call = await invoke(service)
    assert call.failure is None
    assert call.execution.usage.input_tokens == (
        None if usage is None else usage.get("prompt_token_count", 0)
    )
    assert call.execution.model == "reported-gemini"
    assert call.execution.finish_reason == "STOP"
    assert call.execution.provider_request_id is None
    assert service.model.generate_content.call_args.kwargs["request_options"] == {
        "timeout": settings.LLM_REQUEST_TIMEOUT_SECONDS,
        "retry": None,
    }


@pytest.mark.parametrize("status", [200, 429])
async def test_custom_http_metadata_and_failures_are_payload_free(
    monkeypatch, capsys, status
):
    body = {
        "model": "reported-custom",
        "choices": [{"text": "synthetic", "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2},
    }
    response = httpx.Response(
        status,
        json=body if status == 200 else {"error": "private-response-sentinel"},
        headers={"x-request-id": "request-custom"},
        request=httpx.Request("POST", "http://synthetic.invalid/completions"),
    )
    client = AsyncMock()
    client.__aenter__.return_value = client
    client.post.return_value = response
    monkeypatch.setattr(httpx, "AsyncClient", Mock(return_value=client))
    service = CustomModelService()
    with capture_logs() as logs:
        call = await invoke(service)
        call.record(call.failure or "valid")
    assert "private" not in json.dumps(logs) + capsys.readouterr().out
    if status == 200:
        assert call.execution.provider == "custom"
        assert call.execution.model == "reported-custom"
        assert call.execution.provider_request_id == "request-custom"
        assert call.execution.usage.total_tokens is None
    else:
        assert call.failure.code == "rate_limited"
        assert call.execution.provider == "custom"
        assert call.execution.model is None


@pytest.mark.parametrize(
    "service_type,module,client_name,key",
    [
        (OpenAIService, "openai", "AsyncOpenAI", "OPENAI_API_KEY"),
        (GroqService, "groq", "AsyncGroq", "GROQ_API_KEY"),
        (AnthropicService, "anthropic", "AsyncAnthropic", "ANTHROPIC_API_KEY"),
    ],
)
def test_sdk_clients_have_explicit_deadline_and_no_automatic_retries(
    monkeypatch, service_type, module, client_name, key
):
    import importlib

    constructor = Mock()
    monkeypatch.setattr(importlib.import_module(module), client_name, constructor)
    monkeypatch.setattr(settings, key, "synthetic-key")
    service_type()
    assert (
        constructor.call_args.kwargs["timeout"] == settings.LLM_REQUEST_TIMEOUT_SECONDS
    )
    assert constructor.call_args.kwargs["max_retries"] == 0


async def test_deadline_is_one_safe_attempt_and_cancellation_is_not_an_outcome(
    monkeypatch,
):
    monkeypatch.setattr(settings, "LLM_REQUEST_TIMEOUT_SECONDS", 0.01)

    async def slow(**kwargs):
        await asyncio.sleep(1)

    service = SimpleNamespace(
        generate=AsyncMock(side_effect=slow),
        provider="groq",
        model_name="configured-alias",
    )
    call = await invoke(service)
    assert call.failure.code == "timeout"
    assert call.execution.provider == "groq"
    assert call.execution.usage.input_tokens is None
    service.generate.assert_awaited_once()
    service.generate.side_effect = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await invoke(service)


async def test_overlapping_calls_do_not_share_mutable_last_response():
    async def generated(prompt, **kwargs):
        await asyncio.sleep(0 if prompt == "second" else 0.01)
        return ModelResult(
            text=prompt, provider="openai", model=prompt, provider_request_id=prompt
        )

    service = SimpleNamespace(generate=generated)
    calls = await asyncio.gather(
        *(
            call_model(
                service,
                operation="evaluation",
                prompt_version="synthetic-v1",
                prompt=prompt,
            )
            for prompt in ["first", "second"]
        )
    )
    assert [c.execution.model for c in calls] == ["first", "second"]
    assert [c.text for c in calls] == ["first", "second"]
    assert calls[0].execution.call_id != calls[1].execution.call_id


@pytest.mark.parametrize(
    "reason",
    ["length", "max_tokens", "MAX_TOKENS", "content_filter", "SAFETY", "tool_calls"],
)
async def test_complete_looking_evaluation_is_rejected_when_provider_did_not_finish(
    reason,
):
    service = AsyncMock()
    service.generate.return_value = ModelResult(
        text=model_evaluation(), provider="groq", finish_reason=reason
    )
    records = []
    result = await GrillingEngine(service).evaluate_answer(
        "Why?", "Synthetic answer", execution_records=records
    )
    assert result.outcome == "invalid_output" and not hasattr(result, "score")
    assert records[0]["finish_reason"] == reason
    assert records[0]["outcome"] == "invalid_output"


async def test_safe_log_omits_inputs_raw_output_exception_and_sdk_identifiers(capsys):
    service = AsyncMock()
    service.generate.side_effect = [
        ModelResult(
            text="private-output-sentinel",
            model="private-model-sentinel",
            provider_request_id="private-request-sentinel",
        ),
        RuntimeError("private-exception-sentinel"),
    ]
    with capture_logs() as logs:
        first = await invoke(service)
        first.record("invalid_output")
        second = await invoke(service)
        second.record(second.failure)
    serialized = json.dumps(logs) + capsys.readouterr().out
    assert "private-" not in serialized
    assert len(logs) == 2
    assert logs[0]["call_id"] != logs[1]["call_id"]
    assert logs[1]["error_code"] == "unknown"


def test_debug_mode_does_not_enable_sdk_request_body_logging():
    configure_logging(debug=True)
    try:
        for name in ("openai", "anthropic", "groq", "httpx", "httpcore", "google"):
            assert not logging.getLogger(name).isEnabledFor(logging.DEBUG)
    finally:
        configure_logging(debug=False)


@pytest.mark.parametrize("transport", ["rest", "ws"])
@pytest.mark.parametrize(
    "output,expected",
    [
        (model_evaluation(), "valid"),
        ("{}", "invalid_output"),
        (TimeoutError("private-error-sentinel"), "provider_failure"),
        (
            json.dumps(
                {
                    "outcome": "insufficient_evidence",
                    "reasoning": "No evidence",
                    "missing_elements": ["Example"],
                    "suggested_follow_up": None,
                }
            ),
            "insufficient_evidence",
        ),
    ],
)
def test_metadata_survives_graph_rest_ws_and_read_without_new_call(
    interview, transport, output, expected
):
    started = start_interview(interview, max_follow_ups=0)
    initial = started["model_executions"][0]
    assert initial["operation"] == "question_generation"
    assert initial["prompt_version"] == "question-generation-v2"
    assert (
        initial["provider"] is None
    )  # Injected string-only adapter did not report it.
    interview.llm.generate.side_effect = [
        (
            output
            if isinstance(output, Exception)
            else ModelResult(
                text=output,
                provider="openai",
                model="revision-synthetic",
                usage=TokenUsage(input_tokens=10, output_tokens=0),
                finish_reason="stop",
            )
        )
    ]
    result = submit(interview, started["session_id"], transport)
    assert len(result["model_executions"]) == 1
    execution = result["model_executions"][0]
    assert execution["outcome"] == expected
    assert execution["operation"] == "evaluation"
    assert execution["call_id"] != initial["call_id"]
    assert execution["prompt_version"] == "evaluation-v2"
    assert execution["rubric_version"] == "question-aware-seven-dimension-v1"
    assert execution["output_schema_version"] == "model-output-v1"
    assert execution["cost_usd"] is None
    if expected != "provider_failure":
        assert execution["usage"]["output_tokens"] == 0
        assert execution["usage"]["total_tokens"] is None
    restored = assert_restored(interview, started["session_id"], result)
    assert restored["model_executions"] == result["model_executions"]


@pytest.mark.parametrize("follow_up", [None, "Which measurement?"])
def test_followup_metadata_only_records_calls_that_happened(interview, follow_up):
    started = start_interview(interview)
    interview.llm.generate.side_effect = [
        model_evaluation(sufficient=False, follow_up=follow_up),
        '{"question":"Which measurement?"}',
    ]
    result = submit(interview, started["session_id"], "rest")
    assert result["type"] == "follow_up"
    assert [r["operation"] for r in result["model_executions"]] == (
        ["evaluation", "follow_up_generation"] if follow_up is None else ["evaluation"]
    )
    calls = interview.llm.generate.await_count
    skipped = interview.client.post(
        f"/api/v1/sessions/{started['session_id']}/skip"
    ).json()
    assert skipped["model_executions"] == []
    assert interview.llm.generate.await_count == calls


async def test_hybrid_preparation_retains_actual_provider_records():
    service = HybridModelService.__new__(HybridModelService)
    service.preprocessor = AsyncMock()
    service._prepared_contexts = {}
    service.preprocessor.generate.side_effect = [
        ModelResult(text=text, provider="groq", model="reported-groq")
        for text in [
            "Synthetic summary",
            '{"questions":["Why a queue?"]}',
            "Synthetic context",
        ]
    ]
    prepared = await service.prepare_interview_context(
        "synthetic", "Synthetic resume", "tech", 1
    )
    records = prepared["model_executions"]
    assert [r["operation"] for r in records] == [
        "resume_summary",
        "question_generation",
        "question_context",
    ]
    assert all(r["provider"] == "groq" and r["outcome"] == "valid" for r in records)
    engine = GrillingEngine(service, model_type="custom")
    evaluation_records = []
    result = await engine.evaluate_answer(
        "Why?", "Because", execution_records=evaluation_records
    )
    assert result.outcome == "unsupported_mode"
    assert evaluation_records == []
    assert service.preprocessor.generate.await_count == 3


def test_old_checkpoint_has_unknown_execution_history_not_invented_records():
    state = create_initial_state("synthetic", "synthetic")
    del state["model_executions"]
    assert interview_snapshot(state, "synthetic").model_executions == []


@pytest.mark.parametrize(
    "name,code",
    [
        ("DeadlineExceeded", "timeout"),
        ("ResourceExhausted", "rate_limited"),
        ("PermissionDenied", "authentication"),
        ("ServiceUnavailable", "unavailable"),
    ],
)
async def test_gemini_exception_categories_keep_safe_metadata(name, code):
    from google.api_core import exceptions

    service = SimpleNamespace(
        provider="gemini",
        model_name="configured-gemini",
        generate=AsyncMock(
            side_effect=getattr(exceptions, name)("private-error-sentinel")
        ),
    )
    call = await invoke(service)
    call.record(call.failure)
    assert call.failure.code == code
    assert call.execution.provider == "gemini"
    assert "private-error-sentinel" not in call.execution.model_dump_json()


@pytest.mark.parametrize("long_prompt", [False, True])
async def test_hybrid_failure_identifies_the_adapter_that_was_called(long_prompt):
    hybrid = HybridModelService.__new__(HybridModelService)
    hybrid.preprocessor = SimpleNamespace(
        provider="groq",
        model_name="synthetic-groq",
        generate=AsyncMock(side_effect=TimeoutError()),
    )
    hybrid.interviewer = SimpleNamespace(
        provider="custom",
        model_name="synthetic-custom",
        generate=AsyncMock(side_effect=TimeoutError()),
    )
    call = await call_model(
        hybrid,
        operation="question_generation",
        prompt_version="synthetic-v1",
        prompt="x" * (2001 if long_prompt else 10),
    )
    assert call.failure.code == "timeout"
    assert call.execution.provider == ("groq" if long_prompt else "custom")
    assert call.execution.configured_model == (
        "synthetic-groq" if long_prompt else "synthetic-custom"
    )


def test_failed_start_and_followup_keep_execution_evidence(interview):
    interview.llm.generate.side_effect = [TimeoutError("private-start-sentinel")]
    started = start_interview(interview)
    assert started["model_executions"][0]["outcome"] == "provider_failure"
    assert_restored(interview, started["session_id"], started)
    interview.llm.generate.side_effect = [
        '{"questions":["Why a queue?"]}',
        model_evaluation(sufficient=False),
        TimeoutError("private-followup-sentinel"),
    ]
    with interview.client.websocket_connect(
        f"/ws/interview/{started['session_id']}"
    ) as ws:
        ws.receive_json()
        ws.send_json({"type": "start"})
        resumed = ws.receive_json()["data"]
    assert resumed["model_executions"][0]["outcome"] == "valid"
    result = submit(interview, started["session_id"], "rest")
    assert [r["outcome"] for r in result["model_executions"]] == [
        "valid",
        "provider_failure",
    ]
    assert result["evaluation"]["outcome"] == "valid"
    assert result["generation_failure"]["outcome"]["outcome"] == "provider_failure"
    assert_restored(interview, started["session_id"], result)


async def test_model_text_cannot_forge_execution_metadata():
    forged = json.loads(model_evaluation())
    forged["model_executions"] = [
        {"provider": "anthropic", "usage": {"total_tokens": 0}}
    ]
    service = AsyncMock()
    service.generate.return_value = ModelResult(
        text=json.dumps(forged), provider="groq"
    )
    records = []
    result = await GrillingEngine(service).evaluate_answer(
        "Why?", "Synthetic answer", execution_records=records
    )
    assert result.outcome == "invalid_output"
    assert records[0]["provider"] == "groq"
    assert records[0]["usage"]["total_tokens"] is None


async def test_late_sdk_debug_setting_is_suppressed_before_a_request(caplog):
    noisy = logging.getLogger("openai.synthetic_transport")
    noisy.setLevel(logging.DEBUG)

    async def generate(**kwargs):
        noisy.debug("private-request-body-sentinel")
        return ModelResult("synthetic")

    with caplog.at_level(logging.DEBUG):
        await invoke(SimpleNamespace(generate=generate))
    assert "private-request-body-sentinel" not in caplog.text


async def test_voice_provider_errors_do_not_log_audio_or_payloads(monkeypatch, capsys):
    from backend.app.api.routes import websocket
    from backend.app.services.stt_service import DeepgramSTTService
    from backend.app.services.tts_service import ElevenLabsTTSService

    response = httpx.Response(503, text="private-audio-response-sentinel")
    client = AsyncMock()
    client.__aenter__.return_value = client
    client.post.return_value = response
    monkeypatch.setattr(httpx, "AsyncClient", Mock(return_value=client))
    stt = DeepgramSTTService.__new__(DeepgramSTTService)
    stt.api_key = "synthetic-key"
    stt.model = "synthetic-model"
    stt.language = "en"
    tts = ElevenLabsTTSService.__new__(ElevenLabsTTSService)
    tts.VoiceSettings = Mock(return_value={})
    tts.voice_id = "synthetic-voice"
    tts.model = "synthetic-model"
    tts.client = SimpleNamespace(
        text_to_speech=SimpleNamespace(
            convert=Mock(side_effect=RuntimeError("private-tts-error-sentinel"))
        ),
        voices=SimpleNamespace(
            get_all=Mock(side_effect=RuntimeError("private-voices-error-sentinel"))
        ),
    )
    with capture_logs() as logs:
        assert (await stt.transcribe_audio(b"private-audio-sentinel")).text == ""
        assert (await tts.synthesize("private-text-sentinel")).audio_data == b""
        assert [
            chunk async for chunk in tts.synthesize_stream("private-text-sentinel")
        ] == []
        assert await tts.get_available_voices() == []
        monkeypatch.setattr(websocket.settings, "VOICE_ENABLED", True)
        monkeypatch.setattr(
            websocket,
            "get_tts_service",
            Mock(side_effect=RuntimeError("private-init-error-sentinel")),
        )
        assert await websocket.generate_voice_response("private-text-sentinel") is None
    assert "private-" not in json.dumps(logs) + capsys.readouterr().out
    websocket.get_tts_service.assert_called_once()

    assert tts.client.text_to_speech.convert.call_count == 2
    client.post.assert_awaited_once()
