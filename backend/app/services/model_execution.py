"""Request-local results and safe diagnostics; metadata never includes model text."""

import asyncio
import re
from dataclasses import dataclass, field
from time import perf_counter
from uuid import uuid4

from backend.app.config import settings
from backend.app.core.logging_config import get_logger, suppress_provider_payload_logs
from backend.app.models.model_execution import (
    ExecutionOutcome,
    ModelExecution,
    Operation,
    Provider,
    TokenUsage,
)
from backend.app.models.model_outputs import InvalidOutput, ProviderFailure
from backend.app.services.model_failures import provider_failure

logger = get_logger(__name__)


def identifier(value) -> str | None:
    """Keep bounded identifiers in public metadata, never arbitrary SDK objects."""
    if isinstance(value, str) and re.fullmatch(r"[\w./:@+\-]{1,200}", value, re.ASCII):
        return value
    return None


def token_count(value) -> int | None:
    return value if type(value) is int and value >= 0 else None


def field_value(value, name):
    return value.get(name) if isinstance(value, dict) else getattr(value, name, None)


def chat_usage(value) -> TokenUsage:
    """OpenAI/Groq/vLLM usage keys; absent counts remain unknown, including totals."""
    return TokenUsage(
        input_tokens=token_count(field_value(value, "prompt_tokens")),
        output_tokens=token_count(field_value(value, "completion_tokens")),
        total_tokens=token_count(field_value(value, "total_tokens")),
        cache_read_input_tokens=token_count(
            field_value(field_value(value, "prompt_tokens_details"), "cached_tokens")
        ),
    )


@dataclass(frozen=True)
class ModelResult:
    text: str
    provider: Provider | None = None
    configured_model: str | None = None
    model: str | None = None
    provider_request_id: str | None = None
    usage: TokenUsage = field(default_factory=TokenUsage)
    finish_reason: str | None = None


@dataclass
class ModelCall:
    text: str
    execution: ModelExecution
    failure: ProviderFailure | InvalidOutput | None = None

    def record(self, outcome, records: list[dict] | None = None) -> None:
        """Finalize semantic outcome after parsing; log only application-owned fields."""
        name: ExecutionOutcome = (
            outcome if isinstance(outcome, str) else outcome.outcome
        )
        self.execution.outcome = name
        self.execution.error_code = (
            None if isinstance(outcome, str) else getattr(outcome, "code", None)
        )
        if records is not None:
            records.append(self.execution.model_dump(mode="json"))
        # Do not log free-form provider IDs/model labels, exceptions, prompts or text.
        logger.info(
            "model_execution",
            call_id=self.execution.call_id,
            operation=self.execution.operation,
            provider=self.execution.provider,
            outcome=self.execution.outcome,
            error_code=self.execution.error_code,
            latency_ms=self.execution.latency_ms,
            usage=self.execution.usage.model_dump(),
            prompt_version=self.execution.prompt_version,
            rubric_version=self.execution.rubric_version,
            output_schema_version=self.execution.output_schema_version,
        )


async def call_model(
    service,
    *,
    operation: Operation,
    prompt_version: str,
    rubric_version: str | None = None,
    output_schema_version: str | None = None,
    **request,
) -> ModelCall:
    """One bounded attempt. String-only injected adapters honestly lack SDK metadata."""
    provider = getattr(service, "provider", None)
    model_name = getattr(service, "model_name", None)
    identity = getattr(type(service), "execution_identity", None)
    if identity is not None:
        provider, model_name = identity(service, request.get("prompt", ""))
    if provider not in ("anthropic", "openai", "gemini", "groq", "custom"):
        provider = None
    execution = ModelExecution(
        call_id=str(uuid4()),
        operation=operation,
        provider=provider,
        configured_model=identifier(model_name),
        latency_ms=0,
        prompt_version=prompt_version,
        rubric_version=rubric_version,
        output_schema_version=output_schema_version,
        outcome="provider_failure",
    )
    start = perf_counter()
    try:
        suppress_provider_payload_logs()
        async with asyncio.timeout(settings.LLM_REQUEST_TIMEOUT_SECONDS):
            result = await service.generate(**request)
        if isinstance(result, str):
            result = ModelResult(text=result)
        if not isinstance(result, ModelResult):
            raise TypeError("Unsupported provider result")
    except Exception as exc:
        execution.latency_ms = round((perf_counter() - start) * 1000, 3)
        execution.provider_request_id = identifier(getattr(exc, "request_id", None))
        return ModelCall("", execution, provider_failure(exc))

    execution.latency_ms = round((perf_counter() - start) * 1000, 3)
    execution.provider = result.provider or execution.provider
    execution.configured_model = (
        identifier(result.configured_model) or execution.configured_model
    )
    execution.model = identifier(result.model)
    execution.provider_request_id = identifier(result.provider_request_id)
    execution.usage = result.usage
    execution.finish_reason = identifier(result.finish_reason)
    # A complete-looking JSON prefix is still not trustworthy if generation was cut off
    # or blocked. Reject unrecognized explicit stops conservatively. Missing reasons
    # stay unknown and still require strict output parsing.
    failure = None
    if result.finish_reason and result.finish_reason not in {
        "stop",
        "end_turn",
        "stop_sequence",
        "STOP",
    }:
        failure = InvalidOutput(code="schema_violation")
    return ModelCall(result.text, execution, failure)
