"""Safe application-owned classification; never serialize exception messages."""

import httpx

from backend.app.models.model_outputs import ProviderFailure


def provider_failure(exc: Exception) -> ProviderFailure:
    if (
        isinstance(exc, (TimeoutError, httpx.TimeoutException))
        or type(exc).__name__ == "APITimeoutError"
    ):
        return ProviderFailure(code="timeout")
    status = getattr(exc, "status_code", None)
    if status in (401, 403):
        return ProviderFailure(code="authentication")
    if status == 429:
        return ProviderFailure(code="rate_limited")
    if (isinstance(status, int) and status >= 500) or isinstance(
        exc, (ConnectionError, httpx.ConnectError)
    ):
        return ProviderFailure(code="unavailable")
    return ProviderFailure(code="unknown")
