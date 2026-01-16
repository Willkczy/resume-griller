"""
Rate limiting middleware using SlowAPI.
"""

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request, Response


def get_identifier(request: Request) -> str:
    """
    Get identifier for rate limiting.
    Uses remote address by default, can be extended to use API keys or user IDs.
    """
    # For now, use IP address
    # TODO: Use user_id or API key when authentication is implemented
    return get_remote_address(request)


# Create limiter instance
limiter = Limiter(
    key_func=get_identifier,
    default_limits=["100/minute"],  # Global default: 100 requests per minute
    storage_uri="memory://",  # Use in-memory storage (can be Redis in production)
)


def setup_rate_limiting(app):
    """
    Setup rate limiting for the FastAPI app.

    Args:
        app: FastAPI application instance
    """
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
