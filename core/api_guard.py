"""Combined rate limiter + circuit breaker guards for external API calls.

Each guard checks the circuit breaker state, acquires a rate-limit token,
and returns the circuit breaker so callers can record success/failure.

Usage::

    from core.api_guard import yf_guard_sync

    cb = yf_guard_sync()
    try:
        result = yf.Ticker(...).history(...)
        cb.record_success()
    except Exception:
        cb.record_failure()
        raise
"""

import logging

from core.circuit_breaker import CircuitBreaker, circuit_breaker_registry
from core.error_types import DataFetchError
from core.rate_limiter import claude_limiter, kite_limiter, yfinance_limiter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# yfinance guards
# ---------------------------------------------------------------------------

_YF_FAILURE_THRESHOLD = 5
_YF_RECOVERY_TIMEOUT = 300  # 5 minutes


def yf_guard_sync() -> CircuitBreaker:
    """Acquire yfinance rate limit + check circuit breaker (sync)."""
    cb = circuit_breaker_registry.get_or_create(
        "yfinance", failure_threshold=_YF_FAILURE_THRESHOLD, recovery_timeout=_YF_RECOVERY_TIMEOUT,
    )
    if not cb.can_execute():
        raise DataFetchError("yfinance circuit breaker OPEN — too many recent failures")
    yfinance_limiter.acquire_sync()
    return cb


async def yf_guard_async() -> CircuitBreaker:
    """Acquire yfinance rate limit + check circuit breaker (async)."""
    cb = circuit_breaker_registry.get_or_create(
        "yfinance", failure_threshold=_YF_FAILURE_THRESHOLD, recovery_timeout=_YF_RECOVERY_TIMEOUT,
    )
    if not cb.can_execute():
        raise DataFetchError("yfinance circuit breaker OPEN — too many recent failures")
    await yfinance_limiter.acquire()
    return cb


# ---------------------------------------------------------------------------
# Kite guards
# ---------------------------------------------------------------------------

_KITE_FAILURE_THRESHOLD = 5
_KITE_RECOVERY_TIMEOUT = 300


def kite_guard_sync() -> CircuitBreaker:
    """Acquire Kite rate limit + check circuit breaker (sync)."""
    cb = circuit_breaker_registry.get_or_create(
        "kite", failure_threshold=_KITE_FAILURE_THRESHOLD, recovery_timeout=_KITE_RECOVERY_TIMEOUT,
    )
    if not cb.can_execute():
        raise DataFetchError("Kite circuit breaker OPEN — too many recent failures")
    kite_limiter.acquire_sync()
    return cb


# ---------------------------------------------------------------------------
# Claude guards
# ---------------------------------------------------------------------------

_CLAUDE_FAILURE_THRESHOLD = 3
_CLAUDE_RECOVERY_TIMEOUT = 120  # 2 minutes


def claude_guard_sync() -> CircuitBreaker:
    """Acquire Claude rate limit + check circuit breaker (sync)."""
    cb = circuit_breaker_registry.get_or_create(
        "claude", failure_threshold=_CLAUDE_FAILURE_THRESHOLD, recovery_timeout=_CLAUDE_RECOVERY_TIMEOUT,
    )
    if not cb.can_execute():
        raise DataFetchError("Claude circuit breaker OPEN — too many recent failures")
    claude_limiter.acquire_sync()
    return cb
