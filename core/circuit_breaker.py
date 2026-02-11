"""
Circuit breaker pattern for external service calls.

Prevents cascading failures by tracking consecutive errors per service
and temporarily blocking calls once a threshold is reached.

States:
    CLOSED   — normal operation, requests pass through
    OPEN     — failures exceeded threshold, requests blocked
    HALF_OPEN — recovery timeout elapsed, next call is a test
"""

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

STATE_CLOSED = "CLOSED"
STATE_OPEN = "OPEN"
STATE_HALF_OPEN = "HALF_OPEN"


class CircuitBreaker:
    """Tracks consecutive failures for a named service and blocks calls when tripped."""

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 300.0,
    ) -> None:
        self._name = name
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout

        self._state = STATE_CLOSED
        self._failure_count = 0
        self._last_failure_time: float = 0.0

    # -- public interface --

    def record_success(self) -> None:
        """Reset failure count and move to CLOSED."""
        if self._state != STATE_CLOSED:
            logger.info(
                "CircuitBreaker[%s]: %s -> CLOSED after successful call",
                self._name,
                self._state,
            )
        self._failure_count = 0
        self._state = STATE_CLOSED

    def record_failure(self) -> None:
        """Increment failure count; trip to OPEN if threshold reached."""
        self._failure_count += 1
        self._last_failure_time = time.monotonic()

        if self._failure_count >= self._failure_threshold:
            if self._state != STATE_OPEN:
                logger.warning(
                    "CircuitBreaker[%s]: OPEN after %d consecutive failures "
                    "(threshold=%d, recovery_timeout=%.0fs)",
                    self._name,
                    self._failure_count,
                    self._failure_threshold,
                    self._recovery_timeout,
                )
            self._state = STATE_OPEN
        else:
            logger.debug(
                "CircuitBreaker[%s]: failure %d/%d",
                self._name,
                self._failure_count,
                self._failure_threshold,
            )

    def can_execute(self) -> bool:
        """Return True if a call should be allowed through.

        - CLOSED: always True
        - OPEN: True only if recovery_timeout has elapsed (transitions to HALF_OPEN)
        - HALF_OPEN: True (one test call allowed)
        """
        if self._state == STATE_CLOSED:
            return True

        if self._state == STATE_OPEN:
            elapsed = time.monotonic() - self._last_failure_time
            if elapsed >= self._recovery_timeout:
                logger.info(
                    "CircuitBreaker[%s]: OPEN -> HALF_OPEN after %.0fs recovery timeout",
                    self._name,
                    elapsed,
                )
                self._state = STATE_HALF_OPEN
                return True
            return False

        # HALF_OPEN — allow the test call
        return True

    # -- properties --

    @property
    def state(self) -> str:
        """Current state: CLOSED, OPEN, or HALF_OPEN."""
        return self._state

    @property
    def failure_count(self) -> int:
        """Current consecutive failure count."""
        return self._failure_count

    def __repr__(self) -> str:
        return (
            f"CircuitBreaker(name={self._name!r}, state={self._state}, "
            f"failures={self._failure_count}/{self._failure_threshold})"
        )


class CircuitBreakerRegistry:
    """Singleton-style registry mapping service names to CircuitBreaker instances."""

    def __init__(self) -> None:
        self._breakers: dict[str, CircuitBreaker] = {}

    def get_or_create(self, name: str, **kwargs: Any) -> CircuitBreaker:
        """Return existing breaker for *name*, or create one with *kwargs*."""
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(name=name, **kwargs)
            logger.debug("CircuitBreakerRegistry: created breaker %r", name)
        return self._breakers[name]

    def get_all_states(self) -> dict[str, dict]:
        """Return a snapshot of every breaker's state for monitoring.

        Returns:
            dict mapping name to {"state": str, "failure_count": int}.
        """
        return {
            name: {
                "state": cb.state,
                "failure_count": cb.failure_count,
            }
            for name, cb in self._breakers.items()
        }

    def __repr__(self) -> str:
        return f"CircuitBreakerRegistry(breakers={list(self._breakers.keys())})"


# Module-level convenience instance
circuit_breaker_registry = CircuitBreakerRegistry()
