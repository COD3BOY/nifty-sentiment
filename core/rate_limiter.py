"""Token bucket rate limiter and exponential backoff retry decorator.

Provides async-friendly rate limiting for external API calls (yfinance, Kite, Claude)
and a retry decorator with exponential backoff + jitter for transient failures.
"""

import asyncio
import functools
import logging
import random
import time

logger = logging.getLogger(__name__)


class TokenBucketRateLimiter:
    """Async token bucket rate limiter.

    Tokens are added at a fixed rate up to a maximum burst size.
    Callers block (via asyncio.sleep) until a token is available.

    Args:
        rate: Tokens added per second.
        burst: Maximum number of tokens the bucket can hold.
    """

    def __init__(self, rate: float, burst: int) -> None:
        self._rate = rate
        self._burst = burst
        self._tokens = float(burst)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        """Add tokens based on elapsed time since last refill."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._burst, self._tokens + elapsed * self._rate)
        self._last_refill = now

    async def acquire(self) -> None:
        """Block until a token is available, then consume one."""
        while True:
            async with self._lock:
                self._refill()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                # Calculate wait time for the next token
                wait = (1.0 - self._tokens) / self._rate
            await asyncio.sleep(wait)


def async_retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 0.1,
    max_delay: float = 10.0,
    exceptions: tuple = (Exception,),
):
    """Decorator that retries an async function with exponential backoff and jitter.

    On each retry the delay doubles (with random jitter) up to ``max_delay``.
    After ``max_retries`` consecutive failures the last exception is re-raised.

    Args:
        max_retries: Maximum number of retry attempts (not counting the first call).
        base_delay: Initial delay in seconds before the first retry.
        max_delay: Upper bound on the delay between retries.
        exceptions: Tuple of exception types that trigger a retry.
    """

    def decorator(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_retries + 1):
                try:
                    return await fn(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt == max_retries:
                        logger.error(
                            "%s failed after %d attempts: %s",
                            fn.__qualname__,
                            max_retries + 1,
                            exc,
                        )
                        raise
                    # Exponential backoff: base_delay * 2^attempt, capped at max_delay
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    # Add jitter: uniform random between 0 and delay
                    jitter = random.uniform(0, delay)
                    sleep_time = delay + jitter
                    logger.warning(
                        "%s attempt %d/%d failed (%s), retrying in %.2fs",
                        fn.__qualname__,
                        attempt + 1,
                        max_retries + 1,
                        exc,
                        sleep_time,
                    )
                    await asyncio.sleep(sleep_time)
            # Should not reach here, but just in case
            raise last_exc  # type: ignore[misc]

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Pre-configured module-level singletons
# ---------------------------------------------------------------------------

# yfinance: ~5 requests per minute to avoid throttling
yfinance_limiter = TokenBucketRateLimiter(rate=5 / 60, burst=2)

# Kite Connect: 10 requests per second
kite_limiter = TokenBucketRateLimiter(rate=10.0, burst=5)

# Claude (Anthropic API): 1 request per second default
claude_limiter = TokenBucketRateLimiter(rate=1.0, burst=1)
