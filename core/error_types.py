"""Typed exception hierarchy for the NIFTY trading system.

Replaces bare ``except Exception`` with specific error types so callers
can handle different failure modes appropriately.
"""


class NiftyBaseError(Exception):
    """Root exception for all NIFTY system errors."""


class DataFetchError(NiftyBaseError):
    """External API call failed (Kite, yfinance, nsepython, etc.)."""


class DataValidationError(NiftyBaseError):
    """Data quality check failed — data is present but invalid."""


class TradingBlockedError(NiftyBaseError):
    """Trade rejected due to capital, risk, or circuit-breaker limits."""


class ConfigError(NiftyBaseError):
    """Configuration file invalid or missing required keys."""


class AuthenticationError(NiftyBaseError):
    """Kite token expired or API key invalid."""
