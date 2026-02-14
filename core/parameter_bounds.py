"""Hard bounds and safety constants for all tunable strategy parameters.

Every ``param_key`` in ``core/strategy_rules.py`` MUST have an entry in
``PARAMETER_BOUNDS``.  The improvement system refuses to apply a change that
would push a parameter outside its bounds or violate step/drift limits.
"""

from __future__ import annotations

# -------------------------------------------------------------------------
# Parameter bounds — (min, max) for every param_key across all 11 strategies
# -------------------------------------------------------------------------
#
# Keys match ``param_key`` values in strategy_rules.STRATEGY_RULES exactly.
# A param_key may appear in multiple strategies with different defaults;
# the bounds here are the UNION of all reasonable values across strategies.

PARAMETER_BOUNDS: dict[str, tuple[float, float]] = {
    # --- IV thresholds ---
    "iv_low_threshold": (8.0, 25.0),
    "iv_moderate_threshold": (12.0, 35.0),
    "iv_elevated_threshold": (12.0, 30.0),
    "iv_cheap_threshold": (8.0, 25.0),

    # --- IV skew ---
    "iv_skew_threshold": (1.0, 8.0),

    # --- RSI zones ---
    "rsi_neutral_low": (30.0, 50.0),
    "rsi_neutral_high": (50.0, 70.0),
    "rsi_moderate_low": (20.0, 45.0),
    "rsi_moderate_high": (55.0, 80.0),
    "rsi_coiled_low": (40.0, 50.0),
    "rsi_coiled_high": (50.0, 60.0),
    "rsi_overbought_threshold": (60.0, 80.0),
    "rsi_oversold_threshold": (20.0, 40.0),
    "rsi_bullish_low": (40.0, 60.0),
    "rsi_bearish_low": (20.0, 40.0),
    "rsi_bearish_high": (40.0, 60.0),

    # --- PCR zones ---
    "pcr_neutral_low": (0.5, 1.0),
    "pcr_neutral_high": (1.0, 1.8),
    "pcr_bullish_threshold": (0.8, 1.8),
    "pcr_bearish_threshold": (0.3, 0.9),
    "pcr_moderate_threshold": (0.6, 1.4),
    "pcr_balanced_low": (0.6, 1.0),
    "pcr_balanced_high": (1.0, 1.5),

    # --- Bollinger Band width ---
    "bb_width_tight_pct": (0.3, 2.5),
    "bb_width_squeeze_pct": (0.3, 1.5),
    "bb_width_moderate_pct": (0.5, 2.0),

    # --- Max pain proximity ---
    "max_pain_close_pct": (0.1, 0.6),
    "max_pain_near_pct": (0.3, 1.5),

    # --- Debit strategy penalties ---
    "iv_high_penalty_threshold": (12.0, 30.0),
    "bb_width_expanded_pct": (0.8, 3.0),
}

# -------------------------------------------------------------------------
# Safety constants
# -------------------------------------------------------------------------

MAX_STEP_PCT: float = 0.15
"""Maximum percentage change allowed per review cycle (15%)."""

MAX_DRIFT_PCT: float = 0.40
"""Maximum cumulative drift from original default (40%)."""

MIN_SAMPLE_SIZE: int = 15
"""Minimum number of relevant trades before any parameter change is allowed."""

COOLING_PERIOD_TRADES: int = 10
"""Number of trades that must complete after a change before re-evaluation."""

REVERSION_TRIGGER_LOSSES: int = 5
"""Consecutive losses in affected trades that trigger a reversion flag."""

MAX_CHANGES_PER_SESSION: int = 3
"""Maximum parameter changes allowed per review session."""


# -------------------------------------------------------------------------
# Validation helpers
# -------------------------------------------------------------------------

def get_bounds(param_key: str) -> tuple[float, float] | None:
    """Return (min, max) bounds for *param_key*, or ``None`` if unknown."""
    return PARAMETER_BOUNDS.get(param_key)


def is_within_bounds(param_key: str, value: float) -> bool:
    """Check if *value* is within the allowed bounds for *param_key*."""
    bounds = PARAMETER_BOUNDS.get(param_key)
    if bounds is None:
        return True  # Unknown param — no bounds to enforce
    return bounds[0] <= value <= bounds[1]


def check_step_size(current: float, proposed: float) -> bool:
    """Return ``True`` if the change from *current* to *proposed* is within step limit."""
    if current == 0:
        return True
    pct_change = abs(proposed - current) / abs(current)
    return pct_change <= MAX_STEP_PCT


def check_drift(default: float, proposed: float) -> bool:
    """Return ``True`` if *proposed* is within drift limit of *default*."""
    if default == 0:
        return True
    drift = abs(proposed - default) / abs(default)
    return drift <= MAX_DRIFT_PCT


def validate_proposed_value(
    param_key: str,
    default: float,
    current: float,
    proposed: float,
) -> tuple[bool, str]:
    """Run all safety checks on a proposed parameter value.

    Returns ``(allowed, reason)`` where *reason* is empty when allowed.
    """
    # Bounds check
    if not is_within_bounds(param_key, proposed):
        bounds = PARAMETER_BOUNDS[param_key]
        return False, f"{param_key}={proposed} outside bounds [{bounds[0]}, {bounds[1]}]"

    # Step size check
    if not check_step_size(current, proposed):
        pct = abs(proposed - current) / abs(current) * 100 if current else 0
        return False, f"{param_key} step {pct:.1f}% exceeds max {MAX_STEP_PCT * 100:.0f}%"

    # Drift check
    if not check_drift(default, proposed):
        drift = abs(proposed - default) / abs(default) * 100 if default else 0
        return False, f"{param_key} drift {drift:.1f}% from default exceeds max {MAX_DRIFT_PCT * 100:.0f}%"

    return True, ""
