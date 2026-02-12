"""Shared helper functions for options algorithms.

Extracted from jarvis.py, optimus.py, atlas.py to eliminate duplication.
"""

from __future__ import annotations

import math
from datetime import datetime, time as dt_time

from core.options_models import OptionChainData, TechnicalIndicators, TradeLeg
from core.paper_trading_models import PaperTradingState, _IST, _now_ist


def get_strike_data(chain: OptionChainData, strike_price: float):
    """Look up StrikeData by strike price."""
    for s in chain.strikes:
        if s.strike_price == strike_price:
            return s
    return None


def compute_spread_width(legs: list[TradeLeg]) -> float:
    """Compute spread width (distance between strikes) for spread strategies."""
    strikes = sorted(set(leg.strike for leg in legs))
    if len(strikes) >= 2:
        return strikes[-1] - strikes[0]
    return 0.0


def compute_bb_width_pct(technicals: TechnicalIndicators) -> float:
    """Bollinger Band width as percentage of middle band."""
    if technicals.bb_middle > 0:
        return ((technicals.bb_upper - technicals.bb_lower) / technicals.bb_middle) * 100
    return 0.0


def parse_dte(expiry_str: str) -> int:
    """Parse chain.expiry (e.g. '30-Jan-2025') to calendar days from today."""
    try:
        expiry_date = datetime.strptime(expiry_str, "%d-%b-%Y").date()
        today = _now_ist().date()
        return max(0, (expiry_date - today).days)
    except (ValueError, TypeError):
        return 0


def is_breakout(technicals: TechnicalIndicators) -> bool:
    """Spot outside BB + supertrend confirms direction."""
    if technicals.bb_upper <= 0:
        return False
    above_bb = technicals.spot > technicals.bb_upper
    below_bb = technicals.spot < technicals.bb_lower
    if above_bb and technicals.supertrend_direction == 1:
        return True
    if below_bb and technicals.supertrend_direction == -1:
        return True
    return False


def reset_period_tracking(state: PaperTradingState, cfg: dict) -> PaperTradingState:
    """Reset daily/weekly capital tracking when period changes."""
    now = _now_ist()
    today_str = now.strftime("%Y-%m-%d")
    current_capital = state.initial_capital + state.net_realized_pnl
    updates: dict = {}

    # Daily reset
    if state.session_date != today_str:
        updates["session_date"] = today_str
        updates["daily_start_capital"] = current_capital

    # Weekly reset (Monday = 0)
    current_week = now.strftime("%Y-W%W")
    if state.week_start_date != current_week:
        updates["week_start_date"] = current_week
        updates["weekly_start_capital"] = current_capital

    if updates:
        return state.model_copy(update=updates)
    return state


def compute_expected_move(spot: float, iv_or_vix: float, dte: int) -> float:
    """Expected move = spot * (IV/100) * sqrt(DTE/365)."""
    if spot <= 0 or iv_or_vix <= 0 or dte <= 0:
        return 0.0
    return spot * (iv_or_vix / 100.0) * math.sqrt(dte / 365.0)


def is_observation_period(entry_start_time: str = "10:00") -> bool:
    """Check if current time is within the observation window (9:15 to entry_start_time).

    Returns True if the market is open but we're before the entry start time.
    Returns False if the market is closed or past entry start time.
    """
    from core.market_hours import is_market_open
    if not is_market_open():
        return False

    now = _now_ist()
    try:
        h, m = (int(x) for x in entry_start_time.split(":"))
        cutoff = dt_time(h, m)
    except (ValueError, TypeError):
        cutoff = dt_time(10, 0)

    return now.time() < cutoff
