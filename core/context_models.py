"""Pydantic models for multi-level temporal market context.

Five context levels, narrow to broad:
  - SessionContext: today's running intraday metrics
  - DailyContext: end-of-day summary for one trading session
  - WeeklyContext: aggregated from DailyContext rows (Mon-Fri)
  - VolContext: vol regime with persistence tracking
  - MarketContext: composite — what algorithms receive
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class SessionContext(BaseModel):
    """Today's running intraday context."""

    session_open: float = 0.0
    session_high: float = 0.0
    session_low: float = 0.0
    session_range_pct: float = 0.0          # (high-low)/open * 100
    current_vs_vwap_pct: float = 0.0        # (spot-vwap)/vwap * 100
    current_vs_open_pct: float = 0.0        # (spot-open)/open * 100
    rsi_trajectory: str = "flat"            # "rising" | "falling" | "flat"
    ema_alignment: str = "mixed"            # "bullish" | "bearish" | "mixed"
    bb_position: str = "middle"             # "upper" | "middle" | "lower"
    session_trend: str = "range_bound"      # "trending_up" | "trending_down" | "range_bound"
    bars_elapsed: int = 0


class DailyContext(BaseModel):
    """End-of-day summary for one trading session."""

    date: str                               # "YYYY-MM-DD"
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: float = 0.0
    range_pct: float = 0.0                  # (high-low)/open * 100
    body_pct: float = 0.0                   # abs(close-open)/open * 100
    close_vs_open: str = "flat"             # "up" | "down" | "flat"
    candle_type: str = "neutral"            # "bullish_engulf" | "bearish_engulf" | "doji" | "hammer" | "shooting_star" | "neutral"
    gap_from_prev_pct: float = 0.0          # gap vs previous day close
    close_above_ema20: bool = False
    close_above_ema50: bool = False
    ema20: float = 0.0
    ema50: float = 0.0
    rsi_close: float = 50.0
    bb_width_pct: float = 0.0
    volume_vs_20d_avg: float = 1.0          # relative volume
    atm_iv_close: float = 0.0              # from iv_history (if available)


class WeeklyContext(BaseModel):
    """Weekly summary aggregated from DailyContext rows."""

    week_start: str                          # "YYYY-MM-DD" (Monday)
    week_end: str                            # "YYYY-MM-DD" (Friday or last trading day)
    week_open: float = 0.0
    week_high: float = 0.0
    week_low: float = 0.0
    week_close: float = 0.0
    weekly_range_pct: float = 0.0
    weekly_change_pct: float = 0.0           # (close-open)/open * 100
    days_up: int = 0
    days_down: int = 0
    weekly_trend: str = "neutral"            # "bullish" | "bearish" | "neutral"
    avg_daily_range_pct: float = 0.0
    week_vs_prior: str = "inside"            # "higher_high" | "lower_low" | "inside" | "outside"
    ema20_slope: str = "flat"                # "rising" | "falling" | "flat"


class VolContext(BaseModel):
    """Vol regime with persistence tracking."""

    p_rv: float = 0.5
    p_vov: float = 0.5
    p_vrp: float = 0.5
    regime: str = "neutral"                  # "sell_premium" | "buy_premium" | "stand_down" | "neutral"
    regime_since: str = ""                   # date when current regime started
    regime_duration_days: int = 0
    regime_changes_30d: int = 0
    rv_trend: str = "stable"                 # "expanding" | "contracting" | "stable"


class MarketContext(BaseModel):
    """Top-level composite — what algorithms receive."""

    session: SessionContext = Field(default_factory=SessionContext)
    prior_day: DailyContext | None = None
    prior_days: list[DailyContext] = Field(default_factory=list)  # last 5 trading days
    current_week: WeeklyContext | None = None
    prior_week: WeeklyContext | None = None
    vol: VolContext = Field(default_factory=VolContext)
    # Derived convenience fields
    multi_day_trend: str = "neutral"         # from weekly + daily data
    context_bias: str = "neutral"            # composite: "bullish" | "bearish" | "neutral"
