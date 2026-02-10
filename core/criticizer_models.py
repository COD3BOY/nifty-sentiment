"""Pydantic models for the trade criticizer system."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from core.paper_trading_models import _now_ist


class ParameterRecommendation(BaseModel):
    """A single parameter tuning recommendation from the criticizer."""

    strategy_name: str
    parameter_name: str       # e.g. "rsi_neutral_upper_bound"
    current_value: float
    recommended_value: float
    confidence: float         # 0.0-1.0
    reasoning: str
    condition: str            # e.g. "when RSI > 55"


class TradeCritique(BaseModel):
    """Structured critique of a completed trade."""

    trade_id: str
    timestamp: datetime = Field(default_factory=_now_ist)
    overall_grade: str        # excellent/good/acceptable/poor/terrible
    pnl_assessment: dict      # outcome_quality, was_exit_timing_good, risk_reward_actual
    entry_signal_analysis: dict  # signals_that_worked, signals_that_failed, signals_missed
    strategy_fitness: dict    # was_right_strategy, better_strategy, market_regime_match
    parameter_recommendations: list[ParameterRecommendation] = Field(default_factory=list)
    patterns_observed: list[str] = Field(default_factory=list)
    risk_management_notes: str = ""
    summary: str = ""
