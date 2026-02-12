"""V1 Sentinel algorithm — thin wrapper around existing trade logic.

Delegates directly to existing functions with zero modifications to V1 behavior.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from algorithms import register_algorithm
from algorithms.base import TradingAlgorithm
from core.options_models import (
    OptionChainData,
    OptionsAnalytics,
    TechnicalIndicators,
    TradeSuggestion,
)
from core.paper_trading_engine import evaluate_and_manage as _v1_evaluate_and_manage
from core.paper_trading_models import PaperTradingState
from core.trade_strategies import generate_trade_suggestions as _v1_generate_suggestions

if TYPE_CHECKING:
    from core.observation import ObservationSnapshot


@register_algorithm
class SentinelAlgorithm(TradingAlgorithm):
    """Original V1 trading algorithm — wraps existing logic unchanged."""

    name = "sentinel"
    display_name = "V1 Sentinel"
    description = "Original rule-based strategy evaluator with 11 option strategies"

    def generate_suggestions(
        self,
        chain: OptionChainData,
        technicals: TechnicalIndicators,
        analytics: OptionsAnalytics,
        observation: ObservationSnapshot | None = None,
    ) -> list[TradeSuggestion]:
        return _v1_generate_suggestions(analytics, technicals, chain)

    def evaluate_and_manage(
        self,
        state: PaperTradingState,
        suggestions: list[TradeSuggestion] | None,
        chain: OptionChainData | None,
        technicals: TechnicalIndicators | None = None,
        analytics: OptionsAnalytics | None = None,
        lot_size: int | None = None,
        refresh_ts: float = 0.0,
        observation: ObservationSnapshot | None = None,
    ) -> PaperTradingState:
        return _v1_evaluate_and_manage(
            state, suggestions, chain,
            technicals=technicals, analytics=analytics,
            lot_size=lot_size, refresh_ts=refresh_ts,
        )
