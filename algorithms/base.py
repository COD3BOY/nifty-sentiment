"""Abstract base class for pluggable trading algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from core.options_models import (
    OptionChainData,
    OptionsAnalytics,
    TechnicalIndicators,
    TradeSuggestion,
)
from core.paper_trading_models import PaperTradingState

if TYPE_CHECKING:
    from core.context_models import MarketContext
    from core.observation import ObservationSnapshot


class TradingAlgorithm(ABC):
    """Base class for all trading algorithms.

    Each algorithm receives shared market data (chain, technicals, analytics)
    and produces trade suggestions + manages positions independently.
    """

    name: str  # unique ID, e.g. "sentinel", "jarvis"
    display_name: str  # UI label, e.g. "V1 Sentinel"
    description: str  # one-liner

    def __init__(self, config: dict) -> None:
        self.config = config

    @abstractmethod
    def generate_suggestions(
        self,
        chain: OptionChainData,
        technicals: TechnicalIndicators,
        analytics: OptionsAnalytics,
        observation: ObservationSnapshot | None = None,
        context: MarketContext | None = None,
    ) -> list[TradeSuggestion]:
        """Generate trade suggestions from shared market data."""

    @abstractmethod
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
        context: MarketContext | None = None,
    ) -> PaperTradingState:
        """Manage existing positions and optionally open new ones.

        Pure function: state in, state out.
        """
