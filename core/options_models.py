"""Pydantic models for the Intraday Options Desk."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class SignalDirection(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class StrategyName(str, Enum):
    SHORT_STRADDLE = "Short Straddle"
    SHORT_STRANGLE = "Short Strangle"
    LONG_STRADDLE = "Long Straddle"
    LONG_STRANGLE = "Long Strangle"
    BULL_PUT_SPREAD = "Bull Put Spread"
    BEAR_CALL_SPREAD = "Bear Call Spread"
    BULL_CALL_SPREAD = "Bull Call Spread"
    BEAR_PUT_SPREAD = "Bear Put Spread"
    IRON_CONDOR = "Iron Condor"
    LONG_CE = "Long Call (CE)"
    LONG_PE = "Long Put (PE)"


class VolatilityLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    EXTREME = "extreme"


class MomentumLevel(str, Enum):
    OVERSOLD = "oversold"
    WEAK = "weak"
    NEUTRAL = "neutral"
    STRONG = "strong"
    OVERBOUGHT = "overbought"


class StrikeData(BaseModel):
    """Single strike row from the option chain."""

    strike_price: float
    ce_oi: float = 0.0
    ce_change_in_oi: float = 0.0
    ce_volume: int = 0
    ce_iv: float = 0.0
    ce_ltp: float = 0.0
    ce_bid: float = 0.0
    ce_ask: float = 0.0
    pe_oi: float = 0.0
    pe_change_in_oi: float = 0.0
    pe_volume: int = 0
    pe_iv: float = 0.0
    pe_ltp: float = 0.0
    pe_bid: float = 0.0
    pe_ask: float = 0.0


class OptionChainData(BaseModel):
    """Full option chain snapshot for a single expiry."""

    symbol: str = "NIFTY"
    underlying_value: float = 0.0
    expiry: str = ""
    strikes: list[StrikeData] = Field(default_factory=list)
    total_ce_oi: float = 0.0
    total_pe_oi: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class OptionsAnalytics(BaseModel):
    """Computed metrics from the option chain."""

    pcr: float = 0.0
    pcr_label: str = "N/A"
    max_pain: float = 0.0
    support_strike: float = 0.0
    support_oi: float = 0.0
    resistance_strike: float = 0.0
    resistance_oi: float = 0.0
    atm_strike: float = 0.0
    atm_iv: float = 0.0
    iv_skew: float = 0.0


class TechnicalIndicators(BaseModel):
    """Latest-bar technical indicator values."""

    spot: float = 0.0
    spot_change: float = 0.0
    spot_change_pct: float = 0.0
    vwap: float = 0.0
    ema_9: float = 0.0
    ema_21: float = 0.0
    ema_50: float = 0.0
    rsi: float = 0.0
    supertrend: float = 0.0
    supertrend_direction: int = 1  # 1 = bullish, -1 = bearish
    bb_upper: float = 0.0
    bb_middle: float = 0.0
    bb_lower: float = 0.0


class TradeLeg(BaseModel):
    """Single leg of an option trade."""

    action: str  # "SELL" or "BUY"
    instrument: str  # e.g. "NIFTY 23500 CE"
    strike: float
    option_type: str  # "CE" or "PE"
    ltp: float
    lots: int = 1


class TradeSuggestion(BaseModel):
    """A ranked trade suggestion with full context."""

    strategy: StrategyName
    legs: list[TradeLeg]
    direction_bias: str  # "Bullish", "Bearish", "Neutral"
    confidence: str  # "High", "Medium", "Low"
    score: float  # internal ranking score 0-100
    entry_timing: str  # e.g. "Enter now — trend confirmed"
    technicals_to_check: list[str]  # e.g. ["RSI not overbought"]
    expected_outcome: str
    max_profit: str
    max_loss: str
    stop_loss: str
    position_size: str
    reasoning: list[str]  # 3-5 bullet points


class SignalCard(BaseModel):
    """Aggregated signal card for the dashboard."""

    label: str
    direction: SignalDirection = SignalDirection.NEUTRAL
    reasoning: list[str] = Field(default_factory=list)


class OptionsDeskSnapshot(BaseModel):
    """Top-level container for all options desk data."""

    chain: OptionChainData | None = None
    analytics: OptionsAnalytics | None = None
    technicals: TechnicalIndicators | None = None
    signals: list[SignalCard] = Field(default_factory=list)
    trade_suggestions: list[TradeSuggestion] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
