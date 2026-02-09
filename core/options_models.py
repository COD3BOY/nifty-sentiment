"""Pydantic models for the Intraday Options Desk."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class SignalDirection(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


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
    errors: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
