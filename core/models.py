"""Pydantic models for the NIFTY Sentiment Prediction System."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SentimentLevel(str, Enum):
    STRONGLY_BEARISH = "strongly_bearish"
    BEARISH = "bearish"
    SLIGHTLY_BEARISH = "slightly_bearish"
    NEUTRAL = "neutral"
    SLIGHTLY_BULLISH = "slightly_bullish"
    BULLISH = "bullish"
    STRONGLY_BULLISH = "strongly_bullish"


class SentimentScore(BaseModel):
    """A normalized sentiment score from a single data source."""

    source_name: str
    score: float = Field(ge=-1.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    explanation: str = ""
    raw_data: dict[str, Any] = Field(default_factory=dict)
    bullish_factors: list[str] = Field(default_factory=list)
    bearish_factors: list[str] = Field(default_factory=list)


class AggregatedSentiment(BaseModel):
    """The final weighted sentiment from all sources."""

    overall_score: float = Field(ge=-1.0, le=1.0)
    level: SentimentLevel
    confidence: float = Field(ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source_scores: list[SentimentScore] = Field(default_factory=list)
    bullish_factors: list[str] = Field(default_factory=list)
    bearish_factors: list[str] = Field(default_factory=list)
    sources_used: int = 0
    sources_failed: int = 0


class NewsHeadline(BaseModel):
    """A single news headline from Zerodha Pulse or other sources."""

    title: str
    link: str = ""
    published: datetime | None = None
    source: str = ""


class HeadlineSentiment(BaseModel):
    """Sentiment analysis result for a single headline."""

    headline: str
    score: float = Field(ge=-1.0, le=1.0)
    reasoning: str = ""


class ClaudeSentimentResponse(BaseModel):
    """Structured response from Claude sentiment analysis."""

    overall_score: float = Field(ge=-1.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    headline_sentiments: list[HeadlineSentiment] = Field(default_factory=list)
    bullish_factors: list[str] = Field(default_factory=list)
    bearish_factors: list[str] = Field(default_factory=list)
    summary: str = ""


class MarketSnapshot(BaseModel):
    """Snapshot of a market index."""

    ticker: str
    name: str
    current_price: float
    previous_close: float
    change_pct: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class FIIDIIData(BaseModel):
    """FII/DII institutional flow data."""

    date: datetime
    fii_buy: float = 0.0
    fii_sell: float = 0.0
    fii_net: float = 0.0
    dii_buy: float = 0.0
    dii_sell: float = 0.0
    dii_net: float = 0.0
