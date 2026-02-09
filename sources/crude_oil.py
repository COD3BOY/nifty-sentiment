"""Brent crude oil + USD/INR via yfinance."""

import logging

import yfinance as yf

from core.config import get_source_config
from core.models import SentimentScore
from sources import register_source
from sources.base import DataSource

logger = logging.getLogger(__name__)


@register_source
class CrudeOilSource(DataSource):
    @property
    def name(self) -> str:
        return "crude_oil"

    @property
    def source_type(self) -> str:
        return "macro"

    async def fetch_sentiment(self) -> SentimentScore:
        cfg = get_source_config("crude_oil")
        crude_ticker = cfg.get("crude_ticker", "BZ=F")
        inr_ticker = cfg.get("inr_ticker", "INR=X")
        crude_weight = cfg.get("crude_weight", 0.6)
        inr_weight = cfg.get("inr_weight", 0.4)

        crude_score = 0.0
        inr_score = 0.0
        bullish = []
        bearish = []
        raw = {}

        # Brent Crude
        try:
            crude = yf.Ticker(crude_ticker)
            hist = crude.history(period="5d")
            if len(hist) >= 2:
                prev = hist["Close"].iloc[-2]
                current = hist["Close"].iloc[-1]
                change_pct = ((current - prev) / prev) * 100
                # Rising crude is bearish for India (imports 85% of crude)
                crude_score = max(-1.0, min(1.0, -change_pct / 3.0))
                raw["crude_price"] = current
                raw["crude_change_pct"] = change_pct

                if change_pct > 1:
                    bearish.append(f"Brent crude up {change_pct:.1f}% (import cost pressure)")
                elif change_pct < -1:
                    bullish.append(f"Brent crude down {change_pct:.1f}% (import cost relief)")
        except Exception as e:
            logger.warning(f"Crude oil fetch failed: {e}")

        # USD/INR
        try:
            inr = yf.Ticker(inr_ticker)
            hist = inr.history(period="5d")
            if len(hist) >= 2:
                prev = hist["Close"].iloc[-2]
                current = hist["Close"].iloc[-1]
                change_pct = ((current - prev) / prev) * 100
                # Rising USD/INR (weakening rupee) is bearish
                inr_score = max(-1.0, min(1.0, -change_pct / 1.0))
                raw["usdinr"] = current
                raw["usdinr_change_pct"] = change_pct

                if change_pct > 0.2:
                    bearish.append(f"INR weakened {change_pct:.2f}% vs USD")
                elif change_pct < -0.2:
                    bullish.append(f"INR strengthened {change_pct:.2f}% vs USD")
        except Exception as e:
            logger.warning(f"USD/INR fetch failed: {e}")

        combined = crude_score * crude_weight + inr_score * inr_weight
        score = max(-1.0, min(1.0, combined))
        has_data = bool(raw)

        parts = []
        if "crude_price" in raw:
            parts.append(f"Brent: ${raw['crude_price']:.1f} ({raw['crude_change_pct']:+.1f}%)")
        if "usdinr" in raw:
            parts.append(f"USD/INR: {raw['usdinr']:.2f} ({raw['usdinr_change_pct']:+.2f}%)")

        return SentimentScore(
            source_name=self.name,
            score=score,
            confidence=0.7 if has_data else 0.0,
            explanation=", ".join(parts) if parts else "No macro data available",
            raw_data=raw,
            bullish_factors=bullish,
            bearish_factors=bearish,
        )
