"""India VIX fear gauge via yfinance."""

import logging

import yfinance as yf

from core.config import get_source_config
from core.models import SentimentScore
from sources import register_source
from sources.base import DataSource

logger = logging.getLogger(__name__)


@register_source
class VIXSource(DataSource):
    @property
    def name(self) -> str:
        return "vix"

    @property
    def source_type(self) -> str:
        return "market"

    async def fetch_sentiment(self) -> SentimentScore:
        cfg = get_source_config("vix")
        ticker = cfg.get("ticker", "^INDIAVIX")
        spike_threshold = cfg.get("spike_threshold", 10)
        spike_amplification = cfg.get("spike_amplification", 0.3)

        try:
            vix = yf.Ticker(ticker)
            hist = vix.history(period="5d")

            if len(hist) < 2:
                return SentimentScore(
                    source_name=self.name,
                    score=0.0,
                    confidence=0.0,
                    explanation="Insufficient VIX data",
                )

            current_vix = hist["Close"].iloc[-1]
            prev_vix = hist["Close"].iloc[-2]
            day_change_pct = ((current_vix - prev_vix) / prev_vix) * 100

            # VIX scoring: inverse relationship
            # VIX < 12: very bullish, 12-15: bullish, 15-20: neutral, 20-25: bearish, >25: very bearish
            if current_vix < 12:
                base_score = 0.6
            elif current_vix < 15:
                base_score = 0.3
            elif current_vix < 20:
                base_score = 0.0
            elif current_vix < 25:
                base_score = -0.3
            else:
                base_score = -0.6

            # Spike detection: >10% day-over-day rise amplifies bearish signal
            spike_penalty = 0.0
            if day_change_pct > spike_threshold:
                spike_penalty = -spike_amplification
                logger.info(f"VIX spike detected: {day_change_pct:.1f}% rise")

            score = max(-1.0, min(1.0, base_score + spike_penalty))

            bullish = []
            bearish = []
            if current_vix < 15:
                bullish.append(f"Low VIX at {current_vix:.1f} indicates market calm")
            if current_vix > 20:
                bearish.append(f"Elevated VIX at {current_vix:.1f} indicates fear")
            if day_change_pct > spike_threshold:
                bearish.append(f"VIX spiked {day_change_pct:.1f}% day-over-day")
            elif day_change_pct < -spike_threshold:
                bullish.append(f"VIX dropped {day_change_pct:.1f}% day-over-day (fear subsiding)")

            return SentimentScore(
                source_name=self.name,
                score=score,
                confidence=0.85,
                explanation=f"India VIX at {current_vix:.1f} (change: {day_change_pct:+.1f}%)",
                raw_data={
                    "current_vix": current_vix,
                    "prev_vix": prev_vix,
                    "day_change_pct": day_change_pct,
                    "spike_detected": day_change_pct > spike_threshold,
                },
                bullish_factors=bullish,
                bearish_factors=bearish,
            )

        except Exception as e:
            logger.error(f"VIX fetch failed: {e}")
            return SentimentScore(
                source_name=self.name, score=0.0, confidence=0.0,
                explanation=f"Error: {e}",
            )
