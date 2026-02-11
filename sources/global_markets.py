"""Global market indices (S&P 500, Nikkei, Hang Seng, FTSE) via yfinance."""

import logging
from datetime import datetime

import pandas as pd
import yfinance as yf

from core.config import get_source_config
from core.models import MarketSnapshot, SentimentScore
from sources import register_source
from sources.base import DataSource

logger = logging.getLogger(__name__)


@register_source
class GlobalMarketsSource(DataSource):
    @property
    def name(self) -> str:
        return "global_markets"

    @property
    def source_type(self) -> str:
        return "market"

    async def fetch_sentiment(self) -> SentimentScore:
        cfg = get_source_config("global_markets")
        indices_config = cfg.get("indices", {
            "^GSPC": {"name": "S&P 500", "weight": 1.0},
            "^N225": {"name": "Nikkei 225", "weight": 0.7},
            "^HSI": {"name": "Hang Seng", "weight": 0.7},
            "^FTSE": {"name": "FTSE 100", "weight": 0.5},
        })

        snapshots: list[MarketSnapshot] = []
        weighted_sum = 0.0
        total_weight = 0.0
        bullish = []
        bearish = []

        from core.api_guard import yf_guard_async

        for ticker, meta in indices_config.items():
            try:
                cb = await yf_guard_async()
                try:
                    info = yf.Ticker(ticker)
                    hist = info.history(period="2d")
                    cb.record_success()
                except Exception:
                    cb.record_failure()
                    raise
                if len(hist) < 2:
                    continue

                prev_close = hist["Close"].iloc[-2]
                current = hist["Close"].iloc[-1]

                if pd.isna(current) or pd.isna(prev_close) or prev_close == 0:
                    logger.warning("NaN/zero price for %s, skipping", ticker)
                    continue

                current = float(current)
                prev_close = float(prev_close)
                change_pct = ((current - prev_close) / prev_close) * 100

                snap = MarketSnapshot(
                    ticker=ticker,
                    name=meta["name"],
                    current_price=current,
                    previous_close=prev_close,
                    change_pct=change_pct,
                )
                snapshots.append(snap)

                w = meta.get("weight", 1.0)
                weighted_sum += change_pct * w
                total_weight += w

                if change_pct > 0.1:
                    bullish.append(f"{meta['name']} up {change_pct:.2f}%")
                elif change_pct < -0.1:
                    bearish.append(f"{meta['name']} down {change_pct:.2f}%")

            except Exception as e:
                logger.warning(f"Failed to fetch {ticker}: {e}")

        if total_weight == 0:
            return SentimentScore(
                source_name=self.name,
                score=0.0,
                confidence=0.0,
                explanation="No global market data available",
            )

        avg_change = weighted_sum / total_weight
        # Normalize: a 2% move is a strong signal (~1.0/-1.0)
        score = max(-1.0, min(1.0, avg_change / 2.0))
        confidence = min(1.0, len(snapshots) / len(indices_config))

        summary_parts = [f"{s.name}: {s.change_pct:+.2f}%" for s in snapshots]
        explanation = "Global markets: " + ", ".join(summary_parts)

        return SentimentScore(
            source_name=self.name,
            score=score,
            confidence=confidence,
            explanation=explanation,
            raw_data={
                "indices": [s.model_dump(mode="json") for s in snapshots],
                "weighted_avg_change": avg_change,
            },
            bullish_factors=bullish,
            bearish_factors=bearish,
        )
