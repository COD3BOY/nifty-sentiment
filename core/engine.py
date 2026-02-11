"""SentimentEngine - orchestrates parallel data source fetching, weighted averaging, and persistence."""

import asyncio
import logging
from datetime import datetime

from core.config import load_config
from core.database import SentimentDatabase
from core.models import AggregatedSentiment, SentimentLevel, SentimentScore
from sources import discover_sources, get_registry

logger = logging.getLogger(__name__)


def _score_to_level(score: float) -> SentimentLevel:
    """Map a numeric score to a SentimentLevel enum."""
    if score <= -0.6:
        return SentimentLevel.STRONGLY_BEARISH
    elif score <= -0.3:
        return SentimentLevel.BEARISH
    elif score <= -0.1:
        return SentimentLevel.SLIGHTLY_BEARISH
    elif score <= 0.1:
        return SentimentLevel.NEUTRAL
    elif score <= 0.3:
        return SentimentLevel.SLIGHTLY_BULLISH
    elif score <= 0.6:
        return SentimentLevel.BULLISH
    else:
        return SentimentLevel.STRONGLY_BULLISH


class SentimentEngine:
    def __init__(self):
        self.config = load_config()
        self.db = SentimentDatabase()
        discover_sources()
        self._sources = []
        self._init_sources()

    def _init_sources(self) -> None:
        """Instantiate enabled sources from the registry."""
        registry = get_registry()
        sources_config = self.config.get("sources", {})

        for source_name, source_cls in registry.items():
            src_cfg = sources_config.get(source_name, {})
            if src_cfg.get("enabled", False):
                self._sources.append(source_cls(src_cfg))
                logger.info(f"Loaded source: {source_name}")
            else:
                logger.debug(f"Source disabled: {source_name}")

    async def _fetch_one(self, source, timeout: int) -> SentimentScore | None:
        """Fetch sentiment from a single source with timeout."""
        try:
            if not await source.is_available():
                logger.warning(f"Source {source.name} not available, skipping")
                return None
            return await asyncio.wait_for(source.fetch_sentiment(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.error(f"Source {source.name} timed out after {timeout}s")
            return None
        except Exception as e:
            logger.error(f"Source {source.name} failed: {e}")
            return None

    async def compute_sentiment(self, mode: str = "default") -> AggregatedSentiment:
        """Fetch all sources in parallel, compute weighted average, persist.

        Args:
            mode: "default" uses per-source weights, "pre_market" uses boosted
                  weights from config pre_market.weights section.
        """
        default_timeout = self.config.get("engine", {}).get("default_timeout", 30)
        sources_config = self.config.get("sources", {})
        pre_market_weights = self.config.get("pre_market", {}).get("weights", {}) if mode == "pre_market" else {}

        # Run all sources concurrently
        tasks = []
        for source in self._sources:
            src_cfg = sources_config.get(source.name, {})
            timeout = src_cfg.get("timeout", default_timeout)
            tasks.append(self._fetch_one(source, timeout))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect successful results
        scores: list[SentimentScore] = []
        failed = 0
        for i, result in enumerate(results):
            if isinstance(result, BaseException):
                source_name = self._sources[i].name if i < len(self._sources) else "unknown"
                logger.error(f"Source {source_name} raised exception: {result}")
                failed += 1
            elif result is not None and result.confidence > 0:
                scores.append(result)
            else:
                failed += 1

        if not scores:
            sentiment = AggregatedSentiment(
                overall_score=0.0,
                level=SentimentLevel.NEUTRAL,
                confidence=0.0,
                source_scores=[],
                sources_used=0,
                sources_failed=failed,
            )
            self.db.save_aggregated_sentiment(sentiment)
            return sentiment

        # Compute weighted average
        weighted_sum = 0.0
        total_weight = 0.0
        all_bullish = []
        all_bearish = []

        for ss in scores:
            src_cfg = sources_config.get(ss.source_name, {})
            weight = pre_market_weights.get(ss.source_name, src_cfg.get("weight", 1.0))
            # Weight by both configured weight and source confidence
            effective_weight = weight * ss.confidence
            weighted_sum += ss.score * effective_weight
            total_weight += effective_weight

            all_bullish.extend(ss.bullish_factors)
            all_bearish.extend(ss.bearish_factors)

        overall_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        overall_score = max(-1.0, min(1.0, overall_score))

        # Overall confidence: average source confidence weighted by source weight
        conf_sum = sum(
            ss.confidence * pre_market_weights.get(ss.source_name, sources_config.get(ss.source_name, {}).get("weight", 1.0))
            for ss in scores
        )
        conf_weight = sum(
            pre_market_weights.get(ss.source_name, sources_config.get(ss.source_name, {}).get("weight", 1.0))
            for ss in scores
        )
        overall_confidence = conf_sum / conf_weight if conf_weight > 0 else 0.0

        sentiment = AggregatedSentiment(
            overall_score=overall_score,
            level=_score_to_level(overall_score),
            confidence=min(1.0, overall_confidence),
            timestamp=datetime.utcnow(),
            source_scores=scores,
            bullish_factors=all_bullish[:10],
            bearish_factors=all_bearish[:10],
            sources_used=len(scores),
            sources_failed=failed,
        )

        self.db.save_aggregated_sentiment(sentiment)
        return sentiment

    def get_historical(self, days: int = 30) -> list[dict]:
        return self.db.get_historical_sentiment(days)

    def get_accuracy(self, days: int = 30) -> dict:
        return self.db.compute_accuracy(days)

    def update_market_actuals(self) -> None:
        """Fetch and store actual NIFTY close for accuracy tracking."""
        import yfinance as yf
        from core.api_guard import yf_guard_sync

        ticker = self.config.get("engine", {}).get("nifty_ticker", "^NSEI")

        try:
            cb = yf_guard_sync()
            try:
                nifty = yf.Ticker(ticker)
                hist = nifty.history(period="5d")
                cb.record_success()
            except Exception:
                cb.record_failure()
                raise
            for idx, row in hist.iterrows():
                self.db.save_market_actual(
                    date=idx.to_pydatetime(),
                    open_price=row["Open"],
                    close_price=row["Close"],
                )
        except Exception as e:
            logger.error(f"Failed to update market actuals: {e}")
