"""Zerodha Pulse RSS feed news → Claude sentiment analysis."""

import logging
from datetime import datetime

import feedparser
import requests

from analyzers.claude_analyzer import analyze_headlines
from core.config import get_source_config
from core.models import SentimentScore
from sources import register_source
from sources.base import DataSource

logger = logging.getLogger(__name__)


@register_source
class ZerodhaPulseSource(DataSource):
    @property
    def name(self) -> str:
        return "zerodha_pulse"

    @property
    def source_type(self) -> str:
        return "news"

    async def fetch_sentiment(self) -> SentimentScore:
        cfg = get_source_config("zerodha_pulse")
        rss_url = cfg.get("rss_url", "https://pulse.zerodha.com/feed.php")
        max_headlines = cfg.get("max_headlines", 20)

        try:
            resp = requests.get(rss_url, headers={
                "User-Agent": "Mozilla/5.0 (compatible; NiftySentimentBot/1.0)"
            }, timeout=10)
            feed = feedparser.parse(resp.content)
        except requests.exceptions.SSLError:
            logger.warning("Zerodha Pulse SSL error, falling back to feedparser direct fetch")
            feed = feedparser.parse(rss_url)
        except Exception as e:
            logger.warning(f"Failed to fetch Zerodha Pulse RSS: {e}")
            feed = feedparser.parse(rss_url)

        if not feed.entries:
            return SentimentScore(
                source_name=self.name,
                score=0.0,
                confidence=0.0,
                explanation="No headlines available from Zerodha Pulse",
            )

        headlines = [entry.title for entry in feed.entries[:max_headlines]]
        result = await analyze_headlines(headlines)

        return SentimentScore(
            source_name=self.name,
            score=max(-1.0, min(1.0, result.overall_score)),
            confidence=result.confidence,
            explanation=result.summary,
            raw_data={
                "headline_count": len(headlines),
                "headlines": headlines[:5],
                "headline_sentiments": [
                    {"headline": hs.headline, "score": hs.score, "reasoning": hs.reasoning}
                    for hs in result.headline_sentiments[:5]
                ],
            },
            bullish_factors=result.bullish_factors,
            bearish_factors=result.bearish_factors,
        )

    async def is_available(self) -> bool:
        from core.config import get_env
        return bool(get_env("ANTHROPIC_API_KEY"))
