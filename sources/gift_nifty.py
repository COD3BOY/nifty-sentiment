"""GIFT Nifty pre-market gap via Kite Connect."""

import logging

from core.config import get_env, get_source_config
from core.models import SentimentScore
from sources import register_source
from sources.base import DataSource

logger = logging.getLogger(__name__)


@register_source
class GiftNiftySource(DataSource):
    @property
    def name(self) -> str:
        return "gift_nifty"

    @property
    def source_type(self) -> str:
        return "market"

    async def fetch_sentiment(self) -> SentimentScore:
        cfg = get_source_config("gift_nifty")
        gap_pct_scale = cfg.get("gap_pct_scale", 2.0)

        api_key = get_env("KITE_API_KEY")
        access_token = get_env("KITE_ACCESS_TOKEN")

        if not api_key or not access_token:
            return SentimentScore(
                source_name=self.name,
                score=0.0,
                confidence=0.0,
                explanation="Kite Connect credentials not configured",
            )

        try:
            from kiteconnect import KiteConnect
            kite = KiteConnect(api_key=api_key)
            kite.set_access_token(access_token)

            # GIFT Nifty is traded as NIFTY on NSE
            # Get NIFTY previous close
            nifty_quote = kite.quote(["NSE:NIFTY 50"])
            nifty_data = nifty_quote.get("NSE:NIFTY 50", {})
            prev_close = nifty_data.get("ohlc", {}).get("close", 0)

            # Try to get GIFT Nifty LTP (may be available as a separate instrument)
            # Fallback: use NIFTY futures as proxy
            try:
                gift_quote = kite.quote(["NSE:NIFTY FUT"])
                gift_ltp = list(gift_quote.values())[0].get("last_price", prev_close)
            except Exception:
                gift_ltp = nifty_data.get("last_price", prev_close)

            if prev_close == 0:
                return SentimentScore(
                    source_name=self.name,
                    score=0.0,
                    confidence=0.0,
                    explanation="Could not fetch NIFTY previous close",
                )

            gap_pct = ((gift_ltp - prev_close) / prev_close) * 100
            # 1% gap maps to 0.5 score (configurable via gap_pct_scale)
            score = max(-1.0, min(1.0, gap_pct / gap_pct_scale))

            bullish = []
            bearish = []
            if gap_pct > 0.1:
                bullish.append(f"GIFT Nifty indicates {gap_pct:.2f}% gap up opening")
            elif gap_pct < -0.1:
                bearish.append(f"GIFT Nifty indicates {gap_pct:.2f}% gap down opening")

            return SentimentScore(
                source_name=self.name,
                score=score,
                confidence=0.9,
                explanation=f"GIFT Nifty at {gift_ltp:.0f}, prev NIFTY close {prev_close:.0f}, gap {gap_pct:+.2f}%",
                raw_data={
                    "gift_nifty_ltp": gift_ltp,
                    "nifty_prev_close": prev_close,
                    "gap_pct": gap_pct,
                },
                bullish_factors=bullish,
                bearish_factors=bearish,
            )

        except Exception as e:
            logger.error(f"GIFT Nifty fetch failed: {e}")
            return SentimentScore(
                source_name=self.name,
                score=0.0,
                confidence=0.0,
                explanation=f"Error fetching GIFT Nifty: {e}",
            )

    async def is_available(self) -> bool:
        return bool(get_env("KITE_API_KEY") and get_env("KITE_ACCESS_TOKEN"))
