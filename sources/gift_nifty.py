"""NIFTY pre-market / opening gap via NSE data (nsefetch).

Primary: NSE allIndices API for NIFTY 50 index data.
Pre-open (9:00-9:15 IST): NSE pre-open market data for IEP estimate.
Fallback: Kite Connect if nsefetch fails and credentials exist.
"""

import logging
from datetime import datetime, timezone, timedelta

from core.config import get_env, get_source_config
from core.models import SentimentScore
from sources import register_source
from sources.base import DataSource

logger = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))


def _ist_now() -> datetime:
    return datetime.now(IST)


def _is_pre_open() -> bool:
    """Check if current IST time is during pre-open session (9:00-9:15)."""
    now = _ist_now()
    return now.hour == 9 and now.minute < 15


def _fetch_nifty_from_allindices() -> dict | None:
    """Fetch NIFTY 50 data from NSE allIndices API via nsefetch."""
    from nsepython import nsefetch

    url = "https://www.nseindia.com/api/allIndices"
    data = nsefetch(url)
    if not data or "data" not in data:
        return None

    for idx in data["data"]:
        if idx.get("index") == "NIFTY 50":
            return idx
    return None


def _fetch_preopen_iep() -> float | None:
    """Fetch pre-open IEP-weighted estimate from NSE pre-open data.

    Returns estimated opening price based on top NIFTY stocks' IEP,
    or None if data is unavailable.
    """
    from nsepython import nsefetch

    url = "https://www.nseindia.com/api/market-data-pre-open?key=NIFTY"
    data = nsefetch(url)
    if not data:
        return None

    records = data if isinstance(data, list) else data.get("data", [])
    if not records:
        return None

    total_change = 0.0
    count = 0
    for record in records:
        meta = record.get("metadata", {})
        iep = meta.get("iep")
        prev_close = meta.get("previousClose")
        if iep and prev_close and prev_close > 0:
            total_change += ((iep - prev_close) / prev_close) * 100
            count += 1

    if count == 0:
        return None

    return total_change / count


def _fetch_via_kite() -> dict | None:
    """Fallback: fetch NIFTY data via Kite Connect if credentials exist."""
    api_key = get_env("KITE_API_KEY")
    access_token = get_env("KITE_ACCESS_TOKEN")
    if not api_key or not access_token:
        return None

    try:
        from kiteconnect import KiteConnect

        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(access_token)

        quote = kite.quote(["NSE:NIFTY 50"])
        nifty = quote.get("NSE:NIFTY 50", {})
        last = nifty.get("last_price", 0)
        prev_close = nifty.get("ohlc", {}).get("close", 0)
        if prev_close == 0:
            return None

        return {
            "last": last,
            "previousClose": prev_close,
            "open": nifty.get("ohlc", {}).get("open", 0),
            "percentChange": ((last - prev_close) / prev_close) * 100,
            "source": "kite",
        }
    except Exception as e:
        logger.warning(f"Kite Connect fallback failed: {e}")
        return None


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

        nifty_data = None
        data_source = "nse"
        pre_open_change = None

        # 1. Try NSE allIndices
        try:
            nifty_data = _fetch_nifty_from_allindices()
        except Exception as e:
            logger.warning(f"NSE allIndices fetch failed: {e}")

        # 2. During pre-open, try to get IEP data
        if _is_pre_open():
            try:
                pre_open_change = _fetch_preopen_iep()
            except Exception as e:
                logger.warning(f"NSE pre-open fetch failed: {e}")

        # 3. Fallback to Kite if NSE failed
        if nifty_data is None:
            try:
                kite_data = _fetch_via_kite()
                if kite_data:
                    nifty_data = kite_data
                    data_source = "kite"
            except Exception as e:
                logger.warning(f"Kite fallback failed: {e}")

        if nifty_data is None:
            return SentimentScore(
                source_name=self.name,
                score=0.0,
                confidence=0.0,
                explanation="Could not fetch NIFTY data from NSE or Kite",
            )

        # Extract values based on source
        if data_source == "kite":
            last = nifty_data["last"]
            prev_close = nifty_data["previousClose"]
            open_price = nifty_data.get("open", 0)
            gap_pct = nifty_data["percentChange"]
        else:
            last = nifty_data.get("last", 0)
            prev_close = nifty_data.get("previousClose", 0)
            open_price = nifty_data.get("open", 0)
            gap_pct = nifty_data.get("percentChange", 0)

        if prev_close == 0:
            return SentimentScore(
                source_name=self.name,
                score=0.0,
                confidence=0.0,
                explanation="NIFTY previous close is zero — data unavailable",
            )

        # During pre-open, prefer IEP-based estimate if available
        if pre_open_change is not None:
            gap_pct = pre_open_change
            explanation_prefix = f"NIFTY pre-open IEP estimate: {gap_pct:+.2f}% vs prev close {prev_close:.0f}"
            confidence = 0.8
        elif open_price and open_price > 0 and open_price != prev_close:
            gap_pct = ((open_price - prev_close) / prev_close) * 100
            explanation_prefix = f"NIFTY at {last:.0f} (open {open_price:.0f}), prev close {prev_close:.0f}, opening gap {gap_pct:+.2f}%"
            confidence = 0.9
        else:
            explanation_prefix = f"NIFTY at {last:.0f}, prev close {prev_close:.0f}, change {gap_pct:+.2f}%"
            confidence = 0.85

        score = max(-1.0, min(1.0, gap_pct / gap_pct_scale))

        bullish = []
        bearish = []
        if gap_pct > 0.1:
            bullish.append(f"NIFTY indicates {gap_pct:+.2f}% gap up")
        elif gap_pct < -0.1:
            bearish.append(f"NIFTY indicates {gap_pct:+.2f}% gap down")

        if data_source == "kite":
            explanation_prefix += " (via Kite)"

        return SentimentScore(
            source_name=self.name,
            score=score,
            confidence=confidence,
            explanation=explanation_prefix,
            raw_data={
                "nifty_last": last,
                "nifty_prev_close": prev_close,
                "nifty_open": open_price,
                "gap_pct": gap_pct,
                "data_source": data_source,
                "pre_open_iep_used": pre_open_change is not None,
            },
            bullish_factors=bullish,
            bearish_factors=bearish,
        )

    async def is_available(self) -> bool:
        return True
