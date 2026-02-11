"""Intraday OHLCV candle fetcher — Kite primary, yfinance fallback."""

import logging
import os
from datetime import datetime, timedelta, timezone

import pandas as pd

logger = logging.getLogger(__name__)

_IST = timezone(timedelta(hours=5, minutes=30))

# Well-known instrument token for NIFTY 50 index on NSE
_NIFTY_50_TOKEN = 256265

# Map config interval strings to Kite interval values
_INTERVAL_MAP = {
    "1m": "minute",
    "5m": "5minute",
    "10m": "10minute",
    "15m": "15minute",
    "30m": "30minute",
    "60m": "60minute",
    "1d": "day",
}


class IntradayCandleFetcher:
    """Fetches intraday candle data for NIFTY — Kite primary, yfinance fallback."""

    def __init__(self, ticker: str = "^NSEI") -> None:
        self._ticker = ticker

    def _get_kite(self):
        """Create a fresh KiteConnect instance (re-reads env for token refreshes)."""
        from dotenv import load_dotenv
        load_dotenv(override=True)

        api_key = os.environ.get("KITE_API_KEY", "")
        access_token = os.environ.get("KITE_ACCESS_TOKEN", "")
        if not api_key or not access_token:
            return None

        from kiteconnect import KiteConnect

        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(access_token)
        return kite

    def _fetch_kite(self, period: str, interval: str) -> pd.DataFrame:
        """Fetch OHLCV candles from Kite historical data API."""
        from core.api_guard import kite_guard_sync

        kite = self._get_kite()
        if kite is None:
            raise ValueError("Kite credentials not configured")

        kite_interval = _INTERVAL_MAP.get(interval)
        if kite_interval is None:
            raise ValueError(f"Unsupported interval for Kite: {interval}")

        # Convert period string (e.g. "5d") to from/to dates
        now = datetime.now(_IST)
        days = int(period.rstrip("d"))
        from_date = now - timedelta(days=days)

        cb = kite_guard_sync()
        try:
            records = kite.historical_data(
                instrument_token=_NIFTY_50_TOKEN,
                from_date=from_date,
                to_date=now,
                interval=kite_interval,
            )
            cb.record_success()
        except Exception:
            cb.record_failure()
            raise

        if not records:
            raise ValueError("Kite returned empty historical data")

        df = pd.DataFrame(records)
        # Kite returns: date, open, high, low, close, volume
        df = df.rename(columns={
            "date": "Datetime",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        })
        df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True).dt.tz_convert(_IST)
        df = df.set_index("Datetime")

        keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        df = df[keep].copy()
        df.dropna(subset=["Close"], inplace=True)

        logger.info("Candles loaded via Kite (%d bars, %s interval)", len(df), interval)
        return df

    def _fetch_yfinance(self, period: str, interval: str) -> pd.DataFrame:
        """Fetch OHLCV candles from yfinance (fallback)."""
        import yfinance as yf
        from core.api_guard import yf_guard_sync

        cb = yf_guard_sync()
        try:
            tick = yf.Ticker(self._ticker)
            df = tick.history(period=period, interval=interval)
            cb.record_success()
        except Exception:
            cb.record_failure()
            raise

        if df.empty:
            logger.warning("yfinance returned empty DataFrame for %s", self._ticker)
            return df

        # Flatten MultiIndex columns if present (yfinance quirk)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        df = df[keep].copy()
        df.dropna(subset=["Close"], inplace=True)

        logger.info("Candles loaded via yfinance (%d bars, %s interval)", len(df), interval)
        return df

    def fetch(
        self, period: str = "5d", interval: str = "5m"
    ) -> pd.DataFrame:
        """Return OHLCV DataFrame — tries Kite first, falls back to yfinance."""
        try:
            return self._fetch_kite(period, interval)
        except Exception as exc:
            logger.warning("Kite candle fetch failed: %s — trying yfinance", exc)

        return self._fetch_yfinance(period, interval)
