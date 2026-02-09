"""yfinance wrapper for intraday OHLCV candles."""

import logging

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class IntradayCandleFetcher:
    """Fetches intraday candle data for NIFTY via yfinance."""

    def __init__(self, ticker: str = "^NSEI") -> None:
        self._ticker = ticker

    def fetch(
        self, period: str = "5d", interval: str = "5m"
    ) -> pd.DataFrame:
        """Return OHLCV DataFrame with columns: Open, High, Low, Close, Volume."""
        tick = yf.Ticker(self._ticker)
        df = tick.history(period=period, interval=interval)

        if df.empty:
            logger.warning("yfinance returned empty DataFrame for %s", self._ticker)
            return df

        # Flatten MultiIndex columns if present (yfinance quirk)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Keep only OHLCV columns
        keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        df = df[keep].copy()
        df.dropna(subset=["Close"], inplace=True)
        return df
