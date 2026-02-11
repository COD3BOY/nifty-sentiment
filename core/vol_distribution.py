"""Volatility distribution engine for V4 Vol-Optimized algorithm.

Fetches ~3 years of NIFTY close + India VIX via yfinance, computes rolling
realized vol (RV), vol-of-vol (VoV), volatility risk premium (VRP), and
percentile ranks. Provides VolSnapshot for live algorithm use and full
DataFrame for backtesting.

Caching: disk (CSV) + 5-min in-memory TTL.
"""

from __future__ import annotations

import logging
import math
import time as _time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_CACHE_FILE = _DATA_DIR / "vol_distribution_cache.csv"

# In-memory cache
_mem_cache: dict[str, tuple[pd.DataFrame, float]] = {}
_MEM_CACHE_TTL = 300  # 5 minutes


@dataclass
class VolSnapshot:
    """Current volatility regime snapshot for the algorithm."""

    rv_5: float       # 5-day annualized RV (%)
    rv_10: float      # 10-day annualized RV (%)
    rv_20: float      # 20-day annualized RV (%)
    vov_20: float     # vol-of-vol (stdev of RV_5 changes, 20d)
    vix: float        # India VIX
    vrp: float        # VIX - RV_20
    p_rv: float       # percentile rank of RV_20 [0..1]
    p_vov: float      # percentile rank of VoV_20 [0..1]
    p_vrp: float      # percentile rank of VRP [0..1]
    em: float         # expected move (spot * vix/100 * sqrt(dte/365))
    date: str


def _fetch_raw_data(start_date: str = "2022-01-01") -> pd.DataFrame | None:
    """Download NIFTY close + India VIX close via yfinance, merged."""
    try:
        import yfinance as yf

        nifty = yf.download("^NSEI", start=start_date, auto_adjust=True, progress=False)
        vix = yf.download("^INDIAVIX", start=start_date, auto_adjust=True, progress=False)

        if nifty.empty or vix.empty:
            logger.warning("VOL_DIST: Empty data from yfinance (NIFTY=%d, VIX=%d)",
                           len(nifty), len(vix))
            return None

        # Handle potential MultiIndex columns from yfinance
        if isinstance(nifty.columns, pd.MultiIndex):
            nifty.columns = nifty.columns.get_level_values(0)
        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = vix.columns.get_level_values(0)

        df = pd.DataFrame({
            "close": nifty["Close"].squeeze(),
            "vix": vix["Close"].squeeze(),
        })
        df.index = pd.to_datetime(df.index)

        # Forward-fill VIX gaps (VIX may have fewer trading days) — max 5 business days
        df["vix"] = df["vix"].ffill(limit=5)
        df = df.dropna(subset=["close"])

        # Check NIFTY/VIX date overlap
        vix_valid = df["vix"].notna().sum()
        total = len(df)
        if total > 0:
            overlap_pct = vix_valid / total * 100
            if overlap_pct < 80:
                logger.warning(
                    "VOL_DIST: VIX coverage only %.1f%% (%d/%d rows) — VRP may be unreliable",
                    overlap_pct, vix_valid, total,
                )

        logger.info("VOL_DIST: Fetched %d rows (NIFTY: %s to %s)",
                     len(df), df.index[0].date(), df.index[-1].date())
        return df

    except Exception as e:
        logger.error("VOL_DIST: Failed to fetch raw data: %s", e)
        return None


def _compute_rolling_series(df: pd.DataFrame) -> pd.DataFrame:
    """Add log returns, rolling RV, VoV, and VRP columns."""
    df = df.copy()

    # Log returns
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    # Realized volatility: stdev(log_returns, N) * sqrt(252) * 100
    for n in [5, 10, 20]:
        df[f"rv_{n}"] = df["log_return"].rolling(n, min_periods=n).std() * math.sqrt(252) * 100

    # Vol-of-vol: stdev of daily RV_5 changes over 20 days
    df["rv_5_change"] = df["rv_5"].diff()
    df["vov_20"] = df["rv_5_change"].rolling(20, min_periods=10).std()

    # Volatility risk premium: VIX - RV_20
    df["vrp"] = df["vix"] - df["rv_20"]

    return df


def _compute_percentile_ranks(df: pd.DataFrame, lookback: int = 504) -> pd.DataFrame:
    """Compute rolling percentile ranks for rv_20, vov_20, vrp."""
    df = df.copy()

    for col, out_col in [("rv_20", "p_rv"), ("vov_20", "p_vov"), ("vrp", "p_vrp")]:
        ranks = []
        values = df[col].values
        for i in range(len(values)):
            start = max(0, i - lookback + 1)
            window = values[start:i + 1]
            # Filter out NaN and inf
            valid = window[np.isfinite(window)]
            if len(valid) < 60:
                if i == len(values) - 1 and len(valid) > 0:
                    logger.warning(
                        "VOL_DIST: only %d valid values for %s (need 60), using 0.5 default",
                        len(valid), col,
                    )
                ranks.append(np.nan)
            else:
                current = values[i]
                if not np.isfinite(current):
                    ranks.append(np.nan)
                else:
                    count_below = np.sum(valid < current)
                    ranks.append(count_below / (len(valid) - 1) if len(valid) > 1 else 0.5)
        df[out_col] = ranks

    return df


def load_vol_distribution(force_refresh: bool = False) -> pd.DataFrame | None:
    """Load full vol distribution DataFrame.

    Uses disk cache (CSV) + 5-min in-memory TTL cache.
    Returns None on failure.
    """
    now = _time.time()

    # Check in-memory cache
    if not force_refresh and "vol_dist" in _mem_cache:
        cached_df, cached_ts = _mem_cache["vol_dist"]
        if (now - cached_ts) < _MEM_CACHE_TTL:
            return cached_df

    # Check disk cache
    if not force_refresh and _CACHE_FILE.exists():
        try:
            df = pd.read_csv(_CACHE_FILE, index_col=0, parse_dates=True)
            last_date = pd.Timestamp(df.index[-1])
            today = pd.Timestamp.now().normalize()
            # Refresh if stale (more than 1 business day old)
            bday_diff = len(pd.bdate_range(last_date, today)) - 1
            if bday_diff <= 1:
                _mem_cache["vol_dist"] = (df, now)
                logger.info("VOL_DIST: Loaded %d rows from disk cache (last: %s)",
                            len(df), last_date.date())
                return df
            else:
                logger.info("VOL_DIST: Disk cache stale (%d bdays), refreshing...", bday_diff)
        except Exception as e:
            logger.warning("VOL_DIST: Failed to read disk cache: %s", e)

    # Fetch fresh data
    raw = _fetch_raw_data()
    if raw is None:
        return None

    df = _compute_rolling_series(raw)
    df = _compute_percentile_ranks(df)

    # Save to disk cache
    try:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(_CACHE_FILE)
        logger.info("VOL_DIST: Saved %d rows to disk cache", len(df))
    except Exception as e:
        logger.warning("VOL_DIST: Failed to save disk cache: %s", e)

    # Update in-memory cache
    _mem_cache["vol_dist"] = (df, now)
    return df


def get_today_vol_snapshot(spot: float, dte: int = 5) -> VolSnapshot | None:
    """Return latest VolSnapshot for the live algorithm.

    Returns None on failure (algorithm must handle gracefully).
    """
    df = load_vol_distribution()
    if df is None or df.empty:
        return None

    # Get last valid row
    last = df.dropna(subset=["rv_20", "vov_20", "vrp", "p_rv", "p_vov", "p_vrp"])
    if last.empty:
        return None

    row = last.iloc[-1]

    # Check data freshness — reject if older than 2 business days
    last_date = pd.Timestamp(last.index[-1])
    today = pd.Timestamp.now().normalize()
    bday_gap = len(pd.bdate_range(last_date, today)) - 1
    if bday_gap > 2:
        logger.warning(
            "VOL_DIST: last valid row is %d business days old (%s), returning None",
            bday_gap, last_date.date(),
        )
        return None

    vix_val = float(row["vix"]) if np.isfinite(row["vix"]) else 0.0

    # Expected move
    em = 0.0
    if spot > 0 and vix_val > 0 and dte > 0:
        em = spot * (vix_val / 100.0) * math.sqrt(dte / 365.0)

    return VolSnapshot(
        rv_5=float(row["rv_5"]),
        rv_10=float(row["rv_10"]),
        rv_20=float(row["rv_20"]),
        vov_20=float(row["vov_20"]),
        vix=vix_val,
        vrp=float(row["vrp"]),
        p_rv=float(row["p_rv"]),
        p_vov=float(row["p_vov"]),
        p_vrp=float(row["p_vrp"]),
        em=em,
        date=str(last.index[-1].date()),
    )


def get_historical_backtest_data() -> pd.DataFrame | None:
    """Return full DataFrame with NaN rows dropped, for tuning script."""
    df = load_vol_distribution(force_refresh=True)
    if df is None:
        return None
    return df.dropna(subset=["rv_20", "vov_20", "vrp", "p_rv", "p_vov", "p_vrp"]).copy()
