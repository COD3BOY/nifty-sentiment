"""Data assembler — bridges synthetic data to real Pydantic models.

Uses REAL indicator/analytics computation functions on synthetic OHLCV
data.  Only the price data is synthetic; all downstream computation is
identical to production.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from core.indicators import (
    compute_bollinger_bands,
    compute_ema,
    compute_rsi,
    compute_supertrend,
    compute_vwap,
)
from core.observation import compute_observation_snapshot, ObservationSnapshot
from core.options_analytics import build_analytics
from core.options_models import (
    OptionChainData,
    OptionsAnalytics,
    TechnicalIndicators,
)
from core.vol_distribution import VolSnapshot
from simulation.scenario_models import VolRegimeConfig


def compute_technicals(
    df: pd.DataFrame,
    sim_datetime: datetime,
) -> TechnicalIndicators:
    """Compute TechnicalIndicators from a candle DataFrame.

    Uses real indicator functions.  The DataFrame should include
    warmup history — only the last bar's values are extracted.

    Parameters
    ----------
    df : DataFrame
        OHLCV with DatetimeIndex.  Must have enough rows for EMA-50.
    sim_datetime : datetime
        Current simulation time (used for staleness calculation).

    Returns
    -------
    TechnicalIndicators with all fields populated.
    """
    if df is None or len(df) < 2:
        # Return minimal technicals
        return TechnicalIndicators(
            spot=0.0, spot_change=0.0, spot_change_pct=0.0,
            vwap=0.0, ema_9=0.0, ema_20=0.0, ema_21=0.0, ema_50=0.0,
            rsi=50.0, supertrend=0.0, supertrend_direction=0,
            bb_upper=0.0, bb_middle=0.0, bb_lower=0.0,
            data_staleness_minutes=0.0,
        )

    close = df["Close"]
    last_close = float(close.iloc[-1])

    # Spot change from first bar of today
    dates = df.index.date
    today_mask = dates == dates[-1]
    today_df = df[today_mask]
    if len(today_df) > 1:
        first_open = float(today_df["Open"].iloc[0])
        spot_change = last_close - first_open
        spot_change_pct = (spot_change / first_open) * 100.0 if first_open else 0.0
    else:
        spot_change = 0.0
        spot_change_pct = 0.0

    vwap = compute_vwap(df)
    ema9 = compute_ema(close, span=9)
    ema20 = compute_ema(close, span=20)
    ema21 = compute_ema(close, span=21)
    ema50 = compute_ema(close, span=50)
    rsi = compute_rsi(close, period=9)
    st_vals, st_dirs = compute_supertrend(df, period=10, multiplier=3.0)
    bb_upper, bb_mid, bb_lower = compute_bollinger_bands(close, period=20, std_dev=2.0)

    # Staleness is 0 in simulation (data is always "fresh")
    staleness_min = 0.0

    def _safe_float(series, fallback=0.0):
        val = series.iloc[-1] if len(series) > 0 else fallback
        return float(val) if pd.notna(val) else fallback

    return TechnicalIndicators(
        spot=last_close,
        spot_change=round(spot_change, 2),
        spot_change_pct=round(spot_change_pct, 2),
        vwap=_safe_float(vwap, last_close),
        ema_9=_safe_float(ema9, last_close),
        ema_20=_safe_float(ema20, last_close),
        ema_21=_safe_float(ema21, last_close),
        ema_50=_safe_float(ema50, last_close),
        rsi=_safe_float(rsi, 50.0),
        supertrend=_safe_float(st_vals, last_close),
        supertrend_direction=int(st_dirs.iloc[-1]) if len(st_dirs) > 0 and pd.notna(st_dirs.iloc[-1]) else 0,
        bb_upper=_safe_float(bb_upper, last_close),
        bb_middle=_safe_float(bb_mid, last_close),
        bb_lower=_safe_float(bb_lower, last_close),
        data_staleness_minutes=staleness_min,
    )


def compute_analytics(
    chain: OptionChainData,
    iv_percentile: float = 50.0,
) -> OptionsAnalytics:
    """Compute OptionsAnalytics using real ``build_analytics()``.

    Injects ``iv_percentile`` from scenario config since production
    reads this from SQLite.

    Parameters
    ----------
    chain : OptionChainData
        Synthetic option chain.
    iv_percentile : float
        IV percentile to inject (from scenario's ChainConfig).

    Returns
    -------
    OptionsAnalytics with iv_percentile set from scenario.
    """
    analytics = build_analytics(chain)
    analytics.iv_percentile = iv_percentile
    return analytics


def compute_observation(
    df: pd.DataFrame,
    observation_end_time: str = "10:00",
) -> ObservationSnapshot | None:
    """Compute observation snapshot using real ``compute_observation_snapshot()``.

    Parameters
    ----------
    df : DataFrame
        Multi-day OHLCV with DatetimeIndex (needs previous day + today).
    observation_end_time : str
        HH:MM for end of observation window.

    Returns
    -------
    ObservationSnapshot or None if insufficient data.
    """
    try:
        return compute_observation_snapshot(df, observation_end_time)
    except Exception:
        return None


def build_vol_snapshot(
    vol_config: VolRegimeConfig,
    spot: float,
    vix: float,
    sim_date_str: str,
    dte: int = 5,
) -> VolSnapshot:
    """Build a VolSnapshot from scenario config.

    Production calls ``get_today_vol_snapshot()`` which fetches from
    yfinance.  In simulation, we construct directly from scenario params.

    Parameters
    ----------
    vol_config : VolRegimeConfig
        Pre-configured volatility regime percentiles.
    spot : float
        Current spot price.
    vix : float
        Current VIX level.
    sim_date_str : str
        YYYY-MM-DD formatted date.
    dte : int
        Days to expiry for expected move calculation.

    Returns
    -------
    VolSnapshot dataclass instance.
    """
    import math
    em = spot * (vix / 100.0) * math.sqrt(dte / 365.0)

    return VolSnapshot(
        rv_5=vol_config.rv_5,
        rv_10=vol_config.rv_10,
        rv_20=vol_config.rv_20,
        vov_20=vol_config.vov_20,
        vix=vix,
        vrp=vol_config.vrp,
        p_rv=vol_config.p_rv,
        p_vov=vol_config.p_vov,
        p_vrp=vol_config.p_vrp,
        em=round(em, 2),
        date=sim_date_str,
    )


@dataclass
class AssembledTick:
    """All data needed for one algorithm evaluation cycle."""

    chain: OptionChainData
    technicals: TechnicalIndicators
    analytics: OptionsAnalytics
    observation: ObservationSnapshot | None
    vol_snapshot: VolSnapshot
    spot: float
    vix: float
    sim_datetime: datetime
    tick_index: int
