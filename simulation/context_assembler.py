"""Synthetic MarketContext builder for simulation.

Generates synthetic daily history and assembles MarketContext per tick,
reusing all pure computation functions from ``core.context_engine``.

Unlike the live ``ContextEngine`` which bootstraps from yfinance + SQLite,
``SimContextBuilder`` generates everything from scenario parameters so
simulations run with no external dependencies.

Lifecycle:
  1. ``__init__``: generate 55 synthetic daily bars → DailyContext + WeeklyContext + VolContext
  2. ``build_context(full_df, technicals)``: per-tick → MarketContext
  3. ``end_of_day(day_candles, ...)``: multi-day → update prior_days + weekly + vol
"""

from __future__ import annotations

import logging
import math
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd

from core.context_engine import (
    _classify_candle,
    _compute_context_bias,
    _compute_daily_context,
    _compute_multi_day_trend,
    _compute_session_context,
    _compute_weekly_context,
)
from core.context_models import (
    DailyContext,
    MarketContext,
    SessionContext,
    VolContext,
    WeeklyContext,
)
from core.indicators import compute_bollinger_bands, compute_ema, compute_rsi
from simulation.scenario_models import Scenario, VolRegimeConfig

logger = logging.getLogger(__name__)

_IST = timezone(timedelta(hours=5, minutes=30))
_TRADING_DAYS_PER_YEAR = 252


def _classify_vol_regime(p_rv: float, p_vov: float, p_vrp: float) -> str:
    """Classify vol regime from percentiles (same logic as Atlas/context_engine)."""
    if p_rv < 0.4 and p_vrp > 0.5:
        return "sell_premium"
    elif p_rv > 0.7 or p_vov > 0.8:
        return "stand_down"
    elif p_rv > 0.5 and p_vrp < 0.3:
        return "buy_premium"
    return "neutral"


def _generate_synthetic_daily_bars(
    n_bars: int,
    anchor_close: float,
    drift: float,
    seed: int,
) -> pd.DataFrame:
    """Generate synthetic daily OHLCV bars using GBM.

    Parameters
    ----------
    n_bars : int
        Number of daily bars (typically 55 for EMA-50 convergence).
    anchor_close : float
        Target close for the last bar (≈ scenario open_price).
    drift : float
        Annualized drift (from scenario phase config).
    seed : int
        RNG seed (use seed + 7000 to avoid collision with warmup/VIX seeds).

    Returns
    -------
    DataFrame with columns: Open, High, Low, Close, Volume, date (str).
    Index is sequential integers (not DatetimeIndex).
    """
    rng = np.random.default_rng(seed)
    sigma = 0.13  # annualized vol (typical NIFTY)
    dt = 1.0 / _TRADING_DAYS_PER_YEAR

    # Generate closes via GBM
    closes = np.empty(n_bars)
    # Start from a price that drifts to anchor_close
    # Forward-simulate and then scale so last close matches anchor
    z = rng.standard_normal(n_bars)
    log_returns = (drift - 0.5 * sigma ** 2) * dt + sigma * math.sqrt(dt) * z
    cum_returns = np.cumsum(log_returns)
    raw_closes = anchor_close * np.exp(cum_returns - cum_returns[-1])

    closes[:] = raw_closes

    # Generate OHLV from closes
    opens = np.empty(n_bars)
    highs = np.empty(n_bars)
    lows = np.empty(n_bars)
    volumes = rng.uniform(5_000_000, 10_000_000, size=n_bars)

    for i in range(n_bars):
        if i == 0:
            opens[i] = closes[i] * (1.0 + rng.uniform(-0.003, 0.003))
        else:
            opens[i] = closes[i - 1] * (1.0 + rng.uniform(-0.003, 0.003))

        body_range = abs(closes[i] - opens[i])
        wick = body_range * rng.uniform(0.2, 1.0)
        highs[i] = max(opens[i], closes[i]) + wick
        lows[i] = min(opens[i], closes[i]) - wick * rng.uniform(0.3, 1.0)
        # Ensure lows > 0
        lows[i] = max(lows[i], closes[i] * 0.95)

    return pd.DataFrame({
        "Open": opens,
        "High": highs,
        "Low": lows,
        "Close": closes,
        "Volume": volumes,
    })


def _compute_daily_indicators(closes: pd.Series) -> dict:
    """Compute EMA-20, EMA-50, RSI-14, BB-width on a close series."""
    ema20 = compute_ema(closes, 20)
    ema50 = compute_ema(closes, 50)
    rsi = compute_rsi(closes, 14)
    bb_upper, bb_mid, bb_lower = compute_bollinger_bands(closes, 20)
    bb_width = ((bb_upper - bb_lower) / bb_mid * 100.0).fillna(0.0)
    vol_20d = closes.rolling(20, min_periods=1).mean()  # proxy for volume avg
    return {
        "ema20": ema20,
        "ema50": ema50,
        "rsi": rsi,
        "bb_width": bb_width,
        "vol_20d": vol_20d,
    }


def _build_prior_days(
    daily_df: pd.DataFrame,
    indicators: dict,
    dates: list[str],
    n_days: int = 5,
) -> list[DailyContext]:
    """Build DailyContext for the last n_days of a daily DataFrame."""
    n = len(daily_df)
    start = max(0, n - n_days)
    result = []

    for i in range(start, n):
        row = {
            "Open": float(daily_df["Open"].iloc[i]),
            "High": float(daily_df["High"].iloc[i]),
            "Low": float(daily_df["Low"].iloc[i]),
            "Close": float(daily_df["Close"].iloc[i]),
            "Volume": float(daily_df["Volume"].iloc[i]),
            "date": dates[i],
        }

        def _safe(series, idx, fallback=0.0):
            val = float(series.iloc[idx]) if idx < len(series) else fallback
            return val if not math.isnan(val) else fallback

        ema20 = _safe(indicators["ema20"], i)
        ema50 = _safe(indicators["ema50"], i)
        rsi = _safe(indicators["rsi"], i, 50.0)
        bb_w = _safe(indicators["bb_width"], i)

        prev_close = float(daily_df["Close"].iloc[i - 1]) if i > 0 else 0.0
        vol_avg = _safe(indicators["vol_20d"], i, 1.0)

        ctx = _compute_daily_context(row, ema20, ema50, rsi, bb_w, prev_close, vol_avg)
        result.append(ctx)

    # Return most recent first (descending by date)
    result.reverse()
    return result


def _generate_trading_dates(end_date: date, n_days: int) -> list[date]:
    """Generate n_days trading dates ending on or before end_date (skip weekends)."""
    dates = []
    d = end_date
    while len(dates) < n_days:
        if d.weekday() < 5:
            dates.append(d)
        d -= timedelta(days=1)
    dates.reverse()
    return dates


class SimContextBuilder:
    """Builds MarketContext for simulation ticks.

    Stateful: maintains prior_days, weekly context, and vol context
    across ticks and across days (for multi-day simulations).
    """

    def __init__(
        self,
        scenario: Scenario,
        sim_date: date,
        seed: int,
        warmup_df: pd.DataFrame | None = None,
    ):
        self._scenario = scenario
        self._sim_date = sim_date
        self._seed = seed

        # --- A. Generate 55 synthetic daily OHLCV ---
        drift = scenario.price_path.phases[0].drift if scenario.price_path.phases else 0.0
        anchor_close = scenario.price_path.open_price

        n_daily = 55
        self._daily_df = _generate_synthetic_daily_bars(
            n_bars=n_daily,
            anchor_close=anchor_close,
            drift=drift,
            seed=seed + 7000,
        )

        # Generate trading dates for 55 bars ending the day before sim_date
        day_before = sim_date - timedelta(days=1)
        self._daily_dates = _generate_trading_dates(day_before, n_daily)
        self._daily_date_strs = [d.isoformat() for d in self._daily_dates]

        # --- B. Compute indicators on daily close series ---
        close_series = self._daily_df["Close"]
        self._indicators = _compute_daily_indicators(close_series)

        # --- C. Build DailyContext for last 5 days ---
        self._prior_days = _build_prior_days(
            self._daily_df, self._indicators, self._daily_date_strs, n_days=5,
        )

        # --- D. Build WeeklyContext for last 2 weeks ---
        all_daily = _build_prior_days(
            self._daily_df, self._indicators, self._daily_date_strs, n_days=15,
        )
        # all_daily is most-recent-first, reverse for grouping
        all_daily_asc = list(reversed(all_daily))

        from collections import defaultdict
        weeks: dict[str, list[DailyContext]] = defaultdict(list)
        for ctx in all_daily_asc:
            d = datetime.strptime(ctx.date, "%Y-%m-%d")
            monday = d - timedelta(days=d.weekday())
            week_key = monday.strftime("%Y-%m-%d")
            weeks[week_key].append(ctx)

        sorted_weeks = sorted(weeks.keys())
        self._current_week: WeeklyContext | None = None
        self._prior_week: WeeklyContext | None = None

        prior_wctx = None
        weekly_list = []
        for wk in sorted_weeks:
            days = weeks[wk]
            if len(days) < 2:
                continue
            wctx = _compute_weekly_context(days, prior_wctx)
            weekly_list.append(wctx)
            prior_wctx = wctx

        if len(weekly_list) >= 1:
            self._current_week = weekly_list[-1]
        if len(weekly_list) >= 2:
            self._prior_week = weekly_list[-2]

        # --- E. Build VolContext from scenario config ---
        self._vol_config = scenario.vol_regime
        self._vol_ctx = self._build_vol_context(self._vol_config, sim_date)

        logger.info(
            "SIM_CONTEXT: Init %d daily bars, %d prior_days, regime=%s, "
            "multi_day_trend=%s",
            n_daily,
            len(self._prior_days),
            self._vol_ctx.regime,
            _compute_multi_day_trend(
                self._prior_days, self._current_week, self._prior_week,
            ),
        )

    def _build_vol_context(
        self,
        vol_cfg: VolRegimeConfig,
        ref_date: date,
    ) -> VolContext:
        """Build VolContext from scenario VolRegimeConfig."""
        regime = _classify_vol_regime(vol_cfg.p_rv, vol_cfg.p_vov, vol_cfg.p_vrp)
        regime_since = (
            ref_date - timedelta(days=vol_cfg.regime_duration_days)
        ).isoformat()

        return VolContext(
            p_rv=round(vol_cfg.p_rv, 3),
            p_vov=round(vol_cfg.p_vov, 3),
            p_vrp=round(vol_cfg.p_vrp, 3),
            regime=regime,
            regime_since=regime_since,
            regime_duration_days=vol_cfg.regime_duration_days,
            regime_changes_30d=vol_cfg.regime_changes_30d,
            rv_trend=vol_cfg.rv_trend,
        )

    def build_context(
        self,
        full_df: pd.DataFrame,
        technicals,
    ) -> MarketContext:
        """Build MarketContext for the current tick.

        Parameters
        ----------
        full_df : DataFrame
            Progressive candle reveal (warmup + today's bars so far).
        technicals : TechnicalIndicators
            Current tick's technical indicators.

        Returns
        -------
        MarketContext ready for algorithm consumption.
        """
        # Convert technicals to dict for _compute_session_context
        tech_dict = None
        if technicals is not None:
            if hasattr(technicals, "model_dump"):
                tech_dict = technicals.model_dump()
            elif isinstance(technicals, dict):
                tech_dict = technicals

        session = _compute_session_context(full_df, tech_dict)

        prior_day = self._prior_days[0] if self._prior_days else None
        multi_day_trend = _compute_multi_day_trend(
            self._prior_days, self._current_week, self._prior_week,
        )
        context_bias = _compute_context_bias(session, prior_day, self._vol_ctx)

        return MarketContext(
            session=session,
            prior_day=prior_day,
            prior_days=self._prior_days,
            current_week=self._current_week,
            prior_week=self._prior_week,
            vol=self._vol_ctx,
            multi_day_trend=multi_day_trend,
            context_bias=context_bias,
        )

    def end_of_day(
        self,
        day_candles: pd.DataFrame,
        new_vol_regime: VolRegimeConfig | None = None,
        next_date: date | None = None,
    ) -> None:
        """Update context state between multi-day simulation days.

        Parameters
        ----------
        day_candles : DataFrame
            Full day's 375 minute candles (OHLCV with DatetimeIndex).
        new_vol_regime : VolRegimeConfig, optional
            Vol regime for the next day (if different from current).
        next_date : date, optional
            Next simulation date (for regime_since calculation).
        """
        if day_candles is None or day_candles.empty:
            return

        # Extract daily OHLC from minute candles
        day_open = float(day_candles["Open"].iloc[0])
        day_high = float(day_candles["High"].max())
        day_low = float(day_candles["Low"].min())
        day_close = float(day_candles["Close"].iloc[-1])
        day_volume = float(day_candles["Volume"].sum())

        # Determine the date
        dates = day_candles.index.date
        candle_date = max(set(dates))
        date_str = candle_date.isoformat()

        # Append to daily_df for indicator recomputation
        new_row = pd.DataFrame({
            "Open": [day_open],
            "High": [day_high],
            "Low": [day_low],
            "Close": [day_close],
            "Volume": [day_volume],
        })
        self._daily_df = pd.concat([self._daily_df, new_row], ignore_index=True)
        self._daily_dates.append(candle_date)
        self._daily_date_strs.append(date_str)

        # Recompute indicators on full series
        close_series = self._daily_df["Close"]
        self._indicators = _compute_daily_indicators(close_series)

        # Rebuild prior_days (last 5)
        self._prior_days = _build_prior_days(
            self._daily_df, self._indicators, self._daily_date_strs, n_days=5,
        )

        # Rebuild weekly context
        all_daily = _build_prior_days(
            self._daily_df, self._indicators, self._daily_date_strs, n_days=15,
        )
        all_daily_asc = list(reversed(all_daily))

        from collections import defaultdict
        weeks: dict[str, list[DailyContext]] = defaultdict(list)
        for ctx in all_daily_asc:
            d = datetime.strptime(ctx.date, "%Y-%m-%d")
            monday = d - timedelta(days=d.weekday())
            week_key = monday.strftime("%Y-%m-%d")
            weeks[week_key].append(ctx)

        sorted_weeks = sorted(weeks.keys())
        prior_wctx = None
        weekly_list = []
        for wk in sorted_weeks:
            days = weeks[wk]
            if len(days) < 2:
                continue
            wctx = _compute_weekly_context(days, prior_wctx)
            weekly_list.append(wctx)
            prior_wctx = wctx

        if len(weekly_list) >= 1:
            self._current_week = weekly_list[-1]
        if len(weekly_list) >= 2:
            self._prior_week = weekly_list[-2]

        # Update vol context
        if new_vol_regime is not None:
            old_regime = self._vol_ctx.regime
            new_regime = _classify_vol_regime(
                new_vol_regime.p_rv, new_vol_regime.p_vov, new_vol_regime.p_vrp,
            )
            # If regime changed, reset duration; otherwise increment
            if new_regime == old_regime:
                updated_cfg = new_vol_regime.model_copy(update={
                    "regime_duration_days": self._vol_config.regime_duration_days + 1,
                    "regime_changes_30d": self._vol_config.regime_changes_30d,
                })
            else:
                updated_cfg = new_vol_regime.model_copy(update={
                    "regime_duration_days": 1,
                    "regime_changes_30d": self._vol_config.regime_changes_30d + 1,
                })
            self._vol_config = updated_cfg
            ref_date = next_date or (candle_date + timedelta(days=1))
            self._vol_ctx = self._build_vol_context(updated_cfg, ref_date)
        else:
            # Same regime, increment duration
            self._vol_config = self._vol_config.model_copy(update={
                "regime_duration_days": self._vol_config.regime_duration_days + 1,
            })
            ref_date = next_date or (candle_date + timedelta(days=1))
            self._vol_ctx = self._build_vol_context(self._vol_config, ref_date)

        logger.info(
            "SIM_CONTEXT: end_of_day %s — %d prior_days, regime=%s",
            date_str, len(self._prior_days), self._vol_ctx.regime,
        )
