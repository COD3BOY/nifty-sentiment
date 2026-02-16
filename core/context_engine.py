"""Context Engine — multi-level temporal market context.

Computes and serves SessionContext (intraday), DailyContext (daily),
WeeklyContext (weekly), and VolContext (vol regime) to algorithms.

Bootstrap: fetches 60d daily candles from yfinance on startup, computes
DailyContext + WeeklyContext for historical days, persists to SQLite.

Refresh: SessionContext recomputed every 60s from 5-min candles.
DailyContext persisted at EOD. WeeklyContext updated on last day of week.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from core.config import load_config
from core.context_models import (
    DailyContext,
    MarketContext,
    SessionContext,
    VolContext,
    WeeklyContext,
)

logger = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))


# ---------------------------------------------------------------------------
# Pure computation helpers
# ---------------------------------------------------------------------------

def _classify_candle(open_: float, high: float, low: float, close: float) -> str:
    """Simple candle pattern recognition.

    Returns one of: "bullish_engulf", "bearish_engulf", "doji",
    "hammer", "shooting_star", "neutral".
    """
    if open_ <= 0 or high <= 0:
        return "neutral"

    body = abs(close - open_)
    full_range = high - low
    if full_range <= 0:
        return "doji"

    body_ratio = body / full_range

    # Doji: body < 10% of range
    if body_ratio < 0.1:
        return "doji"

    upper_shadow = high - max(open_, close)
    lower_shadow = min(open_, close) - low

    # Hammer: small body at top, long lower shadow (>2x body)
    if lower_shadow > 2 * body and upper_shadow < body:
        return "hammer"

    # Shooting star: small body at bottom, long upper shadow (>2x body)
    if upper_shadow > 2 * body and lower_shadow < body:
        return "shooting_star"

    # Strong bullish: close > open, body > 60% of range
    if close > open_ and body_ratio > 0.6:
        return "bullish_engulf"

    # Strong bearish: close < open, body > 60% of range
    if close < open_ and body_ratio > 0.6:
        return "bearish_engulf"

    return "neutral"


def _compute_daily_context(
    row: dict,
    ema20: float,
    ema50: float,
    rsi: float,
    bb_width_pct: float,
    prev_close: float,
    volume_20d_avg: float,
) -> DailyContext:
    """Compute DailyContext from a single daily candle + indicators."""
    o, h, l, c = row["Open"], row["High"], row["Low"], row["Close"]
    vol = row.get("Volume", 0.0)
    date_str = row["date"]

    range_pct = ((h - l) / o * 100.0) if o > 0 else 0.0
    body_pct = (abs(c - o) / o * 100.0) if o > 0 else 0.0

    if c > o * 1.001:
        close_vs_open = "up"
    elif c < o * 0.999:
        close_vs_open = "down"
    else:
        close_vs_open = "flat"

    gap_pct = ((o - prev_close) / prev_close * 100.0) if prev_close > 0 else 0.0
    vol_ratio = (vol / volume_20d_avg) if volume_20d_avg > 0 else 1.0

    return DailyContext(
        date=date_str,
        open=round(o, 2),
        high=round(h, 2),
        low=round(l, 2),
        close=round(c, 2),
        volume=round(vol, 0),
        range_pct=round(range_pct, 2),
        body_pct=round(body_pct, 2),
        close_vs_open=close_vs_open,
        candle_type=_classify_candle(o, h, l, c),
        gap_from_prev_pct=round(gap_pct, 2),
        close_above_ema20=c > ema20 if ema20 > 0 else False,
        close_above_ema50=c > ema50 if ema50 > 0 else False,
        ema20=round(ema20, 2),
        ema50=round(ema50, 2),
        rsi_close=round(rsi, 2),
        bb_width_pct=round(bb_width_pct, 2),
        volume_vs_20d_avg=round(vol_ratio, 2),
    )


def _compute_weekly_context(
    daily_rows: list[DailyContext],
    prior_week: WeeklyContext | None = None,
) -> WeeklyContext:
    """Aggregate DailyContext rows into a WeeklyContext."""
    if not daily_rows:
        return WeeklyContext(week_start="", week_end="")

    # Sort by date ascending
    days = sorted(daily_rows, key=lambda d: d.date)

    week_open = days[0].open
    week_close = days[-1].close
    week_high = max(d.high for d in days)
    week_low = min(d.low for d in days)

    weekly_range_pct = ((week_high - week_low) / week_open * 100.0) if week_open > 0 else 0.0
    weekly_change_pct = ((week_close - week_open) / week_open * 100.0) if week_open > 0 else 0.0

    days_up = sum(1 for d in days if d.close_vs_open == "up")
    days_down = sum(1 for d in days if d.close_vs_open == "down")

    if weekly_change_pct > 0.3:
        weekly_trend = "bullish"
    elif weekly_change_pct < -0.3:
        weekly_trend = "bearish"
    else:
        weekly_trend = "neutral"

    avg_daily_range = sum(d.range_pct for d in days) / len(days) if days else 0.0

    # Compare with prior week
    week_vs_prior = "inside"
    if prior_week and prior_week.week_high > 0:
        hh = week_high > prior_week.week_high
        ll = week_low < prior_week.week_low
        if hh and ll:
            week_vs_prior = "outside"
        elif hh:
            week_vs_prior = "higher_high"
        elif ll:
            week_vs_prior = "lower_low"
        else:
            week_vs_prior = "inside"

    # EMA20 slope from last day's ema20 vs first day's ema20
    ema20_first = days[0].ema20
    ema20_last = days[-1].ema20
    if ema20_first > 0 and ema20_last > 0:
        slope_pct = (ema20_last - ema20_first) / ema20_first * 100.0
        if slope_pct > 0.1:
            ema20_slope = "rising"
        elif slope_pct < -0.1:
            ema20_slope = "falling"
        else:
            ema20_slope = "flat"
    else:
        ema20_slope = "flat"

    # ISO week Monday for week_start
    first_date = datetime.strptime(days[0].date, "%Y-%m-%d")
    monday = first_date - timedelta(days=first_date.weekday())

    return WeeklyContext(
        week_start=monday.strftime("%Y-%m-%d"),
        week_end=days[-1].date,
        week_open=round(week_open, 2),
        week_high=round(week_high, 2),
        week_low=round(week_low, 2),
        week_close=round(week_close, 2),
        weekly_range_pct=round(weekly_range_pct, 2),
        weekly_change_pct=round(weekly_change_pct, 2),
        days_up=days_up,
        days_down=days_down,
        weekly_trend=weekly_trend,
        avg_daily_range_pct=round(avg_daily_range, 2),
        week_vs_prior=week_vs_prior,
        ema20_slope=ema20_slope,
    )


def _compute_session_context(
    candle_df: pd.DataFrame,
    technicals: dict | None = None,
) -> SessionContext:
    """Compute SessionContext from today's intraday candle DataFrame.

    Parameters
    ----------
    candle_df : 5-min candle DataFrame (today only or multi-day — uses latest day)
    technicals : dict with keys like 'rsi', 'ema_9', 'ema_20', 'ema_50',
                 'bb_upper', 'bb_lower', 'vwap', 'spot' (from TechnicalIndicators)
    """
    if candle_df is None or candle_df.empty:
        return SessionContext()

    # Use only today's data
    dates = candle_df.index.date
    today = max(set(dates))
    today_df = candle_df[dates == today]

    if today_df.empty:
        return SessionContext()

    session_open = float(today_df["Open"].iloc[0])
    session_high = float(today_df["High"].max())
    session_low = float(today_df["Low"].min())
    bars_elapsed = len(today_df)

    range_pct = ((session_high - session_low) / session_open * 100.0) if session_open > 0 else 0.0

    # Current price from technicals or last candle
    spot = 0.0
    if technicals:
        spot = technicals.get("spot", 0.0)
    if spot <= 0:
        spot = float(today_df["Close"].iloc[-1])

    current_vs_open_pct = ((spot - session_open) / session_open * 100.0) if session_open > 0 else 0.0

    # VWAP distance
    current_vs_vwap_pct = 0.0
    if technicals:
        vwap = technicals.get("vwap", 0.0)
        if vwap > 0:
            current_vs_vwap_pct = (spot - vwap) / vwap * 100.0

    # RSI trajectory from last 3 RSI readings (need candle-level RSI)
    rsi_trajectory = "flat"
    if technicals:
        rsi = technicals.get("rsi", 50.0)
        # Simple heuristic: compare to 50
        if rsi > 60:
            rsi_trajectory = "rising"
        elif rsi < 40:
            rsi_trajectory = "falling"

    # EMA alignment
    ema_alignment = "mixed"
    if technicals:
        ema9 = technicals.get("ema_9", 0.0)
        ema20 = technicals.get("ema_20", 0.0)
        ema50 = technicals.get("ema_50", 0.0)
        if ema9 > 0 and ema20 > 0 and ema50 > 0:
            if ema9 > ema20 > ema50:
                ema_alignment = "bullish"
            elif ema9 < ema20 < ema50:
                ema_alignment = "bearish"

    # BB position
    bb_position = "middle"
    if technicals:
        bb_upper = technicals.get("bb_upper", 0.0)
        bb_lower = technicals.get("bb_lower", 0.0)
        if bb_upper > 0 and bb_lower > 0 and bb_upper > bb_lower:
            bb_range = bb_upper - bb_lower
            if bb_range > 0:
                pos = (spot - bb_lower) / bb_range
                if pos > 0.8:
                    bb_position = "upper"
                elif pos < 0.2:
                    bb_position = "lower"

    # Session trend from open-to-current move + range
    session_trend = "range_bound"
    if abs(current_vs_open_pct) > 0.5:
        if current_vs_open_pct > 0:
            session_trend = "trending_up"
        else:
            session_trend = "trending_down"

    return SessionContext(
        session_open=round(session_open, 2),
        session_high=round(session_high, 2),
        session_low=round(session_low, 2),
        session_range_pct=round(range_pct, 2),
        current_vs_vwap_pct=round(current_vs_vwap_pct, 2),
        current_vs_open_pct=round(current_vs_open_pct, 2),
        rsi_trajectory=rsi_trajectory,
        ema_alignment=ema_alignment,
        bb_position=bb_position,
        session_trend=session_trend,
        bars_elapsed=bars_elapsed,
    )


def _compute_vol_context(
    vol_snapshot,
    regime_history: list[dict],
) -> VolContext:
    """Wrap existing VolSnapshot + regime history into VolContext.

    Parameters
    ----------
    vol_snapshot : VolSnapshot from vol_distribution.py (or None)
    regime_history : list of dicts with keys: date, regime, p_rv, p_vov, p_vrp
    """
    if vol_snapshot is None:
        return VolContext()

    p_rv = vol_snapshot.p_rv
    p_vov = vol_snapshot.p_vov
    p_vrp = vol_snapshot.p_vrp

    # Classify regime (same logic as Atlas)
    if p_rv < 0.4 and p_vrp > 0.5:
        regime = "sell_premium"
    elif p_rv > 0.7 or p_vov > 0.8:
        regime = "stand_down"
    elif p_rv > 0.5 and p_vrp < 0.3:
        regime = "buy_premium"
    else:
        regime = "neutral"

    # Regime persistence from history
    regime_since = vol_snapshot.date
    regime_duration_days = 0
    regime_changes_30d = 0

    if regime_history:
        # Count consecutive days of same regime (from most recent backward)
        for entry in regime_history:
            if entry["regime"] == regime:
                regime_since = entry["date"]
                regime_duration_days += 1
            else:
                break

        # Count regime changes in last 30 days
        prev_regime = None
        for entry in regime_history[:30]:
            if prev_regime is not None and entry["regime"] != prev_regime:
                regime_changes_30d += 1
            prev_regime = entry["regime"]

    # RV trend: compare current RV percentile direction
    rv_trend = "stable"
    if len(regime_history) >= 5:
        recent_rv = [e["p_rv"] for e in regime_history[:5]]
        if all(recent_rv[i] >= recent_rv[i + 1] for i in range(len(recent_rv) - 1)):
            rv_trend = "expanding"
        elif all(recent_rv[i] <= recent_rv[i + 1] for i in range(len(recent_rv) - 1)):
            rv_trend = "contracting"

    return VolContext(
        p_rv=round(p_rv, 3),
        p_vov=round(p_vov, 3),
        p_vrp=round(p_vrp, 3),
        regime=regime,
        regime_since=regime_since,
        regime_duration_days=regime_duration_days,
        regime_changes_30d=regime_changes_30d,
        rv_trend=rv_trend,
    )


def _compute_multi_day_trend(
    prior_days: list[DailyContext],
    current_week: WeeklyContext | None,
    prior_week: WeeklyContext | None,
) -> str:
    """Derive overall multi-day trend from daily + weekly data."""
    if not prior_days:
        return "neutral"

    # Score-based approach
    score = 0

    # Recent days direction
    recent = prior_days[:5]
    ups = sum(1 for d in recent if d.close_vs_open == "up")
    downs = sum(1 for d in recent if d.close_vs_open == "down")
    if ups >= 4:
        score += 2
    elif ups >= 3:
        score += 1
    elif downs >= 4:
        score -= 2
    elif downs >= 3:
        score -= 1

    # EMA alignment (last day)
    last_day = prior_days[0]
    if last_day.close_above_ema20 and last_day.close_above_ema50:
        score += 1
    elif not last_day.close_above_ema20 and not last_day.close_above_ema50:
        score -= 1

    # Weekly trend
    if current_week:
        if current_week.weekly_trend == "bullish":
            score += 1
        elif current_week.weekly_trend == "bearish":
            score -= 1

    if score >= 2:
        return "bullish"
    elif score <= -2:
        return "bearish"
    return "neutral"


def _compute_context_bias(
    session: SessionContext,
    prior_day: DailyContext | None,
    vol: VolContext,
) -> str:
    """Composite bias from session, prior day, and vol context."""
    score = 0

    # Session trend
    if session.session_trend == "trending_up":
        score += 1
    elif session.session_trend == "trending_down":
        score -= 1

    # EMA alignment
    if session.ema_alignment == "bullish":
        score += 1
    elif session.ema_alignment == "bearish":
        score -= 1

    # Prior day
    if prior_day:
        if prior_day.close_vs_open == "up" and prior_day.close_above_ema20:
            score += 1
        elif prior_day.close_vs_open == "down" and not prior_day.close_above_ema20:
            score -= 1

    # Vol regime
    if vol.regime == "sell_premium":
        score += 1  # low vol = bullish bias
    elif vol.regime == "stand_down":
        score -= 1  # high vol = bearish bias

    if score >= 2:
        return "bullish"
    elif score <= -2:
        return "bearish"
    return "neutral"


# ---------------------------------------------------------------------------
# Context Engine class
# ---------------------------------------------------------------------------

class ContextEngine:
    """Singleton engine for computing and serving multi-level market context.

    Initialize with config, call bootstrap_history() once at startup.
    Then call update_session() every refresh cycle and get_context() to
    assemble the full MarketContext for algorithms.
    """

    def __init__(self, config: dict | None = None):
        if config is None:
            config = load_config()
        self._cfg = config.get("context_engine", {})
        self._session_ctx = SessionContext()
        self._db = None  # lazy init

    def _get_db(self):
        if self._db is None:
            from core.database import SentimentDatabase
            self._db = SentimentDatabase()
        return self._db

    def bootstrap_history(self) -> int:
        """Fetch daily candles from yfinance and backfill DailyContext + WeeklyContext.

        Returns the number of new DailyContext rows created.
        """
        if not self._cfg.get("daily_context_enabled", True):
            logger.info("CONTEXT: daily_context_enabled=false, skipping bootstrap")
            return 0

        bootstrap_days = self._cfg.get("bootstrap_days", 60)
        ticker = self._cfg.get("daily_candle_ticker", "^NSEI")
        db = self._get_db()

        # Fetch daily candles
        try:
            import yfinance as yf
            from core.api_guard import yf_guard_sync

            cb = yf_guard_sync()
            try:
                df = yf.download(ticker, period=f"{bootstrap_days}d", interval="1d",
                                 auto_adjust=True, progress=False)
                cb.record_success()
            except Exception:
                cb.record_failure()
                raise
        except Exception as e:
            logger.error("CONTEXT: Failed to fetch daily candles: %s", e)
            return 0

        if df is None or df.empty:
            logger.warning("CONTEXT: No daily candle data from yfinance")
            return 0

        # Handle MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Compute indicators on the daily DataFrame
        from core.indicators import compute_ema, compute_rsi, compute_bollinger_bands

        close = df["Close"].squeeze() if isinstance(df["Close"], pd.DataFrame) else df["Close"]
        ema20_series = compute_ema(close, 20)
        ema50_series = compute_ema(close, 50)
        rsi_series = compute_rsi(close, 14)
        bb_upper, bb_middle, bb_lower = compute_bollinger_bands(close, 20)
        bb_width_series = ((bb_upper - bb_lower) / bb_middle * 100.0).fillna(0.0)

        # Volume 20-day moving average
        volume = df["Volume"].squeeze() if isinstance(df["Volume"], pd.DataFrame) else df["Volume"]
        vol_20d_avg = volume.rolling(20, min_periods=1).mean()

        # Build DailyContext for each day not already in SQLite
        new_count = 0
        prev_close = 0.0
        dates = sorted(set(df.index.date))

        for d in dates:
            date_str = str(d)

            # Check if already persisted
            existing = db.get_daily_context(date_str)
            if existing:
                prev_close = existing.close
                continue

            # Get the row data
            mask = df.index.date == d
            day_df = df[mask]
            if day_df.empty:
                continue

            idx = day_df.index[-1]  # last bar of the day (should be only bar for daily)
            row_data = {
                "Open": float(day_df["Open"].iloc[0]),
                "High": float(day_df["High"].max()),
                "Low": float(day_df["Low"].min()),
                "Close": float(day_df["Close"].iloc[-1]),
                "Volume": float(day_df["Volume"].sum()) if "Volume" in day_df.columns else 0.0,
                "date": date_str,
            }

            ema20_val = float(ema20_series.loc[idx]) if idx in ema20_series.index else 0.0
            ema50_val = float(ema50_series.loc[idx]) if idx in ema50_series.index else 0.0
            rsi_val = float(rsi_series.loc[idx]) if idx in rsi_series.index else 50.0
            bb_w_val = float(bb_width_series.loc[idx]) if idx in bb_width_series.index else 0.0
            vol_avg = float(vol_20d_avg.loc[idx]) if idx in vol_20d_avg.index else 0.0

            # Handle NaN
            if math.isnan(ema20_val):
                ema20_val = 0.0
            if math.isnan(ema50_val):
                ema50_val = 0.0
            if math.isnan(rsi_val):
                rsi_val = 50.0
            if math.isnan(bb_w_val):
                bb_w_val = 0.0
            if math.isnan(vol_avg):
                vol_avg = 0.0

            ctx = _compute_daily_context(
                row_data, ema20_val, ema50_val, rsi_val, bb_w_val, prev_close, vol_avg,
            )
            db.save_daily_context(ctx)
            prev_close = ctx.close
            new_count += 1

        logger.info("CONTEXT: Bootstrap created %d new DailyContext rows (%d total days)", new_count, len(dates))

        # Build WeeklyContext for complete weeks
        if self._cfg.get("weekly_context_enabled", True):
            self._rebuild_weekly_contexts(db)

        return new_count

    def _rebuild_weekly_contexts(self, db) -> int:
        """Build WeeklyContext for complete weeks from DailyContext data."""
        # Get all daily contexts
        all_daily = db.get_recent_daily_contexts(n=200)
        if not all_daily:
            return 0

        # Group by ISO week
        from collections import defaultdict
        weeks: dict[str, list[DailyContext]] = defaultdict(list)
        for ctx in all_daily:
            d = datetime.strptime(ctx.date, "%Y-%m-%d")
            monday = d - timedelta(days=d.weekday())
            week_key = monday.strftime("%Y-%m-%d")
            weeks[week_key].append(ctx)

        # Sort weeks
        sorted_weeks = sorted(weeks.keys())
        new_count = 0
        prior_week = None

        for week_key in sorted_weeks:
            days = weeks[week_key]
            if len(days) < 2:
                # Skip incomplete weeks (less than 2 trading days)
                continue

            existing = db.get_weekly_context(week_key)
            if existing:
                prior_week = existing
                continue

            wctx = _compute_weekly_context(days, prior_week)
            db.save_weekly_context(wctx)
            prior_week = wctx
            new_count += 1

        logger.info("CONTEXT: Built %d new WeeklyContext rows", new_count)
        return new_count

    def update_session(
        self,
        candle_df: pd.DataFrame | None = None,
        technicals=None,
        analytics=None,
        observation=None,
    ) -> None:
        """Recompute SessionContext from latest intraday data.

        Called every 60s refresh cycle. Pure computation, no persistence.
        """
        if not self._cfg.get("session_context_enabled", True):
            return

        # Convert TechnicalIndicators Pydantic model to dict for the helper
        tech_dict = None
        if technicals is not None:
            if hasattr(technicals, "model_dump"):
                tech_dict = technicals.model_dump()
            elif isinstance(technicals, dict):
                tech_dict = technicals

        self._session_ctx = _compute_session_context(candle_df, tech_dict)

    def end_of_day(self, technicals=None, analytics=None) -> None:
        """Compute and persist today's DailyContext to SQLite.

        Called when EOD is detected. Also updates WeeklyContext if it's
        the last day of the week (Friday or last trading day).
        """
        if not self._cfg.get("daily_context_enabled", True):
            return

        # We rely on bootstrap to have already populated historical data.
        # For the current day, we'll re-bootstrap to capture today's close.
        try:
            self.bootstrap_history()
        except Exception as e:
            logger.error("CONTEXT: end_of_day bootstrap failed: %s", e)

    def update_vol_context(self) -> None:
        """Update VolContext from vol_distribution and persist regime to SQLite."""
        if not self._cfg.get("vol_context_enabled", True):
            return

        try:
            from core.vol_distribution import get_today_vol_snapshot
            vol_snap = get_today_vol_snapshot(spot=0.0)
            if vol_snap is None:
                return

            db = self._get_db()

            # Classify regime
            p_rv = vol_snap.p_rv
            p_vov = vol_snap.p_vov
            p_vrp = vol_snap.p_vrp

            if p_rv < 0.4 and p_vrp > 0.5:
                regime = "sell_premium"
            elif p_rv > 0.7 or p_vov > 0.8:
                regime = "stand_down"
            elif p_rv > 0.5 and p_vrp < 0.3:
                regime = "buy_premium"
            else:
                regime = "neutral"

            db.save_vol_regime(vol_snap.date, regime, p_rv, p_vov, p_vrp)
        except Exception as e:
            logger.error("CONTEXT: Failed to update vol context: %s", e)

    def get_context(self) -> MarketContext:
        """Assemble full MarketContext from all levels.

        Called every 60s refresh to provide context to algorithms.
        """
        db = self._get_db()

        # Session context (already computed by update_session)
        session = self._session_ctx

        # Daily context from SQLite
        # Note: convert to dicts and re-validate to avoid Streamlit module
        # reload class identity mismatches (DailyContext from a previous
        # module load is a different class object than the current one).
        prior_days = []
        prior_day = None
        if self._cfg.get("daily_context_enabled", True):
            raw_days = db.get_recent_daily_contexts(n=5)
            prior_days = [
                DailyContext.model_validate(d.model_dump(mode="json"))
                for d in raw_days
            ]
            if prior_days:
                prior_day = prior_days[0]

        # Weekly context from SQLite
        current_week = None
        prior_week = None
        if self._cfg.get("weekly_context_enabled", True):
            raw_weeks = db.get_recent_weekly_contexts(n=2)
            recent_weeks = [
                WeeklyContext.model_validate(w.model_dump(mode="json"))
                for w in raw_weeks
            ]
            if len(recent_weeks) >= 1:
                current_week = recent_weeks[0]
            if len(recent_weeks) >= 2:
                prior_week = recent_weeks[1]

        # Vol context
        vol = VolContext()
        if self._cfg.get("vol_context_enabled", True):
            try:
                from core.vol_distribution import get_today_vol_snapshot
                vol_snap = get_today_vol_snapshot(spot=0.0)
                regime_history = db.get_vol_regime_history(days=30)
                vol = _compute_vol_context(vol_snap, regime_history)
            except Exception as e:
                logger.warning("CONTEXT: Failed to compute vol context: %s", e)

        # Derived fields
        multi_day_trend = _compute_multi_day_trend(prior_days, current_week, prior_week)
        context_bias = _compute_context_bias(session, prior_day, vol)

        return MarketContext(
            session=session,
            prior_day=prior_day,
            prior_days=prior_days,
            current_week=current_week,
            prior_week=prior_week,
            vol=vol,
            multi_day_trend=multi_day_trend,
            context_bias=context_bias,
        )
