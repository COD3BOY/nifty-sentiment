"""Tests for the Context Engine — models, computation, persistence, assembly."""

import json
import math
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from core.context_models import (
    DailyContext,
    MarketContext,
    SessionContext,
    VolContext,
    WeeklyContext,
)
from core.context_engine import (
    ContextEngine,
    _classify_candle,
    _compute_context_bias,
    _compute_daily_context,
    _compute_multi_day_trend,
    _compute_session_context,
    _compute_vol_context,
    _compute_weekly_context,
)


# ---------------------------------------------------------------------------
# _classify_candle
# ---------------------------------------------------------------------------

class TestClassifyCandle:
    def test_doji(self):
        # Open == Close, body is 0% of range
        assert _classify_candle(100, 105, 95, 100) == "doji"

    def test_doji_tiny_body(self):
        # Body < 10% of range
        assert _classify_candle(100, 110, 90, 100.5) == "doji"

    def test_bullish_engulf(self):
        # Strong bullish: large body (>60%), close > open
        assert _classify_candle(100, 110, 99, 109) == "bullish_engulf"

    def test_bearish_engulf(self):
        # Strong bearish: large body (>60%), close < open
        assert _classify_candle(109, 110, 99, 100) == "bearish_engulf"

    def test_hammer(self):
        # Body at top, long lower shadow (>2x body), body > 10% of range
        # Range = 20, body = 3 (15%), lower shadow = 12, upper shadow = 5
        assert _classify_candle(107, 110, 90, 110) == "hammer"

    def test_shooting_star(self):
        # Body at bottom, long upper shadow (>2x body), body > 10% of range
        # Range = 20, body = 3 (15%), upper shadow = 12, lower shadow = 5
        assert _classify_candle(93, 110, 90, 90) == "shooting_star"

    def test_neutral(self):
        # Moderate body, balanced shadows
        assert _classify_candle(100, 106, 94, 103) == "neutral"

    def test_zero_range(self):
        assert _classify_candle(100, 100, 100, 100) == "doji"

    def test_bad_input(self):
        assert _classify_candle(0, 0, 0, 0) == "neutral"


# ---------------------------------------------------------------------------
# _compute_daily_context
# ---------------------------------------------------------------------------

class TestComputeDailyContext:
    def test_basic(self):
        row = {
            "Open": 22000.0,
            "High": 22200.0,
            "Low": 21900.0,
            "Close": 22100.0,
            "Volume": 500000.0,
            "date": "2026-02-14",
        }
        ctx = _compute_daily_context(
            row, ema20=22050.0, ema50=21800.0,
            rsi=55.0, bb_width_pct=2.5,
            prev_close=21950.0, volume_20d_avg=450000.0,
        )
        assert ctx.date == "2026-02-14"
        assert ctx.close_vs_open == "up"
        assert ctx.close_above_ema20
        assert ctx.close_above_ema50
        assert ctx.rsi_close == 55.0
        assert ctx.range_pct == pytest.approx((22200 - 21900) / 22000 * 100, abs=0.01)
        assert ctx.gap_from_prev_pct == pytest.approx((22000 - 21950) / 21950 * 100, abs=0.01)
        assert ctx.volume_vs_20d_avg == pytest.approx(500000 / 450000, abs=0.01)

    def test_down_day(self):
        row = {
            "Open": 22100.0,
            "High": 22150.0,
            "Low": 21800.0,
            "Close": 21900.0,
            "Volume": 600000.0,
            "date": "2026-02-14",
        }
        ctx = _compute_daily_context(
            row, ema20=22050.0, ema50=22100.0,
            rsi=35.0, bb_width_pct=3.0,
            prev_close=22050.0, volume_20d_avg=500000.0,
        )
        assert ctx.close_vs_open == "down"
        assert not ctx.close_above_ema20
        assert not ctx.close_above_ema50

    def test_flat_day(self):
        row = {
            "Open": 22000.0,
            "High": 22050.0,
            "Low": 21980.0,
            "Close": 22000.0,
            "Volume": 300000.0,
            "date": "2026-02-14",
        }
        ctx = _compute_daily_context(
            row, ema20=0.0, ema50=0.0,
            rsi=50.0, bb_width_pct=1.0,
            prev_close=22000.0, volume_20d_avg=0.0,
        )
        assert ctx.close_vs_open == "flat"
        assert not ctx.close_above_ema20
        assert ctx.volume_vs_20d_avg == 1.0  # fallback when avg=0

    def test_candle_type_assigned(self):
        row = {
            "Open": 100.0, "High": 110.0, "Low": 99.0, "Close": 109.0,
            "Volume": 100, "date": "2026-01-01",
        }
        ctx = _compute_daily_context(row, 0, 0, 50, 0, 0, 0)
        assert ctx.candle_type == "bullish_engulf"


# ---------------------------------------------------------------------------
# _compute_weekly_context
# ---------------------------------------------------------------------------

class TestComputeWeeklyContext:
    def _make_days(self):
        return [
            DailyContext(date="2026-02-10", open=22000, high=22200, low=21900, close=22100,
                         close_vs_open="up", ema20=21900, range_pct=1.36),
            DailyContext(date="2026-02-11", open=22100, high=22300, low=22000, close=22250,
                         close_vs_open="up", ema20=21950, range_pct=1.36),
            DailyContext(date="2026-02-12", open=22250, high=22400, low=22100, close=22150,
                         close_vs_open="down", ema20=22000, range_pct=1.33),
            DailyContext(date="2026-02-13", open=22150, high=22350, low=22050, close=22300,
                         close_vs_open="up", ema20=22050, range_pct=1.35),
            DailyContext(date="2026-02-14", open=22300, high=22500, low=22200, close=22450,
                         close_vs_open="up", ema20=22100, range_pct=1.35),
        ]

    def test_basic_weekly(self):
        days = self._make_days()
        wctx = _compute_weekly_context(days)
        assert wctx.week_start == "2026-02-09"  # Monday
        assert wctx.week_end == "2026-02-14"
        assert wctx.week_open == 22000
        assert wctx.week_close == 22450
        assert wctx.week_high == 22500
        assert wctx.week_low == 21900
        assert wctx.days_up == 4
        assert wctx.days_down == 1
        assert wctx.weekly_trend == "bullish"
        assert wctx.ema20_slope == "rising"

    def test_bearish_week(self):
        days = [
            DailyContext(date="2026-02-10", open=22500, high=22550, low=22300, close=22350,
                         close_vs_open="down", ema20=22400, range_pct=1.0),
            DailyContext(date="2026-02-11", open=22350, high=22400, low=22100, close=22150,
                         close_vs_open="down", ema20=22350, range_pct=1.3),
            DailyContext(date="2026-02-12", open=22150, high=22200, low=21900, close=21950,
                         close_vs_open="down", ema20=22300, range_pct=1.35),
        ]
        wctx = _compute_weekly_context(days)
        assert wctx.weekly_trend == "bearish"
        assert wctx.days_down == 3
        assert wctx.ema20_slope == "falling"

    def test_vs_prior_higher_high(self):
        prior = WeeklyContext(week_start="2026-02-03", week_end="2026-02-07",
                              week_high=22400, week_low=21800)  # low 21800 < week low 21900
        days = self._make_days()
        wctx = _compute_weekly_context(days, prior_week=prior)
        assert wctx.week_vs_prior == "higher_high"  # 22500 > 22400, 21900 > 21800

    def test_vs_prior_lower_low(self):
        prior = WeeklyContext(week_start="2026-02-03", week_end="2026-02-07",
                              week_high=22600, week_low=22000)
        days = [
            DailyContext(date="2026-02-10", open=22000, high=22100, low=21800, close=21900,
                         close_vs_open="down", ema20=22000, range_pct=1.0),
        ]
        wctx = _compute_weekly_context(days, prior_week=prior)
        assert wctx.week_vs_prior == "lower_low"

    def test_empty_days(self):
        wctx = _compute_weekly_context([])
        assert wctx.week_start == ""


# ---------------------------------------------------------------------------
# _compute_session_context
# ---------------------------------------------------------------------------

class TestComputeSessionContext:
    def _make_df(self):
        """Build a small 5-min candle DF for today."""
        today = pd.Timestamp.now().normalize()
        times = [today + pd.Timedelta(hours=9, minutes=15 + i * 5) for i in range(10)]
        data = {
            "Open": [22000 + i * 10 for i in range(10)],
            "High": [22010 + i * 10 for i in range(10)],
            "Low": [21990 + i * 10 for i in range(10)],
            "Close": [22005 + i * 10 for i in range(10)],
            "Volume": [10000] * 10,
        }
        return pd.DataFrame(data, index=pd.DatetimeIndex(times))

    def test_basic_session(self):
        df = self._make_df()
        tech = {"spot": 22095.0, "vwap": 22050.0, "rsi": 65.0,
                "ema_9": 22080.0, "ema_20": 22060.0, "ema_50": 22040.0,
                "bb_upper": 22150.0, "bb_lower": 21950.0}
        ctx = _compute_session_context(df, tech)
        assert ctx.bars_elapsed == 10
        assert ctx.session_open == 22000.0
        assert ctx.ema_alignment == "bullish"
        assert ctx.rsi_trajectory == "rising"

    def test_empty_df(self):
        ctx = _compute_session_context(pd.DataFrame())
        assert ctx.bars_elapsed == 0
        assert ctx.session_trend == "range_bound"

    def test_none_technicals(self):
        df = self._make_df()
        ctx = _compute_session_context(df, None)
        assert ctx.bars_elapsed == 10
        assert ctx.ema_alignment == "mixed"

    def test_bearish_alignment(self):
        df = self._make_df()
        tech = {"spot": 22050.0, "vwap": 22100.0, "rsi": 35.0,
                "ema_9": 22020.0, "ema_20": 22050.0, "ema_50": 22080.0,
                "bb_upper": 22200.0, "bb_lower": 21900.0}
        ctx = _compute_session_context(df, tech)
        assert ctx.ema_alignment == "bearish"
        assert ctx.rsi_trajectory == "falling"

    def test_bb_upper_position(self):
        df = self._make_df()
        tech = {"spot": 22140.0, "vwap": 22050.0, "rsi": 50.0,
                "ema_9": 0.0, "ema_20": 0.0, "ema_50": 0.0,
                "bb_upper": 22150.0, "bb_lower": 21950.0}
        ctx = _compute_session_context(df, tech)
        assert ctx.bb_position == "upper"


# ---------------------------------------------------------------------------
# _compute_vol_context
# ---------------------------------------------------------------------------

class TestComputeVolContext:
    def test_sell_premium(self):
        snap = MagicMock(p_rv=0.3, p_vov=0.4, p_vrp=0.7, date="2026-02-14")
        ctx = _compute_vol_context(snap, [])
        assert ctx.regime == "sell_premium"
        assert ctx.p_rv == 0.3

    def test_stand_down(self):
        snap = MagicMock(p_rv=0.8, p_vov=0.5, p_vrp=0.4, date="2026-02-14")
        ctx = _compute_vol_context(snap, [])
        assert ctx.regime == "stand_down"

    def test_buy_premium(self):
        snap = MagicMock(p_rv=0.6, p_vov=0.4, p_vrp=0.2, date="2026-02-14")
        ctx = _compute_vol_context(snap, [])
        assert ctx.regime == "buy_premium"

    def test_neutral(self):
        snap = MagicMock(p_rv=0.45, p_vov=0.5, p_vrp=0.45, date="2026-02-14")
        ctx = _compute_vol_context(snap, [])
        assert ctx.regime == "neutral"

    def test_none_snapshot(self):
        ctx = _compute_vol_context(None, [])
        assert ctx.regime == "neutral"
        assert ctx.p_rv == 0.5

    def test_regime_persistence(self):
        snap = MagicMock(p_rv=0.3, p_vov=0.4, p_vrp=0.7, date="2026-02-14")
        history = [
            {"date": "2026-02-14", "regime": "sell_premium", "p_rv": 0.3, "p_vov": 0.4, "p_vrp": 0.7},
            {"date": "2026-02-13", "regime": "sell_premium", "p_rv": 0.32, "p_vov": 0.38, "p_vrp": 0.68},
            {"date": "2026-02-12", "regime": "sell_premium", "p_rv": 0.35, "p_vov": 0.42, "p_vrp": 0.65},
            {"date": "2026-02-11", "regime": "neutral", "p_rv": 0.45, "p_vov": 0.5, "p_vrp": 0.45},
        ]
        ctx = _compute_vol_context(snap, history)
        assert ctx.regime_duration_days == 3
        assert ctx.regime_since == "2026-02-12"

    def test_regime_changes(self):
        snap = MagicMock(p_rv=0.3, p_vov=0.4, p_vrp=0.7, date="2026-02-14")
        history = [
            {"date": "2026-02-14", "regime": "sell_premium", "p_rv": 0.3, "p_vov": 0.4, "p_vrp": 0.7},
            {"date": "2026-02-13", "regime": "neutral", "p_rv": 0.45, "p_vov": 0.5, "p_vrp": 0.45},
            {"date": "2026-02-12", "regime": "sell_premium", "p_rv": 0.3, "p_vov": 0.4, "p_vrp": 0.7},
        ]
        ctx = _compute_vol_context(snap, history)
        assert ctx.regime_changes_30d == 2


# ---------------------------------------------------------------------------
# _compute_multi_day_trend
# ---------------------------------------------------------------------------

class TestComputeMultiDayTrend:
    def test_bullish(self):
        days = [
            DailyContext(date=f"2026-02-1{i}", close_vs_open="up",
                         close_above_ema20=True, close_above_ema50=True)
            for i in range(5)
        ]
        assert _compute_multi_day_trend(days, None, None) == "bullish"

    def test_bearish(self):
        days = [
            DailyContext(date=f"2026-02-1{i}", close_vs_open="down",
                         close_above_ema20=False, close_above_ema50=False)
            for i in range(5)
        ]
        assert _compute_multi_day_trend(days, None, None) == "bearish"

    def test_neutral(self):
        days = [
            DailyContext(date="2026-02-14", close_vs_open="up",
                         close_above_ema20=True, close_above_ema50=False),
            DailyContext(date="2026-02-13", close_vs_open="down",
                         close_above_ema20=False, close_above_ema50=True),
        ]
        assert _compute_multi_day_trend(days, None, None) == "neutral"

    def test_empty(self):
        assert _compute_multi_day_trend([], None, None) == "neutral"


# ---------------------------------------------------------------------------
# _compute_context_bias
# ---------------------------------------------------------------------------

class TestComputeContextBias:
    def test_bullish(self):
        session = SessionContext(session_trend="trending_up", ema_alignment="bullish")
        prior = DailyContext(date="2026-02-14", close_vs_open="up", close_above_ema20=True)
        vol = VolContext(regime="sell_premium")
        assert _compute_context_bias(session, prior, vol) == "bullish"

    def test_bearish(self):
        session = SessionContext(session_trend="trending_down", ema_alignment="bearish")
        prior = DailyContext(date="2026-02-14", close_vs_open="down", close_above_ema20=False)
        vol = VolContext(regime="stand_down")
        assert _compute_context_bias(session, prior, vol) == "bearish"

    def test_neutral(self):
        session = SessionContext()
        assert _compute_context_bias(session, None, VolContext()) == "neutral"


# ---------------------------------------------------------------------------
# ContextEngine — bootstrap + persistence
# ---------------------------------------------------------------------------

class TestContextEngineBootstrap:
    @patch("core.context_engine.ContextEngine._get_db")
    def test_bootstrap_creates_daily_contexts(self, mock_get_db):
        """Bootstrap should create DailyContext rows from yfinance data."""
        # Build a mock daily candle DataFrame
        dates = pd.bdate_range("2026-01-01", periods=10)
        df = pd.DataFrame({
            "Open": [22000 + i * 50 for i in range(10)],
            "High": [22050 + i * 50 for i in range(10)],
            "Low": [21950 + i * 50 for i in range(10)],
            "Close": [22025 + i * 50 for i in range(10)],
            "Volume": [500000] * 10,
        }, index=dates)

        mock_db = MagicMock()
        mock_db.get_daily_context.return_value = None  # nothing persisted yet
        mock_db.get_weekly_context.return_value = None
        mock_db.get_recent_daily_contexts.return_value = []
        mock_get_db.return_value = mock_db

        engine = ContextEngine({"context_engine": {"bootstrap_days": 10}})

        import yfinance as yf
        with patch.object(yf, "download", return_value=df), \
             patch("core.api_guard.yf_guard_sync") as mock_guard:
            mock_cb = MagicMock()
            mock_guard.return_value = mock_cb

            count = engine.bootstrap_history()

        assert count == 10
        assert mock_db.save_daily_context.call_count == 10

    @patch("core.context_engine.ContextEngine._get_db")
    def test_bootstrap_skips_existing(self, mock_get_db):
        """Bootstrap should skip days already in SQLite."""
        dates = pd.bdate_range("2026-01-01", periods=5)
        df = pd.DataFrame({
            "Open": [22000] * 5,
            "High": [22050] * 5,
            "Low": [21950] * 5,
            "Close": [22025] * 5,
            "Volume": [500000] * 5,
        }, index=dates)

        existing = DailyContext(date="2026-01-01", close=22025.0)
        mock_db = MagicMock()
        mock_db.get_daily_context.side_effect = lambda d: existing if d == "2026-01-01" else None
        mock_db.get_weekly_context.return_value = None
        mock_db.get_recent_daily_contexts.return_value = []
        mock_get_db.return_value = mock_db

        engine = ContextEngine({"context_engine": {"bootstrap_days": 5}})

        import yfinance as yf
        with patch.object(yf, "download", return_value=df), \
             patch("core.api_guard.yf_guard_sync") as mock_guard:
            mock_cb = MagicMock()
            mock_guard.return_value = mock_cb

            count = engine.bootstrap_history()

        assert count == 4  # 5 - 1 existing
        assert mock_db.save_daily_context.call_count == 4

    def test_bootstrap_disabled(self):
        engine = ContextEngine({"context_engine": {"daily_context_enabled": False}})
        assert engine.bootstrap_history() == 0


# ---------------------------------------------------------------------------
# ContextEngine — get_context assembly
# ---------------------------------------------------------------------------

class TestGetContext:
    @patch("core.context_engine.ContextEngine._get_db")
    @patch("core.vol_distribution.get_today_vol_snapshot")
    def test_assembly(self, mock_vol_snap, mock_get_db):
        """get_context should assemble from all levels."""
        mock_db = MagicMock()
        prior_day = DailyContext(date="2026-02-14", close=22100, close_vs_open="up",
                                 close_above_ema20=True, close_above_ema50=True)
        mock_db.get_recent_daily_contexts.return_value = [prior_day]
        mock_db.get_recent_weekly_contexts.return_value = [
            WeeklyContext(week_start="2026-02-10", week_end="2026-02-14",
                          weekly_trend="bullish", weekly_change_pct=1.5)
        ]
        mock_db.get_vol_regime_history.return_value = []
        mock_get_db.return_value = mock_db

        mock_vol_snap.return_value = MagicMock(
            p_rv=0.3, p_vov=0.4, p_vrp=0.7, date="2026-02-14",
        )

        engine = ContextEngine()
        engine._session_ctx = SessionContext(session_trend="trending_up", ema_alignment="bullish")

        ctx = engine.get_context()

        assert isinstance(ctx, MarketContext)
        assert ctx.prior_day.date == "2026-02-14"
        assert ctx.current_week.weekly_trend == "bullish"
        assert ctx.vol.regime == "sell_premium"
        assert ctx.session.ema_alignment == "bullish"

    @patch("core.context_engine.ContextEngine._get_db")
    def test_assembly_no_data(self, mock_get_db):
        """get_context should return defaults when no data is available."""
        mock_db = MagicMock()
        mock_db.get_recent_daily_contexts.return_value = []
        mock_db.get_recent_weekly_contexts.return_value = []
        mock_db.get_vol_regime_history.return_value = []
        mock_get_db.return_value = mock_db

        engine = ContextEngine({"context_engine": {"vol_context_enabled": False}})
        ctx = engine.get_context()

        assert ctx.prior_day is None
        assert ctx.current_week is None
        assert ctx.vol.regime == "neutral"
        assert ctx.context_bias == "neutral"


# ---------------------------------------------------------------------------
# ContextEngine — update_session
# ---------------------------------------------------------------------------

class TestUpdateSession:
    def test_update_session(self):
        engine = ContextEngine()
        today = pd.Timestamp.now().normalize()
        times = [today + pd.Timedelta(hours=9, minutes=15 + i * 5) for i in range(5)]
        df = pd.DataFrame({
            "Open": [22000, 22010, 22020, 22030, 22040],
            "High": [22010, 22020, 22030, 22040, 22050],
            "Low": [21990, 22000, 22010, 22020, 22030],
            "Close": [22005, 22015, 22025, 22035, 22045],
            "Volume": [10000] * 5,
        }, index=pd.DatetimeIndex(times))

        engine.update_session(candle_df=df)
        assert engine._session_ctx.bars_elapsed == 5
        assert engine._session_ctx.session_open == 22000.0

    def test_update_session_disabled(self):
        engine = ContextEngine({"context_engine": {"session_context_enabled": False}})
        engine.update_session(candle_df=pd.DataFrame())
        assert engine._session_ctx.bars_elapsed == 0  # unchanged


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompat:
    def test_algorithms_accept_context_none(self):
        """All algorithm signatures accept context=None without error."""
        from algorithms.base import TradingAlgorithm

        # Just verify the ABC signature has context parameter
        import inspect
        sig_gen = inspect.signature(TradingAlgorithm.generate_suggestions)
        sig_eval = inspect.signature(TradingAlgorithm.evaluate_and_manage)

        assert "context" in sig_gen.parameters
        assert sig_gen.parameters["context"].default is None

        assert "context" in sig_eval.parameters
        assert sig_eval.parameters["context"].default is None

    def test_market_context_defaults(self):
        """MarketContext with all defaults should be valid."""
        ctx = MarketContext()
        assert ctx.prior_day is None
        assert ctx.prior_days == []
        assert ctx.vol.regime == "neutral"
        assert ctx.context_bias == "neutral"


# ---------------------------------------------------------------------------
# Model serialization
# ---------------------------------------------------------------------------

class TestModelSerialization:
    def test_daily_context_json_roundtrip(self):
        ctx = DailyContext(
            date="2026-02-14", open=22000, high=22200, low=21900, close=22100,
            close_vs_open="up", candle_type="bullish_engulf",
        )
        data = ctx.model_dump(mode="json")
        restored = DailyContext.model_validate(data)
        assert restored.date == "2026-02-14"
        assert restored.candle_type == "bullish_engulf"

    def test_weekly_context_json_roundtrip(self):
        ctx = WeeklyContext(week_start="2026-02-10", week_end="2026-02-14",
                            weekly_trend="bullish", weekly_change_pct=1.5)
        data = ctx.model_dump(mode="json")
        restored = WeeklyContext.model_validate(data)
        assert restored.weekly_trend == "bullish"

    def test_market_context_json_roundtrip(self):
        ctx = MarketContext(
            session=SessionContext(session_trend="trending_up"),
            prior_day=DailyContext(date="2026-02-14"),
            vol=VolContext(regime="sell_premium"),
            context_bias="bullish",
        )
        data = ctx.model_dump(mode="json")
        restored = MarketContext.model_validate(data)
        assert restored.context_bias == "bullish"
        assert restored.vol.regime == "sell_premium"


# ---------------------------------------------------------------------------
# Database persistence (integration-style tests with mock)
# ---------------------------------------------------------------------------

class TestDatabasePersistence:
    @patch("core.context_engine.ContextEngine._get_db")
    def test_end_of_day_calls_bootstrap(self, mock_get_db):
        """end_of_day should re-run bootstrap to capture today's close."""
        engine = ContextEngine()
        with patch.object(engine, "bootstrap_history") as mock_bootstrap:
            engine.end_of_day()
            mock_bootstrap.assert_called_once()

    @patch("core.context_engine.ContextEngine._get_db")
    def test_update_vol_context_persists(self, mock_get_db):
        """update_vol_context should save regime to database."""
        mock_db = MagicMock()
        mock_get_db.return_value = mock_db

        engine = ContextEngine()

        with patch("core.vol_distribution.get_today_vol_snapshot") as mock_vol:
            mock_vol.return_value = MagicMock(
                p_rv=0.3, p_vov=0.4, p_vrp=0.7, date="2026-02-14",
            )
            engine.update_vol_context()

        mock_db.save_vol_regime.assert_called_once_with(
            "2026-02-14", "sell_premium", 0.3, 0.4, 0.7,
        )
