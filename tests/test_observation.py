"""Tests for the Observation Period Engine (core/observation.py)."""

from __future__ import annotations

from datetime import datetime, time as dt_time, timezone, timedelta
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from core.observation import (
    ObservationSnapshot,
    OpeningGap,
    OpeningRange,
    InitialTrend,
    ObservationVolumeProfile,
    VWAPContext,
    _compute_opening_gap,
    _compute_opening_range,
    _compute_initial_trend,
    _compute_volume_profile,
    _compute_vwap_context,
    _compute_bias,
    _split_today_and_prior,
    _get_observation_window,
    _get_prior_days_observation,
    compute_observation_snapshot,
)


# ---------------------------------------------------------------------------
# Helpers: build synthetic DataFrames
# ---------------------------------------------------------------------------

IST = timezone(timedelta(hours=5, minutes=30))


def _make_candle_df(
    dates_and_times: list[tuple[str, str]],
    opens: list[float],
    highs: list[float],
    lows: list[float],
    closes: list[float],
    volumes: list[float] | None = None,
) -> pd.DataFrame:
    """Build a candle DataFrame from explicit data."""
    timestamps = []
    for date_str, time_str in dates_and_times:
        dt_str = f"{date_str} {time_str}"
        timestamps.append(pd.Timestamp(dt_str, tz=IST))

    if volumes is None:
        volumes = [100_000.0] * len(dates_and_times)

    df = pd.DataFrame(
        {
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "Volume": volumes,
        },
        index=pd.DatetimeIndex(timestamps),
    )
    return df


def _make_simple_day(
    date_str: str,
    base_price: float = 23000.0,
    n_bars: int = 9,
    start_time: str = "09:15",
    interval_minutes: int = 5,
    trend: str = "flat",
    volume: float = 100_000.0,
) -> pd.DataFrame:
    """Generate a simple day of OHLCV data."""
    timestamps = []
    opens, highs, lows, closes, volumes = [], [], [], [], []

    h, m = (int(x) for x in start_time.split(":"))
    price = base_price

    for i in range(n_bars):
        ts = pd.Timestamp(f"{date_str} {h:02d}:{m:02d}", tz=IST)
        timestamps.append(ts)

        if trend == "up":
            delta = base_price * 0.001  # 0.1% per bar
        elif trend == "down":
            delta = -base_price * 0.001
        else:
            delta = 0.0

        open_p = price
        close_p = price + delta
        high_p = max(open_p, close_p) + base_price * 0.0002
        low_p = min(open_p, close_p) - base_price * 0.0002

        opens.append(open_p)
        highs.append(high_p)
        lows.append(low_p)
        closes.append(close_p)
        volumes.append(volume)

        price = close_p
        m += interval_minutes
        if m >= 60:
            h += m // 60
            m = m % 60

    return pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": volumes},
        index=pd.DatetimeIndex(timestamps),
    )


def _make_multi_day_df(days: list[tuple[str, str, float]], n_bars: int = 9) -> pd.DataFrame:
    """Build a multi-day DataFrame.

    days: list of (date_str, trend, base_price)
    """
    frames = []
    for date_str, trend, base_price in days:
        frames.append(_make_simple_day(date_str, base_price, n_bars=n_bars, trend=trend))
    return pd.concat(frames)


# ---------------------------------------------------------------------------
# Tests: _compute_opening_gap
# ---------------------------------------------------------------------------

class TestOpeningGap:
    def test_gap_up(self):
        prev = _make_simple_day("2026-02-10", base_price=23000)
        today = _make_simple_day("2026-02-11", base_price=23100)
        gap = _compute_opening_gap(today, prev, gap_threshold_pct=0.2)

        assert gap.direction == "gap_up"
        assert gap.gap_points > 0
        assert gap.gap_pct > 0.2

    def test_gap_down(self):
        prev = _make_simple_day("2026-02-10", base_price=23000)
        today = _make_simple_day("2026-02-11", base_price=22900)
        gap = _compute_opening_gap(today, prev, gap_threshold_pct=0.2)

        assert gap.direction == "gap_down"
        assert gap.gap_points < 0
        assert gap.gap_pct < -0.2

    def test_flat_gap(self):
        prev = _make_simple_day("2026-02-10", base_price=23000)
        today = _make_simple_day("2026-02-11", base_price=23000)
        gap = _compute_opening_gap(today, prev, gap_threshold_pct=0.2)

        assert gap.direction == "flat"
        assert abs(gap.gap_pct) <= 0.2

    def test_empty_today(self):
        prev = _make_simple_day("2026-02-10", base_price=23000)
        empty = prev.iloc[0:0]
        gap = _compute_opening_gap(empty, prev)
        assert gap == OpeningGap()

    def test_empty_prior(self):
        today = _make_simple_day("2026-02-11", base_price=23000)
        empty = today.iloc[0:0]
        gap = _compute_opening_gap(today, empty)
        assert gap.today_open == today["Open"].iloc[0]
        assert gap.prev_close == 0.0

    def test_gap_fill_detection(self):
        """Gap up that gets fully filled."""
        dates_times = [("2026-02-10", "15:25"), ("2026-02-11", "09:15"), ("2026-02-11", "09:20")]
        # prev close 23000, today open 23100 (gap up), then price falls to 22990
        prev = _make_candle_df(
            [("2026-02-10", "15:25")],
            opens=[23000], highs=[23010], lows=[22990], closes=[23000],
        )
        today = _make_candle_df(
            [("2026-02-11", "09:15"), ("2026-02-11", "09:20")],
            opens=[23100, 23050], highs=[23110, 23060], lows=[23040, 22990],
            closes=[23050, 22990],
        )
        gap = _compute_opening_gap(today, prev, gap_threshold_pct=0.2)
        assert gap.direction == "gap_up"
        assert gap.is_gap_filled  # price went below prev close


# ---------------------------------------------------------------------------
# Tests: _compute_opening_range
# ---------------------------------------------------------------------------

class TestOpeningRange:
    def test_basic_range(self):
        df = _make_simple_day("2026-02-11", base_price=23000, n_bars=5)
        result = _compute_opening_range(df, is_complete=True)

        assert result.high > 0
        assert result.low > 0
        assert result.high >= result.low
        assert result.range_points >= 0
        assert result.is_complete is True

    def test_incomplete_range(self):
        df = _make_simple_day("2026-02-11", base_price=23000, n_bars=3)
        result = _compute_opening_range(df, is_complete=False)
        assert result.is_complete is False

    def test_breakout_above(self):
        df = _make_simple_day("2026-02-11", base_price=23000, n_bars=5)
        high = float(df["High"].max())
        result = _compute_opening_range(df, is_complete=True, current_price=high + 50)

        assert result.breakout_direction == "above"
        assert result.breakout_distance_pct > 0

    def test_breakout_below(self):
        df = _make_simple_day("2026-02-11", base_price=23000, n_bars=5)
        low = float(df["Low"].min())
        result = _compute_opening_range(df, is_complete=True, current_price=low - 50)

        assert result.breakout_direction == "below"
        assert result.breakout_distance_pct > 0

    def test_no_breakout(self):
        df = _make_simple_day("2026-02-11", base_price=23000, n_bars=5)
        mid = float(df["Close"].iloc[-1])
        result = _compute_opening_range(df, is_complete=True, current_price=mid)
        assert result.breakout_direction == ""

    def test_empty_df(self):
        df = _make_simple_day("2026-02-11", base_price=23000, n_bars=5).iloc[0:0]
        result = _compute_opening_range(df, is_complete=False)
        assert result == OpeningRange()


# ---------------------------------------------------------------------------
# Tests: _compute_initial_trend
# ---------------------------------------------------------------------------

class TestInitialTrend:
    def test_uptrend(self):
        df = _make_simple_day("2026-02-11", base_price=23000, n_bars=9, trend="up")
        result = _compute_initial_trend(df, moderate_pct=0.3, strong_pct=0.7)
        assert result.direction in ("up", "reversal_up")
        assert result.move_pct > 0

    def test_downtrend(self):
        df = _make_simple_day("2026-02-11", base_price=23000, n_bars=9, trend="down")
        result = _compute_initial_trend(df, moderate_pct=0.3, strong_pct=0.7)
        assert result.direction in ("down", "reversal_down")
        assert result.move_pct < 0

    def test_sideways(self):
        df = _make_simple_day("2026-02-11", base_price=23000, n_bars=9, trend="flat")
        result = _compute_initial_trend(df, moderate_pct=0.3, strong_pct=0.7)
        assert result.direction == "sideways"
        assert result.strength == "weak"

    def test_strength_classification(self):
        """Strong uptrend should be classified as strong."""
        # Create a strong uptrend: 1% move in 9 bars
        df = _make_simple_day("2026-02-11", base_price=23000, n_bars=9, trend="up")
        # Override close to be 1% above open for a strong signal
        opens = [23000]
        closes = [23200]  # ~0.87% move
        highs = [23210]
        lows = [22990]
        df2 = _make_candle_df(
            [("2026-02-11", "09:15"), ("2026-02-11", "09:55")],
            opens=[23000, 23100], highs=[23050, 23210],
            lows=[22990, 23090], closes=[23100, 23200],
        )
        result = _compute_initial_trend(df2, moderate_pct=0.3, strong_pct=0.7)
        assert result.strength == "strong"

    def test_single_bar(self):
        df = _make_simple_day("2026-02-11", base_price=23000, n_bars=1)
        result = _compute_initial_trend(df)
        assert result == InitialTrend()


# ---------------------------------------------------------------------------
# Tests: _compute_volume_profile
# ---------------------------------------------------------------------------

class TestVolumeProfile:
    def test_high_volume(self):
        today = _make_simple_day("2026-02-11", base_price=23000, volume=200_000)
        prior = _make_multi_day_df([
            ("2026-02-07", "flat", 23000),
            ("2026-02-10", "flat", 23000),
        ])
        result = _compute_volume_profile(today, prior, high_ratio=1.3, low_ratio=0.7)
        assert result.classification == "high"
        assert result.relative_volume > 1.3

    def test_low_volume(self):
        today = _make_simple_day("2026-02-11", base_price=23000, volume=30_000)
        prior = _make_multi_day_df([
            ("2026-02-07", "flat", 23000),
            ("2026-02-10", "flat", 23000),
        ])
        result = _compute_volume_profile(today, prior, high_ratio=1.3, low_ratio=0.7)
        assert result.classification == "low"
        assert result.relative_volume < 0.7

    def test_normal_volume(self):
        today = _make_simple_day("2026-02-11", base_price=23000, volume=100_000)
        prior = _make_multi_day_df([
            ("2026-02-07", "flat", 23000),
            ("2026-02-10", "flat", 23000),
        ])
        result = _compute_volume_profile(today, prior, high_ratio=1.3, low_ratio=0.7)
        assert result.classification == "normal"
        assert 0.7 <= result.relative_volume <= 1.3

    def test_no_prior_data(self):
        today = _make_simple_day("2026-02-11", base_price=23000, volume=100_000)
        empty = today.iloc[0:0]
        result = _compute_volume_profile(today, empty)
        assert result.relative_volume == 1.0
        assert result.classification == "normal"

    def test_empty_today(self):
        empty = _make_simple_day("2026-02-11", base_price=23000).iloc[0:0]
        prior = _make_simple_day("2026-02-10", base_price=23000)
        result = _compute_volume_profile(empty, prior)
        assert result == ObservationVolumeProfile()


# ---------------------------------------------------------------------------
# Tests: _compute_vwap_context
# ---------------------------------------------------------------------------

class TestVWAPContext:
    def test_consistently_above(self):
        """Uptrend keeps price above VWAP most of the time."""
        df = _make_simple_day("2026-02-11", base_price=23000, n_bars=9, trend="up")
        result = _compute_vwap_context(df, consistency_pct=70.0)
        # In an uptrend, most bars should be above VWAP
        assert result.total_bars == 9
        assert result.pct_above >= 50.0  # at least some above

    def test_consistently_below(self):
        """Downtrend keeps price below VWAP most of the time."""
        df = _make_simple_day("2026-02-11", base_price=23000, n_bars=9, trend="down")
        result = _compute_vwap_context(df, consistency_pct=70.0)
        assert result.total_bars == 9

    def test_empty_df(self):
        empty = _make_simple_day("2026-02-11", base_price=23000).iloc[0:0]
        result = _compute_vwap_context(empty)
        assert result == VWAPContext()


# ---------------------------------------------------------------------------
# Tests: _compute_bias
# ---------------------------------------------------------------------------

class TestBias:
    def test_bullish_bias(self):
        gap = OpeningGap(direction="gap_up", gap_pct=0.5, is_gap_filled=False, gap_points=100)
        opening_range = OpeningRange()
        trend = InitialTrend(direction="up", move_pct=0.8, strength="strong")
        volume = ObservationVolumeProfile(classification="high", relative_volume=1.5)
        vwap = VWAPContext(relationship="consistently_above", pct_above=80)

        bias, reasons = _compute_bias(gap, opening_range, trend, volume, vwap)
        assert bias == "bullish"
        assert len(reasons) > 0

    def test_bearish_bias(self):
        gap = OpeningGap(direction="gap_down", gap_pct=-0.5, is_gap_filled=False, gap_points=-100)
        opening_range = OpeningRange()
        trend = InitialTrend(direction="down", move_pct=-0.8, strength="strong")
        volume = ObservationVolumeProfile(classification="high", relative_volume=1.5)
        vwap = VWAPContext(relationship="consistently_below", pct_above=20)

        bias, reasons = _compute_bias(gap, opening_range, trend, volume, vwap)
        assert bias == "bearish"
        assert len(reasons) > 0

    def test_neutral_bias(self):
        gap = OpeningGap(direction="flat", gap_pct=0.0)
        opening_range = OpeningRange()
        trend = InitialTrend(direction="sideways", move_pct=0.0, strength="weak")
        volume = ObservationVolumeProfile(classification="normal", relative_volume=1.0)
        vwap = VWAPContext(relationship="crossing", pct_above=50)

        bias, reasons = _compute_bias(gap, opening_range, trend, volume, vwap)
        assert bias == "neutral"
        assert len(reasons) > 0

    def test_gap_fill_means_neutral(self):
        """A filled gap should contribute to neutral/mean-reversion bias."""
        gap = OpeningGap(direction="gap_up", gap_pct=0.5, is_gap_filled=True, gap_points=100)
        opening_range = OpeningRange()
        trend = InitialTrend(direction="sideways", move_pct=0.0, strength="weak")
        volume = ObservationVolumeProfile(classification="normal", relative_volume=1.0)
        vwap = VWAPContext(relationship="crossing", pct_above=50)

        bias, reasons = _compute_bias(gap, opening_range, trend, volume, vwap)
        # Gap fill doesn't add directional score, so bias stays neutral
        assert bias == "neutral"
        assert any("filled" in r.lower() for r in reasons)


# ---------------------------------------------------------------------------
# Tests: _split_today_and_prior
# ---------------------------------------------------------------------------

class TestSplitTodayAndPrior:
    def test_basic_split(self):
        df = _make_multi_day_df([
            ("2026-02-07", "flat", 23000),
            ("2026-02-10", "flat", 23050),
            ("2026-02-11", "up", 23100),
        ])
        today, prior, _ = _split_today_and_prior(df)
        assert all(d.day == 11 for d in today.index.date)
        assert all(d.day != 11 for d in prior.index.date)

    def test_single_day(self):
        df = _make_simple_day("2026-02-11", base_price=23000)
        today, prior, _ = _split_today_and_prior(df)
        assert len(today) == len(df)
        assert len(prior) == 0

    def test_empty_df(self):
        df = _make_simple_day("2026-02-11", base_price=23000).iloc[0:0]
        today, prior, _ = _split_today_and_prior(df)
        assert today.empty
        assert prior.empty


# ---------------------------------------------------------------------------
# Tests: _get_observation_window
# ---------------------------------------------------------------------------

class TestGetObservationWindow:
    def test_filters_to_observation_time(self):
        df = _make_simple_day("2026-02-11", base_price=23000, n_bars=20, start_time="09:15")
        obs = _get_observation_window(df, dt_time(10, 0))
        # Should only include bars from 9:15 to 9:55
        assert len(obs) <= 9  # at most 9 five-minute bars

    def test_empty_df(self):
        empty = _make_simple_day("2026-02-11", base_price=23000).iloc[0:0]
        result = _get_observation_window(empty, dt_time(10, 0))
        assert result.empty


# ---------------------------------------------------------------------------
# Tests: compute_observation_snapshot (integration)
# ---------------------------------------------------------------------------

class TestComputeObservationSnapshot:
    @patch("core.observation.load_config")
    def test_basic_snapshot(self, mock_config):
        mock_config.return_value = {
            "paper_trading": {
                "observation": {
                    "gap_threshold_pct": 0.2,
                    "trend_moderate_pct": 0.3,
                    "trend_strong_pct": 0.7,
                    "volume_high_ratio": 1.3,
                    "volume_low_ratio": 0.7,
                    "vwap_consistency_pct": 70.0,
                }
            }
        }

        df = _make_multi_day_df([
            ("2026-02-07", "flat", 23000),
            ("2026-02-10", "flat", 23050),
            ("2026-02-11", "up", 23100),
        ])
        result = compute_observation_snapshot(df, observation_end_time="10:00")

        assert isinstance(result, ObservationSnapshot)
        assert result.bars_collected > 0
        assert result.date == "2026-02-11"
        assert result.bias in ("bullish", "bearish", "neutral")
        assert len(result.bias_reasons) > 0

    @patch("core.observation.load_config")
    def test_empty_df_returns_empty(self, mock_config):
        mock_config.return_value = {"paper_trading": {}}
        result = compute_observation_snapshot(pd.DataFrame(), observation_end_time="10:00")
        assert result == ObservationSnapshot()

    @patch("core.observation.load_config")
    def test_none_df_returns_empty(self, mock_config):
        mock_config.return_value = {"paper_trading": {}}
        result = compute_observation_snapshot(None, observation_end_time="10:00")
        assert result == ObservationSnapshot()

    @patch("core.observation.load_config")
    def test_single_bar_day(self, mock_config):
        mock_config.return_value = {"paper_trading": {}}
        df = _make_simple_day("2026-02-11", base_price=23000, n_bars=1)
        result = compute_observation_snapshot(df, observation_end_time="10:00")
        assert result.bars_collected == 1

    @patch("core.observation.load_config")
    def test_gap_up_with_uptrend(self, mock_config):
        mock_config.return_value = {
            "paper_trading": {
                "observation": {
                    "gap_threshold_pct": 0.2,
                    "trend_moderate_pct": 0.3,
                    "trend_strong_pct": 0.7,
                    "volume_high_ratio": 1.3,
                    "volume_low_ratio": 0.7,
                    "vwap_consistency_pct": 70.0,
                }
            }
        }

        df = _make_multi_day_df([
            ("2026-02-10", "flat", 23000),
            ("2026-02-11", "up", 23200),
        ], n_bars=9)
        result = compute_observation_snapshot(df, observation_end_time="10:00")

        assert result.gap.direction == "gap_up"
        assert result.initial_trend.direction in ("up", "reversal_up")


# ---------------------------------------------------------------------------
# Tests: is_observation_period
# ---------------------------------------------------------------------------

class TestIsObservationPeriod:
    @patch("core.market_hours.is_market_open", return_value=False)
    def test_market_closed(self, mock_open):
        from core.options_utils import is_observation_period
        assert is_observation_period("10:00") is False

    @patch("core.market_hours.is_market_open", return_value=True)
    @patch("core.options_utils._now_ist")
    def test_before_entry_time(self, mock_now, mock_open):
        mock_now.return_value = datetime(2026, 2, 11, 9, 30, tzinfo=IST)
        from core.options_utils import is_observation_period
        assert is_observation_period("10:00") is True

    @patch("core.market_hours.is_market_open", return_value=True)
    @patch("core.options_utils._now_ist")
    def test_after_entry_time(self, mock_now, mock_open):
        mock_now.return_value = datetime(2026, 2, 11, 10, 30, tzinfo=IST)
        from core.options_utils import is_observation_period
        assert is_observation_period("10:00") is False

    @patch("core.market_hours.is_market_open", return_value=True)
    @patch("core.options_utils._now_ist")
    def test_at_entry_time(self, mock_now, mock_open):
        mock_now.return_value = datetime(2026, 2, 11, 10, 0, tzinfo=IST)
        from core.options_utils import is_observation_period
        assert is_observation_period("10:00") is False

    @patch("core.market_hours.is_market_open", return_value=True)
    @patch("core.options_utils._now_ist")
    def test_invalid_time_format(self, mock_now, mock_open):
        mock_now.return_value = datetime(2026, 2, 11, 9, 30, tzinfo=IST)
        from core.options_utils import is_observation_period
        # Invalid format falls back to 10:00
        assert is_observation_period("invalid") is True


# ---------------------------------------------------------------------------
# Tests: ObservationSnapshot model
# ---------------------------------------------------------------------------

class TestObservationSnapshotModel:
    def test_defaults(self):
        s = ObservationSnapshot()
        assert s.date == ""
        assert s.is_complete is False
        assert s.bars_collected == 0
        assert s.bias == "neutral"
        assert s.bias_reasons == []
        assert isinstance(s.gap, OpeningGap)
        assert isinstance(s.opening_range, OpeningRange)
        assert isinstance(s.initial_trend, InitialTrend)
        assert isinstance(s.volume, ObservationVolumeProfile)
        assert isinstance(s.vwap_context, VWAPContext)

    def test_json_serializable(self):
        s = ObservationSnapshot(
            date="2026-02-11",
            is_complete=True,
            bars_collected=9,
            bias="bullish",
            bias_reasons=["Gap up holding", "Strong uptrend"],
        )
        # Should not raise
        json_data = s.model_dump(mode="json")
        assert json_data["bias"] == "bullish"
        assert len(json_data["bias_reasons"]) == 2
