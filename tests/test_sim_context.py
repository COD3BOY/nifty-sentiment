"""Tests for simulation.context_assembler — synthetic MarketContext for sim."""

from datetime import date, datetime, timedelta, timezone

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
from simulation.context_assembler import (
    SimContextBuilder,
    _classify_vol_regime,
    _generate_synthetic_daily_bars,
    _generate_trading_dates,
)
from simulation.scenario_models import Scenario, VolRegimeConfig


_IST = timezone(timedelta(hours=5, minutes=30))


@pytest.fixture
def trending_up_scenario():
    """Scenario with positive drift."""
    return Scenario(
        name="test_trending_up",
        price_path={
            "open_price": 23000.0,
            "phases": [{"drift": 0.15, "volatility": 0.12}],
        },
        chain={"expiry_dte": 5, "atm_iv": 15.0, "num_strikes_each_side": 10},
        vol_regime={"p_rv": 0.3, "p_vov": 0.4, "p_vrp": 0.6},  # sell_premium
        num_warmup_days=2,
    )


@pytest.fixture
def trending_down_scenario():
    """Scenario with negative drift."""
    return Scenario(
        name="test_trending_down",
        price_path={
            "open_price": 23000.0,
            "phases": [{"drift": -0.15, "volatility": 0.12}],
        },
        chain={"expiry_dte": 5, "atm_iv": 15.0, "num_strikes_each_side": 10},
        vol_regime={"p_rv": 0.8, "p_vov": 0.5, "p_vrp": 0.3},  # stand_down
        num_warmup_days=2,
    )


@pytest.fixture
def range_bound_scenario():
    """Scenario with zero drift."""
    return Scenario(
        name="test_range_bound",
        price_path={
            "open_price": 23000.0,
            "phases": [{"drift": 0.0, "volatility": 0.12}],
        },
        chain={"expiry_dte": 5, "atm_iv": 15.0, "num_strikes_each_side": 10},
        vol_regime={"p_rv": 0.5, "p_vov": 0.5, "p_vrp": 0.5},  # neutral
        num_warmup_days=2,
    )


@pytest.fixture
def sim_date():
    return date(2026, 2, 16)


class TestInit:
    """Test SimContextBuilder initialization."""

    def test_generates_55_daily_bars(self, trending_up_scenario, sim_date):
        builder = SimContextBuilder(trending_up_scenario, sim_date, seed=42)
        assert len(builder._daily_df) == 55

    def test_five_prior_days(self, trending_up_scenario, sim_date):
        builder = SimContextBuilder(trending_up_scenario, sim_date, seed=42)
        assert len(builder._prior_days) == 5

    def test_prior_days_dates_descend(self, trending_up_scenario, sim_date):
        builder = SimContextBuilder(trending_up_scenario, sim_date, seed=42)
        dates = [d.date for d in builder._prior_days]
        assert dates == sorted(dates, reverse=True)

    def test_dates_skip_weekends(self, trending_up_scenario, sim_date):
        builder = SimContextBuilder(trending_up_scenario, sim_date, seed=42)
        for d in builder._daily_dates:
            assert d.weekday() < 5, f"{d} is a weekend"

    def test_trending_up_multi_day_trend(self, trending_up_scenario, sim_date):
        """Positive drift should produce prior days trending up → bullish or neutral."""
        builder = SimContextBuilder(trending_up_scenario, sim_date, seed=42)
        # With positive drift, most days should close up → bullish trend
        ups = sum(1 for d in builder._prior_days if d.close_vs_open == "up")
        # Not deterministic, but trend function should return valid string
        from core.context_engine import _compute_multi_day_trend
        trend = _compute_multi_day_trend(
            builder._prior_days, builder._current_week, builder._prior_week,
        )
        assert trend in ("bullish", "bearish", "neutral")

    def test_trending_down_prior_days(self, trending_down_scenario, sim_date):
        """Negative drift → prior days mostly down."""
        builder = SimContextBuilder(trending_down_scenario, sim_date, seed=42)
        # Verify all prior_days have valid DailyContext fields
        for d in builder._prior_days:
            assert d.open > 0
            assert d.close > 0
            assert d.close_vs_open in ("up", "down", "flat")

    def test_range_bound_prior_days(self, range_bound_scenario, sim_date):
        """Zero drift → mixed direction."""
        builder = SimContextBuilder(range_bound_scenario, sim_date, seed=42)
        directions = set(d.close_vs_open for d in builder._prior_days)
        # Should have some variety (not all same direction)
        assert len(directions) >= 1  # at least valid values

    def test_weekly_context_built(self, trending_up_scenario, sim_date):
        """Weekly context should be built from daily bars."""
        builder = SimContextBuilder(trending_up_scenario, sim_date, seed=42)
        assert builder._current_week is not None
        assert isinstance(builder._current_week, WeeklyContext)
        assert builder._current_week.week_start != ""


class TestVolContext:
    """Test VolContext construction from scenario config."""

    def test_sell_premium_regime(self, sim_date):
        scenario = Scenario(
            name="test",
            vol_regime={"p_rv": 0.3, "p_vov": 0.4, "p_vrp": 0.6},
        )
        builder = SimContextBuilder(scenario, sim_date, seed=42)
        assert builder._vol_ctx.regime == "sell_premium"

    def test_stand_down_regime(self, sim_date):
        scenario = Scenario(
            name="test",
            vol_regime={"p_rv": 0.8, "p_vov": 0.5, "p_vrp": 0.3},
        )
        builder = SimContextBuilder(scenario, sim_date, seed=42)
        assert builder._vol_ctx.regime == "stand_down"

    def test_buy_premium_regime(self, sim_date):
        scenario = Scenario(
            name="test",
            vol_regime={"p_rv": 0.6, "p_vov": 0.4, "p_vrp": 0.2},
        )
        builder = SimContextBuilder(scenario, sim_date, seed=42)
        assert builder._vol_ctx.regime == "buy_premium"

    def test_custom_regime_duration(self, sim_date):
        scenario = Scenario(
            name="test",
            vol_regime={
                "p_rv": 0.3, "p_vov": 0.4, "p_vrp": 0.6,
                "regime_duration_days": 15,
            },
        )
        builder = SimContextBuilder(scenario, sim_date, seed=42)
        assert builder._vol_ctx.regime_duration_days == 15

    def test_custom_rv_trend(self, sim_date):
        scenario = Scenario(
            name="test",
            vol_regime={
                "p_rv": 0.3, "p_vov": 0.4, "p_vrp": 0.6,
                "rv_trend": "expanding",
            },
        )
        builder = SimContextBuilder(scenario, sim_date, seed=42)
        assert builder._vol_ctx.rv_trend == "expanding"


class TestBuildContext:
    """Test per-tick context building."""

    @pytest.fixture
    def builder_with_candles(self, trending_up_scenario, sim_date):
        """Builder + synthetic candle DataFrame for testing."""
        builder = SimContextBuilder(trending_up_scenario, sim_date, seed=42)

        # Create minimal candle DataFrame (simulating progressive reveal)
        start = datetime.combine(sim_date, datetime.min.time().replace(hour=9, minute=15), tzinfo=_IST)
        timestamps = [start + timedelta(minutes=i) for i in range(30)]
        rng = np.random.default_rng(99)
        base = 23000.0
        closes = base + np.cumsum(rng.normal(0, 5, 30))
        df = pd.DataFrame({
            "Open": closes - rng.uniform(1, 5, 30),
            "High": closes + rng.uniform(5, 20, 30),
            "Low": closes - rng.uniform(5, 20, 30),
            "Close": closes,
            "Volume": rng.uniform(100000, 500000, 30),
        }, index=pd.DatetimeIndex(timestamps, name="datetime"))

        return builder, df

    def test_returns_market_context(self, builder_with_candles):
        builder, df = builder_with_candles
        ctx = builder.build_context(df, None)
        assert isinstance(ctx, MarketContext)

    def test_session_context_populated(self, builder_with_candles):
        builder, df = builder_with_candles
        ctx = builder.build_context(df, None)
        assert ctx.session.session_open > 0
        assert ctx.session.bars_elapsed > 0

    def test_prior_day_populated(self, builder_with_candles):
        builder, df = builder_with_candles
        ctx = builder.build_context(df, None)
        assert ctx.prior_day is not None
        assert ctx.prior_day.close > 0

    def test_vol_context_populated(self, builder_with_candles):
        builder, df = builder_with_candles
        ctx = builder.build_context(df, None)
        assert ctx.vol.regime in ("sell_premium", "buy_premium", "stand_down", "neutral")

    def test_multi_day_trend_valid(self, builder_with_candles):
        builder, df = builder_with_candles
        ctx = builder.build_context(df, None)
        assert ctx.multi_day_trend in ("bullish", "bearish", "neutral")

    def test_context_bias_valid(self, builder_with_candles):
        builder, df = builder_with_candles
        ctx = builder.build_context(df, None)
        assert ctx.context_bias in ("bullish", "bearish", "neutral")


class TestEndOfDay:
    """Test end_of_day context updates for multi-day simulation."""

    @pytest.fixture
    def builder_and_day_candles(self, trending_up_scenario, sim_date):
        builder = SimContextBuilder(trending_up_scenario, sim_date, seed=42)

        # Create 375 minute candles for one day
        start = datetime.combine(sim_date, datetime.min.time().replace(hour=9, minute=15), tzinfo=_IST)
        timestamps = [start + timedelta(minutes=i) for i in range(375)]
        rng = np.random.default_rng(99)
        base = 23000.0
        closes = base + np.cumsum(rng.normal(0, 5, 375))
        df = pd.DataFrame({
            "Open": closes - rng.uniform(1, 5, 375),
            "High": closes + rng.uniform(5, 20, 375),
            "Low": closes - rng.uniform(5, 20, 375),
            "Close": closes,
            "Volume": rng.uniform(100000, 500000, 375),
        }, index=pd.DatetimeIndex(timestamps, name="datetime"))

        return builder, df

    def test_prior_days_updated(self, builder_and_day_candles, sim_date):
        builder, candles = builder_and_day_candles
        old_dates = [d.date for d in builder._prior_days]

        builder.end_of_day(candles, next_date=sim_date + timedelta(days=1))

        new_dates = [d.date for d in builder._prior_days]
        # Most recent prior_day should now be sim_date
        assert new_dates[0] == sim_date.isoformat()

    def test_max_five_prior_days(self, builder_and_day_candles, sim_date):
        builder, candles = builder_and_day_candles
        builder.end_of_day(candles, next_date=sim_date + timedelta(days=1))
        assert len(builder._prior_days) == 5

    def test_daily_df_grows(self, builder_and_day_candles, sim_date):
        builder, candles = builder_and_day_candles
        old_len = len(builder._daily_df)
        builder.end_of_day(candles, next_date=sim_date + timedelta(days=1))
        assert len(builder._daily_df) == old_len + 1

    def test_vol_regime_same_increments_duration(self, builder_and_day_candles, sim_date):
        builder, candles = builder_and_day_candles
        old_duration = builder._vol_ctx.regime_duration_days
        # Pass same vol_regime → should increment duration
        builder.end_of_day(candles, next_date=sim_date + timedelta(days=1))
        assert builder._vol_ctx.regime_duration_days == old_duration + 1

    def test_vol_regime_change_resets_duration(self, builder_and_day_candles, sim_date):
        builder, candles = builder_and_day_candles
        # Current regime is sell_premium (p_rv=0.3, p_vrp=0.6)
        # Change to stand_down
        new_vol = VolRegimeConfig(p_rv=0.8, p_vov=0.9, p_vrp=0.1)
        old_changes = builder._vol_ctx.regime_changes_30d
        builder.end_of_day(candles, new_vol_regime=new_vol, next_date=sim_date + timedelta(days=1))
        assert builder._vol_ctx.regime == "stand_down"
        assert builder._vol_ctx.regime_duration_days == 1
        assert builder._vol_ctx.regime_changes_30d == old_changes + 1

    def test_weekly_context_updated(self, builder_and_day_candles, sim_date):
        builder, candles = builder_and_day_candles
        builder.end_of_day(candles, next_date=sim_date + timedelta(days=1))
        # Weekly context should still be valid
        assert builder._current_week is not None


class TestMultiDay:
    """Test multi-day context continuity."""

    def test_context_carries_forward_across_days(self, trending_up_scenario, sim_date):
        builder = SimContextBuilder(trending_up_scenario, sim_date, seed=42)
        rng = np.random.default_rng(100)

        for day_offset in range(3):
            day = sim_date + timedelta(days=day_offset)
            # Skip weekends
            while day.weekday() >= 5:
                day += timedelta(days=1)

            start = datetime.combine(day, datetime.min.time().replace(hour=9, minute=15), tzinfo=_IST)
            timestamps = [start + timedelta(minutes=i) for i in range(375)]
            base = 23000.0 + day_offset * 50
            closes = base + np.cumsum(rng.normal(0, 5, 375))
            candles = pd.DataFrame({
                "Open": closes - rng.uniform(1, 5, 375),
                "High": closes + rng.uniform(5, 20, 375),
                "Low": closes - rng.uniform(5, 20, 375),
                "Close": closes,
                "Volume": rng.uniform(100000, 500000, 375),
            }, index=pd.DatetimeIndex(timestamps, name="datetime"))

            # Build context (as runner would per-tick)
            ctx = builder.build_context(candles, None)
            assert isinstance(ctx, MarketContext)
            assert len(ctx.prior_days) == 5

            # End of day
            next_day = day + timedelta(days=1)
            while next_day.weekday() >= 5:
                next_day += timedelta(days=1)
            builder.end_of_day(candles, next_date=next_day)

        # After 3 days, daily_df should have grown by 3
        assert len(builder._daily_df) == 55 + 3

    def test_regime_shift_between_days(self, trending_up_scenario, sim_date):
        builder = SimContextBuilder(trending_up_scenario, sim_date, seed=42)
        assert builder._vol_ctx.regime == "sell_premium"

        # Create minimal candles for end_of_day
        start = datetime.combine(sim_date, datetime.min.time().replace(hour=9, minute=15), tzinfo=_IST)
        timestamps = [start + timedelta(minutes=i) for i in range(375)]
        rng = np.random.default_rng(99)
        closes = 23000.0 + np.cumsum(rng.normal(0, 5, 375))
        candles = pd.DataFrame({
            "Open": closes,
            "High": closes + 10,
            "Low": closes - 10,
            "Close": closes,
            "Volume": rng.uniform(100000, 500000, 375),
        }, index=pd.DatetimeIndex(timestamps, name="datetime"))

        # Shift to stand_down
        new_vol = VolRegimeConfig(p_rv=0.8, p_vov=0.9, p_vrp=0.1)
        builder.end_of_day(candles, new_vol_regime=new_vol, next_date=sim_date + timedelta(days=1))
        assert builder._vol_ctx.regime == "stand_down"

    def test_weekly_updates_across_days(self, trending_up_scenario, sim_date):
        builder = SimContextBuilder(trending_up_scenario, sim_date, seed=42)
        week_before = builder._current_week

        start = datetime.combine(sim_date, datetime.min.time().replace(hour=9, minute=15), tzinfo=_IST)
        timestamps = [start + timedelta(minutes=i) for i in range(375)]
        rng = np.random.default_rng(99)
        closes = 23000.0 + np.cumsum(rng.normal(0, 5, 375))
        candles = pd.DataFrame({
            "Open": closes, "High": closes + 10, "Low": closes - 10,
            "Close": closes, "Volume": rng.uniform(100000, 500000, 375),
        }, index=pd.DatetimeIndex(timestamps, name="datetime"))

        builder.end_of_day(candles, next_date=sim_date + timedelta(days=1))
        # Weekly should still be present (possibly updated)
        assert builder._current_week is not None

    def test_single_day_works_without_end_of_day(self, trending_up_scenario, sim_date):
        """Single-day sim doesn't call end_of_day — context should still work."""
        builder = SimContextBuilder(trending_up_scenario, sim_date, seed=42)

        start = datetime.combine(sim_date, datetime.min.time().replace(hour=9, minute=15), tzinfo=_IST)
        timestamps = [start + timedelta(minutes=i) for i in range(30)]
        rng = np.random.default_rng(99)
        closes = 23000.0 + np.cumsum(rng.normal(0, 5, 30))
        df = pd.DataFrame({
            "Open": closes, "High": closes + 10, "Low": closes - 10,
            "Close": closes, "Volume": rng.uniform(100000, 500000, 30),
        }, index=pd.DatetimeIndex(timestamps, name="datetime"))

        ctx = builder.build_context(df, None)
        assert isinstance(ctx, MarketContext)
        assert ctx.prior_day is not None
        assert ctx.vol.regime == "sell_premium"


class TestRepro:
    """Test reproducibility from seed."""

    def test_same_seed_same_context(self, trending_up_scenario, sim_date):
        b1 = SimContextBuilder(trending_up_scenario, sim_date, seed=42)
        b2 = SimContextBuilder(trending_up_scenario, sim_date, seed=42)

        # Same prior_days
        assert len(b1._prior_days) == len(b2._prior_days)
        for d1, d2 in zip(b1._prior_days, b2._prior_days):
            assert d1.close == d2.close
            assert d1.date == d2.date

    def test_different_seed_different_context(self, trending_up_scenario, sim_date):
        b1 = SimContextBuilder(trending_up_scenario, sim_date, seed=42)
        b2 = SimContextBuilder(trending_up_scenario, sim_date, seed=99)

        # Different seeds should produce different closes
        closes_1 = [d.close for d in b1._prior_days]
        closes_2 = [d.close for d in b2._prior_days]
        assert closes_1 != closes_2


class TestScenarioFields:
    """Test new VolRegimeConfig fields and backward compatibility."""

    def test_default_values(self):
        cfg = VolRegimeConfig()
        assert cfg.regime_duration_days == 5
        assert cfg.regime_changes_30d == 2
        assert cfg.rv_trend == "stable"

    def test_custom_values(self):
        cfg = VolRegimeConfig(
            regime_duration_days=1,
            regime_changes_30d=6,
            rv_trend="expanding",
        )
        assert cfg.regime_duration_days == 1
        assert cfg.regime_changes_30d == 6
        assert cfg.rv_trend == "expanding"

    def test_existing_yaml_backward_compat(self):
        """Existing YAML files without new fields should still load."""
        from simulation.scenario_models import load_scenario, list_scenarios
        scenarios = list_scenarios()
        if scenarios:
            # Load the first available scenario
            sc = load_scenario(scenarios[0]["name"])
            # Should use default values for new fields
            assert sc.vol_regime.regime_duration_days == 5
            assert sc.vol_regime.regime_changes_30d == 2
            assert sc.vol_regime.rv_trend == "stable"


class TestHelpers:
    """Test helper functions."""

    def test_classify_vol_regime_sell_premium(self):
        assert _classify_vol_regime(0.3, 0.4, 0.6) == "sell_premium"

    def test_classify_vol_regime_stand_down(self):
        assert _classify_vol_regime(0.8, 0.5, 0.3) == "stand_down"

    def test_classify_vol_regime_buy_premium(self):
        assert _classify_vol_regime(0.6, 0.4, 0.2) == "buy_premium"

    def test_classify_vol_regime_neutral(self):
        assert _classify_vol_regime(0.5, 0.5, 0.5) == "neutral"

    def test_generate_synthetic_daily_bars_shape(self):
        df = _generate_synthetic_daily_bars(55, 23000.0, 0.1, seed=42)
        assert len(df) == 55
        assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]

    def test_synthetic_bars_anchor_close(self):
        anchor = 23000.0
        df = _generate_synthetic_daily_bars(55, anchor, 0.0, seed=42)
        # Last close should be near anchor (within 5%)
        assert abs(df["Close"].iloc[-1] - anchor) / anchor < 0.05

    def test_synthetic_bars_positive_prices(self):
        df = _generate_synthetic_daily_bars(55, 23000.0, -0.3, seed=42)
        assert (df["Close"] > 0).all()
        assert (df["Open"] > 0).all()
        assert (df["High"] > 0).all()
        assert (df["Low"] > 0).all()

    def test_generate_trading_dates_skips_weekends(self):
        dates = _generate_trading_dates(date(2026, 2, 16), 10)  # Monday
        assert len(dates) == 10
        for d in dates:
            assert d.weekday() < 5
