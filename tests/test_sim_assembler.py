"""Tests for simulation.data_assembler — data model bridging."""

from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from simulation.data_assembler import (
    build_vol_snapshot,
    compute_analytics,
    compute_observation,
    compute_technicals,
)
from simulation.chain_synthesizer import build_chain
from simulation.price_engine import generate_price_path, generate_warmup_days
from simulation.scenario_models import ChainConfig, Scenario, VolRegimeConfig

_IST = timezone(timedelta(hours=5, minutes=30))


@pytest.fixture
def scenario():
    return Scenario(name="test")


@pytest.fixture
def ohlcv_df(scenario):
    """Multi-day OHLCV DataFrame with warmup + today."""
    warmup = generate_warmup_days(scenario, date(2026, 2, 10), seed=42, num_days=4)
    today = generate_price_path(scenario, date(2026, 2, 10), seed=42)
    return pd.concat([warmup, today])


class TestComputeTechnicals:
    """Test compute_technicals with synthetic data."""

    def test_returns_valid_technicals(self, ohlcv_df):
        sim_dt = datetime(2026, 2, 10, 12, 0, tzinfo=_IST)
        technicals = compute_technicals(ohlcv_df, sim_dt)

        assert technicals.spot > 0
        assert 0 <= technicals.rsi <= 100
        assert technicals.ema_9 > 0
        assert technicals.ema_50 > 0
        assert technicals.bb_upper >= technicals.bb_lower

    def test_spot_near_last_close(self, ohlcv_df):
        sim_dt = datetime(2026, 2, 10, 12, 0, tzinfo=_IST)
        technicals = compute_technicals(ohlcv_df, sim_dt)
        last_close = ohlcv_df["Close"].iloc[-1]
        assert technicals.spot == last_close

    def test_progressive_reveal(self, scenario):
        """Technicals should change as more candles are revealed."""
        warmup = generate_warmup_days(scenario, date(2026, 2, 10), seed=42, num_days=4)
        today = generate_price_path(scenario, date(2026, 2, 10), seed=42)

        # 50 bars revealed
        df_50 = pd.concat([warmup, today.iloc[:50]])
        t50 = compute_technicals(df_50, datetime(2026, 2, 10, 10, 5, tzinfo=_IST))

        # 200 bars revealed
        df_200 = pd.concat([warmup, today.iloc[:200]])
        t200 = compute_technicals(df_200, datetime(2026, 2, 10, 12, 35, tzinfo=_IST))

        # Spot values should differ
        assert t50.spot != t200.spot

    def test_handles_empty_df(self):
        technicals = compute_technicals(pd.DataFrame(), datetime.now(_IST))
        assert technicals.spot == 0.0
        assert technicals.rsi == 50.0

    def test_staleness_is_zero(self, ohlcv_df):
        technicals = compute_technicals(ohlcv_df, datetime(2026, 2, 10, 12, 0, tzinfo=_IST))
        assert technicals.data_staleness_minutes == 0.0


class TestComputeAnalytics:
    """Test compute_analytics with synthetic chain."""

    def test_analytics_with_iv_percentile(self):
        config = ChainConfig(atm_iv=15.0, num_strikes_each_side=10)
        sim_dt = datetime(2026, 2, 10, 10, 30, tzinfo=_IST)
        chain = build_chain(23000, sim_dt, config, seed=42)

        analytics = compute_analytics(chain, iv_percentile=75.0)
        assert analytics.iv_percentile == 75.0
        assert analytics.pcr > 0
        assert analytics.atm_strike > 0


class TestComputeObservation:
    """Test compute_observation with synthetic data."""

    def test_observation_from_synthetic_data(self):
        """Observation should work with multi-day synthetic candles."""
        scenario = Scenario(name="test")
        warmup = generate_warmup_days(scenario, date(2026, 2, 10), seed=42, num_days=2)
        today = generate_price_path(scenario, date(2026, 2, 10), seed=42)

        # Use first 60 bars of today (up to ~10:15)
        df = pd.concat([warmup, today.iloc[:60]])
        obs = compute_observation(df)

        if obs is not None:
            assert obs.bars_collected > 0
            assert obs.bias in ("bullish", "bearish", "neutral")


class TestBuildVolSnapshot:
    """Test build_vol_snapshot."""

    def test_basic_construction(self):
        config = VolRegimeConfig(
            rv_5=12.0, rv_10=13.0, rv_20=14.0,
            vov_20=3.0, vrp=2.0,
            p_rv=0.5, p_vov=0.5, p_vrp=0.5,
        )
        vs = build_vol_snapshot(config, 23000.0, 14.0, "2026-02-10", dte=5)

        assert vs.rv_5 == 12.0
        assert vs.vix == 14.0
        assert vs.em > 0
        assert vs.date == "2026-02-10"

    def test_expected_move_formula(self):
        config = VolRegimeConfig()
        vs = build_vol_snapshot(config, 23000.0, 14.0, "2026-02-10", dte=5)
        import math
        expected_em = 23000 * (14.0 / 100.0) * math.sqrt(5 / 365.0)
        assert abs(vs.em - expected_em) < 0.1
