"""Tests for simulation.price_engine — price path generation."""

from datetime import date

import numpy as np
import pytest

from simulation.price_engine import (
    _generate_minute_closes,
    _u_shape_volume,
    compute_log_returns,
    generate_price_path,
    generate_vix_path,
    generate_warmup_days,
)
from simulation.scenario_models import PhaseConfig, Scenario


@pytest.fixture
def default_scenario():
    return Scenario(name="test_default")


@pytest.fixture
def trending_up_scenario():
    return Scenario(
        name="test_trend_up",
        price_path={
            "open_price": 23000.0,
            "phases": [{"drift": 0.30, "volatility": 0.12}],
        },
    )


class TestMinuteCloses:
    """Test _generate_minute_closes."""

    def test_output_length(self):
        phases = [PhaseConfig()]
        rng = np.random.default_rng(42)
        closes = _generate_minute_closes(phases, 23000.0, rng)
        assert len(closes) == 375

    def test_first_close_near_open(self):
        phases = [PhaseConfig(volatility=0.10)]
        rng = np.random.default_rng(42)
        closes = _generate_minute_closes(phases, 23000.0, rng)
        # First close should be within 1% of open
        assert abs(closes[0] - 23000) / 23000 < 0.01

    def test_reproducibility(self):
        phases = [PhaseConfig()]
        closes1 = _generate_minute_closes(phases, 23000.0, np.random.default_rng(42))
        closes2 = _generate_minute_closes(phases, 23000.0, np.random.default_rng(42))
        np.testing.assert_array_equal(closes1, closes2)

    def test_different_seeds_different_paths(self):
        phases = [PhaseConfig()]
        closes1 = _generate_minute_closes(phases, 23000.0, np.random.default_rng(42))
        closes2 = _generate_minute_closes(phases, 23000.0, np.random.default_rng(99))
        assert not np.array_equal(closes1, closes2)

    def test_mean_reversion_stays_near_level(self):
        phases = [PhaseConfig(
            mean_reversion_kappa=20.0,
            mean_reversion_level=23000.0,
            volatility=0.10,
        )]
        rng = np.random.default_rng(42)
        closes = _generate_minute_closes(phases, 23000.0, rng)
        # Mean-reverting path should stay within ~3% of level
        assert np.max(np.abs(closes - 23000) / 23000) < 0.05

    def test_multi_phase(self):
        phases = [
            PhaseConfig(start_minute=0, end_minute=200, volatility=0.05),
            PhaseConfig(start_minute=200, end_minute=375, volatility=0.30),
        ]
        rng = np.random.default_rng(42)
        closes = _generate_minute_closes(phases, 23000.0, rng)
        assert len(closes) == 375
        # Second half should have higher variance
        returns_first = np.diff(np.log(closes[:200]))
        returns_second = np.diff(np.log(closes[200:]))
        assert np.std(returns_second) > np.std(returns_first) * 1.5


class TestVolumeProfile:
    """Test _u_shape_volume."""

    def test_total_volume_matches(self):
        rng = np.random.default_rng(42)
        volumes = _u_shape_volume(375, 5_000_000, 2.0, rng)
        assert abs(volumes.sum() - 5_000_000) < 1.0  # floating point

    def test_u_shape_pattern(self):
        rng = np.random.default_rng(42)
        volumes = _u_shape_volume(375, 5_000_000, 3.0, rng)
        # First and last 30 bars should have higher avg volume than middle
        first_30 = volumes[:30].mean()
        middle = volumes[150:220].mean()
        last_30 = volumes[-30:].mean()
        assert first_30 > middle
        assert last_30 > middle


class TestGeneratePricePath:
    """Test generate_price_path end-to-end."""

    def test_returns_dataframe(self, default_scenario):
        df = generate_price_path(default_scenario, date(2026, 2, 10), seed=42)
        assert len(df) == 375
        assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]

    def test_ohlc_consistency(self, default_scenario):
        df = generate_price_path(default_scenario, date(2026, 2, 10), seed=42)
        # High >= max(Open, Close) and Low <= min(Open, Close)
        assert (df["High"] >= df[["Open", "Close"]].max(axis=1)).all()
        assert (df["Low"] <= df[["Open", "Close"]].min(axis=1)).all()

    def test_datetime_index(self, default_scenario):
        df = generate_price_path(default_scenario, date(2026, 2, 10), seed=42)
        assert df.index.name == "datetime"
        # First timestamp should be 9:15
        assert df.index[0].hour == 9
        assert df.index[0].minute == 15

    def test_reproducibility(self, default_scenario):
        df1 = generate_price_path(default_scenario, date(2026, 2, 10), seed=42)
        df2 = generate_price_path(default_scenario, date(2026, 2, 10), seed=42)
        np.testing.assert_array_almost_equal(df1["Close"].values, df2["Close"].values)


class TestVIXGeneration:
    """Test generate_vix_path."""

    def test_output_length(self, default_scenario):
        spot_returns = np.random.default_rng(42).standard_normal(375) * 0.001
        vix = generate_vix_path(default_scenario, spot_returns, seed=42)
        assert len(vix) == 375

    def test_vix_starts_at_open_level(self, default_scenario):
        spot_returns = np.random.default_rng(42).standard_normal(375) * 0.001
        vix = generate_vix_path(default_scenario, spot_returns, seed=42)
        assert vix[0] == default_scenario.vix.open_level

    def test_vix_floor(self, default_scenario):
        spot_returns = np.random.default_rng(42).standard_normal(375) * 0.001
        vix = generate_vix_path(default_scenario, spot_returns, seed=42)
        assert np.all(vix >= 8.0)

    def test_vix_spikes(self):
        scenario = Scenario(
            name="test_spike",
            vix={"open_level": 14.0, "mean_level": 14.0, "spikes": {100: 25.0}},
        )
        spot_returns = np.zeros(375)
        vix = generate_vix_path(scenario, spot_returns, seed=42)
        # VIX should be elevated around tick 100
        assert vix[104] > 20.0


class TestWarmupDays:
    """Test generate_warmup_days."""

    def test_generates_correct_number_of_days(self, default_scenario):
        df = generate_warmup_days(default_scenario, date(2026, 2, 10), seed=42, num_days=4)
        # Should have 4 × 375 = 1500 rows
        assert len(df) == 4 * 375

    def test_warmup_is_calm(self, default_scenario):
        df = generate_warmup_days(default_scenario, date(2026, 2, 10), seed=42)
        returns = np.diff(np.log(df["Close"].values))
        # Warmup should have low volatility
        assert np.std(returns) < 0.005  # < 0.5% per minute std


class TestLogReturns:
    def test_first_return_is_zero(self):
        closes = np.array([100, 101, 99, 102])
        returns = compute_log_returns(closes)
        assert returns[0] == 0.0

    def test_correct_returns(self):
        closes = np.array([100, 110])
        returns = compute_log_returns(closes)
        expected = np.log(110 / 100)
        assert abs(returns[1] - expected) < 1e-10
