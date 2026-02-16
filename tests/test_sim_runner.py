"""Tests for simulation.runner — end-to-end simulation execution."""

from datetime import date

import pytest

from simulation.runner import get_algorithm, run_day, _make_initial_state
from simulation.scenario_models import Scenario, load_scenario


@pytest.fixture
def simple_scenario():
    """Simple trending-up scenario for fast testing."""
    return Scenario(
        name="test_simple",
        price_path={
            "open_price": 23000.0,
            "phases": [{"drift": 0.10, "volatility": 0.12}],
        },
        chain={"expiry_dte": 5, "atm_iv": 15.0, "num_strikes_each_side": 10},
        num_warmup_days=2,
    )


class TestGetAlgorithm:
    """Test algorithm instantiation."""

    def test_get_sentinel(self):
        algo = get_algorithm("sentinel")
        assert algo.name == "sentinel"

    def test_get_jarvis(self):
        algo = get_algorithm("jarvis")
        assert algo.name == "jarvis"

    def test_get_optimus(self):
        algo = get_algorithm("optimus")
        assert algo.name == "optimus"

    def test_get_atlas(self):
        algo = get_algorithm("atlas")
        assert algo.name == "atlas"

    def test_unknown_algo_raises(self):
        with pytest.raises(ValueError, match="Unknown algorithm"):
            get_algorithm("nonexistent")


class TestMakeInitialState:
    """Test initial state creation."""

    def test_fresh_state(self):
        sc = Scenario(name="test")
        state = _make_initial_state(sc, "test_algo")
        assert state.initial_capital > 0
        assert len(state.open_positions) == 0
        assert len(state.trade_log) == 0
        assert state.total_realized_pnl == 0.0


class TestRunDay:
    """Test single-day simulation execution."""

    def test_run_produces_result(self, simple_scenario):
        """Basic smoke test: run sentinel through a simple scenario."""
        result = run_day(
            scenario=simple_scenario,
            algo_name="sentinel",
            sim_date=date(2026, 2, 10),
            seed=42,
            speed=1e9,  # max speed (no sleeping)
        )

        assert result.sim_date == date(2026, 2, 10)
        assert result.algo_name == "sentinel"
        assert result.seed == 42
        assert len(result.candles) == 375
        assert len(result.vix_path) == 375
        assert result.wall_clock_seconds >= 0
        assert result.final_state is not None

    def test_tick_snapshots_recorded(self, simple_scenario):
        result = run_day(
            scenario=simple_scenario,
            algo_name="sentinel",
            sim_date=date(2026, 2, 10),
            seed=42,
            speed=1e9,
        )

        # Should have tick snapshots (one per minute)
        assert len(result.tick_snapshots) > 0
        # Spot values should be positive
        for ts in result.tick_snapshots:
            assert ts.spot > 0
            assert ts.vix > 0

    def test_eod_close(self, simple_scenario):
        """All positions should be closed at end of day."""
        result = run_day(
            scenario=simple_scenario,
            algo_name="sentinel",
            sim_date=date(2026, 2, 10),
            seed=42,
            speed=1e9,
        )
        assert len(result.final_state.open_positions) == 0

    def test_reproducibility(self, simple_scenario):
        """Same seed should produce identical results."""
        r1 = run_day(simple_scenario, "sentinel", date(2026, 2, 10), seed=42, speed=1e9)
        r2 = run_day(simple_scenario, "sentinel", date(2026, 2, 10), seed=42, speed=1e9)

        assert len(r1.trades) == len(r2.trades)
        assert len(r1.tick_snapshots) == len(r2.tick_snapshots)
        # Spot values should be identical
        for ts1, ts2 in zip(r1.tick_snapshots[:10], r2.tick_snapshots[:10]):
            assert ts1.spot == ts2.spot

    def test_callback_called(self, simple_scenario):
        """Verify callback is invoked each tick."""
        call_count = [0]

        def counter(tick_idx, state, spot, vix, sim_time):
            call_count[0] += 1

        run_day(
            simple_scenario, "sentinel", date(2026, 2, 10),
            seed=42, speed=1e9, callback=counter,
        )
        assert call_count[0] > 300  # ~375 ticks expected


class TestScenarioLoading:
    """Test scenario YAML loading."""

    def test_load_normal_trending_up(self):
        sc = load_scenario("normal/trending_up")
        assert sc.name == "trending_up"
        assert sc.price_path.open_price > 0

    def test_load_crisis_flash_crash(self):
        sc = load_scenario("crisis/flash_crash")
        assert sc.name == "flash_crash"
        assert len(sc.price_path.phases) > 1  # multi-phase

    def test_load_nonexistent_raises(self):
        with pytest.raises(KeyError):
            load_scenario("normal/nonexistent_scenario")

    def test_list_scenarios(self):
        from simulation.scenario_models import list_scenarios
        scenarios = list_scenarios()
        assert len(scenarios) > 20  # 28+ expected
        names = [s["name"] for s in scenarios]
        assert "normal/trending_up" in names
        assert "crisis/flash_crash" in names
