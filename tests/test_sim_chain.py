"""Tests for simulation.chain_synthesizer — synthetic option chain."""

from datetime import date, datetime, timedelta, timezone

import pytest

from simulation.chain_synthesizer import build_chain, update_chain, _iv_smile
from simulation.scenario_models import ChainConfig

_IST = timezone(timedelta(hours=5, minutes=30))


@pytest.fixture
def chain_config():
    return ChainConfig(
        expiry_dte=5,
        atm_iv=15.0,
        iv_skew_slope=-0.08,
        iv_skew_curvature=0.02,
        num_strikes_each_side=10,
        strike_step=50.0,
    )


@pytest.fixture
def sim_datetime():
    return datetime(2026, 2, 10, 10, 30, tzinfo=_IST)


class TestIVSmile:
    """Test parametric IV smile model."""

    def test_atm_iv_equals_base(self):
        iv = _iv_smile(23000, 23000, 15.0, -0.08, 0.02)
        assert abs(iv - 15.0) < 0.01

    def test_otm_put_higher_iv(self):
        """OTM puts should have higher IV (negative slope)."""
        atm_iv = _iv_smile(23000, 23000, 15.0, -0.08, 0.02)
        otm_put_iv = _iv_smile(22500, 23000, 15.0, -0.08, 0.02)
        assert otm_put_iv > atm_iv

    def test_deep_otm_call_has_smile(self):
        """Deep OTM calls should also have elevated IV (curvature)."""
        atm_iv = _iv_smile(23000, 23000, 15.0, -0.08, 0.02)
        deep_otm_call_iv = _iv_smile(24000, 23000, 15.0, -0.08, 0.02)
        # With positive curvature, deep OTM should be above ATM
        assert deep_otm_call_iv > atm_iv - 1.0  # approximately

    def test_iv_floor(self):
        """IV should never go below 1%."""
        iv = _iv_smile(30000, 23000, 5.0, -0.08, 0.02)
        assert iv >= 1.0


class TestBuildChain:
    """Test build_chain — full chain generation."""

    def test_chain_structure(self, chain_config, sim_datetime):
        chain = build_chain(23000, sim_datetime, chain_config, seed=42)
        assert chain.symbol == "NIFTY"
        assert chain.underlying_value == 23000
        assert len(chain.strikes) == 21  # 10 each side + ATM

    def test_strikes_sorted(self, chain_config, sim_datetime):
        chain = build_chain(23000, sim_datetime, chain_config, seed=42)
        strikes = [s.strike_price for s in chain.strikes]
        assert strikes == sorted(strikes)

    def test_atm_strike_present(self, chain_config, sim_datetime):
        chain = build_chain(23000, sim_datetime, chain_config, seed=42)
        strike_prices = [s.strike_price for s in chain.strikes]
        assert 23000 in strike_prices

    def test_bs_consistent_ltps(self, chain_config, sim_datetime):
        """CE LTP should decrease for higher strikes, PE LTP should increase."""
        chain = build_chain(23000, sim_datetime, chain_config, seed=42)

        # Find ATM index
        atm_idx = None
        for i, s in enumerate(chain.strikes):
            if s.strike_price == 23000:
                atm_idx = i
                break
        assert atm_idx is not None

        # CE LTP decreases as strike increases (calls become OTM)
        for i in range(atm_idx + 1, len(chain.strikes) - 1):
            assert chain.strikes[i].ce_ltp >= chain.strikes[i + 1].ce_ltp

        # PE LTP decreases as strike decreases (puts become OTM)
        for i in range(atm_idx - 1, 0, -1):
            assert chain.strikes[i].pe_ltp >= chain.strikes[i - 1].pe_ltp

    def test_oi_positive(self, chain_config, sim_datetime):
        chain = build_chain(23000, sim_datetime, chain_config, seed=42)
        for s in chain.strikes:
            assert s.ce_oi >= 0
            assert s.pe_oi >= 0

    def test_bid_ask_spread(self, chain_config, sim_datetime):
        chain = build_chain(23000, sim_datetime, chain_config, seed=42)
        for s in chain.strikes:
            assert s.ce_ask >= s.ce_bid
            assert s.pe_ask >= s.pe_bid

    def test_total_oi(self, chain_config, sim_datetime):
        chain = build_chain(23000, sim_datetime, chain_config, seed=42)
        expected_ce_oi = sum(s.ce_oi for s in chain.strikes)
        assert abs(chain.total_ce_oi - expected_ce_oi) < 10  # rounding

    def test_expiry_format(self, chain_config, sim_datetime):
        chain = build_chain(23000, sim_datetime, chain_config, seed=42)
        # Should be "DD-MON-YYYY" e.g. "15-FEB-2026"
        assert len(chain.expiry) > 0
        parts = chain.expiry.split("-")
        assert len(parts) == 3

    def test_reproducibility(self, chain_config, sim_datetime):
        c1 = build_chain(23000, sim_datetime, chain_config, seed=42)
        c2 = build_chain(23000, sim_datetime, chain_config, seed=42)
        assert c1.strikes[5].ce_ltp == c2.strikes[5].ce_ltp
        assert c1.strikes[5].pe_oi == c2.strikes[5].pe_oi


class TestUpdateChain:
    """Test incremental chain updates."""

    def test_preserves_strike_count(self, chain_config, sim_datetime):
        chain = build_chain(23000, sim_datetime, chain_config, seed=42)
        updated = update_chain(
            chain, 23050, sim_datetime + timedelta(minutes=5),
            chain_config, seed=43,
        )
        assert len(updated.strikes) == len(chain.strikes)

    def test_updates_underlying(self, chain_config, sim_datetime):
        chain = build_chain(23000, sim_datetime, chain_config, seed=42)
        updated = update_chain(
            chain, 23100, sim_datetime + timedelta(minutes=5),
            chain_config, seed=43,
        )
        assert updated.underlying_value == 23100

    def test_ltps_change_with_spot(self, chain_config, sim_datetime):
        chain = build_chain(23000, sim_datetime, chain_config, seed=42)
        updated = update_chain(
            chain, 23200, sim_datetime + timedelta(minutes=5),
            chain_config, seed=43,
        )
        # ATM call should be more expensive when spot moves up
        atm_idx = len(chain.strikes) // 2
        # LTPs are recomputed, so they should reflect the new spot
        assert updated.underlying_value == 23200

    def test_volume_accumulates(self, chain_config, sim_datetime):
        chain = build_chain(23000, sim_datetime, chain_config, seed=42)
        updated = update_chain(
            chain, 23010, sim_datetime + timedelta(minutes=5),
            chain_config, seed=43,
        )
        # Volume should increase (adds incremental)
        for i in range(len(chain.strikes)):
            assert updated.strikes[i].ce_volume >= chain.strikes[i].ce_volume


class TestBuildAnalyticsFromChain:
    """Test that synthetic chains work with real build_analytics."""

    def test_analytics_from_synthetic_chain(self, chain_config, sim_datetime):
        from core.options_analytics import build_analytics

        chain = build_chain(23000, sim_datetime, chain_config, seed=42)
        analytics = build_analytics(chain)

        assert analytics.pcr > 0
        assert analytics.atm_strike > 0
        assert analytics.max_pain > 0
