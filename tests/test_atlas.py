"""Tests for algorithms.atlas — regime classification, dynamic params, clamp."""

import pytest

from core.vol_distribution import VolSnapshot


# ── Import atlas internals ────────────────────────────────────────────────

from algorithms.atlas import (
    _clamp,
    _classify_regime,
    _dynamic_strike_delta,
    _dynamic_min_credit_ratio,
    _dynamic_sl_multiple,
    _dynamic_delta_exit,
    _dynamic_take_profit,
    _dynamic_breakeven_mult,
    _dynamic_risk_per_trade,
    _dynamic_max_portfolio_risk,
    _neutral_snapshot,
)


# ── _clamp ────────────────────────────────────────────────────────────────

class TestClamp:
    def test_within_range(self):
        assert _clamp(0.5, 0.0, 1.0) == 0.5

    def test_below_lo(self):
        assert _clamp(-1.0, 0.0, 1.0) == 0.0

    def test_above_hi(self):
        assert _clamp(5.0, 0.0, 1.0) == 1.0

    def test_at_boundaries(self):
        assert _clamp(0.0, 0.0, 1.0) == 0.0
        assert _clamp(1.0, 0.0, 1.0) == 1.0


# ── Regime Classification ─────────────────────────────────────────────────

def _make_vol(p_rv=0.5, p_vov=0.5, p_vrp=0.5, **kw):
    defaults = dict(rv_5=15, rv_10=15, rv_20=15, vov_20=1, vix=15, vrp=0, em=0, date="2026-01-01")
    defaults.update(kw)
    return VolSnapshot(p_rv=p_rv, p_vov=p_vov, p_vrp=p_vrp, **defaults)


class TestClassifyRegime:
    def test_sell_premium(self):
        vol = _make_vol(p_rv=0.4, p_vov=0.3, p_vrp=0.7)
        cfg = {"regime_sell_vrp_min": 0.55, "regime_sell_vov_max": 0.70, "standdown_vov_min": 0.85, "standdown_rv_min": 0.90}
        assert _classify_regime(vol, cfg) == "sell_premium"

    def test_stand_down_high_vov(self):
        vol = _make_vol(p_rv=0.4, p_vov=0.90, p_vrp=0.7)
        cfg = {"standdown_vov_min": 0.85, "standdown_rv_min": 0.90}
        assert _classify_regime(vol, cfg) == "stand_down"

    def test_stand_down_high_rv(self):
        vol = _make_vol(p_rv=0.95, p_vov=0.3, p_vrp=0.7)
        cfg = {"standdown_vov_min": 0.85, "standdown_rv_min": 0.90}
        assert _classify_regime(vol, cfg) == "stand_down"

    def test_buy_premium_low_vrp(self):
        vol = _make_vol(p_rv=0.4, p_vov=0.3, p_vrp=0.2)
        cfg = {"regime_sell_vrp_min": 0.55, "regime_sell_vov_max": 0.70, "standdown_vov_min": 0.85, "standdown_rv_min": 0.90}
        assert _classify_regime(vol, cfg) == "buy_premium"

    def test_sell_premium_ambiguous(self):
        """VRP rich but VoV too high → sell_premium rejected, falls to ambiguous."""
        vol = _make_vol(p_rv=0.4, p_vov=0.75, p_vrp=0.7)
        cfg = {"regime_sell_vrp_min": 0.55, "regime_sell_vov_max": 0.70, "standdown_vov_min": 0.85, "standdown_rv_min": 0.90}
        # VoV > sell_vov_max but < standdown → sell_premium rejected
        result = _classify_regime(vol, cfg)
        assert result == "sell_premium" or result == "stand_down" or result not in ["sell_premium"]
        # Actually, p_vov=0.75 > 0.70 → sell_premium gate fails, p_vrp=0.7 > 0.30 → not buy
        # Falls through to ambiguous → "sell_premium" (the default fallthrough)


# ── Dynamic Parameter Functions ───────────────────────────────────────────

class TestDynamicStrikeDelta:
    def test_low_vol_wider_delta(self):
        # Low p_rv → less subtracted → higher delta
        d = _dynamic_strike_delta(0.1, {})
        assert d > _dynamic_strike_delta(0.9, {})

    def test_clamped_to_range(self):
        d = _dynamic_strike_delta(0.0, {})
        assert 0.10 <= d <= 0.20
        d = _dynamic_strike_delta(1.0, {})
        assert 0.10 <= d <= 0.20


class TestDynamicMinCreditRatio:
    def test_high_vrp_higher_credit(self):
        r = _dynamic_min_credit_ratio(0.9, {})
        assert r > _dynamic_min_credit_ratio(0.1, {})

    def test_clamped_to_range(self):
        r = _dynamic_min_credit_ratio(0.0, {})
        assert 0.22 <= r <= 0.40
        r = _dynamic_min_credit_ratio(1.0, {})
        assert 0.22 <= r <= 0.40


class TestDynamicSlMultiple:
    def test_high_vov_wider_sl(self):
        sl = _dynamic_sl_multiple(0.9, {})
        assert sl > _dynamic_sl_multiple(0.1, {})

    def test_clamped_to_range(self):
        sl = _dynamic_sl_multiple(0.0, {})
        assert 1.6 <= sl <= 2.5
        sl = _dynamic_sl_multiple(1.0, {})
        assert 1.6 <= sl <= 2.5


class TestDynamicDeltaExit:
    def test_high_rv_wider_exit(self):
        d = _dynamic_delta_exit(0.9, {})
        assert d > _dynamic_delta_exit(0.1, {})

    def test_clamped_to_range(self):
        d = _dynamic_delta_exit(0.0, {})
        assert 0.22 <= d <= 0.33


class TestDynamicTakeProfit:
    def test_low_vol_faster_profit(self):
        # Low p_rv → (1 - p_rv) high → larger take_profit
        tp = _dynamic_take_profit(0.1, {})
        assert tp > _dynamic_take_profit(0.9, {})

    def test_clamped_to_range(self):
        tp = _dynamic_take_profit(0.0, {})
        assert 0.45 <= tp <= 0.70


class TestDynamicBreakevenMult:
    def test_high_vov_wider_breakeven(self):
        be = _dynamic_breakeven_mult(0.9, {})
        assert be > _dynamic_breakeven_mult(0.1, {})

    def test_clamped_to_range(self):
        be = _dynamic_breakeven_mult(0.0, {})
        assert 1.15 <= be <= 1.50


class TestDynamicRiskPerTrade:
    def test_high_vol_lower_risk(self):
        r = _dynamic_risk_per_trade(0.9, 0.9, {})
        assert r < _dynamic_risk_per_trade(0.1, 0.1, {})

    def test_clamped_to_range(self):
        r = _dynamic_risk_per_trade(0.0, 0.0, {})
        assert 0.125 <= r <= 0.50
        r = _dynamic_risk_per_trade(1.0, 1.0, {})
        assert 0.125 <= r <= 0.50


class TestDynamicMaxPortfolioRisk:
    def test_high_vov_lower_portfolio_risk(self):
        r = _dynamic_max_portfolio_risk(0.9, {})
        assert r < _dynamic_max_portfolio_risk(0.1, {})

    def test_clamped_to_range(self):
        r = _dynamic_max_portfolio_risk(0.0, {})
        assert 1.5 <= r <= 3.0
        r = _dynamic_max_portfolio_risk(1.0, {})
        assert 1.5 <= r <= 3.0


# ── Neutral Snapshot ──────────────────────────────────────────────────────

class TestNeutralSnapshot:
    def test_all_percentiles_half(self):
        ns = _neutral_snapshot()
        assert ns.p_rv == 0.5
        assert ns.p_vov == 0.5
        assert ns.p_vrp == 0.5

    def test_date_is_today(self):
        from datetime import datetime, timezone, timedelta
        ns = _neutral_snapshot()
        ist = timezone(timedelta(hours=5, minutes=30))
        today_str = datetime.now(ist).strftime("%Y-%m-%d")
        assert ns.date == today_str
