"""Tests for the daily review / self-improvement system.

Covers:
- Trade classification (A/B/C/D)
- Signal extraction and reliability
- Parameter calibration (MAE/MFE)
- Safety rail validation
- Parameter bounds
- Improvement ledger lifecycle
"""

from __future__ import annotations

import pytest
from datetime import datetime, timedelta, timezone

from core.paper_trading_models import TradeRecord, PaperTradingState
from core.improvement_models import (
    ImprovementProposal,
    ReviewSession,
    SignalReliability,
    TradeClassification,
)
from core.parameter_bounds import (
    PARAMETER_BOUNDS,
    check_drift,
    check_step_size,
    is_within_bounds,
    validate_proposed_value,
)
from analyzers.daily_review import (
    _extract_signals,
    _check_regime_fit,
    calibrate_parameters,
    check_safety_rails,
    classify_trade,
    compute_signal_reliability,
    run_daily_review,
)

_IST = timezone(timedelta(hours=5, minutes=30))


# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------

def _make_trade(
    strategy: str = "Iron Condor",
    strategy_type: str = "credit",
    direction_bias: str = "Neutral",
    realized_pnl: float = 500.0,
    score: float = 65.0,
    entry_context: dict | None = None,
    max_drawdown: float = 200.0,
    max_favorable: float = 600.0,
    exit_reason: str = "closed_profit_target",
    profit_target_amount: float = 500.0,
    stop_loss_amount: float = 1000.0,
    trade_id: str = "t001",
) -> TradeRecord:
    """Create a test trade record with customizable fields."""
    now = datetime.now(_IST)
    if entry_context is None:
        entry_context = {
            "spot": 23000.0,
            "vwap": 22980.0,
            "ema_9": 23010.0,
            "ema_21": 22990.0,
            "ema_50": 22950.0,
            "rsi": 52.0,
            "supertrend_direction": 1,
            "pcr": 1.05,
            "atm_iv": 15.0,
            "bb_upper": 23100.0,
            "bb_middle": 23000.0,
            "bb_lower": 22900.0,
            "max_pain": 23000.0,
        }
    return TradeRecord(
        id=trade_id,
        strategy=strategy,
        strategy_type=strategy_type,
        direction_bias=direction_bias,
        confidence="High",
        score=score,
        legs_summary=[{"action": "SELL", "strike": 23000, "option_type": "CE"}],
        lots=1,
        entry_time=now - timedelta(hours=2),
        exit_time=now,
        exit_reason=exit_reason,
        realized_pnl=realized_pnl,
        execution_cost=50.0,
        net_pnl=realized_pnl - 50.0,
        margin_required=100000.0,
        net_premium=150.0,
        stop_loss_amount=stop_loss_amount,
        profit_target_amount=profit_target_amount,
        entry_context=entry_context,
        spot_at_entry=23000.0,
        spot_at_exit=23020.0,
        max_drawdown=max_drawdown,
        max_favorable=max_favorable,
    )


# -------------------------------------------------------------------------
# Trade classification tests
# -------------------------------------------------------------------------

class TestTradeClassification:
    def test_category_a_good_entry_profitable(self):
        """Sound entry + profit = Category A."""
        trade = _make_trade(realized_pnl=500, score=65)
        c = classify_trade(trade)
        assert c.category == "A"
        assert c.profitable is True
        assert c.signal_alignment_count >= 3

    def test_category_b_bad_entry_profitable(self):
        """Flawed entry + profit = Category B (lucky)."""
        ctx = {
            "spot": 23000.0, "vwap": 23200.0,  # Spot far below VWAP
            "ema_9": 22900.0, "ema_21": 23000.0, "ema_50": 23100.0,  # Bearish alignment
            "rsi": 28.0,  # Oversold (bad for neutral)
            "supertrend_direction": -1,
            "pcr": 0.5,  # Skewed
            "atm_iv": 8.0,  # Too low for credit
            "bb_upper": 23500.0, "bb_middle": 23000.0, "bb_lower": 22500.0,  # Wide
            "max_pain": 23500.0,
        }
        trade = _make_trade(realized_pnl=500, score=25, entry_context=ctx)
        c = classify_trade(trade)
        assert c.category == "B"
        assert c.profitable is True
        assert c.signal_alignment_count < 3

    def test_category_c_good_entry_unprofitable(self):
        """Sound entry + loss = Category C (bad luck)."""
        trade = _make_trade(realized_pnl=-300, score=65)
        c = classify_trade(trade)
        assert c.category == "C"
        assert c.profitable is False
        assert c.signal_alignment_count >= 3

    def test_category_d_bad_entry_unprofitable(self):
        """Flawed entry + loss = Category D (process failure)."""
        ctx = {
            "spot": 23000.0, "vwap": 23200.0,
            "ema_9": 22900.0, "ema_21": 23000.0, "ema_50": 23100.0,
            "rsi": 28.0,
            "supertrend_direction": -1,
            "pcr": 0.5,
            "atm_iv": 8.0,
            "bb_upper": 23500.0, "bb_middle": 23000.0, "bb_lower": 22500.0,
            "max_pain": 23500.0,
        }
        trade = _make_trade(realized_pnl=-500, score=25, entry_context=ctx)
        c = classify_trade(trade)
        assert c.category == "D"
        assert c.profitable is False

    def test_no_entry_context_defaults_gracefully(self):
        """Missing entry context should not crash."""
        trade = _make_trade(realized_pnl=100, entry_context={})
        c = classify_trade(trade)
        assert c.category in ("A", "B", "C", "D")
        assert c.signal_alignment_count == 0

    def test_classification_notes_low_alignment(self):
        """Notes should flag low signal alignment."""
        ctx = {
            "spot": 23000.0, "vwap": 23200.0,
            "rsi": 75.0,
            "pcr": 0.4,
        }
        trade = _make_trade(realized_pnl=-100, entry_context=ctx, score=20)
        c = classify_trade(trade)
        assert any("aligned signals" in n.lower() for n in c.notes)


# -------------------------------------------------------------------------
# Signal extraction tests
# -------------------------------------------------------------------------

class TestSignalExtraction:
    def test_bullish_signals_for_bull_put_spread(self):
        """Bullish context should align with Bull Put Spread."""
        ctx = {
            "spot": 23100.0, "vwap": 23000.0,
            "ema_9": 23100.0, "ema_21": 23050.0, "ema_50": 23000.0,
            "rsi": 55.0,
            "supertrend_direction": 1,
            "pcr": 1.3,
            "atm_iv": 16.0,
            "bb_upper": 23200.0, "bb_middle": 23050.0, "bb_lower": 22900.0,
            "max_pain": 23050.0,
        }
        present, aligned, opposed = _extract_signals(
            ctx, "Bull Put Spread", "credit", "Bullish"
        )
        assert len(aligned) >= 4
        assert "ema_alignment" in aligned
        assert "supertrend" in aligned
        assert "spot_vs_vwap" in aligned

    def test_bearish_signals_for_bear_call_spread(self):
        """Bearish context should align with Bear Call Spread."""
        ctx = {
            "spot": 22900.0, "vwap": 23000.0,
            "ema_9": 22900.0, "ema_21": 22950.0, "ema_50": 23000.0,
            "rsi": 42.0,
            "supertrend_direction": -1,
            "pcr": 0.6,
            "atm_iv": 16.0,
            "bb_upper": 23200.0, "bb_middle": 23050.0, "bb_lower": 22900.0,
            "max_pain": 22800.0,
        }
        present, aligned, opposed = _extract_signals(
            ctx, "Bear Call Spread", "credit", "Bearish"
        )
        assert "ema_alignment" in aligned
        assert "supertrend" in aligned

    def test_neutral_signals_for_iron_condor(self):
        """Neutral context should align with Iron Condor."""
        ctx = {
            "spot": 23000.0, "vwap": 22990.0,
            "ema_9": 23005.0, "ema_21": 23010.0, "ema_50": 22990.0,
            "rsi": 50.0,
            "supertrend_direction": 1,
            "pcr": 1.0,
            "atm_iv": 16.0,
            "bb_upper": 23100.0, "bb_middle": 23000.0, "bb_lower": 22900.0,
            "max_pain": 23010.0,
        }
        present, aligned, opposed = _extract_signals(
            ctx, "Iron Condor", "credit", "Neutral"
        )
        assert "rsi" in aligned
        assert "pcr" in aligned

    def test_empty_context_returns_empty(self):
        """Empty context → no signals."""
        present, aligned, opposed = _extract_signals({}, "Iron Condor", "credit", "Neutral")
        assert present == []
        assert aligned == []
        assert opposed == []


# -------------------------------------------------------------------------
# Strategy-regime fit tests
# -------------------------------------------------------------------------

class TestRegimeFit:
    def test_credit_in_squeeze_low_iv_bad_fit(self):
        """Credit strategy in BB squeeze + low IV → bad fit."""
        ctx = {
            "atm_iv": 10.0,
            "bb_upper": 23030.0, "bb_middle": 23000.0, "bb_lower": 22970.0,
        }
        assert _check_regime_fit(ctx, "Short Straddle", "credit") is False

    def test_credit_in_normal_conditions_good_fit(self):
        """Credit strategy in normal BB + moderate IV → good fit."""
        ctx = {
            "atm_iv": 16.0,
            "bb_upper": 23200.0, "bb_middle": 23000.0, "bb_lower": 22800.0,
        }
        assert _check_regime_fit(ctx, "Short Straddle", "credit") is True

    def test_debit_in_expanded_high_iv_bad_fit(self):
        """Debit strategy when BB already expanded + high IV → bad fit."""
        ctx = {
            "atm_iv": 25.0,
            "bb_upper": 23500.0, "bb_middle": 23000.0, "bb_lower": 22500.0,
        }
        assert _check_regime_fit(ctx, "Long Straddle", "debit") is False

    def test_debit_in_squeeze_good_fit(self):
        """Debit strategy in BB squeeze → good fit."""
        ctx = {
            "atm_iv": 12.0,
            "bb_upper": 23080.0, "bb_middle": 23000.0, "bb_lower": 22920.0,
        }
        assert _check_regime_fit(ctx, "Long Straddle", "debit") is True

    def test_no_context_assumes_fit(self):
        """Missing context → assume good fit."""
        assert _check_regime_fit({}, "Iron Condor", "credit") is True


# -------------------------------------------------------------------------
# Signal reliability tests
# -------------------------------------------------------------------------

class TestSignalReliability:
    def test_reliability_computation(self):
        """Verify signal reliability metrics over multiple trades."""
        trades = [
            _make_trade(trade_id=f"t{i:03d}", realized_pnl=500 if i % 2 == 0 else -300)
            for i in range(10)
        ]
        reliability = compute_signal_reliability(trades, window_days=30)
        assert len(reliability) > 0
        for sig in reliability:
            assert sig.total_trades > 0
            assert 0 <= sig.predictive_accuracy <= 1.0

    def test_empty_trades_returns_empty(self):
        """No trades → empty reliability list."""
        assert compute_signal_reliability([], window_days=20) == []

    def test_signal_reliability_model_properties(self):
        """Test SignalReliability property computations."""
        sr = SignalReliability(
            signal_name="rsi",
            total_trades=20,
            confirming_wins=8,
            confirming_losses=4,
            opposing_wins=3,
            opposing_losses=5,
        )
        assert sr.predictive_accuracy == pytest.approx(8 / 12)
        assert sr.information_coefficient == pytest.approx((8 + 5 - 4 - 3) / 20)

    def test_signal_reliability_zero_division(self):
        """Zero trades should not raise."""
        sr = SignalReliability(signal_name="test", total_trades=0)
        assert sr.predictive_accuracy == 0.0
        assert sr.information_coefficient == 0.0


# -------------------------------------------------------------------------
# Parameter calibration tests
# -------------------------------------------------------------------------

class TestParameterCalibration:
    def test_calibration_flags_tight_sl(self):
        """Should flag SL as too tight when most SL exits reversed."""
        trades = []
        for i in range(20):
            trades.append(_make_trade(
                trade_id=f"t{i:03d}",
                realized_pnl=-500,
                exit_reason="closed_stop_loss",
                max_favorable=200.0,  # Was positive before SL hit
                max_drawdown=1000.0,
            ))
        calibrations = calibrate_parameters(trades)
        assert len(calibrations) >= 1
        iron_condor_cal = [c for c in calibrations if c.strategy_name == "Iron Condor"]
        assert len(iron_condor_cal) == 1
        assert iron_condor_cal[0].sl_hit_then_reversed_pct == 100.0
        assert "too tight" in iron_condor_cal[0].suggestion_reason.lower()

    def test_calibration_flags_conservative_pt(self):
        """Should flag PT as too conservative when MFE far exceeds PT."""
        trades = []
        for i in range(20):
            trades.append(_make_trade(
                trade_id=f"t{i:03d}",
                realized_pnl=500,
                exit_reason="closed_profit_target",
                max_favorable=1500.0,  # 3x the PT amount
                profit_target_amount=500.0,
            ))
        calibrations = calibrate_parameters(trades)
        iron_condor_cal = [c for c in calibrations if c.strategy_name == "Iron Condor"]
        assert len(iron_condor_cal) == 1
        assert iron_condor_cal[0].pt_exceeded_pct == 100.0
        assert "conservative" in iron_condor_cal[0].suggestion_reason.lower()

    def test_calibration_insufficient_trades(self):
        """Below min sample size → no calibration output."""
        trades = [_make_trade(trade_id=f"t{i:03d}") for i in range(5)]
        calibrations = calibrate_parameters(trades)
        assert len(calibrations) == 0  # 5 < 15 min sample


# -------------------------------------------------------------------------
# Safety rail tests
# -------------------------------------------------------------------------

class TestSafetyRails:
    def test_insufficient_evidence_blocked(self):
        """Proposal with < 15 trades should be blocked."""
        proposal = ImprovementProposal(
            strategy_name="Iron Condor",
            parameter_name="rsi_neutral_low",
            current_value=40.0,
            proposed_value=38.0,
            default_value=40.0,
            change_pct=-0.05,
            evidence_trade_count=8,
            evidence_summary="test",
            expected_impact="test",
            reversion_criteria="test",
            classification="calibration",
            confidence=0.7,
        )
        allowed, reason = check_safety_rails(proposal, {}, 0)
        assert not allowed
        assert "insufficient" in reason.lower()

    def test_step_size_exceeded_blocked(self):
        """Proposal exceeding 15% step should be blocked."""
        proposal = ImprovementProposal(
            strategy_name="Iron Condor",
            parameter_name="rsi_neutral_low",
            current_value=40.0,
            proposed_value=30.0,  # 25% change
            default_value=40.0,
            change_pct=-0.25,
            evidence_trade_count=20,
            evidence_summary="test",
            expected_impact="test",
            reversion_criteria="test",
            classification="calibration",
            confidence=0.7,
        )
        allowed, reason = check_safety_rails(proposal, {}, 0)
        assert not allowed
        assert "step" in reason.lower()

    def test_cooling_period_blocked(self):
        """Proposal for parameter in cooling should be blocked."""
        proposal = ImprovementProposal(
            strategy_name="Iron Condor",
            parameter_name="rsi_neutral_low",
            current_value=40.0,
            proposed_value=38.0,
            default_value=40.0,
            change_pct=-0.05,
            evidence_trade_count=20,
            evidence_summary="test",
            expected_impact="test",
            reversion_criteria="test",
            classification="calibration",
            confidence=0.7,
        )
        cooling = {"Iron Condor.rsi_neutral_low": 5}
        allowed, reason = check_safety_rails(proposal, cooling, 0)
        assert not allowed
        assert "cooling" in reason.lower()

    def test_session_budget_blocked(self):
        """More than max_changes_per_session should be blocked."""
        proposal = ImprovementProposal(
            strategy_name="Iron Condor",
            parameter_name="rsi_neutral_low",
            current_value=40.0,
            proposed_value=38.0,
            default_value=40.0,
            change_pct=-0.05,
            evidence_trade_count=20,
            evidence_summary="test",
            expected_impact="test",
            reversion_criteria="test",
            classification="calibration",
            confidence=0.7,
        )
        allowed, reason = check_safety_rails(proposal, {}, 3)
        assert not allowed
        assert "budget" in reason.lower()

    def test_valid_proposal_passes(self):
        """Valid proposal should pass all rails."""
        proposal = ImprovementProposal(
            strategy_name="Iron Condor",
            parameter_name="rsi_neutral_low",
            current_value=40.0,
            proposed_value=38.0,
            default_value=40.0,
            change_pct=-0.05,
            evidence_trade_count=20,
            evidence_summary="test",
            expected_impact="test",
            reversion_criteria="test",
            classification="calibration",
            confidence=0.7,
        )
        allowed, reason = check_safety_rails(proposal, {}, 0)
        assert allowed
        assert reason == ""


# -------------------------------------------------------------------------
# Parameter bounds tests
# -------------------------------------------------------------------------

class TestParameterBounds:
    def test_all_strategy_params_have_bounds(self):
        """Every param_key in strategy_rules should have bounds defined."""
        from core.strategy_rules import STRATEGY_RULES
        missing = []
        for strat, rules in STRATEGY_RULES.items():
            for rule in rules.get("scoring_rules", []):
                pk = rule.get("param_key")
                if pk and pk not in PARAMETER_BOUNDS:
                    missing.append(f"{strat}.{pk}")
        assert missing == [], f"Missing bounds for: {missing}"

    def test_bounds_are_valid_ranges(self):
        """All bounds should have min < max."""
        for key, (lo, hi) in PARAMETER_BOUNDS.items():
            assert lo < hi, f"{key}: min={lo} >= max={hi}"

    def test_defaults_within_bounds(self):
        """Every default value should be within its bounds."""
        from core.strategy_rules import STRATEGY_RULES
        violations = []
        for strat, rules in STRATEGY_RULES.items():
            for rule in rules.get("scoring_rules", []):
                pk = rule.get("param_key")
                default = rule.get("default")
                if pk and default is not None:
                    if not is_within_bounds(pk, default):
                        violations.append(f"{strat}.{pk}={default}")
        assert violations == [], f"Defaults outside bounds: {violations}"

    def test_check_step_size(self):
        assert check_step_size(40.0, 44.0) is True   # 10%
        assert check_step_size(40.0, 47.0) is False   # 17.5%
        assert check_step_size(0, 10) is True          # Zero base → allow

    def test_check_drift(self):
        assert check_drift(40.0, 50.0) is True    # 25% drift
        assert check_drift(40.0, 60.0) is False   # 50% drift
        assert check_drift(0, 10) is True          # Zero base → allow

    def test_validate_proposed_value_valid(self):
        allowed, reason = validate_proposed_value("rsi_neutral_low", 40.0, 40.0, 38.0)
        assert allowed
        assert reason == ""

    def test_validate_proposed_value_out_of_bounds(self):
        allowed, reason = validate_proposed_value("rsi_neutral_low", 40.0, 40.0, 25.0)
        assert not allowed
        assert "bounds" in reason.lower()

    def test_validate_proposed_value_excessive_step(self):
        allowed, reason = validate_proposed_value("rsi_neutral_low", 40.0, 40.0, 32.0)
        assert not allowed
        assert "step" in reason.lower()


# -------------------------------------------------------------------------
# Full review session tests
# -------------------------------------------------------------------------

class TestRunDailyReview:
    def test_empty_state(self):
        """Empty state should produce a review with 0 trades."""
        state = PaperTradingState()
        session = run_daily_review(state, "atlas")
        assert session.trades_reviewed == 0
        assert "No closed trades" in session.notes[0]

    def test_review_with_trades(self):
        """Review with trades should produce classifications and signal data."""
        trades = [
            _make_trade(trade_id=f"t{i:03d}", realized_pnl=500 if i < 7 else -300)
            for i in range(10)
        ]
        state = PaperTradingState(trade_log=trades)
        session = run_daily_review(state, "atlas")
        assert session.trades_reviewed == 10
        assert len(session.classifications) == 10
        assert sum(session.classification_summary.values()) == 10
        assert len(session.signal_reliability) > 0

    def test_review_classification_summary(self):
        """Summary counts should match individual classifications."""
        trades = [_make_trade(trade_id=f"t{i:03d}") for i in range(5)]
        state = PaperTradingState(trade_log=trades)
        session = run_daily_review(state, "jarvis")
        total = sum(session.classification_summary.values())
        assert total == len(session.classifications)

    def test_review_flags_category_d(self):
        """Review should note Category D trades in notes."""
        ctx_bad = {
            "spot": 23000.0, "vwap": 23200.0,
            "ema_9": 22900.0, "ema_21": 23000.0, "ema_50": 23100.0,
            "rsi": 28.0,
            "supertrend_direction": -1,
            "pcr": 0.5,
            "atm_iv": 8.0,
            "bb_upper": 23500.0, "bb_middle": 23000.0, "bb_lower": 22500.0,
            "max_pain": 23500.0,
        }
        trades = [
            _make_trade(
                trade_id=f"t{i:03d}",
                realized_pnl=-500,
                score=20,
                entry_context=ctx_bad,
            )
            for i in range(5)
        ]
        state = PaperTradingState(trade_log=trades)
        session = run_daily_review(state, "atlas")
        assert any("Category D" in n for n in session.notes)


# -------------------------------------------------------------------------
# Improvement models tests
# -------------------------------------------------------------------------

class TestImprovementModels:
    def test_trade_classification_creation(self):
        tc = TradeClassification(
            trade_id="t001",
            strategy="Iron Condor",
            category="A",
            profitable=True,
            signal_alignment_count=5,
            signal_opposition_count=1,
            strategy_regime_fit=True,
            sizing_compliant=True,
            entry_quality_score=75.0,
        )
        assert tc.category == "A"
        assert tc.entry_quality_score == 75.0

    def test_improvement_proposal_defaults(self):
        p = ImprovementProposal(
            strategy_name="Iron Condor",
            parameter_name="rsi_neutral_low",
            current_value=40.0,
            proposed_value=38.0,
            default_value=40.0,
            change_pct=-0.05,
            evidence_trade_count=20,
            evidence_summary="test evidence",
            expected_impact="fewer false entries",
            reversion_criteria="5 consecutive losses",
            classification="calibration",
            confidence=0.7,
        )
        assert p.status == "proposed"
        assert len(p.id) > 0

    def test_review_session_creation(self):
        rs = ReviewSession(
            date="2026-02-12",
            algorithm="atlas",
            trades_reviewed=10,
            classification_summary={"A": 5, "B": 2, "C": 2, "D": 1},
        )
        assert rs.trades_reviewed == 10
        assert rs.classification_summary["A"] == 5
