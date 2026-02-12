"""Daily trade review engine — classifies trades, analyzes signals, and generates proposals.

All functions are pure (no side effects except logging). The ``run_daily_review``
orchestrator produces a ``ReviewSession`` that can be inspected before persisting.
"""

from __future__ import annotations

import logging
import statistics
from datetime import datetime, timedelta, timezone

from core.improvement_models import (
    ImprovementProposal,
    ParameterCalibration,
    ReviewSession,
    SignalReliability,
    TradeClassification,
)
from core.paper_trading_models import PaperTradingState, TradeRecord, _now_ist
from core.parameter_bounds import (
    MIN_SAMPLE_SIZE,
    validate_proposed_value,
)
from core.strategy_rules import STRATEGY_RULES

logger = logging.getLogger(__name__)

_IST = timezone(timedelta(hours=5, minutes=30))

# -------------------------------------------------------------------------
# Signal extraction from entry_context
# -------------------------------------------------------------------------

# Known signals and how to extract their state from a MarketContextSnapshot dict.
# Each extractor returns (signal_name, value, is_confirming_for_direction).

_CREDIT_STRATEGIES = {
    "Short Straddle", "Short Strangle", "Iron Condor",
    "Bull Put Spread", "Bear Call Spread",
}
_DEBIT_STRATEGIES = {
    "Long Straddle", "Long Strangle", "Bull Call Spread",
    "Bear Put Spread", "Long Call (CE)", "Long Put (PE)",
}
_BULLISH_STRATEGIES = {
    "Bull Put Spread", "Bull Call Spread", "Long Call (CE)",
}
_BEARISH_STRATEGIES = {
    "Bear Call Spread", "Bear Put Spread", "Long Put (PE)",
}
_NEUTRAL_STRATEGIES = {
    "Short Straddle", "Short Strangle", "Long Straddle",
    "Long Strangle", "Iron Condor",
}


def _extract_signals(
    ctx: dict, strategy: str, strategy_type: str, direction_bias: str,
) -> tuple[list[str], list[str], list[str]]:
    """Extract signals from entry_context and classify as aligned/opposed.

    Returns ``(all_signals, aligned, opposed)`` — lists of signal names.
    """
    if not ctx:
        return [], [], []

    signals_present: list[str] = []
    aligned: list[str] = []
    opposed: list[str] = []

    rsi = ctx.get("rsi", 0)
    spot = ctx.get("spot", 0)
    vwap = ctx.get("vwap", 0)
    ema_9 = ctx.get("ema_9", 0)
    ema_21 = ctx.get("ema_21", 0)
    ema_50 = ctx.get("ema_50", 0)
    supertrend_dir = ctx.get("supertrend_direction", 0)
    pcr = ctx.get("pcr", 0)
    atm_iv = ctx.get("atm_iv", 0)
    bb_upper = ctx.get("bb_upper", 0)
    bb_lower = ctx.get("bb_lower", 0)
    bb_middle = ctx.get("bb_middle", 0)
    max_pain = ctx.get("max_pain", 0)

    # --- RSI signal ---
    if rsi > 0:
        signals_present.append("rsi")
        if strategy in _NEUTRAL_STRATEGIES:
            if 40 <= rsi <= 60:
                aligned.append("rsi")
            else:
                opposed.append("rsi")
        elif strategy in _BULLISH_STRATEGIES:
            if 40 <= rsi < 70:
                aligned.append("rsi")
            elif rsi >= 70:
                opposed.append("rsi")
        elif strategy in _BEARISH_STRATEGIES:
            if 30 < rsi <= 60:
                aligned.append("rsi")
            elif rsi <= 30:
                opposed.append("rsi")

    # --- EMA alignment ---
    if ema_9 > 0 and ema_21 > 0 and ema_50 > 0:
        signals_present.append("ema_alignment")
        ema_bullish = ema_9 > ema_21 > ema_50
        ema_bearish = ema_9 < ema_21 < ema_50

        if strategy in _BULLISH_STRATEGIES:
            (aligned if ema_bullish else opposed).append("ema_alignment")
        elif strategy in _BEARISH_STRATEGIES:
            (aligned if ema_bearish else opposed).append("ema_alignment")
        elif strategy in _NEUTRAL_STRATEGIES:
            # Neutral prefers no strong trend
            if not ema_bullish and not ema_bearish:
                aligned.append("ema_alignment")
            else:
                opposed.append("ema_alignment")

    # --- Supertrend direction ---
    if supertrend_dir != 0:
        signals_present.append("supertrend")
        if strategy in _BULLISH_STRATEGIES:
            (aligned if supertrend_dir == 1 else opposed).append("supertrend")
        elif strategy in _BEARISH_STRATEGIES:
            (aligned if supertrend_dir == -1 else opposed).append("supertrend")
        elif strategy in _NEUTRAL_STRATEGIES:
            # Neutral is okay with either direction for credit; debit wants squeeze
            aligned.append("supertrend")

    # --- Spot vs VWAP ---
    if spot > 0 and vwap > 0:
        signals_present.append("spot_vs_vwap")
        above_vwap = spot > vwap
        if strategy in _BULLISH_STRATEGIES:
            (aligned if above_vwap else opposed).append("spot_vs_vwap")
        elif strategy in _BEARISH_STRATEGIES:
            (aligned if not above_vwap else opposed).append("spot_vs_vwap")
        elif strategy in _NEUTRAL_STRATEGIES:
            # Neutral prefers spot near VWAP
            pct_diff = abs(spot - vwap) / vwap * 100 if vwap else 0
            (aligned if pct_diff < 0.5 else opposed).append("spot_vs_vwap")

    # --- PCR ---
    if pcr > 0:
        signals_present.append("pcr")
        if strategy in _BULLISH_STRATEGIES:
            (aligned if pcr > 1.0 else opposed).append("pcr")
        elif strategy in _BEARISH_STRATEGIES:
            (aligned if pcr < 0.8 else opposed).append("pcr")
        elif strategy in _NEUTRAL_STRATEGIES:
            (aligned if 0.8 <= pcr <= 1.2 else opposed).append("pcr")

    # --- ATM IV level ---
    if atm_iv > 0:
        signals_present.append("atm_iv")
        if strategy_type == "credit":
            # Credit strategies prefer moderate-to-high IV
            (aligned if atm_iv >= 14 else opposed).append("atm_iv")
        else:
            # Debit strategies prefer low IV (cheap premiums)
            (aligned if atm_iv < 18 else opposed).append("atm_iv")

    # --- BB width ---
    if bb_upper > 0 and bb_lower > 0 and bb_middle > 0:
        signals_present.append("bb_width")
        bb_width_pct = (bb_upper - bb_lower) / bb_middle * 100 if bb_middle else 0
        if strategy_type == "credit":
            # Credit wants tight/range-bound
            (aligned if bb_width_pct < 1.5 else opposed).append("bb_width")
        else:
            # Debit wants squeeze → breakout
            (aligned if bb_width_pct < 1.0 else opposed).append("bb_width")

    # --- Max pain proximity ---
    if max_pain > 0 and spot > 0:
        signals_present.append("max_pain")
        mp_pct = abs(spot - max_pain) / spot * 100
        if strategy in _NEUTRAL_STRATEGIES:
            (aligned if mp_pct < 0.5 else opposed).append("max_pain")
        else:
            # Directional: being away from max pain is fine
            aligned.append("max_pain")

    return signals_present, aligned, opposed


# -------------------------------------------------------------------------
# Strategy-regime fit
# -------------------------------------------------------------------------

def _check_regime_fit(
    ctx: dict, strategy: str, strategy_type: str,
) -> bool:
    """Check if the strategy type matched the vol/price regime at entry."""
    if not ctx:
        return True  # No data to judge — assume fit

    atm_iv = ctx.get("atm_iv", 0)
    bb_upper = ctx.get("bb_upper", 0)
    bb_lower = ctx.get("bb_lower", 0)
    bb_middle = ctx.get("bb_middle", 0)
    bb_width_pct = (
        (bb_upper - bb_lower) / bb_middle * 100 if bb_middle else 0
    )

    if strategy_type == "credit":
        # Credit strategies need range-bound or high IV environment
        # Bad fit: very tight BB (squeeze → imminent breakout) with low IV
        if bb_width_pct < 0.6 and atm_iv < 14:
            return False  # Squeeze + low IV = bad for credit
        return True
    else:
        # Debit strategies need breakout potential or cheap premiums
        # Bad fit: wide BB (already expanded) with high IV (expensive)
        if bb_width_pct > 2.0 and atm_iv > 20:
            return False  # Expanded + expensive = bad for debit
        return True


# -------------------------------------------------------------------------
# Trade classification
# -------------------------------------------------------------------------

def classify_trade(
    record: TradeRecord,
) -> TradeClassification:
    """Classify a single closed trade on the entry-quality / outcome matrix.

    Returns a ``TradeClassification`` with category A/B/C/D.
    """
    ctx = record.entry_context or {}
    profitable = record.realized_pnl > 0

    signals_present, aligned, opposed = _extract_signals(
        ctx, record.strategy, record.strategy_type, record.direction_bias,
    )
    regime_fit = _check_regime_fit(ctx, record.strategy, record.strategy_type)

    # Position sizing compliance (basic check)
    sizing_ok = True  # We don't have capital info per-trade, assume ok

    # Entry quality score: weighted combination
    alignment_score = (
        (len(aligned) / max(len(signals_present), 1)) * 60 if signals_present else 30
    )
    regime_score = 25 if regime_fit else 0
    # Score from strategy evaluator
    eval_score = min(record.score / 100 * 15, 15) if record.score > 0 else 0
    entry_quality = min(alignment_score + regime_score + eval_score, 100)

    # Classification: "technically sound" = quality >= 50 AND >= 3 aligned signals
    technically_sound = entry_quality >= 50 and len(aligned) >= 3

    if technically_sound and profitable:
        category = "A"
    elif not technically_sound and profitable:
        category = "B"
    elif technically_sound and not profitable:
        category = "C"
    else:
        category = "D"

    notes: list[str] = []
    if len(aligned) < 3:
        notes.append(f"Only {len(aligned)} aligned signals (need 3+)")
    if not regime_fit:
        notes.append("Strategy-regime mismatch at entry")
    if len(opposed) > len(aligned):
        notes.append("More opposing signals than aligned at entry")

    return TradeClassification(
        trade_id=record.id,
        strategy=record.strategy,
        category=category,
        profitable=profitable,
        signal_alignment_count=len(aligned),
        signal_opposition_count=len(opposed),
        strategy_regime_fit=regime_fit,
        sizing_compliant=sizing_ok,
        entry_quality_score=round(entry_quality, 1),
        signals_present=signals_present,
        signals_aligned=aligned,
        signals_opposed=opposed,
        notes=notes,
    )


# -------------------------------------------------------------------------
# Signal reliability analysis
# -------------------------------------------------------------------------

def compute_signal_reliability(
    trades: list[TradeRecord],
    window_days: int = 20,
) -> list[SignalReliability]:
    """Compute per-signal reliability metrics over a rolling window.

    Returns a list sorted by predictive accuracy (worst first).
    """
    now = _now_ist()
    cutoff = now - timedelta(days=window_days)

    recent_trades = [
        t for t in trades
        if t.exit_time and t.exit_time.replace(tzinfo=_IST if t.exit_time.tzinfo is None else t.exit_time.tzinfo) >= cutoff
    ]

    # Accumulate per-signal stats
    stats: dict[str, dict[str, int]] = {}

    for trade in recent_trades:
        ctx = trade.entry_context or {}
        won = trade.realized_pnl > 0
        _, aligned, opposed = _extract_signals(
            ctx, trade.strategy, trade.strategy_type, trade.direction_bias,
        )

        all_signals = set(aligned) | set(opposed)
        for sig in all_signals:
            if sig not in stats:
                stats[sig] = {
                    "total": 0,
                    "confirming_wins": 0,
                    "confirming_losses": 0,
                    "opposing_wins": 0,
                    "opposing_losses": 0,
                }
            stats[sig]["total"] += 1
            if sig in aligned:
                if won:
                    stats[sig]["confirming_wins"] += 1
                else:
                    stats[sig]["confirming_losses"] += 1
            elif sig in opposed:
                if won:
                    stats[sig]["opposing_wins"] += 1
                else:
                    stats[sig]["opposing_losses"] += 1

    result = [
        SignalReliability(
            signal_name=name,
            total_trades=s["total"],
            confirming_wins=s["confirming_wins"],
            confirming_losses=s["confirming_losses"],
            opposing_wins=s["opposing_wins"],
            opposing_losses=s["opposing_losses"],
        )
        for name, s in stats.items()
    ]

    # Sort by predictive accuracy ascending (worst signals first)
    result.sort(key=lambda r: r.predictive_accuracy)
    return result


# -------------------------------------------------------------------------
# Parameter calibration (SL/PT via MAE/MFE)
# -------------------------------------------------------------------------

def calibrate_parameters(
    trades: list[TradeRecord],
) -> list[ParameterCalibration]:
    """Analyze MAE/MFE distributions to suggest SL/PT calibration.

    Groups trades by strategy and produces calibration entries for strategies
    with enough data.
    """
    from collections import defaultdict

    by_strategy: dict[str, list[TradeRecord]] = defaultdict(list)
    for t in trades:
        by_strategy[t.strategy].append(t)

    results: list[ParameterCalibration] = []

    for strategy_name, strades in by_strategy.items():
        if len(strades) < MIN_SAMPLE_SIZE:
            continue

        winners = [t for t in strades if t.realized_pnl > 0]
        losers = [t for t in strades if t.realized_pnl <= 0]

        # MAE distributions
        winner_maes = [t.max_drawdown for t in winners if t.max_drawdown > 0]
        loser_maes = [t.max_drawdown for t in losers if t.max_drawdown > 0]

        mae_med_w = statistics.median(winner_maes) if winner_maes else 0
        mae_med_l = statistics.median(loser_maes) if loser_maes else 0

        # MFE for winners
        winner_mfes = [t.max_favorable for t in winners if t.max_favorable > 0]
        mfe_med = statistics.median(winner_mfes) if winner_mfes else 0

        # SL analysis: what % of SL trades reversed (had positive MFE)?
        sl_trades = [
            t for t in strades if t.exit_reason == "closed_stop_loss"
        ]
        sl_then_reversed = [t for t in sl_trades if t.max_favorable > 0]
        sl_rev_pct = (
            len(sl_then_reversed) / len(sl_trades) * 100 if sl_trades else 0
        )

        # PT analysis: what % of winners exceeded PT significantly?
        pt_exceeded = 0
        if winners:
            for w in winners:
                if w.profit_target_amount > 0 and w.max_favorable > 0:
                    ratio = w.max_favorable / w.profit_target_amount
                    if ratio > 1.2:
                        pt_exceeded += 1
        pt_exc_pct = pt_exceeded / len(winners) * 100 if winners else 0

        # Get current SL/PT values from strategy rules (if available)
        rules = STRATEGY_RULES.get(strategy_name, {})
        strategy_type = rules.get("type", "")

        results.append(ParameterCalibration(
            parameter_name="stop_loss_calibration",
            strategy_name=strategy_name,
            current_value=0,  # SL is in algorithm config, not strategy_rules
            default_value=0,
            evidence_trades=len(strades),
            sl_hit_then_reversed_pct=round(sl_rev_pct, 1),
            pt_exceeded_pct=round(pt_exc_pct, 1),
            mae_median_winners=round(mae_med_w, 2),
            mae_median_losers=round(mae_med_l, 2),
            mfe_median=round(mfe_med, 2),
            suggestion_reason=(
                "SL may be too tight"
                if sl_rev_pct > 50
                else (
                    "PT may be too conservative"
                    if pt_exc_pct > 60
                    else ""
                )
            ),
        ))

    return results


# -------------------------------------------------------------------------
# Safety rail check
# -------------------------------------------------------------------------

def check_safety_rails(
    proposal: ImprovementProposal,
    cooling_periods: dict[str, int],
    session_proposal_count: int,
) -> tuple[bool, str]:
    """Validate a proposal against all safety rails.

    Returns ``(allowed, reason_if_blocked)``.
    """
    # Sample size
    if proposal.evidence_trade_count < MIN_SAMPLE_SIZE:
        return False, (
            f"Insufficient evidence: {proposal.evidence_trade_count} trades "
            f"(need {MIN_SAMPLE_SIZE})"
        )

    # Parameter bounds + step size + drift
    allowed, reason = validate_proposed_value(
        proposal.parameter_name,
        proposal.default_value,
        proposal.current_value,
        proposal.proposed_value,
    )
    if not allowed:
        return False, reason

    # Cooling period
    key = f"{proposal.strategy_name}.{proposal.parameter_name}"
    if key in cooling_periods:
        return False, (
            f"Cooling period: {cooling_periods[key]} trades remaining for {key}"
        )

    # Session budget
    from core.parameter_bounds import MAX_CHANGES_PER_SESSION
    if session_proposal_count >= MAX_CHANGES_PER_SESSION:
        return False, (
            f"Session budget exhausted ({MAX_CHANGES_PER_SESSION} max)"
        )

    return True, ""


# -------------------------------------------------------------------------
# Review session orchestrator
# -------------------------------------------------------------------------

def run_daily_review(
    state: PaperTradingState,
    algo_name: str,
    strategy_rules: dict | None = None,
    window_days: int = 20,
) -> ReviewSession:
    """Run a full daily review on closed trades.

    This is a pure function — it reads state and produces a ``ReviewSession``
    without any side effects. Persist the session separately if desired.
    """
    if strategy_rules is None:
        strategy_rules = STRATEGY_RULES

    now = _now_ist()
    today = now.strftime("%Y-%m-%d")

    trades = state.trade_log
    if not trades:
        return ReviewSession(
            date=today,
            algorithm=algo_name,
            trades_reviewed=0,
            notes=["No closed trades to review"],
        )

    # --- Step 1: Classify all trades ---
    classifications: list[TradeClassification] = []
    for trade in trades:
        classifications.append(classify_trade(trade))

    cat_summary = {"A": 0, "B": 0, "C": 0, "D": 0}
    for c in classifications:
        cat_summary[c.category] += 1

    # --- Step 2: Signal reliability ---
    signal_reliability = compute_signal_reliability(trades, window_days)

    # --- Step 3: Parameter calibration ---
    calibrations = calibrate_parameters(trades)

    # --- Step 4: Generate notes ---
    notes: list[str] = []

    # Flag low-reliability signals
    for sig in signal_reliability:
        if sig.total_trades >= 10 and sig.predictive_accuracy < 0.45:
            notes.append(
                f"Signal '{sig.signal_name}' has {sig.predictive_accuracy:.0%} "
                f"accuracy over {sig.total_trades} trades — worse than coin flip"
            )

    # Flag SL/PT issues
    for cal in calibrations:
        if cal.sl_hit_then_reversed_pct > 50:
            notes.append(
                f"{cal.strategy_name}: {cal.sl_hit_then_reversed_pct:.0f}% of "
                f"SL exits reversed — SL may be too tight"
            )
        if cal.pt_exceeded_pct > 60:
            notes.append(
                f"{cal.strategy_name}: {cal.pt_exceeded_pct:.0f}% of winners "
                f"exceeded PT significantly — PT may be too conservative"
            )

    # Category D analysis
    d_trades = [c for c in classifications if c.category == "D"]
    if d_trades:
        notes.append(
            f"{len(d_trades)} Category D trades (flawed entry + loss) — "
            f"investigate for structural patterns"
        )

    # Category B analysis
    b_trades = [c for c in classifications if c.category == "B"]
    if b_trades:
        notes.append(
            f"{len(b_trades)} Category B trades (flawed entry but profitable) — "
            f"these are 'lucky' trades; fix the process"
        )

    return ReviewSession(
        date=today,
        algorithm=algo_name,
        trades_reviewed=len(trades),
        classification_summary=cat_summary,
        classifications=classifications,
        signal_reliability=signal_reliability,
        parameter_calibrations=calibrations,
        proposals=[],  # Proposals are generated by Claude Code after reviewing
        notes=notes,
    )
