"""V4 Atlas algorithm — Distribution-based dynamic thresholds.

Every threshold is a function of NIFTY's own historical volatility distribution.
Replaces fixed constants with percentile-based gates derived from realized vol,
vol-of-vol, and the volatility risk premium.

Key features:
  - 9 dynamic parameter functions of (pRV, pVoV, pVRP)
  - Regime classifier: sell_premium / buy_premium / stand_down
  - All regime/parameter decisions driven by vol distribution percentiles
  - Fallback to neutral snapshot (percentiles = 0.5) when data unavailable
"""

from __future__ import annotations

import logging
import math
import time as _time
from datetime import datetime, timedelta, timezone

from algorithms import register_algorithm
from algorithms.base import TradingAlgorithm
from core.event_calendar import is_near_event
from core.greeks import bs_delta, compute_pop
from core.market_hours import is_market_open
from core.options_analytics import compute_liquidity_score
from core.options_models import (
    OptionChainData,
    OptionsAnalytics,
    StrategyName,
    TechnicalIndicators,
    TradeLeg,
    TradeSuggestion,
)
from core.paper_trading_engine import (
    check_exit_conditions,
    classify_strategy,
    close_position,
    compute_execution_cost,
    compute_net_premium,
    open_position,
    update_position_ltp,
)
from core.paper_trading_models import (
    PaperPosition,
    PaperTradingState,
    PositionStatus,
    StrategyType,
    TradeRecord,
)
from core.vol_distribution import VolSnapshot, get_today_vol_snapshot

logger = logging.getLogger(__name__)

_IST = timezone(timedelta(hours=5, minutes=30))

# ---------------------------------------------------------------------------
# Strategy classification sets (same as optimus.py)
# ---------------------------------------------------------------------------

_ATLAS_CREDIT = {
    StrategyName.IRON_CONDOR,
    StrategyName.SHORT_STRANGLE,
    StrategyName.BULL_PUT_SPREAD,
    StrategyName.BEAR_CALL_SPREAD,
}

_ATLAS_DEBIT = {
    StrategyName.BULL_CALL_SPREAD,
    StrategyName.BEAR_PUT_SPREAD,
}

_ATLAS_PRIMARY = {
    StrategyName.IRON_CONDOR,
}

_ATLAS_SECONDARY = {
    StrategyName.SHORT_STRANGLE,
    StrategyName.BULL_CALL_SPREAD,
    StrategyName.BEAR_PUT_SPREAD,
}

_SPREAD_STRATEGIES = {
    StrategyName.IRON_CONDOR,
    StrategyName.BULL_PUT_SPREAD,
    StrategyName.BEAR_CALL_SPREAD,
    StrategyName.BULL_CALL_SPREAD,
    StrategyName.BEAR_PUT_SPREAD,
}


def _now_ist() -> datetime:
    return datetime.now(_IST)


# ---------------------------------------------------------------------------
# Neutral fallback snapshot
# ---------------------------------------------------------------------------

def _neutral_snapshot() -> VolSnapshot:
    """Return a neutral snapshot with all percentiles at 0.5."""
    return VolSnapshot(
        rv_5=15.0, rv_10=15.0, rv_20=15.0,
        vov_20=1.0, vix=15.0, vrp=0.0,
        p_rv=0.5, p_vov=0.5, p_vrp=0.5,
        em=0.0, date=_now_ist().strftime("%Y-%m-%d"),
    )


# ---------------------------------------------------------------------------
# 9 Dynamic Parameter Functions
# ---------------------------------------------------------------------------

def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def _dynamic_strike_delta(p_rv: float, cfg: dict) -> float:
    """Short strike delta target: lower delta when vol is high."""
    base = cfg.get("strike_delta_base", 0.20)
    coeff = cfg.get("strike_delta_coeff", 0.08)
    return _clamp(base - coeff * p_rv, 0.10, 0.20)


def _dynamic_min_credit_ratio(p_vrp: float, cfg: dict) -> float:
    """Min credit/width gate: higher when VRP is rich."""
    base = cfg.get("credit_ratio_base", 0.22)
    coeff = cfg.get("credit_ratio_coeff", 0.16)
    return _clamp(base + coeff * p_vrp, 0.22, 0.40)


def _dynamic_sl_multiple(p_vov: float, cfg: dict) -> float:
    """Stop loss multiplier: wider when vol-of-vol is high."""
    base = cfg.get("sl_multiple_base", 1.6)
    coeff = cfg.get("sl_multiple_coeff", 0.8)
    return _clamp(base + coeff * p_vov, 1.6, 2.5)


def _dynamic_delta_exit(p_rv: float, cfg: dict) -> float:
    """Emergency delta exit threshold: tighter when vol is high."""
    base = cfg.get("delta_exit_base", 0.22)
    coeff = cfg.get("delta_exit_coeff", 0.10)
    return _clamp(base + coeff * p_rv, 0.22, 0.33)


def _dynamic_take_profit(p_rv: float, cfg: dict) -> float:
    """Profit capture %: take profits faster when vol is low."""
    base = cfg.get("take_profit_base", 0.45)
    coeff = cfg.get("take_profit_coeff", 0.20)
    return _clamp(base + coeff * (1 - p_rv), 0.45, 0.70)


def _dynamic_breakeven_mult(p_vov: float, cfg: dict) -> float:
    """Expected move coverage requirement: wider when VoV is high."""
    base = cfg.get("breakeven_mult_base", 1.15)
    coeff = cfg.get("breakeven_mult_coeff", 0.30)
    return _clamp(base + coeff * p_vov, 1.15, 1.50)


def _dynamic_risk_per_trade(p_rv: float, p_vov: float, cfg: dict) -> float:
    """Per-trade risk budget as % of account. Shrinks in high vol."""
    base_pct = cfg.get("risk_per_trade_base_pct", 0.50)
    a = cfg.get("risk_per_trade_rv_coeff", 0.5)
    b = cfg.get("risk_per_trade_vov_coeff", 0.5)
    factor = _clamp((1 - a * p_rv) * (1 - b * p_vov), 0.25, 1.0)
    return _clamp(base_pct * factor, 0.125, 0.50)


def _dynamic_max_portfolio_risk(p_vov: float, cfg: dict) -> float:
    """Portfolio risk cap as %. Shrinks when VoV is high."""
    base_pct = cfg.get("max_portfolio_risk_base_pct", 3.0)
    coeff = cfg.get("max_portfolio_risk_coeff", 0.5)
    return _clamp(base_pct * (1 - coeff * p_vov), 1.5, 3.0)


def _classify_regime(vol: VolSnapshot, cfg: dict) -> str:
    """Classify volatility regime.

    Returns: "sell_premium", "buy_premium", or "stand_down"
    """
    # Priority: stand_down (safety first)
    if vol.p_vov >= cfg.get("standdown_vov_min", 0.85):
        return "stand_down"
    if vol.p_rv >= cfg.get("standdown_rv_min", 0.90):
        return "stand_down"

    # Sell premium: VRP rich and vol stable
    sell_vrp_min = cfg.get("regime_sell_vrp_min", 0.55)
    sell_vov_max = cfg.get("regime_sell_vov_max", 0.70)
    if vol.p_vrp >= sell_vrp_min and vol.p_vov <= sell_vov_max:
        return "sell_premium"

    # Buy premium: VRP cheap (options underpriced)
    if vol.p_vrp <= 0.30:
        return "buy_premium"

    # Ambiguous
    return "stand_down"


def _get_all_dynamic_params(vol: VolSnapshot, cfg: dict) -> dict:
    """Compute all 9 dynamic parameters and return as dict for logging/display."""
    return {
        "strike_delta": round(_dynamic_strike_delta(vol.p_rv, cfg), 4),
        "min_credit_ratio": round(_dynamic_min_credit_ratio(vol.p_vrp, cfg), 4),
        "sl_multiple": round(_dynamic_sl_multiple(vol.p_vov, cfg), 2),
        "delta_exit": round(_dynamic_delta_exit(vol.p_rv, cfg), 4),
        "take_profit": round(_dynamic_take_profit(vol.p_rv, cfg), 4),
        "breakeven_mult": round(_dynamic_breakeven_mult(vol.p_vov, cfg), 3),
        "risk_per_trade_pct": round(_dynamic_risk_per_trade(vol.p_rv, vol.p_vov, cfg), 4),
        "max_portfolio_risk_pct": round(_dynamic_max_portfolio_risk(vol.p_vov, cfg), 2),
        "regime": _classify_regime(vol, cfg),
    }


# ---------------------------------------------------------------------------
# Helpers (shared with hedge.py pattern)
# ---------------------------------------------------------------------------

def _compute_expected_move(spot: float, vix: float, dte: int) -> float:
    if spot <= 0 or vix <= 0 or dte <= 0:
        return 0.0
    return spot * (vix / 100.0) * math.sqrt(dte / 365.0)


def _parse_dte(expiry_str: str) -> int:
    try:
        expiry_date = datetime.strptime(expiry_str, "%d-%b-%Y").date()
        today = _now_ist().date()
        return max(0, (expiry_date - today).days)
    except (ValueError, TypeError):
        return 0


def _get_strike_data(chain: OptionChainData, strike_price: float):
    for s in chain.strikes:
        if s.strike_price == strike_price:
            return s
    return None


def _compute_spread_width(legs: list[TradeLeg]) -> float:
    strikes = sorted(set(leg.strike for leg in legs))
    if len(strikes) >= 2:
        return strikes[-1] - strikes[0]
    return 0.0


def _compute_bb_width_pct(technicals: TechnicalIndicators) -> float:
    if technicals.bb_middle > 0:
        return ((technicals.bb_upper - technicals.bb_lower) / technicals.bb_middle) * 100
    return 0.0


def _is_breakout(technicals: TechnicalIndicators) -> bool:
    if technicals.bb_upper <= 0:
        return False
    above_bb = technicals.spot > technicals.bb_upper
    below_bb = technicals.spot < technicals.bb_lower
    if above_bb and technicals.supertrend_direction == 1:
        return True
    if below_bb and technicals.supertrend_direction == -1:
        return True
    return False


def _reset_period_tracking(state: PaperTradingState, cfg: dict) -> PaperTradingState:
    now = _now_ist()
    today_str = now.strftime("%Y-%m-%d")
    current_capital = state.initial_capital + state.net_realized_pnl
    updates: dict = {}

    if state.session_date != today_str:
        updates["session_date"] = today_str
        updates["daily_start_capital"] = current_capital

    current_week = now.strftime("%Y-W%W")
    if state.week_start_date != current_week:
        updates["week_start_date"] = current_week
        updates["weekly_start_capital"] = current_capital

    if updates:
        return state.model_copy(update=updates)
    return state


# ===================================================================
# Trade Construction — Iron Condor (dynamic parameters)
# ===================================================================

def _build_atlas_iron_condor(
    chain: OptionChainData,
    analytics: OptionsAnalytics,
    technicals: TechnicalIndicators,
    vol: VolSnapshot,
    cfg: dict,
) -> TradeSuggestion | None:
    """Build iron condor using dynamic delta/credit/breakeven targets."""
    spot = chain.underlying_value
    if spot <= 0:
        return None

    dte = _parse_dte(chain.expiry)
    if dte <= 0:
        return None

    # Dynamic parameters
    delta_target = _dynamic_strike_delta(vol.p_rv, cfg)
    delta_min = max(0.08, delta_target - 0.04)
    delta_max = delta_target + 0.04
    min_credit_ratio = _dynamic_min_credit_ratio(vol.p_vrp, cfg)
    be_mult = _dynamic_breakeven_mult(vol.p_vov, cfg)

    widths = cfg.get("ic_widths", [50, 100])
    max_rr = cfg.get("ic_max_risk_reward", 2.0)

    expected_move = _compute_expected_move(spot, vol.vix, dte)

    best_candidate: TradeSuggestion | None = None
    best_score = 0.0

    ce_candidates = []
    pe_candidates = []
    for sd in chain.strikes:
        if sd.strike_price <= spot and sd.pe_delta != 0:
            pe_delta = abs(sd.pe_delta)
            if delta_min <= pe_delta <= delta_max and sd.pe_ltp > 0:
                pe_candidates.append(sd)
        if sd.strike_price >= spot and sd.ce_delta != 0:
            ce_delta = abs(sd.ce_delta)
            if delta_min <= ce_delta <= delta_max and sd.ce_ltp > 0:
                ce_candidates.append(sd)

    for short_ce in ce_candidates:
        for short_pe in pe_candidates:
            for width in widths:
                long_ce_strike = short_ce.strike_price + width
                long_pe_strike = short_pe.strike_price - width

                long_ce_sd = _get_strike_data(chain, long_ce_strike)
                long_pe_sd = _get_strike_data(chain, long_pe_strike)

                if long_ce_sd is None or long_pe_sd is None:
                    continue
                if long_ce_sd.ce_ltp <= 0 or long_pe_sd.pe_ltp <= 0:
                    continue

                credit = (
                    short_ce.ce_ltp + short_pe.pe_ltp
                    - long_ce_sd.ce_ltp - long_pe_sd.pe_ltp
                )
                if credit <= 0:
                    continue

                credit_pct = credit / width
                if credit_pct < min_credit_ratio:
                    continue

                max_loss_per_unit = width - credit
                if credit > 0:
                    rr = max_loss_per_unit / credit
                else:
                    continue
                if rr > max_rr:
                    continue

                # Breakeven check with dynamic multiplier
                upper_be = short_ce.strike_price + credit
                lower_be = short_pe.strike_price - credit
                if expected_move > 0:
                    upper_dist = upper_be - spot
                    lower_dist = spot - lower_be
                    if upper_dist < be_mult * expected_move or lower_dist < be_mult * expected_move:
                        continue

                avg_delta = (abs(short_ce.ce_delta) + abs(short_pe.pe_delta)) / 2
                score = credit_pct * (1 - avg_delta) * 10000

                if score > best_score:
                    best_score = score
                    lot_size = cfg.get("lot_size", 75)
                    best_candidate = TradeSuggestion(
                        strategy=StrategyName.IRON_CONDOR,
                        legs=[
                            TradeLeg(action="SELL", instrument=f"NIFTY {int(short_ce.strike_price)} CE",
                                     strike=short_ce.strike_price, option_type="CE", ltp=short_ce.ce_ltp),
                            TradeLeg(action="BUY", instrument=f"NIFTY {int(long_ce_strike)} CE",
                                     strike=long_ce_strike, option_type="CE", ltp=long_ce_sd.ce_ltp),
                            TradeLeg(action="SELL", instrument=f"NIFTY {int(short_pe.strike_price)} PE",
                                     strike=short_pe.strike_price, option_type="PE", ltp=short_pe.pe_ltp),
                            TradeLeg(action="BUY", instrument=f"NIFTY {int(long_pe_strike)} PE",
                                     strike=long_pe_strike, option_type="PE", ltp=long_pe_sd.pe_ltp),
                        ],
                        direction_bias="Neutral",
                        confidence="High",
                        score=round(score, 1),
                        entry_timing="Enter now — vol regime favors premium selling",
                        technicals_to_check=[
                            f"pRV: {vol.p_rv:.2f}, pVoV: {vol.p_vov:.2f}, pVRP: {vol.p_vrp:.2f}",
                            f"Dynamic delta target: {delta_target:.3f}",
                            f"Dynamic credit ratio: {min_credit_ratio:.3f}",
                        ],
                        expected_outcome=f"Profit if NIFTY stays between {int(lower_be)}-{int(upper_be)}",
                        max_profit=f"₹{credit * lot_size:,.0f} per lot",
                        max_loss=f"₹{max_loss_per_unit * lot_size:,.0f} per lot",
                        stop_loss=f"Dynamic SL: {_dynamic_sl_multiple(vol.p_vov, cfg):.1f}x credit | delta > {_dynamic_delta_exit(vol.p_rv, cfg):.3f}",
                        position_size="Per Atlas dynamic sizing",
                        reasoning=[
                            f"Iron condor: SELL {int(short_pe.strike_price)} PE / {int(short_ce.strike_price)} CE, width {width}",
                            f"Net credit: ₹{credit:.1f} ({credit_pct * 100:.1f}% of width)",
                            f"RR: 1:{rr:.1f}, breakevens: {int(lower_be)}-{int(upper_be)}",
                            f"Short deltas: CE={abs(short_ce.ce_delta):.3f}, PE={abs(short_pe.pe_delta):.3f}",
                            f"Vol regime: pRV={vol.p_rv:.2f} pVoV={vol.p_vov:.2f} pVRP={vol.p_vrp:.2f}",
                            f"Expected move: ±{expected_move:.0f} pts ({dte} DTE), BE mult: {be_mult:.2f}x",
                        ],
                    )

    return best_candidate


# ===================================================================
# Hedged Strangle Construction (dynamic parameters)
# ===================================================================

def _build_atlas_hedged_strangle(
    chain: OptionChainData,
    analytics: OptionsAnalytics,
    technicals: TechnicalIndicators,
    vol: VolSnapshot,
    cfg: dict,
) -> TradeSuggestion | None:
    """Build hedged strangle using vol distribution regime check."""
    spot = chain.underlying_value
    if spot <= 0:
        return None

    # Regime check: require rich VRP instead of fixed IV percentile
    if vol.p_vrp < 0.60:
        return None

    # Dynamic delta range
    delta_target = _dynamic_strike_delta(vol.p_rv, cfg)
    delta_min = max(0.06, delta_target - 0.05)
    delta_max = delta_target + 0.03
    min_credit_pct = cfg.get("strangle_min_credit_pct_spot", 2.0)
    wing_offset = cfg.get("strangle_wing_offset", 200)

    best_candidate: TradeSuggestion | None = None
    best_score = 0.0

    ce_candidates = []
    pe_candidates = []
    for sd in chain.strikes:
        if sd.strike_price >= spot and sd.ce_delta != 0:
            ce_delta = abs(sd.ce_delta)
            if delta_min <= ce_delta <= delta_max and sd.ce_ltp > 0:
                ce_candidates.append(sd)
        if sd.strike_price <= spot and sd.pe_delta != 0:
            pe_delta = abs(sd.pe_delta)
            if delta_min <= pe_delta <= delta_max and sd.pe_ltp > 0:
                pe_candidates.append(sd)

    for short_ce in ce_candidates:
        for short_pe in pe_candidates:
            long_ce_strike = short_ce.strike_price + wing_offset
            long_pe_strike = short_pe.strike_price - wing_offset

            long_ce_sd = _get_strike_data(chain, long_ce_strike)
            long_pe_sd = _get_strike_data(chain, long_pe_strike)

            if long_ce_sd is None or long_pe_sd is None:
                continue
            if long_ce_sd.ce_ltp <= 0 or long_pe_sd.pe_ltp <= 0:
                continue

            credit = (
                short_ce.ce_ltp + short_pe.pe_ltp
                - long_ce_sd.ce_ltp - long_pe_sd.pe_ltp
            )
            if credit <= 0:
                continue

            credit_pct_spot = (credit / spot) * 100
            if credit_pct_spot < min_credit_pct:
                continue

            max_loss_per_unit = wing_offset - credit
            if credit > 0:
                rr = max_loss_per_unit / credit
            else:
                continue

            lot_size = cfg.get("lot_size", 75)
            avg_delta = (abs(short_ce.ce_delta) + abs(short_pe.pe_delta)) / 2
            score = credit_pct_spot * (1 - avg_delta) * 100

            if score > best_score:
                best_score = score
                best_candidate = TradeSuggestion(
                    strategy=StrategyName.SHORT_STRANGLE,
                    legs=[
                        TradeLeg(action="SELL", instrument=f"NIFTY {int(short_ce.strike_price)} CE",
                                 strike=short_ce.strike_price, option_type="CE", ltp=short_ce.ce_ltp),
                        TradeLeg(action="SELL", instrument=f"NIFTY {int(short_pe.strike_price)} PE",
                                 strike=short_pe.strike_price, option_type="PE", ltp=short_pe.pe_ltp),
                        TradeLeg(action="BUY", instrument=f"NIFTY {int(long_ce_strike)} CE",
                                 strike=long_ce_strike, option_type="CE", ltp=long_ce_sd.ce_ltp),
                        TradeLeg(action="BUY", instrument=f"NIFTY {int(long_pe_strike)} PE",
                                 strike=long_pe_strike, option_type="PE", ltp=long_pe_sd.pe_ltp),
                    ],
                    direction_bias="Neutral",
                    confidence="High",
                    score=round(score, 1),
                    entry_timing="Enter now — rich VRP with stable vol regime",
                    technicals_to_check=[
                        f"pVRP: {vol.p_vrp:.2f} (rich — options overpriced)",
                        f"pVoV: {vol.p_vov:.2f} (stable)",
                        f"Dynamic delta: {delta_target:.3f}",
                    ],
                    expected_outcome=f"Profit if NIFTY stays between {int(short_pe.strike_price)}-{int(short_ce.strike_price)}",
                    max_profit=f"₹{credit * lot_size:,.0f} per lot",
                    max_loss=f"₹{max_loss_per_unit * lot_size:,.0f} per lot",
                    stop_loss=f"Dynamic SL: {_dynamic_sl_multiple(vol.p_vov, cfg):.1f}x | delta > {_dynamic_delta_exit(vol.p_rv, cfg):.3f}",
                    position_size="Per Atlas dynamic sizing",
                    reasoning=[
                        f"Hedged strangle: SELL {int(short_pe.strike_price)} PE / {int(short_ce.strike_price)} CE",
                        f"Protective wings at {int(long_pe_strike)} PE / {int(long_ce_strike)} CE (offset {wing_offset})",
                        f"Net credit: ₹{credit:.1f} ({credit_pct_spot:.2f}% of spot)",
                        f"Vol regime: pVRP={vol.p_vrp:.2f} (rich), pVoV={vol.p_vov:.2f} (stable)",
                        f"Short deltas: CE={abs(short_ce.ce_delta):.3f}, PE={abs(short_pe.pe_delta):.3f}",
                    ],
                )

    return best_candidate


# ===================================================================
# Debit Spread Construction (buy_premium regime)
# ===================================================================

def _build_atlas_debit_spread(
    chain: OptionChainData,
    analytics: OptionsAnalytics,
    technicals: TechnicalIndicators,
    vol: VolSnapshot,
    cfg: dict,
) -> TradeSuggestion | None:
    """Build debit spread in buy_premium regime (cheap options)."""
    spot = chain.underlying_value
    if spot <= 0:
        return None

    # Breakout must be confirmed for directional debit
    if not _is_breakout(technicals):
        return None

    min_rr = cfg.get("debit_min_risk_reward", 2.0)
    lot_size = cfg.get("lot_size", 75)

    is_bullish = technicals.supertrend_direction == 1

    if is_bullish:
        strategy = StrategyName.BULL_CALL_SPREAD
        atm_strike = analytics.atm_strike
        atm_sd = _get_strike_data(chain, atm_strike)
        if atm_sd is None or atm_sd.ce_ltp <= 0:
            return None

        for width in [100, 50]:
            sell_strike = atm_strike + width
            sell_sd = _get_strike_data(chain, sell_strike)
            if sell_sd is None or sell_sd.ce_ltp <= 0:
                continue

            net_debit = atm_sd.ce_ltp - sell_sd.ce_ltp
            if net_debit <= 0:
                continue

            max_profit = width - net_debit
            rr = max_profit / net_debit
            if rr < min_rr:
                continue

            score = rr * 20

            return TradeSuggestion(
                strategy=strategy,
                legs=[
                    TradeLeg(action="BUY", instrument=f"NIFTY {int(atm_strike)} CE",
                             strike=atm_strike, option_type="CE", ltp=atm_sd.ce_ltp),
                    TradeLeg(action="SELL", instrument=f"NIFTY {int(sell_strike)} CE",
                             strike=sell_strike, option_type="CE", ltp=sell_sd.ce_ltp),
                ],
                direction_bias="Bullish",
                confidence="Medium",
                score=round(score, 1),
                entry_timing="Enter now — buy_premium regime with bullish breakout",
                technicals_to_check=[
                    f"pVRP: {vol.p_vrp:.2f} (cheap — options underpriced)",
                    f"Spot: {spot:.0f} > BB upper: {technicals.bb_upper:.0f}",
                    f"Supertrend: bullish",
                ],
                expected_outcome=f"Profit if NIFTY moves above {int(atm_strike + net_debit)}",
                max_profit=f"₹{max_profit * lot_size:,.0f} per lot",
                max_loss=f"₹{net_debit * lot_size:,.0f} per lot",
                stop_loss=f"Exit at {cfg.get('debit_sl_premium_loss_pct', 50.0)}% loss or {cfg.get('debit_max_sessions', 3)} sessions",
                position_size="Per Atlas dynamic sizing",
                reasoning=[
                    f"Bull call spread: BUY {int(atm_strike)} CE, SELL {int(sell_strike)} CE",
                    f"Net debit: ₹{net_debit:.1f}, max profit: ₹{max_profit:.1f} (RR 1:{rr:.1f})",
                    f"Buy premium regime: pVRP={vol.p_vrp:.2f} (options cheap)",
                    f"Breakout confirmed: spot {spot:.0f} above BB {technicals.bb_upper:.0f}",
                ],
            )
    else:
        strategy = StrategyName.BEAR_PUT_SPREAD
        atm_strike = analytics.atm_strike
        atm_sd = _get_strike_data(chain, atm_strike)
        if atm_sd is None or atm_sd.pe_ltp <= 0:
            return None

        for width in [100, 50]:
            sell_strike = atm_strike - width
            sell_sd = _get_strike_data(chain, sell_strike)
            if sell_sd is None or sell_sd.pe_ltp <= 0:
                continue

            net_debit = atm_sd.pe_ltp - sell_sd.pe_ltp
            if net_debit <= 0:
                continue

            max_profit = width - net_debit
            rr = max_profit / net_debit
            if rr < min_rr:
                continue

            score = rr * 20

            return TradeSuggestion(
                strategy=strategy,
                legs=[
                    TradeLeg(action="BUY", instrument=f"NIFTY {int(atm_strike)} PE",
                             strike=atm_strike, option_type="PE", ltp=atm_sd.pe_ltp),
                    TradeLeg(action="SELL", instrument=f"NIFTY {int(sell_strike)} PE",
                             strike=sell_strike, option_type="PE", ltp=sell_sd.pe_ltp),
                ],
                direction_bias="Bearish",
                confidence="Medium",
                score=round(score, 1),
                entry_timing="Enter now — buy_premium regime with bearish breakout",
                technicals_to_check=[
                    f"pVRP: {vol.p_vrp:.2f} (cheap — options underpriced)",
                    f"Spot: {spot:.0f} < BB lower: {technicals.bb_lower:.0f}",
                    f"Supertrend: bearish",
                ],
                expected_outcome=f"Profit if NIFTY moves below {int(atm_strike - net_debit)}",
                max_profit=f"₹{max_profit * lot_size:,.0f} per lot",
                max_loss=f"₹{net_debit * lot_size:,.0f} per lot",
                stop_loss=f"Exit at {cfg.get('debit_sl_premium_loss_pct', 50.0)}% loss or {cfg.get('debit_max_sessions', 3)} sessions",
                position_size="Per Atlas dynamic sizing",
                reasoning=[
                    f"Bear put spread: BUY {int(atm_strike)} PE, SELL {int(sell_strike)} PE",
                    f"Net debit: ₹{net_debit:.1f}, max profit: ₹{max_profit:.1f} (RR 1:{rr:.1f})",
                    f"Buy premium regime: pVRP={vol.p_vrp:.2f} (options cheap)",
                    f"Breakout confirmed: spot {spot:.0f} below BB {technicals.bb_lower:.0f}",
                ],
            )

    return None


# ===================================================================
# Position Sizing (dynamic risk budget)
# ===================================================================

def _compute_atlas_lots(
    suggestion: TradeSuggestion,
    state: PaperTradingState,
    vol: VolSnapshot,
    cfg: dict,
) -> tuple[int, str | None]:
    """Compute lots using dynamic risk-per-trade budget."""
    lot_size = cfg.get("lot_size", 75)
    account_value = state.initial_capital + state.net_realized_pnl

    # Estimate max loss per lot
    if suggestion.strategy in _SPREAD_STRATEGIES:
        width = _compute_spread_width(suggestion.legs)
        sell_prem = sum(leg.ltp for leg in suggestion.legs if leg.action == "SELL")
        buy_prem = sum(leg.ltp for leg in suggestion.legs if leg.action == "BUY")
        net_credit_per_unit = sell_prem - buy_prem
        max_loss_per_lot = (width - max(0, net_credit_per_unit)) * lot_size
    elif suggestion.strategy in _ATLAS_CREDIT:
        net_per_unit = sum(
            leg.ltp * (1 if leg.action == "SELL" else -1) for leg in suggestion.legs
        )
        sl_mult = _dynamic_sl_multiple(vol.p_vov, cfg)
        max_loss_per_lot = abs(net_per_unit) * sl_mult * lot_size
    else:
        max_loss_per_lot = abs(sum(
            leg.ltp * (1 if leg.action == "BUY" else -1) for leg in suggestion.legs
        )) * lot_size

    if max_loss_per_lot <= 0:
        return 0, "cannot compute max loss per lot"

    # Dynamic risk budget
    risk_pct = _dynamic_risk_per_trade(vol.p_rv, vol.p_vov, cfg)
    risk_budget = (risk_pct / 100) * account_value
    lots = math.floor(risk_budget / max_loss_per_lot)

    # Dynamic portfolio risk check
    max_port_risk_pct = _dynamic_max_portfolio_risk(vol.p_vov, cfg)
    existing_risk = sum(p.stop_loss_amount for p in state.open_positions if p.status == PositionStatus.OPEN)
    new_risk = max_loss_per_lot * max(1, lots)
    if account_value > 0 and (existing_risk + new_risk) / account_value * 100 > max_port_risk_pct:
        allowed_risk = (max_port_risk_pct / 100) * account_value - existing_risk
        if allowed_risk <= 0:
            return 0, f"portfolio risk at limit {max_port_risk_pct:.1f}%"
        lots = math.floor(allowed_risk / max_loss_per_lot)

    # Consecutive loss reduction
    reduce_threshold = cfg.get("consecutive_loss_reduce_threshold", 3)
    if state.consecutive_losses >= reduce_threshold:
        lots = max(1, lots // 2)
        logger.info("ATLAS: %d consecutive losses — reducing to %d lots", state.consecutive_losses, lots)

    if lots < 1:
        return 0, f"max_loss_per_lot ₹{max_loss_per_lot:.0f} > risk budget ₹{risk_budget:.0f}"

    return lots, None


# ===================================================================
# Liquidity + Fill Filters (reuses hedge pattern)
# ===================================================================

def _check_atlas_liquidity(
    suggestion: TradeSuggestion,
    chain: OptionChainData,
    cfg: dict,
) -> str | None:
    leg_strikes = [leg.strike for leg in suggestion.legs]
    min_oi = cfg.get("min_oi", 15_000)
    min_vol = cfg.get("min_volume", 2_000)
    max_ba = cfg.get("max_bid_ask_pct", 3.0)
    _liq_score, liq_reject = compute_liquidity_score(chain, leg_strikes, min_oi, min_vol, max_ba)
    if liq_reject:
        return f"liquidity: {liq_reject}"
    return None


def _apply_atlas_fills(
    suggestion: TradeSuggestion,
    chain: OptionChainData,
    cfg: dict,
) -> tuple[list[TradeLeg], str | None]:
    slippage = cfg.get("slippage_per_leg", 0.05)
    new_legs: list[TradeLeg] = []

    for leg in suggestion.legs:
        sd = _get_strike_data(chain, leg.strike)
        if sd is None:
            new_legs.append(leg)
            continue

        if leg.action == "SELL":
            bid = sd.ce_bid if leg.option_type == "CE" else sd.pe_bid
            fill_price = max(0.05, (bid if bid > 0 else leg.ltp) - slippage)
        else:
            ask = sd.ce_ask if leg.option_type == "CE" else sd.pe_ask
            fill_price = (ask if ask > 0 else leg.ltp) + slippage

        new_legs.append(leg.model_copy(update={"ltp": round(fill_price, 2)}))

    # Post-fill RR check
    max_rr = cfg.get("post_fill_max_rr", 2.0)
    sell_prem = sum(l.ltp for l in new_legs if l.action == "SELL")
    buy_prem = sum(l.ltp for l in new_legs if l.action == "BUY")
    net_credit = sell_prem - buy_prem

    if suggestion.strategy in _ATLAS_CREDIT and net_credit <= 0:
        return new_legs, "no net credit after realistic fills"

    if suggestion.strategy in _SPREAD_STRATEGIES and net_credit > 0:
        width = _compute_spread_width(new_legs)
        if width > 0:
            rr = (width - net_credit) / net_credit
            if rr > max_rr:
                return new_legs, f"post-fill RR {rr:.1f}:1 > max {max_rr}:1"

    return new_legs, None


# ===================================================================
# Global Hard Gates
# ===================================================================

def _check_atlas_gates(
    state: PaperTradingState,
    suggestion: TradeSuggestion,
    chain: OptionChainData,
    cfg: dict,
) -> str | None:
    account_value = state.initial_capital + state.net_realized_pnl

    # Daily loss shutdown
    daily_loss_pct = cfg.get("daily_loss_shutdown_pct", 1.25)
    if state.daily_start_capital > 0:
        daily_pnl = account_value - state.daily_start_capital
        if daily_pnl <= -(state.daily_start_capital * daily_loss_pct / 100):
            return f"daily loss shutdown: {abs(daily_pnl):.0f} > {daily_loss_pct}%"

    # Weekly drawdown shutdown
    weekly_dd_pct = cfg.get("weekly_drawdown_shutdown_pct", 2.0)
    if state.weekly_start_capital > 0:
        weekly_pnl = account_value - state.weekly_start_capital
        if weekly_pnl <= -(state.weekly_start_capital * weekly_dd_pct / 100):
            return f"weekly drawdown shutdown: {abs(weekly_pnl):.0f} > {weekly_dd_pct}%"

    # Trading halted
    if state.trading_halted:
        return "trading halted by portfolio defense"

    # Max margin utilization
    max_margin_util = cfg.get("max_margin_utilization_pct", 25.0)
    if account_value > 0:
        margin_util = (state.margin_in_use / account_value) * 100
        if margin_util > max_margin_util:
            return f"margin utilization {margin_util:.1f}% > {max_margin_util}%"

    # Block naked positions
    sell_legs = [leg for leg in suggestion.legs if leg.action == "SELL"]
    buy_legs = [leg for leg in suggestion.legs if leg.action == "BUY"]
    if sell_legs and not buy_legs:
        return "ATLAS blocks naked positions — protective legs required"

    return None


# ===================================================================
# Portfolio Structure
# ===================================================================

def _check_atlas_portfolio_structure(
    state: PaperTradingState,
    suggestion: TradeSuggestion,
    cfg: dict,
) -> str | None:
    max_open = cfg.get("max_open_positions", 4)
    open_positions = [p for p in state.open_positions if p.status == PositionStatus.OPEN]
    if len(open_positions) >= max_open:
        return f"max {max_open} open positions reached"

    max_ic = cfg.get("max_iron_condors", 3)
    ic_count = sum(1 for p in open_positions if p.strategy == StrategyName.IRON_CONDOR.value)
    if suggestion.strategy == StrategyName.IRON_CONDOR and ic_count >= max_ic:
        return f"max {max_ic} iron condors reached"

    max_dir = cfg.get("max_directional_positions", 1)
    dir_strategies = {StrategyName.BULL_CALL_SPREAD.value, StrategyName.BEAR_PUT_SPREAD.value}
    dir_count = sum(1 for p in open_positions if p.strategy in dir_strategies)
    if suggestion.strategy.value in dir_strategies and dir_count >= max_dir:
        return f"max {max_dir} directional position(s) reached"

    # Allocation tracking
    tracker = state.allocation_tracker or {}
    primary_used = tracker.get("primary_used", 0.0)
    secondary_used = tracker.get("secondary_used", 0.0)
    account_value = state.initial_capital + state.net_realized_pnl
    primary_limit = account_value * cfg.get("primary_allocation_pct", 80.0) / 100
    secondary_limit = account_value * cfg.get("secondary_allocation_pct", 20.0) / 100

    if suggestion.strategy in _ATLAS_PRIMARY:
        if primary_used >= primary_limit:
            return f"primary allocation exhausted ({primary_used:.0f}/{primary_limit:.0f})"
    elif suggestion.strategy in _ATLAS_SECONDARY:
        if secondary_used >= secondary_limit:
            return f"secondary allocation exhausted ({secondary_used:.0f}/{secondary_limit:.0f})"

    return None


# ===================================================================
# Stop Loss Rules (dynamic)
# ===================================================================

def _check_atlas_credit_sl(
    position: PaperPosition,
    chain: OptionChainData,
    vol: VolSnapshot,
    cfg: dict,
) -> PositionStatus | None:
    """Dynamic credit SL: sl_mult(pVoV) x credit OR delta > delta_exit(pRV)."""
    pnl = position.total_unrealized_pnl
    sl_mult = _dynamic_sl_multiple(vol.p_vov, cfg)

    if position.net_premium > 0 and pnl <= -(abs(position.net_premium) * sl_mult):
        logger.info("ATLAS SL: %s loss %.0f >= %.1fx credit %.0f",
                     position.strategy, abs(pnl), sl_mult, abs(position.net_premium))
        return PositionStatus.CLOSED_STOP_LOSS

    # Dynamic delta exit
    delta_exit = _dynamic_delta_exit(vol.p_rv, cfg)
    for leg in position.legs:
        if leg.action != "SELL":
            continue
        sd = _get_strike_data(chain, leg.strike)
        if sd is None:
            continue
        delta = abs(sd.ce_delta if leg.option_type == "CE" else sd.pe_delta)
        if delta >= delta_exit:
            logger.info("ATLAS SL: delta %.3f >= dynamic exit %.3f on %s %d",
                         delta, delta_exit, leg.option_type, int(leg.strike))
            return PositionStatus.CLOSED_STOP_LOSS

    return None


def _check_atlas_strangle_sl(
    position: PaperPosition,
    chain: OptionChainData,
    vol: VolSnapshot,
    cfg: dict,
) -> PositionStatus | None:
    """Strangle SL: premium doubles OR dynamic delta exit."""
    pnl = position.total_unrealized_pnl

    if position.net_premium > 0 and pnl <= -abs(position.net_premium):
        logger.info("ATLAS SL: strangle premium doubled — loss %.0f >= credit %.0f",
                     abs(pnl), abs(position.net_premium))
        return PositionStatus.CLOSED_STOP_LOSS

    for leg in position.legs:
        if leg.action == "SELL" and leg.entry_ltp > 0:
            if leg.current_ltp >= 2 * leg.entry_ltp:
                logger.info("ATLAS SL: strangle leg %s %d premium doubled (%.1f -> %.1f)",
                             leg.option_type, int(leg.strike), leg.entry_ltp, leg.current_ltp)
                return PositionStatus.CLOSED_STOP_LOSS

    # Dynamic delta exit
    delta_exit = _dynamic_delta_exit(vol.p_rv, cfg)
    for leg in position.legs:
        if leg.action != "SELL":
            continue
        sd = _get_strike_data(chain, leg.strike)
        if sd is None:
            continue
        delta = abs(sd.ce_delta if leg.option_type == "CE" else sd.pe_delta)
        if delta >= delta_exit:
            logger.info("ATLAS SL: strangle delta %.3f >= exit %.3f",
                         delta, delta_exit)
            return PositionStatus.CLOSED_STOP_LOSS

    return None


def _check_atlas_debit_sl(
    position: PaperPosition,
    cfg: dict,
) -> PositionStatus | None:
    """Debit SL: 50% premium loss + session limit."""
    pnl = position.total_unrealized_pnl
    sl_pct = cfg.get("debit_sl_premium_loss_pct", 50.0)

    if position.net_premium < 0 and pnl <= -(abs(position.net_premium) * sl_pct / 100):
        return PositionStatus.CLOSED_STOP_LOSS

    max_sessions = cfg.get("debit_max_sessions", 3)
    if position.sessions_held >= max_sessions:
        return PositionStatus.CLOSED_TIME_LIMIT

    return None


# ===================================================================
# Dynamic Take Profit
# ===================================================================

def _check_atlas_take_profit(
    position: PaperPosition,
    vol: VolSnapshot,
    cfg: dict,
) -> PositionStatus | None:
    """Dynamic take profit: capture % varies with vol."""
    pnl = position.total_unrealized_pnl
    strategy_type = classify_strategy(position.strategy)

    if strategy_type == StrategyType.CREDIT and position.net_premium > 0:
        tp_pct = _dynamic_take_profit(vol.p_rv, cfg)
        target = abs(position.net_premium) * tp_pct
        if pnl >= target:
            logger.info("ATLAS TP: %s pnl %.0f >= %.0f%% of credit (%.0f)",
                         position.strategy, pnl, tp_pct * 100, target)
            return PositionStatus.CLOSED_PROFIT_TARGET

    return None


# ===================================================================
# Enrichment
# ===================================================================

def _enrich_atlas_suggestion(
    suggestion: TradeSuggestion,
    chain: OptionChainData,
    analytics: OptionsAnalytics,
    technicals: TechnicalIndicators,
    vol: VolSnapshot,
    cfg: dict,
    lots: int,
) -> TradeSuggestion:
    lot_size = cfg.get("lot_size", 75)

    sell_prem = sum(leg.ltp for leg in suggestion.legs if leg.action == "SELL")
    buy_prem = sum(leg.ltp for leg in suggestion.legs if leg.action == "BUY")
    net = sell_prem - buy_prem
    width = _compute_spread_width(suggestion.legs)
    is_credit = suggestion.strategy in _ATLAS_CREDIT

    if is_credit:
        if width > 0:
            max_profit = net * lot_size * lots
            max_loss = (width - net) * lot_size * lots
        else:
            max_profit = net * lot_size * lots
            sl_mult = _dynamic_sl_multiple(vol.p_vov, cfg)
            max_loss = net * sl_mult * lot_size * lots
        rr = max_loss / max_profit if max_profit > 0 else 0.0
    else:
        net_debit = buy_prem - sell_prem
        if width > 0:
            max_profit = (width - net_debit) * lot_size * lots
            max_loss = net_debit * lot_size * lots
        else:
            max_profit = 0.0
            max_loss = net_debit * lot_size * lots
        rr = max_profit / max_loss if max_loss > 0 else 0.0

    # POP
    pop_val = 0.0
    for leg in suggestion.legs:
        if leg.action == "SELL":
            sd = _get_strike_data(chain, leg.strike)
            if sd:
                delta = abs(sd.ce_delta if leg.option_type == "CE" else sd.pe_delta)
                if is_credit:
                    pop_val = compute_pop("credit", short_strike_delta=delta)
                break

    # Liquidity
    leg_strikes = [leg.strike for leg in suggestion.legs]
    liq_score, _ = compute_liquidity_score(
        chain, leg_strikes,
        min_oi=cfg.get("min_oi", 15_000),
        min_volume=cfg.get("min_volume", 2_000),
        max_ba_pct=cfg.get("max_bid_ask_pct", 3.0),
    )

    # Dynamic params info
    dyn_params = _get_all_dynamic_params(vol, cfg)
    account_value = cfg.get("initial_capital", 2_500_000)
    risk_pct = (max_loss / account_value * 100) if account_value > 0 else 0

    extra_reasoning = [
        f"Vol snapshot: RV20={vol.rv_20:.1f}% VIX={vol.vix:.1f} VRP={vol.vrp:.1f}",
        f"Percentiles: pRV={vol.p_rv:.2f} pVoV={vol.p_vov:.2f} pVRP={vol.p_vrp:.2f}",
        f"Dynamic: delta={dyn_params['strike_delta']:.3f} SL={dyn_params['sl_multiple']:.1f}x TP={dyn_params['take_profit']:.0%}",
        f"Risk: {risk_pct:.2f}% of capital (₹{max_loss:,.0f} max loss for {lots} lots)",
    ]

    return suggestion.model_copy(update={
        "risk_reward_ratio": round(rr, 2),
        "pop": pop_val,
        "max_loss_numeric": round(max_loss, 0),
        "max_profit_numeric": round(max_profit, 0),
        "liquidity_score": liq_score,
        "net_credit_debit": round(net * lot_size * lots, 0),
        "reasoning": list(suggestion.reasoning) + extra_reasoning,
    })


# ===================================================================
# Main Algorithm Class
# ===================================================================

@register_algorithm
class AtlasAlgorithm(TradingAlgorithm):
    """V4 Atlas — Distribution-based dynamic thresholds."""

    name = "atlas"
    display_name = "V4 Atlas"
    description = "Dynamic thresholds from NIFTY's own volatility distribution"

    # ---------------------------------------------------------------
    # generate_suggestions
    # ---------------------------------------------------------------

    def generate_suggestions(
        self,
        chain: OptionChainData,
        technicals: TechnicalIndicators,
        analytics: OptionsAnalytics,
    ) -> list[TradeSuggestion]:
        cfg = self.config
        accepted: list[TradeSuggestion] = []
        rejected: list[TradeSuggestion] = []

        # Fetch VolSnapshot
        spot = chain.underlying_value
        dte = _parse_dte(chain.expiry) if chain.expiry else 5
        vol = get_today_vol_snapshot(spot, dte)
        if vol is None:
            vol = _neutral_snapshot()
            vol = VolSnapshot(
                rv_5=vol.rv_5, rv_10=vol.rv_10, rv_20=vol.rv_20,
                vov_20=vol.vov_20, vix=analytics.atm_iv if analytics.atm_iv > 0 else 15.0,
                vrp=vol.vrp, p_rv=vol.p_rv, p_vov=vol.p_vov, p_vrp=vol.p_vrp,
                em=_compute_expected_move(spot, analytics.atm_iv, dte),
                date=vol.date,
            )
            logger.warning("ATLAS: VolSnapshot unavailable, using neutral fallback")

        # Classify regime
        regime = _classify_regime(vol, cfg)
        dyn_params = _get_all_dynamic_params(vol, cfg)

        logger.info(
            "ATLAS regime=%s | pRV=%.2f pVoV=%.2f pVRP=%.2f | delta=%.3f credit=%.3f SL=%.1fx TP=%.0f%% BE=%.2fx",
            regime, vol.p_rv, vol.p_vov, vol.p_vrp,
            dyn_params["strike_delta"], dyn_params["min_credit_ratio"],
            dyn_params["sl_multiple"], dyn_params["take_profit"] * 100,
            dyn_params["breakeven_mult"],
        )

        # Macro event check
        macro_override = cfg.get("macro_event_override", False)
        if not macro_override:
            near_event, event_label = is_near_event(days_ahead=1)
            if near_event and regime == "sell_premium":
                logger.info("ATLAS: Macro event within 24h: %s — downgrading to stand_down", event_label)
                regime = "stand_down"

        if regime == "sell_premium":
            # Build IC
            ic = _build_atlas_iron_condor(chain, analytics, technicals, vol, cfg)
            if ic is not None:
                liq_reject = _check_atlas_liquidity(ic, chain, cfg)
                if liq_reject:
                    rejected.append(ic.model_copy(update={"rejection_reason": liq_reject, "score": 0.0}))
                else:
                    new_legs, fill_reject = _apply_atlas_fills(ic, chain, cfg)
                    if fill_reject:
                        rejected.append(ic.model_copy(update={"rejection_reason": fill_reject, "score": 0.0}))
                    else:
                        ic = ic.model_copy(update={"legs": new_legs})
                        enriched = _enrich_atlas_suggestion(ic, chain, analytics, technicals, vol, cfg, lots=1)
                        accepted.append(enriched)
                        logger.info("ATLAS ACCEPTED: %s | RR=%.2f | POP=%.1f%% | liq=%.0f",
                                    enriched.strategy.value, enriched.risk_reward_ratio,
                                    enriched.pop, enriched.liquidity_score)

            # Build hedged strangle
            hs = _build_atlas_hedged_strangle(chain, analytics, technicals, vol, cfg)
            if hs is not None:
                liq_reject = _check_atlas_liquidity(hs, chain, cfg)
                if liq_reject:
                    rejected.append(hs.model_copy(update={"rejection_reason": liq_reject, "score": 0.0}))
                else:
                    new_legs, fill_reject = _apply_atlas_fills(hs, chain, cfg)
                    if fill_reject:
                        rejected.append(hs.model_copy(update={"rejection_reason": fill_reject, "score": 0.0}))
                    else:
                        hs = hs.model_copy(update={"legs": new_legs})
                        enriched = _enrich_atlas_suggestion(hs, chain, analytics, technicals, vol, cfg, lots=1)
                        accepted.append(enriched)
                        logger.info("ATLAS ACCEPTED: %s | RR=%.2f | POP=%.1f%% | liq=%.0f",
                                    enriched.strategy.value, enriched.risk_reward_ratio,
                                    enriched.pop, enriched.liquidity_score)

        elif regime == "buy_premium":
            ds = _build_atlas_debit_spread(chain, analytics, technicals, vol, cfg)
            if ds is not None:
                liq_reject = _check_atlas_liquidity(ds, chain, cfg)
                if liq_reject:
                    rejected.append(ds.model_copy(update={"rejection_reason": liq_reject, "score": 0.0}))
                else:
                    new_legs, fill_reject = _apply_atlas_fills(ds, chain, cfg)
                    if fill_reject:
                        rejected.append(ds.model_copy(update={"rejection_reason": fill_reject, "score": 0.0}))
                    else:
                        ds = ds.model_copy(update={"legs": new_legs})
                        enriched = _enrich_atlas_suggestion(ds, chain, analytics, technicals, vol, cfg, lots=1)
                        accepted.append(enriched)
                        logger.info("ATLAS ACCEPTED: %s | RR=%.2f | POP=%.1f%% | liq=%.0f",
                                    enriched.strategy.value, enriched.risk_reward_ratio,
                                    enriched.pop, enriched.liquidity_score)

        else:
            logger.info("ATLAS: Stand-down regime — no new suggestions")

        return sorted(accepted, key=lambda s: s.score, reverse=True) + rejected

    # ---------------------------------------------------------------
    # evaluate_and_manage
    # ---------------------------------------------------------------

    def evaluate_and_manage(
        self,
        state: PaperTradingState,
        suggestions: list[TradeSuggestion] | None,
        chain: OptionChainData | None,
        technicals: TechnicalIndicators | None = None,
        analytics: OptionsAnalytics | None = None,
        lot_size: int | None = None,
        refresh_ts: float = 0.0,
    ) -> PaperTradingState:
        # Market-open guard: skip all trading logic when market is closed
        if not is_market_open():
            return state

        if lot_size is None:
            lot_size = self.config.get("lot_size", 75)
        cfg = self.config

        # Reset daily/weekly
        state = _reset_period_tracking(state, cfg)

        # Fetch VolSnapshot
        spot = chain.underlying_value if chain else 0.0
        dte = _parse_dte(chain.expiry) if chain and chain.expiry else 5
        vol = get_today_vol_snapshot(spot, dte)
        if vol is None:
            vol = _neutral_snapshot()
            logger.warning("ATLAS: VolSnapshot unavailable during management, using neutral fallback")

        regime = _classify_regime(vol, cfg)
        dyn_params = _get_all_dynamic_params(vol, cfg)

        # Store regime + dynamic params on state
        state = state.model_copy(update={
            "vol_regime": regime,
            "vol_snapshot_ts": _time.time(),
            "vol_dynamic_params": dyn_params,
        })

        # --- Phase 1: Manage existing positions ---
        still_open: list[PaperPosition] = []
        new_records: list[TradeRecord] = []
        added_realized = 0.0
        added_costs = 0.0
        new_pending_critiques: list[str] = []

        for position in state.open_positions:
            if position.status != PositionStatus.OPEN:
                continue

            if chain:
                position = update_position_ltp(position, chain)

            pnl = position.total_unrealized_pnl
            position = position.model_copy(update={
                "peak_pnl": max(position.peak_pnl, pnl),
                "trough_pnl": min(position.trough_pnl, pnl),
            })

            exit_reason = None
            strategy_type = classify_strategy(position.strategy)

            # Dynamic stop losses
            if chain:
                if strategy_type == StrategyType.CREDIT:
                    if position.strategy == StrategyName.SHORT_STRANGLE.value:
                        exit_reason_sl = _check_atlas_strangle_sl(position, chain, vol, cfg)
                    else:
                        exit_reason_sl = _check_atlas_credit_sl(position, chain, vol, cfg)
                    if exit_reason_sl:
                        exit_reason = exit_reason_sl
                elif strategy_type == StrategyType.DEBIT:
                    exit_reason_sl = _check_atlas_debit_sl(position, cfg)
                    if exit_reason_sl:
                        exit_reason = exit_reason_sl

            # Dynamic take profit
            if exit_reason is None:
                tp = _check_atlas_take_profit(position, vol, cfg)
                if tp:
                    exit_reason = tp

            # Regime-change exit: if regime flips to stand_down while holding credit
            if exit_reason is None and regime == "stand_down" and strategy_type == StrategyType.CREDIT:
                # Only close if position is profitable (don't force-close at a loss)
                if pnl > 0:
                    logger.info("ATLAS: Regime changed to stand_down — closing profitable credit %s",
                                position.strategy)
                    exit_reason = PositionStatus.CLOSED_PROFIT_TARGET

            # Breakout detection for credit positions
            if exit_reason is None and chain and technicals:
                is_credit_range = position.strategy in {
                    StrategyName.IRON_CONDOR.value,
                    StrategyName.SHORT_STRANGLE.value,
                }
                if is_credit_range and _is_breakout(technicals):
                    logger.info("ATLAS: Breakout detected — closing %s", position.strategy)
                    exit_reason = PositionStatus.CLOSED_STOP_LOSS

            # Standard exit conditions (EOD)
            if exit_reason is None:
                exit_reason = check_exit_conditions(position)

            if exit_reason:
                _closed_pos, record = close_position(
                    position, exit_reason,
                    technicals=technicals, analytics=analytics, chain=chain,
                )
                logger.info("ATLAS CLOSED: %s | reason=%s | pnl=%.2f",
                             record.strategy,
                             exit_reason.value if hasattr(exit_reason, 'value') else exit_reason,
                             record.net_pnl)
                new_records.append(record)
                added_realized += record.realized_pnl
                added_costs += record.execution_cost
                if record.entry_context:
                    new_pending_critiques.append(record.id)
            else:
                still_open.append(position)

        # --- Portfolio defense: consecutive losses ---
        consecutive_losses = state.consecutive_losses
        trading_halted = state.trading_halted

        for record in new_records:
            if record.net_pnl < 0:
                consecutive_losses += 1
            else:
                consecutive_losses = 0

        halt_after = cfg.get("consecutive_loss_halt", 5)
        if consecutive_losses >= halt_after:
            trading_halted = True
            logger.warning("ATLAS HALTED: %d consecutive losses >= %d", consecutive_losses, halt_after)

        current_capital = state.initial_capital + state.total_realized_pnl + added_realized - (state.total_execution_costs + added_costs)
        peak_capital = max(state.peak_capital, current_capital)

        state = state.model_copy(update={
            "open_positions": still_open,
            "trade_log": state.trade_log + new_records,
            "total_realized_pnl": state.total_realized_pnl + added_realized,
            "total_execution_costs": state.total_execution_costs + added_costs,
            "pending_critiques": state.pending_critiques + new_pending_critiques,
            "consecutive_losses": consecutive_losses,
            "trading_halted": trading_halted,
            "peak_capital": peak_capital,
        })

        # Initialize session start capital
        if state.session_start_capital == 0.0:
            state = state.model_copy(update={
                "session_start_capital": state.initial_capital + state.net_realized_pnl,
            })

        # Check daily/weekly shutdown
        account_value = state.initial_capital + state.net_realized_pnl
        daily_loss_pct = cfg.get("daily_loss_shutdown_pct", 1.25)
        if state.daily_start_capital > 0:
            daily_pnl = account_value - state.daily_start_capital
            if daily_pnl <= -(state.daily_start_capital * daily_loss_pct / 100):
                logger.warning("ATLAS SHUTDOWN: daily loss %.0f > %.1f%%", abs(daily_pnl), daily_loss_pct)
                return state

        weekly_dd_pct = cfg.get("weekly_drawdown_shutdown_pct", 2.0)
        if state.weekly_start_capital > 0:
            weekly_pnl = account_value - state.weekly_start_capital
            if weekly_pnl <= -(state.weekly_start_capital * weekly_dd_pct / 100):
                logger.warning("ATLAS SHUTDOWN: weekly drawdown %.0f > %.1f%%", abs(weekly_pnl), weekly_dd_pct)
                return state

        # --- Phase 2: Open new positions ---
        if not state.is_auto_trading or not suggestions or not chain:
            return state

        if state.trading_halted:
            logger.info("ATLAS: Trading halted — skipping new positions")
            return state

        # Data staleness guard
        if technicals and technicals.data_staleness_minutes > 20:
            return state

        # Cooldown
        now_ts = _time.time()
        if state.last_trade_opened_ts > 0 and (now_ts - state.last_trade_opened_ts) < 60:
            return state

        for suggestion in suggestions:
            if suggestion.rejection_reason:
                continue
            if suggestion.score <= 0:
                continue

            # Global gates
            gate_reason = _check_atlas_gates(state, suggestion, chain, cfg)
            if gate_reason:
                logger.info("ATLAS GATE BLOCKED: %s — %s", suggestion.strategy.value, gate_reason)
                continue

            # Portfolio structure
            struct_reason = _check_atlas_portfolio_structure(state, suggestion, cfg)
            if struct_reason:
                logger.info("ATLAS STRUCTURE BLOCKED: %s — %s", suggestion.strategy.value, struct_reason)
                continue

            # Skip duplicates
            held = {p.strategy for p in state.open_positions if p.status == PositionStatus.OPEN}
            if suggestion.strategy.value in held:
                continue

            if state.capital_remaining <= 0:
                break

            # Dynamic position sizing
            vol_lots, size_reject = _compute_atlas_lots(suggestion, state, vol, cfg)
            if size_reject:
                logger.info("ATLAS SIZING REJECTED: %s — %s", suggestion.strategy.value, size_reject)
                continue

            # Open position
            position = open_position(
                suggestion, lot_size, state.capital_remaining,
                expiry=chain.expiry,
                technicals=technicals, analytics=analytics, chain=chain,
            )
            if vol_lots > 0 and vol_lots != position.lots:
                position = position.model_copy(update={"lots": vol_lots})

            position = position.model_copy(update={
                "entry_date": _now_ist().strftime("%Y-%m-%d"),
            })

            logger.info(
                "ATLAS OPENED: %s | lots=%d | margin=%.0f | premium=%.2f | regime=%s",
                position.strategy, position.lots, position.margin_required,
                position.net_premium, regime,
            )

            # Update allocation tracker
            tracker = dict(state.allocation_tracker or {})
            if suggestion.strategy in _ATLAS_PRIMARY:
                tracker["primary_used"] = tracker.get("primary_used", 0.0) + position.margin_required
            elif suggestion.strategy in _ATLAS_SECONDARY:
                tracker["secondary_used"] = tracker.get("secondary_used", 0.0) + position.margin_required

            state = state.model_copy(update={
                "open_positions": state.open_positions + [position],
                "last_trade_opened_ts": _time.time(),
                "allocation_tracker": tracker,
            })
            break  # max 1 trade per cycle

        return state
