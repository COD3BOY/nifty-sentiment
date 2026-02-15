"""V3 Optimus algorithm — Capital preservation options trading.

Conservative credit strategies (iron condors, hedged strangles) with strict
drawdown limits. Builds own trade candidates by scanning the option chain
for precise delta/width/credit targets — independent of V1 suggestions.

Key constraints:
  - Max 8% annual drawdown, 1.25% daily, 2% weekly
  - 0.5% risk per trade, 3% max portfolio risk
  - No naked (unlimited-risk) positions
  - Real India VIX via yfinance for regime detection
  - Active delta monitoring (0.25 warn, 0.30 exit)
"""

from __future__ import annotations

import logging
import math
import time as _time
from typing import TYPE_CHECKING

from algorithms import register_algorithm
from algorithms.base import TradingAlgorithm
from core.config import load_config
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
    _now_ist,
)
from core.options_utils import (
    get_strike_data as _get_strike_data,
    compute_spread_width as _compute_spread_width,
    compute_bb_width_pct as _compute_bb_width_pct,
    parse_dte as _parse_dte,
    is_breakout as _is_breakout,
    reset_period_tracking as _reset_period_tracking,
    compute_expected_move as _compute_expected_move,
    is_observation_period as _is_observation_period,
)

if TYPE_CHECKING:
    from core.context_models import MarketContext
    from core.observation import ObservationSnapshot

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Strategy classification sets
# ---------------------------------------------------------------------------

_OPTIMUS_CREDIT = {
    StrategyName.IRON_CONDOR,
    StrategyName.SHORT_STRANGLE,
    StrategyName.BULL_PUT_SPREAD,
    StrategyName.BEAR_CALL_SPREAD,
}

_OPTIMUS_DEBIT = {
    StrategyName.BULL_CALL_SPREAD,
    StrategyName.BEAR_PUT_SPREAD,
}

_OPTIMUS_PRIMARY = {
    StrategyName.IRON_CONDOR,
}

_OPTIMUS_SECONDARY = {
    StrategyName.SHORT_STRANGLE,  # hedged strangle (with wings)
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


# ---------------------------------------------------------------------------
# VIX fetching with 5-minute cache
# ---------------------------------------------------------------------------

_vix_cache: dict[str, tuple[float, float]] = {}  # ticker -> (value, timestamp)
_VIX_CACHE_TTL = 300  # 5 minutes


def _fetch_india_vix(cfg: dict) -> float | None:
    """Fetch India VIX via yfinance with 5-minute cache. Returns None on failure."""
    ticker = cfg.get("vix_ticker", "^INDIAVIX")
    now = _time.time()

    cached = _vix_cache.get(ticker)
    if cached and (now - cached[1]) < _VIX_CACHE_TTL:
        return cached[0]

    try:
        import yfinance as yf
        vix_data = yf.Ticker(ticker)
        hist = vix_data.history(period="1d")
        if hist.empty:
            return None
        vix_val = float(hist["Close"].iloc[-1])
        _vix_cache[ticker] = (vix_val, now)
        return vix_val
    except Exception as e:
        logger.warning("OPTIMUS: Failed to fetch VIX (%s): %s", ticker, e)
        return None


def _is_range_bound(technicals: TechnicalIndicators) -> bool:
    """Spot within Bollinger Bands and near VWAP."""
    if technicals.bb_upper <= 0 or technicals.bb_lower <= 0:
        return False
    in_bb = technicals.bb_lower <= technicals.spot <= technicals.bb_upper
    near_vwap = True
    if technicals.vwap > 0:
        near_vwap = abs(technicals.spot - technicals.vwap) / technicals.vwap * 100 < 0.5
    return in_bb and near_vwap


def _is_mean_reverting(technicals: TechnicalIndicators) -> bool:
    """RSI neutral (40-60) + BB contracting (width < 1.5%)."""
    rsi_neutral = 40 <= technicals.rsi <= 60
    bb_width = _compute_bb_width_pct(technicals)
    bb_contracting = 0 < bb_width < 1.5
    return rsi_neutral and bb_contracting


# ===================================================================
# Section 3A: Credit Regime Filter
# ===================================================================

def _check_credit_regime(
    analytics: OptionsAnalytics,
    technicals: TechnicalIndicators,
    vix: float | None,
    cfg: dict,
) -> tuple[str | None, bool]:
    """Check if credit regime conditions are met.

    Returns (rejection_reason | None, is_half_size).
    """
    is_half_size = False

    # IV percentile check
    min_iv_pct = cfg.get("credit_min_iv_percentile", 55.0)
    if analytics.iv_percentile < min_iv_pct:
        return f"IV percentile {analytics.iv_percentile:.1f} < {min_iv_pct} (need elevated IV for credit)", False

    # VIX band check
    if vix is not None:
        vix_no_credit = cfg.get("vix_no_credit_below", 14.0)
        vix_max = cfg.get("vix_credit_max", 24.0)
        vix_half = cfg.get("vix_half_size_above", 28.0)

        if vix < vix_no_credit:
            return f"VIX {vix:.1f} < {vix_no_credit} — IV too low for credit strategies", False

        if vix > vix_half:
            is_half_size = True
            logger.info("OPTIMUS: VIX %.1f > %.1f — half-size mode", vix, vix_half)

        if vix > vix_max and not is_half_size:
            # Between vix_max and vix_half — still allow but log warning
            logger.info("OPTIMUS: VIX %.1f > %.1f — elevated, proceed with caution", vix, vix_max)

    # BB width check — market should be range-bound
    max_bb_width = cfg.get("credit_max_bb_width_pct", 3.0)
    bb_width = _compute_bb_width_pct(technicals)
    if bb_width > max_bb_width:
        return f"BB width {bb_width:.1f}% > {max_bb_width}% — market too volatile for credit", False

    # Macro event check
    macro_override = cfg.get("macro_event_override", False)
    if not macro_override:
        near_event, event_label = is_near_event(days_ahead=1)
        if near_event:
            return f"Macro event within 24h: {event_label} — blocking credit strategies", False

    return None, is_half_size


# ===================================================================
# Section 3B: Iron Condor Construction
# ===================================================================

def _build_iron_condor(
    chain: OptionChainData,
    analytics: OptionsAnalytics,
    technicals: TechnicalIndicators,
    cfg: dict,
) -> TradeSuggestion | None:
    """Build an iron condor by scanning the chain for optimal strikes.

    Scans for short CE/PE strikes with delta 0.12-0.18, tries widths [50, 100].
    """
    spot = chain.underlying_value
    if spot <= 0:
        return None

    dte = _parse_dte(chain.expiry)
    if dte <= 0:
        return None

    delta_min = cfg.get("ic_short_delta_min", 0.12)
    delta_max = cfg.get("ic_short_delta_max", 0.18)
    widths = cfg.get("ic_widths", [50, 100])
    min_credit_pct_width = cfg.get("ic_min_credit_pct_width", 35.0)
    max_rr = cfg.get("ic_max_risk_reward", 2.0)
    be_mult = cfg.get("ic_breakeven_expected_move_mult", 1.3)

    expected_move = _compute_expected_move(spot, analytics.atm_iv, dte)

    best_candidate: TradeSuggestion | None = None
    best_score = 0.0

    # Find CE short strikes with delta in range
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

                # Net credit
                credit = (
                    short_ce.ce_ltp + short_pe.pe_ltp
                    - long_ce_sd.ce_ltp - long_pe_sd.pe_ltp
                )
                if credit <= 0:
                    continue

                # Credit as % of width
                credit_pct = (credit / width) * 100
                if credit_pct < min_credit_pct_width:
                    continue

                # Risk:Reward
                max_loss_per_unit = width - credit
                if credit > 0:
                    rr = max_loss_per_unit / credit
                else:
                    continue
                if rr > max_rr:
                    continue

                # Breakeven check: breakevens should be >= 1.3x expected move from spot
                upper_be = short_ce.strike_price + credit
                lower_be = short_pe.strike_price - credit
                if expected_move > 0:
                    upper_dist = upper_be - spot
                    lower_dist = spot - lower_be
                    if upper_dist < be_mult * expected_move or lower_dist < be_mult * expected_move:
                        continue

                # Score: higher credit % + lower delta = better
                avg_delta = (abs(short_ce.ce_delta) + abs(short_pe.pe_delta)) / 2
                score = credit_pct * (1 - avg_delta) * 100

                if score > best_score:
                    best_score = score
                    lot_size = cfg.get("lot_size", 65)
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
                        entry_timing="Enter now — range-bound market with elevated IV",
                        technicals_to_check=[
                            f"RSI: {technicals.rsi:.1f}",
                            f"BB width: {_compute_bb_width_pct(technicals):.2f}%",
                            f"IV percentile: {analytics.iv_percentile:.1f}",
                        ],
                        expected_outcome=f"Profit if NIFTY stays between {int(lower_be)}-{int(upper_be)}",
                        max_profit=f"₹{credit * lot_size:,.0f} per lot",
                        max_loss=f"₹{max_loss_per_unit * lot_size:,.0f} per lot",
                        stop_loss=f"Exit at {cfg.get('credit_sl_multiplier', 1.8):.1f}x credit or short delta > {cfg.get('delta_exit_threshold', 0.30)}",
                        position_size="Per hedge sizing rules",
                        reasoning=[
                            f"Iron condor: SELL {int(short_pe.strike_price)} PE / {int(short_ce.strike_price)} CE, width {width}",
                            f"Net credit: ₹{credit:.1f} ({credit_pct:.1f}% of width)",
                            f"RR: 1:{rr:.1f}, breakevens: {int(lower_be)}-{int(upper_be)}",
                            f"Short deltas: CE={abs(short_ce.ce_delta):.3f}, PE={abs(short_pe.pe_delta):.3f}",
                            f"Expected move: ±{expected_move:.0f} pts ({dte} DTE)",
                        ],
                    )

    return best_candidate


# ===================================================================
# Section 4: Hedged Strangle Construction
# ===================================================================

def _build_hedged_strangle(
    chain: OptionChainData,
    analytics: OptionsAnalytics,
    technicals: TechnicalIndicators,
    cfg: dict,
) -> TradeSuggestion | None:
    """Build a hedged strangle (short strangle + protective wings).

    Extra regime requirements: IV percentile >= 65, VIX 16-22, mean-reversion.
    """
    spot = chain.underlying_value
    if spot <= 0:
        return None

    # Extra IV percentile requirement for strangles
    min_iv_pct = cfg.get("strangle_min_iv_percentile", 65.0)
    if analytics.iv_percentile < min_iv_pct:
        return None

    # Mean-reversion check
    if not _is_mean_reverting(technicals):
        return None

    delta_min = cfg.get("strangle_delta_min", 0.10)
    delta_max = cfg.get("strangle_delta_max", 0.15)
    min_credit_pct = cfg.get("strangle_min_credit_pct_spot", 2.0)
    wing_offset = cfg.get("strangle_wing_offset", 200)

    best_candidate: TradeSuggestion | None = None
    best_score = 0.0

    # Find short CE strikes
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
            # Protective wings
            long_ce_strike = short_ce.strike_price + wing_offset
            long_pe_strike = short_pe.strike_price - wing_offset

            long_ce_sd = _get_strike_data(chain, long_ce_strike)
            long_pe_sd = _get_strike_data(chain, long_pe_strike)

            if long_ce_sd is None or long_pe_sd is None:
                continue
            if long_ce_sd.ce_ltp <= 0 or long_pe_sd.pe_ltp <= 0:
                continue

            # Net credit (short strangle - protective wings)
            credit = (
                short_ce.ce_ltp + short_pe.pe_ltp
                - long_ce_sd.ce_ltp - long_pe_sd.pe_ltp
            )
            if credit <= 0:
                continue

            # Credit >= 2% of spot
            credit_pct_spot = (credit / spot) * 100
            if credit_pct_spot < min_credit_pct:
                continue

            # Max loss is wing_offset - credit (on either side)
            max_loss_per_unit = wing_offset - credit
            if credit > 0:
                rr = max_loss_per_unit / credit
            else:
                continue

            lot_size = cfg.get("lot_size", 65)
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
                    entry_timing="Enter now — mean-reverting market with elevated IV",
                    technicals_to_check=[
                        f"RSI: {technicals.rsi:.1f} (neutral zone)",
                        f"BB width: {_compute_bb_width_pct(technicals):.2f}% (contracting)",
                        f"IV percentile: {analytics.iv_percentile:.1f} (elevated)",
                    ],
                    expected_outcome=f"Profit if NIFTY stays between {int(short_pe.strike_price)}-{int(short_ce.strike_price)}",
                    max_profit=f"₹{credit * lot_size:,.0f} per lot",
                    max_loss=f"₹{max_loss_per_unit * lot_size:,.0f} per lot",
                    stop_loss=f"Exit if premium doubles or short delta > {cfg.get('delta_exit_threshold', 0.30)}",
                    position_size="Per hedge sizing rules",
                    reasoning=[
                        f"Hedged strangle: SELL {int(short_pe.strike_price)} PE / {int(short_ce.strike_price)} CE",
                        f"Protective wings at {int(long_pe_strike)} PE / {int(long_ce_strike)} CE (offset {wing_offset})",
                        f"Net credit: ₹{credit:.1f} ({credit_pct_spot:.2f}% of spot)",
                        f"Short deltas: CE={abs(short_ce.ce_delta):.3f}, PE={abs(short_pe.pe_delta):.3f}",
                        f"Mean-reverting regime: RSI {technicals.rsi:.1f}, BB width {_compute_bb_width_pct(technicals):.2f}%",
                    ],
                )

    return best_candidate


# ===================================================================
# Section 5: Debit Spread Construction
# ===================================================================

def _build_debit_spread(
    chain: OptionChainData,
    analytics: OptionsAnalytics,
    technicals: TechnicalIndicators,
    cfg: dict,
) -> TradeSuggestion | None:
    """Build a debit spread when IV is low and breakout is confirmed."""
    spot = chain.underlying_value
    if spot <= 0:
        return None

    # IV percentile must be low
    max_iv_pct = cfg.get("debit_max_iv_percentile", 40.0)
    if analytics.iv_percentile > max_iv_pct:
        return None

    # Breakout must be confirmed
    if not _is_breakout(technicals):
        return None

    min_rr = cfg.get("debit_min_risk_reward", 2.0)
    max_risk_pct = cfg.get("debit_max_risk_pct", 0.4)
    lot_size = cfg.get("lot_size", 65)

    # Determine direction from supertrend
    is_bullish = technicals.supertrend_direction == 1

    if is_bullish:
        strategy = StrategyName.BULL_CALL_SPREAD
        # Buy ATM call, sell OTM call
        atm_strike = analytics.atm_strike
        atm_sd = _get_strike_data(chain, atm_strike)
        if atm_sd is None or atm_sd.ce_ltp <= 0:
            return None

        # Try widths of 50 and 100
        for width in [100, 50]:
            sell_strike = atm_strike + width
            sell_sd = _get_strike_data(chain, sell_strike)
            if sell_sd is None or sell_sd.ce_ltp <= 0:
                continue

            net_debit = atm_sd.ce_ltp - sell_sd.ce_ltp
            if net_debit <= 0:
                continue

            max_profit = width - net_debit
            if net_debit > 0:
                rr = max_profit / net_debit
            else:
                continue

            if rr < min_rr:
                continue

            score = rr * 20  # simple scoring

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
                entry_timing="Enter now — bullish breakout confirmed",
                technicals_to_check=[
                    f"Spot: {spot:.0f} > BB upper: {technicals.bb_upper:.0f}",
                    f"Supertrend: bullish",
                    f"IV percentile: {analytics.iv_percentile:.1f} (low — good for debit)",
                ],
                expected_outcome=f"Profit if NIFTY moves above {int(atm_strike + net_debit)}",
                max_profit=f"₹{max_profit * lot_size:,.0f} per lot",
                max_loss=f"₹{net_debit * lot_size:,.0f} per lot",
                stop_loss=f"Exit at {cfg.get('debit_sl_premium_loss_pct', 50.0)}% premium loss or after {cfg.get('debit_max_sessions', 3)} sessions",
                position_size="Per hedge sizing rules",
                reasoning=[
                    f"Bull call spread: BUY {int(atm_strike)} CE, SELL {int(sell_strike)} CE",
                    f"Net debit: ₹{net_debit:.1f}, max profit: ₹{max_profit:.1f} (RR 1:{rr:.1f})",
                    f"Breakout confirmed: spot {spot:.0f} above BB upper {technicals.bb_upper:.0f}",
                    f"Low IV ({analytics.iv_percentile:.1f}th percentile) favors debit strategies",
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
            if net_debit > 0:
                rr = max_profit / net_debit
            else:
                continue

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
                entry_timing="Enter now — bearish breakout confirmed",
                technicals_to_check=[
                    f"Spot: {spot:.0f} < BB lower: {technicals.bb_lower:.0f}",
                    f"Supertrend: bearish",
                    f"IV percentile: {analytics.iv_percentile:.1f} (low — good for debit)",
                ],
                expected_outcome=f"Profit if NIFTY moves below {int(atm_strike - net_debit)}",
                max_profit=f"₹{max_profit * lot_size:,.0f} per lot",
                max_loss=f"₹{net_debit * lot_size:,.0f} per lot",
                stop_loss=f"Exit at {cfg.get('debit_sl_premium_loss_pct', 50.0)}% premium loss or after {cfg.get('debit_max_sessions', 3)} sessions",
                position_size="Per hedge sizing rules",
                reasoning=[
                    f"Bear put spread: BUY {int(atm_strike)} PE, SELL {int(sell_strike)} PE",
                    f"Net debit: ₹{net_debit:.1f}, max profit: ₹{max_profit:.1f} (RR 1:{rr:.1f})",
                    f"Breakout confirmed: spot {spot:.0f} below BB lower {technicals.bb_lower:.0f}",
                    f"Low IV ({analytics.iv_percentile:.1f}th percentile) favors debit strategies",
                ],
            )

    return None


# ===================================================================
# Section 1+6: Position Sizing
# ===================================================================

def _compute_optimus_lots(
    suggestion: TradeSuggestion,
    state: PaperTradingState,
    cfg: dict,
    is_half_size: bool = False,
) -> tuple[int, str | None]:
    """Compute lots using 0.5% risk rule. Returns (lots, rejection_reason)."""
    lot_size = cfg.get("lot_size", 65)
    account_value = state.initial_capital + state.net_realized_pnl

    # Estimate max loss per lot
    if suggestion.strategy in _SPREAD_STRATEGIES:
        width = _compute_spread_width(suggestion.legs)
        sell_prem = sum(leg.ltp for leg in suggestion.legs if leg.action == "SELL")
        buy_prem = sum(leg.ltp for leg in suggestion.legs if leg.action == "BUY")
        net_credit_per_unit = sell_prem - buy_prem
        max_loss_per_lot = (width - max(0, net_credit_per_unit)) * lot_size
    elif suggestion.strategy in _OPTIMUS_CREDIT:
        net_per_unit = sum(
            leg.ltp * (1 if leg.action == "SELL" else -1) for leg in suggestion.legs
        )
        sl_mult = cfg.get("credit_sl_multiplier", 1.8)
        max_loss_per_lot = abs(net_per_unit) * sl_mult * lot_size
    else:
        max_loss_per_lot = abs(sum(
            leg.ltp * (1 if leg.action == "BUY" else -1) for leg in suggestion.legs
        )) * lot_size

    if max_loss_per_lot <= 0:
        return 0, "cannot compute max loss per lot"

    # Risk budget: 0.5% of account
    max_risk_pct = cfg.get("max_risk_per_trade_pct", 0.5)
    risk_budget = (max_risk_pct / 100) * account_value
    lots = math.floor(risk_budget / max_loss_per_lot)

    # Portfolio risk check: sum of all open max losses <= 3% of account
    max_portfolio_risk_pct = cfg.get("max_portfolio_risk_pct", 3.0)
    existing_risk = sum(p.stop_loss_amount for p in state.open_positions if p.status == PositionStatus.OPEN)
    new_risk = max_loss_per_lot * max(1, lots)
    if account_value > 0 and (existing_risk + new_risk) / account_value * 100 > max_portfolio_risk_pct:
        # Reduce lots to fit within portfolio risk
        allowed_risk = (max_portfolio_risk_pct / 100) * account_value - existing_risk
        if allowed_risk <= 0:
            return 0, f"portfolio risk {(existing_risk / account_value * 100):.1f}% already at limit {max_portfolio_risk_pct}%"
        lots = math.floor(allowed_risk / max_loss_per_lot)

    # Consecutive loss reduction
    reduce_threshold = cfg.get("consecutive_loss_reduce_threshold", 3)
    if state.consecutive_losses >= reduce_threshold:
        lots = max(1, lots // 2)
        logger.info("OPTIMUS: %d consecutive losses — reducing to %d lots", state.consecutive_losses, lots)

    # Half-size mode (high VIX)
    if is_half_size:
        lots = max(1, lots // 2)

    if lots < 1:
        return 0, f"max_loss_per_lot ₹{max_loss_per_lot:.0f} > 0.5% risk budget ₹{risk_budget:.0f}"

    return lots, None


# ===================================================================
# Section 2: Global Hard Gates
# ===================================================================

def _check_optimus_gates(
    state: PaperTradingState,
    suggestion: TradeSuggestion,
    chain: OptionChainData,
    cfg: dict,
) -> str | None:
    """Check all global hard risk limits. Returns rejection reason or None."""
    account_value = state.initial_capital + state.net_realized_pnl
    lot_size = cfg.get("lot_size", 65)

    # 1.25% daily loss -> hard shutdown
    daily_loss_pct = cfg.get("daily_loss_shutdown_pct", 1.25)
    if state.daily_start_capital > 0:
        daily_pnl = account_value - state.daily_start_capital
        if daily_pnl <= -(state.daily_start_capital * daily_loss_pct / 100):
            return f"daily loss shutdown: {abs(daily_pnl):.0f} > {daily_loss_pct}% of daily capital"

    # 2% weekly drawdown -> stop for week
    weekly_dd_pct = cfg.get("weekly_drawdown_shutdown_pct", 2.0)
    if state.weekly_start_capital > 0:
        weekly_pnl = account_value - state.weekly_start_capital
        if weekly_pnl <= -(state.weekly_start_capital * weekly_dd_pct / 100):
            return f"weekly drawdown shutdown: {abs(weekly_pnl):.0f} > {weekly_dd_pct}% of weekly capital"

    # Trading halted check
    if state.trading_halted:
        return "trading halted by portfolio defense (consecutive losses)"

    # 25% max margin utilization
    max_margin_util = cfg.get("max_margin_utilization_pct", 25.0)
    if account_value > 0:
        margin_util = (state.margin_in_use / account_value) * 100
        if margin_util > max_margin_util:
            return f"margin utilization {margin_util:.1f}% > max {max_margin_util}%"

    # Block naked unlimited-risk positions (verify BUY legs present for all strategies)
    sell_legs = [leg for leg in suggestion.legs if leg.action == "SELL"]
    buy_legs = [leg for leg in suggestion.legs if leg.action == "BUY"]
    if sell_legs and not buy_legs:
        return "OPTIMUS blocks naked (unlimited-risk) positions — protective legs required"

    return None


# ===================================================================
# Section 7: Liquidity + Execution Filters
# ===================================================================

def _check_optimus_liquidity(
    suggestion: TradeSuggestion,
    chain: OptionChainData,
    cfg: dict,
) -> str | None:
    """Check liquidity requirements for hedge trades."""
    leg_strikes = [leg.strike for leg in suggestion.legs]
    min_oi = cfg.get("min_oi", 15_000)
    min_vol = cfg.get("min_volume", 2_000)
    max_ba = cfg.get("max_bid_ask_pct", 3.0)
    _liq_score, liq_reject = compute_liquidity_score(chain, leg_strikes, min_oi, min_vol, max_ba)
    if liq_reject:
        return f"liquidity: {liq_reject}"
    return None


def _apply_optimus_fills(
    suggestion: TradeSuggestion,
    chain: OptionChainData,
    cfg: dict,
) -> tuple[list[TradeLeg], str | None]:
    """Apply bid/ask pricing + slippage. Returns (new_legs, rejection_reason)."""
    slippage = cfg.get("slippage_per_leg", 0.05)
    new_legs: list[TradeLeg] = []

    for leg in suggestion.legs:
        sd = _get_strike_data(chain, leg.strike)
        if sd is None:
            new_legs.append(leg)
            continue

        if leg.action == "SELL":
            bid = sd.ce_bid if leg.option_type == "CE" else sd.pe_bid
            if bid > 0:
                fill_price = max(0.05, bid - slippage)
            else:
                fill_price = max(0.05, leg.ltp - slippage)
        else:
            ask = sd.ce_ask if leg.option_type == "CE" else sd.pe_ask
            if ask > 0:
                fill_price = ask + slippage
            else:
                fill_price = leg.ltp + slippage

        new_legs.append(leg.model_copy(update={"ltp": round(fill_price, 2)}))

    # Post-fill RR check
    max_rr = cfg.get("post_fill_max_rr", 2.0)
    sell_prem = sum(l.ltp for l in new_legs if l.action == "SELL")
    buy_prem = sum(l.ltp for l in new_legs if l.action == "BUY")
    net_credit = sell_prem - buy_prem

    if suggestion.strategy in _OPTIMUS_CREDIT and net_credit <= 0:
        return new_legs, "no net credit after realistic fills"

    if suggestion.strategy in _SPREAD_STRATEGIES and net_credit > 0:
        width = _compute_spread_width(new_legs)
        if width > 0:
            rr = (width - net_credit) / net_credit
            if rr > max_rr:
                return new_legs, f"post-fill RR {rr:.1f}:1 > max {max_rr}:1"

    return new_legs, None


# ===================================================================
# Section 8: Portfolio Structure
# ===================================================================

def _check_portfolio_structure(
    state: PaperTradingState,
    suggestion: TradeSuggestion,
    cfg: dict,
) -> str | None:
    """Check portfolio structure limits. Returns rejection reason or None."""
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

    if suggestion.strategy in _OPTIMUS_PRIMARY:
        if primary_used >= primary_limit:
            return f"primary allocation exhausted ({primary_used:.0f}/{primary_limit:.0f})"
    elif suggestion.strategy in _OPTIMUS_SECONDARY:
        if secondary_used >= secondary_limit:
            return f"secondary allocation exhausted ({secondary_used:.0f}/{secondary_limit:.0f})"

    return None


# ===================================================================
# Section 9: Defensive Adjustments (Position Management)
# ===================================================================

def _check_defensive_adjustments(
    position: PaperPosition,
    chain: OptionChainData,
    technicals: TechnicalIndicators,
    cfg: dict,
) -> tuple[str | None, str]:
    """Check if position needs defensive adjustment.

    Returns (exit_reason | None, note for reasoning).
    """
    delta_exit = cfg.get("delta_exit_threshold", 0.30)
    delta_warn = cfg.get("delta_adjustment_threshold", 0.25)
    notes: list[str] = []

    for leg in position.legs:
        if leg.action != "SELL":
            continue
        sd = _get_strike_data(chain, leg.strike)
        if sd is None:
            continue

        delta = abs(sd.ce_delta if leg.option_type == "CE" else sd.pe_delta)

        if delta >= delta_exit:
            return (
                f"short {leg.option_type} {int(leg.strike)} delta {delta:.3f} >= {delta_exit} — forced exit",
                f"Delta breach: {leg.option_type} {int(leg.strike)} delta={delta:.3f}",
            )

        if delta >= delta_warn:
            notes.append(
                f"WARNING: {leg.option_type} {int(leg.strike)} delta {delta:.3f} approaching {delta_exit} — consider rolling"
            )
            logger.warning("OPTIMUS: %s", notes[-1])

    # Market breakout detection — close all range-bound positions
    is_credit_range = position.strategy in {
        StrategyName.IRON_CONDOR.value,
        StrategyName.SHORT_STRANGLE.value,
    }
    if is_credit_range and _is_breakout(technicals):
        return (
            f"market breakout detected — closing range-bound {position.strategy}",
            "Breakout: spot outside BB with supertrend confirmation",
        )

    return None, "; ".join(notes) if notes else ""


# ===================================================================
# Section 3C: Stop Loss Rules
# ===================================================================

def _check_optimus_credit_sl(
    position: PaperPosition,
    chain: OptionChainData,
    technicals: TechnicalIndicators,
    cfg: dict,
) -> PositionStatus | None:
    """Check hedge credit stop loss: 1.8x credit OR delta > 0.30 OR technical breach."""
    pnl = position.total_unrealized_pnl
    sl_mult = cfg.get("credit_sl_multiplier", 1.8)

    # 1.8x credit loss
    if position.net_premium > 0 and pnl <= -(abs(position.net_premium) * sl_mult):
        logger.info("OPTIMUS SL: %s loss %.0f >= %.1fx credit %.0f",
                     position.strategy, abs(pnl), sl_mult, abs(position.net_premium))
        return PositionStatus.CLOSED_STOP_LOSS

    # Delta breach (checked in defensive adjustments, but double-check here)
    delta_exit = cfg.get("delta_exit_threshold", 0.30)
    for leg in position.legs:
        if leg.action != "SELL":
            continue
        sd = _get_strike_data(chain, leg.strike)
        if sd is None:
            continue
        delta = abs(sd.ce_delta if leg.option_type == "CE" else sd.pe_delta)
        if delta >= delta_exit:
            logger.info("OPTIMUS SL: %s delta %.3f >= %.2f on %s %d",
                         position.strategy, delta, delta_exit, leg.option_type, int(leg.strike))
            return PositionStatus.CLOSED_STOP_LOSS

    return None


def _check_optimus_strangle_sl(
    position: PaperPosition,
    chain: OptionChainData,
    cfg: dict,
) -> PositionStatus | None:
    """Strangle SL: premium doubles OR delta > 0.30 OR total loss = 1x credit."""
    pnl = position.total_unrealized_pnl

    # Total loss = 1x credit received (premium doubles means loss = credit)
    if position.net_premium > 0 and pnl <= -abs(position.net_premium):
        logger.info("OPTIMUS SL: strangle premium doubled — loss %.0f >= credit %.0f",
                     abs(pnl), abs(position.net_premium))
        return PositionStatus.CLOSED_STOP_LOSS

    # Per-leg check: if any sell leg's current premium > 2x entry
    for leg in position.legs:
        if leg.action == "SELL" and leg.entry_ltp > 0:
            if leg.current_ltp >= 2 * leg.entry_ltp:
                logger.info("OPTIMUS SL: strangle leg %s %d premium doubled (%.1f -> %.1f)",
                             leg.option_type, int(leg.strike), leg.entry_ltp, leg.current_ltp)
                return PositionStatus.CLOSED_STOP_LOSS

    return None


def _check_optimus_debit_sl(
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
# Enrichment
# ===================================================================

def _enrich_optimus_suggestion(
    suggestion: TradeSuggestion,
    chain: OptionChainData,
    analytics: OptionsAnalytics,
    technicals: TechnicalIndicators,
    vix: float | None,
    cfg: dict,
    lots: int,
) -> TradeSuggestion:
    """Populate numeric fields and add regime justification to reasoning."""
    lot_size = cfg.get("lot_size", 65)

    sell_prem = sum(leg.ltp for leg in suggestion.legs if leg.action == "SELL")
    buy_prem = sum(leg.ltp for leg in suggestion.legs if leg.action == "BUY")
    net = sell_prem - buy_prem
    width = _compute_spread_width(suggestion.legs)
    is_credit = suggestion.strategy in _OPTIMUS_CREDIT

    if is_credit:
        if width > 0:
            max_profit = net * lot_size * lots
            max_loss = (width - net) * lot_size * lots
        else:
            max_profit = net * lot_size * lots
            max_loss = net * 1.8 * lot_size * lots
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

    # Regime justification reasoning
    account_value = cfg.get("initial_capital", 2_500_000)
    risk_pct = (max_loss / account_value * 100) if account_value > 0 else 0
    bb_width = _compute_bb_width_pct(technicals)
    extra_reasoning = [
        f"Regime: VIX={vix:.1f if vix else 'N/A'}, IV%ile={analytics.iv_percentile:.1f}, BB width={bb_width:.2f}%",
        f"Range-bound: {'Yes' if _is_range_bound(technicals) else 'No'}",
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
class OptimusAlgorithm(TradingAlgorithm):
    """V3 Optimus algorithm — Capital preservation with strict drawdown limits."""

    name = "optimus"
    display_name = "V3 Optimus"
    description = "Capital preservation: 1.5-3% monthly, max 8% annual drawdown"

    # ---------------------------------------------------------------
    # generate_suggestions — Build own candidates from chain scan
    # ---------------------------------------------------------------

    def generate_suggestions(
        self,
        chain: OptionChainData,
        technicals: TechnicalIndicators,
        analytics: OptionsAnalytics,
        observation: ObservationSnapshot | None = None,
        context: MarketContext | None = None,
    ) -> list[TradeSuggestion]:
        """Generate hedge trade candidates independently (no V1 dependency)."""
        if not is_market_open():
            return []

        cfg = self.config
        accepted: list[TradeSuggestion] = []
        rejected: list[TradeSuggestion] = []

        # Fetch India VIX
        vix = _fetch_india_vix(cfg)
        if vix is None:
            # Fallback to ATM IV as proxy
            vix = analytics.atm_iv
            logger.info("OPTIMUS: VIX fetch failed, using ATM IV %.1f as proxy", vix)

        # --- Credit strategies ---
        credit_rejection, is_half_size = _check_credit_regime(analytics, technicals, vix, cfg)

        if credit_rejection is None:
            # Build iron condor
            ic = _build_iron_condor(chain, analytics, technicals, cfg)
            if ic is not None:
                # Liquidity filter
                liq_reject = _check_optimus_liquidity(ic, chain, cfg)
                if liq_reject:
                    rejected.append(ic.model_copy(update={"rejection_reason": liq_reject, "score": 0.0}))
                    logger.info("OPTIMUS REJECTED: %s — %s", ic.strategy.value, liq_reject)
                else:
                    # Realistic fills
                    new_legs, fill_reject = _apply_optimus_fills(ic, chain, cfg)
                    if fill_reject:
                        rejected.append(ic.model_copy(update={"rejection_reason": fill_reject, "score": 0.0}))
                        logger.info("OPTIMUS REJECTED: %s — %s", ic.strategy.value, fill_reject)
                    else:
                        ic = ic.model_copy(update={"legs": new_legs})
                        enriched = _enrich_optimus_suggestion(ic, chain, analytics, technicals, vix, cfg, lots=1)
                        accepted.append(enriched)
                        logger.info(
                            "OPTIMUS ACCEPTED: %s | RR=%.2f | POP=%.1f%% | liq=%.0f",
                            enriched.strategy.value, enriched.risk_reward_ratio,
                            enriched.pop, enriched.liquidity_score,
                        )

            # Build hedged strangle
            hs = _build_hedged_strangle(chain, analytics, technicals, cfg)
            if hs is not None:
                liq_reject = _check_optimus_liquidity(hs, chain, cfg)
                if liq_reject:
                    rejected.append(hs.model_copy(update={"rejection_reason": liq_reject, "score": 0.0}))
                    logger.info("OPTIMUS REJECTED: %s — %s", hs.strategy.value, liq_reject)
                else:
                    new_legs, fill_reject = _apply_optimus_fills(hs, chain, cfg)
                    if fill_reject:
                        rejected.append(hs.model_copy(update={"rejection_reason": fill_reject, "score": 0.0}))
                        logger.info("OPTIMUS REJECTED: %s — %s", hs.strategy.value, fill_reject)
                    else:
                        hs = hs.model_copy(update={"legs": new_legs})
                        enriched = _enrich_optimus_suggestion(hs, chain, analytics, technicals, vix, cfg, lots=1)
                        accepted.append(enriched)
                        logger.info(
                            "OPTIMUS ACCEPTED: %s | RR=%.2f | POP=%.1f%% | liq=%.0f",
                            enriched.strategy.value, enriched.risk_reward_ratio,
                            enriched.pop, enriched.liquidity_score,
                        )
        else:
            logger.info("OPTIMUS: Credit regime blocked — %s", credit_rejection)

        # --- Debit strategies ---
        ds = _build_debit_spread(chain, analytics, technicals, cfg)
        if ds is not None:
            liq_reject = _check_optimus_liquidity(ds, chain, cfg)
            if liq_reject:
                rejected.append(ds.model_copy(update={"rejection_reason": liq_reject, "score": 0.0}))
                logger.info("OPTIMUS REJECTED: %s — %s", ds.strategy.value, liq_reject)
            else:
                new_legs, fill_reject = _apply_optimus_fills(ds, chain, cfg)
                if fill_reject:
                    rejected.append(ds.model_copy(update={"rejection_reason": fill_reject, "score": 0.0}))
                    logger.info("OPTIMUS REJECTED: %s — %s", ds.strategy.value, fill_reject)
                else:
                    ds = ds.model_copy(update={"legs": new_legs})
                    enriched = _enrich_optimus_suggestion(ds, chain, analytics, technicals, vix, cfg, lots=1)
                    accepted.append(enriched)
                    logger.info(
                        "OPTIMUS ACCEPTED: %s | RR=%.2f | POP=%.1f%% | liq=%.0f",
                        enriched.strategy.value, enriched.risk_reward_ratio,
                        enriched.pop, enriched.liquidity_score,
                    )

        return sorted(accepted, key=lambda s: s.score, reverse=True) + rejected

    # ---------------------------------------------------------------
    # evaluate_and_manage — Manage positions + open new
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
        observation: ObservationSnapshot | None = None,
        context: MarketContext | None = None,
    ) -> PaperTradingState:
        """Manage positions with hedge-specific risk rules."""
        # Market-open guard: skip all trading logic when market is closed
        if not is_market_open():
            return state

        if lot_size is None:
            lot_size = self.config.get("lot_size", 65)
        cfg = self.config

        # Reset daily/weekly period tracking
        state = _reset_period_tracking(state, cfg)

        # Fetch VIX for regime awareness during management
        vix = _fetch_india_vix(cfg) if chain else None
        _, is_half_size = (None, False)
        if analytics and technicals and vix is not None:
            _, is_half_size = _check_credit_regime(analytics, technicals, vix, cfg)

        # Build trade status notes for dashboard visibility
        notes: list[str] = []

        # Add observation context to notes
        if observation:
            notes.append(f"Observation: {observation.bias} bias ({observation.bars_collected} bars)")
            if observation.initial_trend.strength != "weak":
                notes.append(f"Trend: {observation.initial_trend.direction} ({observation.initial_trend.strength})")
            if observation.volume.classification != "normal":
                notes.append(f"Volume: {observation.volume.classification} ({observation.volume.relative_volume:.1f}x)")

        if vix is not None:
            notes.append(f"VIX: {vix:.1f}")
            vix_min = cfg.get("vix_no_credit_below", 14.0)
            vix_half = cfg.get("vix_half_size_above", 28.0)
            if vix < vix_min:
                notes.append(f"Credit strategies blocked: VIX {vix:.1f} < minimum {vix_min}")
            elif vix > vix_half:
                notes.append(f"Half-size mode: VIX {vix:.1f} > {vix_half}")
        else:
            notes.append("VIX: unavailable")
        if analytics:
            notes.append(f"IV percentile: {analytics.iv_percentile:.1f}")
            min_iv_pct = cfg.get("credit_min_iv_percentile", 55.0)
            if analytics.iv_percentile < min_iv_pct:
                notes.append(f"Credit blocked: IV percentile {analytics.iv_percentile:.1f} < {min_iv_pct}")

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

            # Hedge-specific stop losses
            if chain:
                if strategy_type == StrategyType.CREDIT:
                    if position.strategy == StrategyName.SHORT_STRANGLE.value:
                        exit_reason_sl = _check_optimus_strangle_sl(position, chain, cfg)
                    else:
                        exit_reason_sl = _check_optimus_credit_sl(position, chain, technicals or TechnicalIndicators(), cfg)
                    if exit_reason_sl:
                        exit_reason = exit_reason_sl
                elif strategy_type == StrategyType.DEBIT:
                    exit_reason_sl = _check_optimus_debit_sl(position, cfg)
                    if exit_reason_sl:
                        exit_reason = exit_reason_sl

                # Defensive adjustments (delta monitoring, breakout closure)
                if exit_reason is None and technicals:
                    def_exit, def_note = _check_defensive_adjustments(position, chain, technicals, cfg)
                    if def_exit:
                        logger.info("OPTIMUS DEFENSIVE: %s — %s", position.strategy, def_exit)
                        exit_reason = PositionStatus.CLOSED_STOP_LOSS
                    elif def_note:
                        logger.info("OPTIMUS NOTE: %s — %s", position.strategy, def_note)

            # Standard exit conditions (PT, EOD)
            if exit_reason is None:
                exit_reason = check_exit_conditions(position)

            if exit_reason:
                _closed_pos, record = close_position(
                    position, exit_reason,
                    technicals=technicals, analytics=analytics, chain=chain,
                )
                logger.info(
                    "OPTIMUS CLOSED: %s | reason=%s | pnl=%.2f",
                    record.strategy, exit_reason.value if hasattr(exit_reason, 'value') else exit_reason,
                    record.net_pnl,
                )
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
            logger.warning("OPTIMUS HALTED: %d consecutive losses >= %d", consecutive_losses, halt_after)

        # Track peak capital
        current_capital = state.initial_capital + state.total_realized_pnl + added_realized - (state.total_execution_costs + added_costs)
        peak_capital = max(state.peak_capital, current_capital)

        # Build intermediate state
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

        # --- Initialize session start capital ---
        if state.session_start_capital == 0.0:
            state = state.model_copy(update={
                "session_start_capital": state.initial_capital + state.net_realized_pnl,
            })

        # --- Check daily/weekly shutdown ---
        account_value = state.initial_capital + state.net_realized_pnl
        daily_loss_pct = cfg.get("daily_loss_shutdown_pct", 1.25)
        if state.daily_start_capital > 0:
            daily_pnl = account_value - state.daily_start_capital
            if daily_pnl <= -(state.daily_start_capital * daily_loss_pct / 100):
                logger.warning("OPTIMUS SHUTDOWN: daily loss %.0f > %.1f%%", abs(daily_pnl), daily_loss_pct)
                notes.append(f"Blocked: Daily loss shutdown (loss {abs(daily_pnl):.0f} > {daily_loss_pct}%)")
                return state.model_copy(update={"trade_status_notes": notes})

        weekly_dd_pct = cfg.get("weekly_drawdown_shutdown_pct", 2.0)
        if state.weekly_start_capital > 0:
            weekly_pnl = account_value - state.weekly_start_capital
            if weekly_pnl <= -(state.weekly_start_capital * weekly_dd_pct / 100):
                logger.warning("OPTIMUS SHUTDOWN: weekly drawdown %.0f > %.1f%%", abs(weekly_pnl), weekly_dd_pct)
                notes.append(f"Blocked: Weekly drawdown shutdown (loss {abs(weekly_pnl):.0f} > {weekly_dd_pct}%)")
                return state.model_copy(update={"trade_status_notes": notes})

        # --- Phase 2: Open new positions ---
        # Observation period guard: collect data, don't trade yet
        entry_start = cfg.get("entry_start_time",
                              load_config().get("paper_trading", {}).get("entry_start_time", "10:00"))
        if _is_observation_period(entry_start):
            notes.append(f"Observation period active — collecting data before {entry_start}")
            return state.model_copy(update={"trade_status_notes": notes})

        if not state.is_auto_trading:
            notes.append("Auto-trading is OFF")
            return state.model_copy(update={"trade_status_notes": notes})
        if not suggestions:
            notes.append("No trade suggestions available")
            return state.model_copy(update={"trade_status_notes": notes})
        if not chain:
            notes.append("No option chain data available")
            return state.model_copy(update={"trade_status_notes": notes})

        if state.trading_halted:
            logger.info("OPTIMUS: Trading halted — skipping new positions")
            notes.append(f"Blocked: Trading halted ({state.consecutive_losses} consecutive losses)")
            return state.model_copy(update={"trade_status_notes": notes})

        # Data staleness guard
        if technicals and technicals.data_staleness_minutes > 20:
            notes.append(f"Blocked: Data stale ({technicals.data_staleness_minutes:.0f} min > 20 min limit)")
            return state.model_copy(update={"trade_status_notes": notes})

        # Cooldown
        now_ts = _time.time()
        if state.last_trade_opened_ts > 0 and (now_ts - state.last_trade_opened_ts) < 60:
            notes.append("Trade cooldown active (60s between trades)")
            return state.model_copy(update={"trade_status_notes": notes})

        opened_trade = False
        for suggestion in suggestions:
            if suggestion.rejection_reason:
                continue
            if suggestion.score <= 0:
                continue

            # Global gates
            gate_reason = _check_optimus_gates(state, suggestion, chain, cfg)
            if gate_reason:
                logger.info("OPTIMUS GATE BLOCKED: %s — %s", suggestion.strategy.value, gate_reason)
                notes.append(f"Blocked {suggestion.strategy.value}: {gate_reason}")
                continue

            # Portfolio structure
            struct_reason = _check_portfolio_structure(state, suggestion, cfg)
            if struct_reason:
                logger.info("OPTIMUS STRUCTURE BLOCKED: %s — %s", suggestion.strategy.value, struct_reason)
                notes.append(f"Blocked {suggestion.strategy.value}: {struct_reason}")
                continue

            # Skip duplicates
            held = {p.strategy for p in state.open_positions if p.status == PositionStatus.OPEN}
            if suggestion.strategy.value in held:
                notes.append(f"Blocked {suggestion.strategy.value}: already holding same strategy")
                continue

            if state.capital_remaining <= 0:
                notes.append("Blocked: no capital remaining")
                break

            # Hedge position sizing
            hedge_lots, size_reject = _compute_optimus_lots(suggestion, state, cfg, is_half_size)
            if size_reject:
                logger.info("OPTIMUS SIZING REJECTED: %s — %s", suggestion.strategy.value, size_reject)
                notes.append(f"Blocked {suggestion.strategy.value}: sizing — {size_reject}")
                continue

            # Open position
            position = open_position(
                suggestion, lot_size, state.capital_remaining,
                expiry=chain.expiry,
                technicals=technicals, analytics=analytics, chain=chain,
            )
            # Override lots with hedge sizing
            if hedge_lots > 0 and hedge_lots != position.lots:
                position = position.model_copy(update={"lots": hedge_lots})

            position = position.model_copy(update={
                "entry_date": _now_ist().strftime("%Y-%m-%d"),
            })

            logger.info(
                "OPTIMUS OPENED: %s | lots=%d | margin=%.0f | premium=%.2f",
                position.strategy, position.lots, position.margin_required, position.net_premium,
            )

            # Update allocation tracker
            tracker = dict(state.allocation_tracker or {})
            if suggestion.strategy in _OPTIMUS_PRIMARY:
                tracker["primary_used"] = tracker.get("primary_used", 0.0) + position.margin_required
            elif suggestion.strategy in _OPTIMUS_SECONDARY:
                tracker["secondary_used"] = tracker.get("secondary_used", 0.0) + position.margin_required

            state = state.model_copy(update={
                "open_positions": state.open_positions + [position],
                "last_trade_opened_ts": _time.time(),
                "allocation_tracker": tracker,
            })
            notes.append(f"Opened: {position.strategy} ({position.lots} lots)")
            opened_trade = True
            break  # max 1 trade per cycle

        if not opened_trade and not any("Opened:" in n for n in notes):
            notes.append("All suggestions blocked by gates/structure/sizing")

        return state.model_copy(update={"trade_status_notes": notes})
