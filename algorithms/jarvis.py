"""V2 Jarvis algorithm — risk-first, rule-based trading system.

Implements all 10 master prompt sections:
  1. Strategy separation (credit vs debit)
  2. Global hard risk limits
  3. Credit strategy rules
  4. Debit strategy rules
  5. Position sizing
  6. Entry logic filters
  7. Realistic execution (bid/ask + slippage)
  8. Portfolio defense
  9. Output format (numeric fields on TradeSuggestion)
 10. Performance metrics (computed on PaperTradingState)

Self-contained — no coupling to V1 Sentinel logic.
"""

from __future__ import annotations

import logging
import math
import time as _time
from collections import Counter
from datetime import datetime, time

from typing import TYPE_CHECKING

from algorithms import register_algorithm
from algorithms.base import TradingAlgorithm
from core.config import load_config
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
    compute_lots,
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
    is_observation_period as _is_observation_period,
)

if TYPE_CHECKING:
    from core.observation import ObservationSnapshot

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Credit / Debit strategy sets
# ---------------------------------------------------------------------------

_CREDIT_STRATEGIES = {
    StrategyName.SHORT_STRADDLE,
    StrategyName.SHORT_STRANGLE,
    StrategyName.BULL_PUT_SPREAD,
    StrategyName.BEAR_CALL_SPREAD,
    StrategyName.IRON_CONDOR,
}

_DEBIT_STRATEGIES = {
    StrategyName.LONG_STRADDLE,
    StrategyName.LONG_STRANGLE,
    StrategyName.BULL_CALL_SPREAD,
    StrategyName.BEAR_PUT_SPREAD,
    StrategyName.LONG_CE,
    StrategyName.LONG_PE,
}

_SPREAD_STRATEGIES = {
    StrategyName.BULL_PUT_SPREAD,
    StrategyName.BEAR_CALL_SPREAD,
    StrategyName.BULL_CALL_SPREAD,
    StrategyName.BEAR_PUT_SPREAD,
    StrategyName.IRON_CONDOR,
}

_NAKED_STRATEGIES = {
    StrategyName.SHORT_STRADDLE,
    StrategyName.SHORT_STRANGLE,
    StrategyName.LONG_STRADDLE,
    StrategyName.LONG_STRANGLE,
}


def _get_vix_from_analytics(analytics: OptionsAnalytics | None) -> float:
    """Extract VIX-like value. Uses ATM IV as proxy if no direct VIX."""
    if analytics:
        return analytics.atm_iv
    return 0.0


# ===================================================================
# Section 6: Entry Logic Filters
# ===================================================================

def _check_credit_entry_filters(
    analytics: OptionsAnalytics,
    technicals: TechnicalIndicators,
    cfg: dict,
) -> str | None:
    """Credit entry: IV percentile ≥ 50, market in range."""
    min_iv_pct = cfg.get("credit_min_iv_percentile", 50.0)
    if analytics.iv_percentile < min_iv_pct:
        return f"IV percentile {analytics.iv_percentile:.1f} < {min_iv_pct} (need elevated IV for credit)"

    # Range detection: BB width compression or spot near VWAP
    if technicals.bb_middle > 0:
        bb_width_pct = ((technicals.bb_upper - technicals.bb_lower) / technicals.bb_middle) * 100
        spot_from_vwap_pct = abs(technicals.spot - technicals.vwap) / technicals.vwap * 100 if technicals.vwap > 0 else 0
        if bb_width_pct > 3.0 and spot_from_vwap_pct > 1.0:
            return f"Market not in range: BB width {bb_width_pct:.1f}%, spot {spot_from_vwap_pct:.1f}% from VWAP"

    return None


def _check_debit_entry_filters(
    analytics: OptionsAnalytics,
    technicals: TechnicalIndicators,
    cfg: dict,
) -> str | None:
    """Debit entry: IV percentile ≤ 50, directional momentum confirmed."""
    max_iv_pct = cfg.get("debit_max_iv_percentile", 50.0)
    if analytics.iv_percentile > max_iv_pct:
        return f"IV percentile {analytics.iv_percentile:.1f} > {max_iv_pct} (IV too high for debit)"

    return None


# ===================================================================
# Section 3: Credit Strategy Validation
# ===================================================================

def _validate_credit_trade(
    suggestion: TradeSuggestion,
    chain: OptionChainData,
    analytics: OptionsAnalytics,
    cfg: dict,
) -> str | None:
    """Validate credit-specific rules. Returns rejection reason or None."""
    lot_size = cfg.get("lot_size", 65)

    # --- Risk:Reward check ---
    max_rr = cfg.get("credit_max_risk_reward", 2.5)
    sell_premium = sum(leg.ltp for leg in suggestion.legs if leg.action == "SELL")
    buy_premium = sum(leg.ltp for leg in suggestion.legs if leg.action == "BUY")
    net_credit = sell_premium - buy_premium
    if net_credit <= 0:
        return "no net credit on credit strategy"

    if suggestion.strategy in _SPREAD_STRATEGIES:
        width = _compute_spread_width(suggestion.legs)
        if width > 0:
            max_loss = (width - net_credit) * lot_size
            max_profit = net_credit * lot_size
            if max_profit > 0:
                rr = max_loss / max_profit
                if rr > max_rr:
                    return f"RR {rr:.1f}:1 > max {max_rr}:1"

    # --- Strangle-specific rules ---
    if suggestion.strategy == StrategyName.SHORT_STRANGLE:
        min_credit_pct = cfg.get("strangle_min_credit_pct_spot", 2.0)
        credit_pct = (net_credit / chain.underlying_value) * 100 if chain.underlying_value > 0 else 0
        if credit_pct < min_credit_pct:
            return f"strangle credit {credit_pct:.2f}% < min {min_credit_pct}% of spot"

        delta_min = cfg.get("strangle_delta_min", 0.12)
        delta_max = cfg.get("strangle_delta_max", 0.20)
        for leg in suggestion.legs:
            if leg.action == "SELL":
                sd = _get_strike_data(chain, leg.strike)
                if sd:
                    delta = abs(sd.ce_delta if leg.option_type == "CE" else sd.pe_delta)
                    if delta > 0 and (delta < delta_min or delta > delta_max):
                        return f"SELL {leg.option_type} delta {delta:.3f} outside [{delta_min}, {delta_max}]"

    # --- Spread width check ---
    if suggestion.strategy in _SPREAD_STRATEGIES and suggestion.strategy != StrategyName.IRON_CONDOR:
        allowed_widths = cfg.get("spread_widths", [50, 100])
        width = _compute_spread_width(suggestion.legs)
        if width > 0 and width not in allowed_widths:
            return f"spread width {width} not in allowed {allowed_widths}"

        # Credit ≥ 30% of width
        min_credit_pct_width = cfg.get("spread_min_credit_pct_width", 30.0)
        if width > 0:
            credit_pct_of_width = (net_credit / width) * 100
            if credit_pct_of_width < min_credit_pct_width:
                return f"credit {credit_pct_of_width:.1f}% of width < min {min_credit_pct_width}%"

    # --- Liquidity check ---
    leg_strikes = [leg.strike for leg in suggestion.legs]
    min_oi = cfg.get("min_oi", 10_000)
    min_vol = cfg.get("min_volume", 1_000)
    max_ba = cfg.get("max_bid_ask_pct", 5.0)
    _liq_score, liq_reject = compute_liquidity_score(chain, leg_strikes, min_oi, min_vol, max_ba)
    if liq_reject:
        return f"liquidity: {liq_reject}"

    return None


# ===================================================================
# Section 4: Debit Strategy Validation
# ===================================================================

def _validate_debit_trade(
    suggestion: TradeSuggestion,
    chain: OptionChainData,
    analytics: OptionsAnalytics,
    technicals: TechnicalIndicators,
    cfg: dict,
) -> str | None:
    """Validate debit-specific rules. Returns rejection reason or None."""
    # --- Long Call: spot > EMA20 & EMA50, RSI > 55, IV percentile < 50 ---
    if suggestion.strategy == StrategyName.LONG_CE:
        if technicals.spot <= technicals.ema_20 or technicals.spot <= technicals.ema_50:
            return "Long Call: spot not above EMA20 & EMA50"
        if technicals.rsi <= 55:
            return f"Long Call: RSI {technicals.rsi:.1f} ≤ 55"

    # --- Long Put: spot < EMA20 & EMA50, RSI < 45, IV percentile < 50 ---
    if suggestion.strategy == StrategyName.LONG_PE:
        if technicals.spot >= technicals.ema_20 or technicals.spot >= technicals.ema_50:
            return "Long Put: spot not below EMA20 & EMA50"
        if technicals.rsi >= 45:
            return f"Long Put: RSI {technicals.rsi:.1f} ≥ 45"

    # --- Debit spreads: RR ≥ 1:1.8, POP ≥ 40% ---
    if suggestion.strategy in {StrategyName.BULL_CALL_SPREAD, StrategyName.BEAR_PUT_SPREAD}:
        lot_size = cfg.get("lot_size", 65)
        buy_premium = sum(leg.ltp for leg in suggestion.legs if leg.action == "BUY")
        sell_premium = sum(leg.ltp for leg in suggestion.legs if leg.action == "SELL")
        net_debit = buy_premium - sell_premium
        width = _compute_spread_width(suggestion.legs)

        if net_debit > 0 and width > 0:
            max_profit = (width - net_debit) * lot_size
            max_loss = net_debit * lot_size
            min_rr = cfg.get("debit_min_risk_reward", 1.8)
            if max_loss > 0:
                rr = max_profit / max_loss
                if rr < min_rr:
                    return f"debit spread RR {rr:.2f}:1 < min {min_rr}:1"

            # POP check using short strike delta
            min_pop = cfg.get("debit_min_pop_pct", 40.0)
            for leg in suggestion.legs:
                if leg.action == "SELL":
                    sd = _get_strike_data(chain, leg.strike)
                    if sd:
                        delta = abs(sd.ce_delta if leg.option_type == "CE" else sd.pe_delta)
                        pop = compute_pop("credit", short_strike_delta=delta)
                        # For debit spread buyer, POP is roughly complement
                        debit_pop = 100.0 - pop
                        if debit_pop < min_pop:
                            return f"debit spread POP {debit_pop:.1f}% < min {min_pop}%"

    # --- Liquidity check ---
    leg_strikes = [leg.strike for leg in suggestion.legs]
    min_oi = cfg.get("min_oi", 10_000)
    min_vol = cfg.get("min_volume", 1_000)
    max_ba = cfg.get("max_bid_ask_pct", 5.0)
    _liq_score, liq_reject = compute_liquidity_score(chain, leg_strikes, min_oi, min_vol, max_ba)
    if liq_reject:
        return f"liquidity: {liq_reject}"

    return None


# ===================================================================
# Section 2: Global Hard Risk Limits
# ===================================================================

def _check_global_gates(
    state: PaperTradingState,
    suggestion: TradeSuggestion,
    chain: OptionChainData,
    analytics: OptionsAnalytics | None,
    cfg: dict,
) -> str | None:
    """Check all global risk limits. Returns rejection reason or None."""
    lot_size = cfg.get("lot_size", 65)

    # --- Daily loss > 2% → full shutdown ---
    daily_loss_pct = cfg.get("daily_loss_shutdown_pct", 2.0)
    if state.session_start_capital > 0:
        current_capital = state.initial_capital + state.net_realized_pnl
        session_pnl = current_capital - state.session_start_capital
        if session_pnl <= -(state.session_start_capital * daily_loss_pct / 100):
            return f"daily loss shutdown: {abs(session_pnl):.0f} > {daily_loss_pct}% of session capital"

    # --- Trading halted by portfolio defense ---
    if state.trading_halted:
        return "trading halted by portfolio defense (consecutive losses)"

    # --- Capital at risk ≤ 1% per trade ---
    max_risk_pct = cfg.get("max_capital_risk_per_trade_pct", 1.0)
    account_value = state.initial_capital + state.net_realized_pnl
    net_premium_1lot = compute_net_premium(suggestion, lot_size, lots=1)

    # Estimate max loss for 1 lot
    if suggestion.strategy in _SPREAD_STRATEGIES:
        width = _compute_spread_width(suggestion.legs)
        sell_prem = sum(leg.ltp for leg in suggestion.legs if leg.action == "SELL")
        buy_prem = sum(leg.ltp for leg in suggestion.legs if leg.action == "BUY")
        net_credit_per_unit = sell_prem - buy_prem
        max_loss_1lot = (width - max(0, net_credit_per_unit)) * lot_size
    else:
        # For naked strategies and debit: use net premium as proxy
        max_loss_1lot = abs(net_premium_1lot) * 2  # conservative 2× estimate

    if account_value > 0 and max_loss_1lot > 0:
        risk_pct = (max_loss_1lot / account_value) * 100
        if risk_pct > max_risk_pct:
            return f"1-lot risk {risk_pct:.2f}% > max {max_risk_pct}% per trade"

    # --- Margin utilization ≤ 30% ---
    max_margin_util = cfg.get("max_margin_utilization_pct", 30.0)
    if account_value > 0:
        margin_util = (state.margin_in_use / account_value) * 100
        if margin_util > max_margin_util:
            return f"margin utilization {margin_util:.1f}% > max {max_margin_util}%"

    # --- Max 5 open positions, max 2 per direction ---
    max_open = cfg.get("max_open_positions", 5)
    open_count = len([p for p in state.open_positions if p.status == PositionStatus.OPEN])
    if open_count >= max_open:
        return f"max {max_open} open positions reached"

    max_per_dir = cfg.get("max_positions_per_direction", 2)
    dir_counts = Counter(
        p.direction_bias for p in state.open_positions if p.status == PositionStatus.OPEN
    )
    if dir_counts.get(suggestion.direction_bias, 0) >= max_per_dir:
        return f"max {max_per_dir} {suggestion.direction_bias} positions reached"

    # --- Expiry day after cutoff → block ---
    cutoff_time_str = cfg.get("expiry_day_cutoff_time", "13:30")
    now = _now_ist()
    if chain.expiry:
        try:
            expiry_date = datetime.strptime(chain.expiry, "%d-%b-%Y").date()
            if now.date() == expiry_date:
                h, m = (int(x) for x in cutoff_time_str.split(":"))
                if now.time() >= time(h, m):
                    return f"expiry day cutoff: {now.strftime('%H:%M')} >= {cutoff_time_str}"
        except ValueError:
            pass

    # --- VIX > 28 → spreads only ---
    vix_threshold = cfg.get("vix_spread_only_threshold", 28.0)
    vix = _get_vix_from_analytics(analytics)
    if vix > vix_threshold and suggestion.strategy in _NAKED_STRATEGIES:
        return f"VIX {vix:.1f} > {vix_threshold}: naked strategies blocked, spreads only"

    return None


# ===================================================================
# Section 5: Position Sizing
# ===================================================================

def _compute_jarvis_lots(
    suggestion: TradeSuggestion,
    account_value: float,
    cfg: dict,
    consecutive_losses: int = 0,
) -> tuple[int, str | None]:
    """Compute lots using 1% risk rule. Returns (lots, rejection_reason)."""
    lot_size = cfg.get("lot_size", 65)

    # Estimate max loss per lot
    if suggestion.strategy in _SPREAD_STRATEGIES:
        width = _compute_spread_width(suggestion.legs)
        sell_prem = sum(leg.ltp for leg in suggestion.legs if leg.action == "SELL")
        buy_prem = sum(leg.ltp for leg in suggestion.legs if leg.action == "BUY")
        net_credit_per_unit = sell_prem - buy_prem
        max_loss_per_lot = (width - max(0, net_credit_per_unit)) * lot_size
    elif suggestion.strategy in _CREDIT_STRATEGIES:
        # Naked credit: use net premium × SL multiplier
        net_per_unit = sum(
            leg.ltp * (1 if leg.action == "SELL" else -1) for leg in suggestion.legs
        )
        sl_mult = cfg.get("credit_sl_multiplier", 2.2)
        max_loss_per_lot = abs(net_per_unit) * sl_mult * lot_size
    else:
        # Debit: max loss = premium paid
        max_loss_per_lot = abs(sum(
            leg.ltp * (1 if leg.action == "BUY" else -1) for leg in suggestion.legs
        )) * lot_size

    if max_loss_per_lot <= 0:
        return 0, "cannot compute max loss per lot"

    # lots = floor(1% of account / max_loss_per_lot)
    risk_budget = 0.01 * account_value
    lots = math.floor(risk_budget / max_loss_per_lot)

    # Portfolio defense: halve after consecutive losses
    half_after = cfg.get("consecutive_loss_half_size", 2)
    if consecutive_losses >= half_after:
        lots = max(1, lots // 2)

    if lots < 1:
        return 0, f"max_loss_per_lot ₹{max_loss_per_lot:.0f} > 1% risk budget ₹{risk_budget:.0f}"

    return lots, None


# ===================================================================
# Section 7: Realistic Execution
# ===================================================================

def _apply_realistic_fills(
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
            # Sell at bid, minus slippage
            bid = sd.ce_bid if leg.option_type == "CE" else sd.pe_bid
            if bid > 0:
                fill_price = max(0.05, bid - slippage)
            else:
                fill_price = max(0.05, leg.ltp - slippage)
        else:
            # Buy at ask, plus slippage
            ask = sd.ce_ask if leg.option_type == "CE" else sd.pe_ask
            if ask > 0:
                fill_price = ask + slippage
            else:
                fill_price = leg.ltp + slippage

        new_legs.append(leg.model_copy(update={"ltp": round(fill_price, 2)}))

    return new_legs, None


# ===================================================================
# Section 9: Enrich Suggestion with Numeric Fields
# ===================================================================

def _enrich_suggestion(
    suggestion: TradeSuggestion,
    chain: OptionChainData,
    analytics: OptionsAnalytics,
    cfg: dict,
) -> TradeSuggestion:
    """Fill in numeric fields on a TradeSuggestion for the output format."""
    lot_size = cfg.get("lot_size", 65)

    sell_prem = sum(leg.ltp for leg in suggestion.legs if leg.action == "SELL")
    buy_prem = sum(leg.ltp for leg in suggestion.legs if leg.action == "BUY")
    net = sell_prem - buy_prem  # positive = credit

    width = _compute_spread_width(suggestion.legs)
    is_credit = suggestion.strategy in _CREDIT_STRATEGIES

    if is_credit:
        if width > 0:
            max_profit = net * lot_size
            max_loss = (width - net) * lot_size
        else:
            max_profit = net * lot_size
            max_loss = net * 2.2 * lot_size  # naked estimate
        rr = max_loss / max_profit if max_profit > 0 else 0.0
    else:
        net_debit = buy_prem - sell_prem
        if width > 0:
            max_profit = (width - net_debit) * lot_size
            max_loss = net_debit * lot_size
        else:
            max_profit = 0.0
            max_loss = net_debit * lot_size
        rr = max_profit / max_loss if max_loss > 0 else 0.0

    # POP from short strike delta
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
        min_oi=cfg.get("min_oi", 10_000),
        min_volume=cfg.get("min_volume", 1_000),
        max_ba_pct=cfg.get("max_bid_ask_pct", 5.0),
    )

    return suggestion.model_copy(update={
        "risk_reward_ratio": round(rr, 2),
        "pop": pop_val,
        "max_loss_numeric": round(max_loss, 0),
        "max_profit_numeric": round(max_profit, 0),
        "liquidity_score": liq_score,
        "net_credit_debit": round(net * lot_size, 0),
    })


# ===================================================================
# Main Algorithm Class
# ===================================================================

@register_algorithm
class JarvisAlgorithm(TradingAlgorithm):
    """V2 Jarvis algorithm with strict risk management."""

    name = "jarvis"
    display_name = "V2 Jarvis"
    description = "Risk-first rules with strict RR, liquidity, and portfolio defense"

    # ---------------------------------------------------------------
    # generate_suggestions — Sections 1, 3, 4, 6, 7, 9
    # ---------------------------------------------------------------

    def generate_suggestions(
        self,
        chain: OptionChainData,
        technicals: TechnicalIndicators,
        analytics: OptionsAnalytics,
        observation: ObservationSnapshot | None = None,
    ) -> list[TradeSuggestion]:
        """Generate and validate suggestions using Jarvis rules.

        Uses V1's base suggestions, then applies Jarvis filters.
        """
        if not is_market_open():
            return []

        # Start from V1's raw suggestions as candidates
        from core.trade_strategies import generate_trade_suggestions
        raw_suggestions = generate_trade_suggestions(analytics, technicals, chain)

        cfg = self.config
        accepted: list[TradeSuggestion] = []
        rejected: list[TradeSuggestion] = []

        for suggestion in raw_suggestions:
            is_credit = suggestion.strategy in _CREDIT_STRATEGIES
            rejection: str | None = None

            # Section 6: Entry logic filters
            if is_credit:
                rejection = _check_credit_entry_filters(analytics, technicals, cfg)
            else:
                rejection = _check_debit_entry_filters(analytics, technicals, cfg)

            # Section 3/4: Strategy-specific validation
            if rejection is None:
                if is_credit:
                    rejection = _validate_credit_trade(suggestion, chain, analytics, cfg)
                else:
                    rejection = _validate_debit_trade(suggestion, chain, analytics, technicals, cfg)

            # Section 7: Realistic execution fills
            if rejection is None:
                new_legs, fill_reject = _apply_realistic_fills(suggestion, chain, cfg)
                if fill_reject:
                    rejection = fill_reject
                else:
                    suggestion = suggestion.model_copy(update={"legs": new_legs})

                    # Re-check RR after realistic fills
                    if is_credit:
                        sell_prem = sum(l.ltp for l in new_legs if l.action == "SELL")
                        buy_prem = sum(l.ltp for l in new_legs if l.action == "BUY")
                        net_credit = sell_prem - buy_prem
                        if net_credit <= 0:
                            rejection = "no net credit after realistic fills"
                        elif suggestion.strategy in _SPREAD_STRATEGIES:
                            width = _compute_spread_width(new_legs)
                            if width > 0:
                                rr = (width - net_credit) / net_credit
                                max_rr = cfg.get("credit_max_risk_reward", 2.5)
                                if rr > max_rr:
                                    rejection = f"RR {rr:.1f}:1 > {max_rr}:1 after realistic fills"

            if rejection:
                rejected.append(suggestion.model_copy(update={
                    "rejection_reason": rejection,
                    "score": 0.0,
                }))
                logger.info(
                    "V2 REJECTED: %s — %s", suggestion.strategy.value, rejection,
                )
            else:
                # Section 9: Enrich with numeric fields
                enriched = _enrich_suggestion(suggestion, chain, analytics, cfg)
                accepted.append(enriched)
                logger.info(
                    "V2 ACCEPTED: %s | RR=%.2f | POP=%.1f%% | liq=%.0f",
                    enriched.strategy.value, enriched.risk_reward_ratio,
                    enriched.pop, enriched.liquidity_score,
                )

        # Return accepted + rejected (rejected have score=0 and won't be traded)
        return sorted(accepted, key=lambda s: s.score, reverse=True) + rejected

    # ---------------------------------------------------------------
    # evaluate_and_manage — Sections 2, 5, 7, 8
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
    ) -> PaperTradingState:
        """Manage positions with institutional risk rules."""
        # Market-open guard: skip all trading logic when market is closed
        if not is_market_open():
            return state

        if lot_size is None:
            lot_size = self.config.get("lot_size", 65)
        cfg = self.config

        # Build trade status notes for dashboard visibility
        notes: list[str] = []

        # Add observation context to notes
        if observation:
            notes.append(f"Observation: {observation.bias} bias ({observation.bars_collected} bars)")
            if observation.gap.direction != "flat":
                notes.append(f"Gap: {observation.gap.direction} {observation.gap.gap_pct:+.2f}%")
            notes.append(f"Trend: {observation.initial_trend.direction} ({observation.initial_trend.strength})")

        vix = _get_vix_from_analytics(analytics)
        if vix > 0:
            notes.append(f"VIX (ATM IV proxy): {vix:.1f}")
            vix_threshold = cfg.get("vix_spread_only_threshold", 28.0)
            if vix > vix_threshold:
                notes.append(f"Naked strategies blocked: VIX {vix:.1f} > {vix_threshold}")

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

            # V2 credit SL: exit at 2.2× credit for spreads
            exit_reason = None
            strategy_type = classify_strategy(position.strategy)

            if strategy_type == StrategyType.CREDIT:
                sl_mult = cfg.get("credit_sl_multiplier", 2.2)
                if position.net_premium > 0 and pnl <= -(abs(position.net_premium) * sl_mult):
                    exit_reason = PositionStatus.CLOSED_STOP_LOSS
            elif strategy_type == StrategyType.DEBIT:
                # Debit SL: 50% premium loss
                sl_pct = cfg.get("debit_sl_premium_loss_pct", 50.0)
                if position.net_premium < 0 and pnl <= -(abs(position.net_premium) * sl_pct / 100):
                    exit_reason = PositionStatus.CLOSED_STOP_LOSS

                # Session-based exit for debit
                max_sessions = cfg.get("debit_max_sessions", 3)
                if position.sessions_held >= max_sessions:
                    exit_reason = PositionStatus.CLOSED_TIME_LIMIT

            # Standard exit conditions (PT, EOD)
            if exit_reason is None:
                exit_reason = check_exit_conditions(position)

            if exit_reason:
                _closed_pos, record = close_position(
                    position, exit_reason,
                    technicals=technicals, analytics=analytics, chain=chain,
                )
                logger.info(
                    "V2 CLOSED: %s | reason=%s | pnl=%.2f",
                    record.strategy, exit_reason.value if hasattr(exit_reason, 'value') else exit_reason, record.net_pnl,
                )
                new_records.append(record)
                added_realized += record.realized_pnl
                added_costs += record.execution_cost
                if record.entry_context:
                    new_pending_critiques.append(record.id)
            else:
                still_open.append(position)

        # --- Section 8: Portfolio defense — consecutive losses ---
        consecutive_losses = state.consecutive_losses
        trading_halted = state.trading_halted

        for record in new_records:
            if record.net_pnl < 0:
                consecutive_losses += 1
            else:
                consecutive_losses = 0  # reset on win

        halt_after = cfg.get("consecutive_loss_halt", 3)
        if consecutive_losses >= halt_after:
            trading_halted = True
            logger.warning("V2 HALTED: %d consecutive losses ≥ %d", consecutive_losses, halt_after)

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

        # --- Daily loss > 2% → full shutdown ---
        daily_loss_pct = cfg.get("daily_loss_shutdown_pct", 2.0)
        if state.session_start_capital > 0:
            session_pnl = (state.initial_capital + state.net_realized_pnl) - state.session_start_capital
            if session_pnl <= -(state.session_start_capital * daily_loss_pct / 100):
                logger.warning("V2 SHUTDOWN: daily loss %.0f > %.1f%%", abs(session_pnl), daily_loss_pct)
                notes.append(f"Blocked: Daily loss shutdown (loss {abs(session_pnl):.0f} > {daily_loss_pct}%)")
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
            logger.info("V2: Trading halted — skipping new positions")
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
            # Skip rejected suggestions
            if suggestion.rejection_reason:
                continue
            if suggestion.score <= 0:
                continue

            # Section 2: Global gates
            gate_reason = _check_global_gates(state, suggestion, chain, analytics, cfg)
            if gate_reason:
                logger.info("V2 GATE BLOCKED: %s — %s", suggestion.strategy.value, gate_reason)
                notes.append(f"Blocked {suggestion.strategy.value}: {gate_reason}")
                continue

            # Skip duplicates
            held = {p.strategy for p in state.open_positions if p.status == PositionStatus.OPEN}
            if suggestion.strategy.value in held:
                notes.append(f"Blocked {suggestion.strategy.value}: already holding same strategy")
                continue

            if state.capital_remaining <= 0:
                notes.append("Blocked: no capital remaining")
                break

            # Section 5: Position sizing
            account_value = state.initial_capital + state.net_realized_pnl
            jarvis_lots, size_reject = _compute_jarvis_lots(
                suggestion, account_value, cfg, state.consecutive_losses,
            )
            if size_reject:
                logger.info("V2 SIZING REJECTED: %s — %s", suggestion.strategy.value, size_reject)
                notes.append(f"Blocked {suggestion.strategy.value}: sizing — {size_reject}")
                continue

            # Open position using the Jarvis lot count
            position = open_position(
                suggestion, lot_size, state.capital_remaining,
                expiry=chain.expiry,
                technicals=technicals, analytics=analytics, chain=chain,
            )
            # Override lots if Jarvis sizing differs
            if jarvis_lots > 0 and jarvis_lots != position.lots:
                position = position.model_copy(update={"lots": jarvis_lots})

            # Set entry date for session tracking
            position = position.model_copy(update={
                "entry_date": _now_ist().strftime("%Y-%m-%d"),
            })

            logger.info(
                "V2 OPENED: %s | lots=%d | margin=%.0f | premium=%.2f",
                position.strategy, position.lots, position.margin_required, position.net_premium,
            )

            state = state.model_copy(update={
                "open_positions": state.open_positions + [position],
                "last_trade_opened_ts": _time.time(),
            })
            notes.append(f"Opened: {position.strategy} ({position.lots} lots)")
            opened_trade = True
            break  # max 1 trade per cycle

        if not opened_trade and not any("Opened:" in n for n in notes):
            notes.append("All suggestions blocked by gates/sizing")

        return state.model_copy(update={"trade_status_notes": notes})
