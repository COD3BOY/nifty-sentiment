"""Pure business logic for paper trading simulation.

No Streamlit imports — all functions take data in and return data out.
Designed for future reuse in algorithmic trading.
"""

from __future__ import annotations

import logging
import math
import re
import time as _time
from collections import Counter
from datetime import datetime, time, timedelta, timezone
from pathlib import Path

from core.paper_trading_models import _now_ist

from core.config import load_config
from core.kite_margins import get_kite_charges, get_kite_margin
from core.margin_cache import get_cached_margin, update_cached_margin
from core.options_models import (
    OptionChainData,
    OptionsAnalytics,
    StrategyName,
    TechnicalIndicators,
    TradeSuggestion,
)
from core.paper_trading_models import (
    MarginSource,
    PaperPosition,
    PaperTradingState,
    PositionLeg,
    PositionStatus,
    StrategyType,
    TradeRecord,
)
from core.trade_context import MarketContextSnapshot, build_leg_contexts

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_CREDIT_STRATEGIES = {
    StrategyName.SHORT_STRADDLE,
    StrategyName.SHORT_STRANGLE,
    StrategyName.BULL_PUT_SPREAD,
    StrategyName.BEAR_CALL_SPREAD,
    StrategyName.IRON_CONDOR,
}


def _pt_cfg() -> dict:
    return load_config().get("paper_trading", {})


def _lot_size() -> int:
    return _pt_cfg().get("lot_size", 25)


# ---------------------------------------------------------------------------
# Strategy classification
# ---------------------------------------------------------------------------

def classify_strategy(name: str | StrategyName) -> StrategyType:
    """Classify a strategy as credit or debit."""
    if isinstance(name, str):
        try:
            name = StrategyName(name)
        except ValueError:
            # Fallback: check if it looks like a credit strategy name
            upper = name.upper().replace(" ", "_")
            for cs in _CREDIT_STRATEGIES:
                if upper in cs.name or cs.name in upper:
                    return StrategyType.CREDIT
            return StrategyType.DEBIT
    return StrategyType.CREDIT if name in _CREDIT_STRATEGIES else StrategyType.DEBIT


# ---------------------------------------------------------------------------
# Premium & risk computation
# ---------------------------------------------------------------------------

def compute_net_premium(
    suggestion: TradeSuggestion, lot_size: int, lots: int = 1,
) -> float:
    """Compute net premium from legs. SELL adds, BUY subtracts.

    Per-leg lots from the suggestion are ignored; we use the `lots` param
    (number of lots computed by position sizing) applied uniformly.
    """
    total = 0.0
    for leg in suggestion.legs:
        if leg.action == "SELL":
            total += leg.ltp * lots * lot_size
        else:
            total -= leg.ltp * lots * lot_size
    return total


def _parse_currency(text: str) -> float | None:
    """Extract numeric value from currency strings like '₹5,000' or '₹12,500'."""
    match = re.search(r"[\u20b9$]?\s*([\d,]+(?:\.\d+)?)", text)
    if match:
        try:
            return float(match.group(1).replace(",", ""))
        except ValueError:
            return None
    return None


def compute_risk_params(
    suggestion: TradeSuggestion, lot_size: int, lots: int = 1,
) -> tuple[float, float, float]:
    """Compute (net_premium, stop_loss_amount, profit_target_amount).

    Credit strategies: SL = parsed max_loss scaled by lots (or fallback multiplier), PT = credit_pct of premium
    Debit strategies: SL = debit_premium_pct of premium paid, PT = debit_pct of premium paid
    """
    cfg = _pt_cfg()
    net_premium = compute_net_premium(suggestion, lot_size, lots)
    strategy_type = classify_strategy(suggestion.strategy)

    if strategy_type == StrategyType.CREDIT:
        # Stop loss
        parsed_loss = _parse_currency(suggestion.max_loss)
        credit_fallback = cfg.get("stop_loss", {}).get("credit_fallback_multiplier", 1.0)
        if parsed_loss and parsed_loss > 0:
            # parsed max_loss is for 1 lot — scale by lots
            sl = parsed_loss * lots
        else:
            sl = abs(net_premium) * credit_fallback

        # Profit target
        credit_pt_pct = cfg.get("profit_target", {}).get("credit_pct", 0.30)
        pt = abs(net_premium) * credit_pt_pct
    else:
        # Debit: premium paid is negative net_premium
        premium_paid = abs(net_premium)
        debit_sl_pct = cfg.get("stop_loss", {}).get("debit_premium_pct", 0.40)
        sl = premium_paid * debit_sl_pct

        debit_pt_pct = cfg.get("profit_target", {}).get("debit_pct", 0.50)
        pt = premium_paid * debit_pt_pct

    return net_premium, sl, pt


# ---------------------------------------------------------------------------
# Position sizing & execution costs
# ---------------------------------------------------------------------------

def _strategy_margin_key(strategy: str | StrategyName) -> str:
    """Map a StrategyName to the margin_estimates config key."""
    if isinstance(strategy, StrategyName):
        strategy = strategy.value
    return strategy.lower().replace(" ", "_")


_NOTIONAL_MARGIN_PCT: dict[str, float] = {
    "short_straddle": 0.10,    # ~150K at NIFTY 23000
    "short_strangle": 0.09,    # ~135K
    "bull_put_spread": 0.04,   # ~60K
    "bear_call_spread": 0.04,
    "iron_condor": 0.03,       # ~45K
    # debit strategies: 0.0 → falls through to premium fallback
}


def _compute_notional_margin(
    strategy_key: str, suggestion: TradeSuggestion, lot_size: int,
) -> float | None:
    """Estimate margin as a percentage of notional value (avg strike * lot_size).

    Returns None for debit strategies (pct == 0) so the caller falls through.
    """
    pct = _NOTIONAL_MARGIN_PCT.get(strategy_key, 0.0)
    if pct == 0.0:
        return None
    strikes = [leg.strike for leg in suggestion.legs]
    if not strikes:
        return None
    avg_strike = sum(strikes) / len(strikes)
    return pct * avg_strike * lot_size


def compute_lots(
    strategy: str | StrategyName,
    available_capital: float,
    lot_size: int,
    suggestion: TradeSuggestion,
    expiry: str | None = None,
) -> tuple[int, float, str]:
    """Compute how many lots to trade based on capital and margin requirements.

    Two-pass approach:
      Pass 1: Use tiered margin fallback to estimate lot count.
              Kite live → Cached SPAN (24h) → Notional heuristic → Static config → Premium
      Pass 2: Call Kite with actual lots for true SPAN margin (hedge benefits).

    Returns (lots, total_margin, margin_source).
    """
    cfg = _pt_cfg()
    max_lots = cfg.get("max_lots", 4)
    utilization = cfg.get("capital_utilization", 0.70)

    margin_per_lot: float = 0
    margin_source: str = MarginSource.STATIC.value
    key = _strategy_margin_key(strategy)
    legs_dicts = [
        {"action": leg.action, "strike": leg.strike, "option_type": leg.option_type, "ltp": leg.ltp}
        for leg in suggestion.legs
    ]

    # --- Pass 1: Tiered fallback for 1-lot margin ---

    # Tier 1: Kite live SPAN
    if expiry:
        kite_margin_1 = get_kite_margin(legs_dicts, expiry, lots=1, lot_size=lot_size)
        if kite_margin_1 is not None:
            margin_per_lot = kite_margin_1
            margin_source = MarginSource.KITE.value
            # Persist to disk cache for future fallback
            update_cached_margin(key, margin_per_lot)

    # Tier 2: Cached SPAN from disk (24h TTL)
    if margin_per_lot == 0:
        cached = get_cached_margin(key)
        if cached is not None:
            margin_per_lot = cached
            margin_source = MarginSource.CACHED.value

    # Tier 3: Notional heuristic
    if margin_per_lot == 0:
        heuristic = _compute_notional_margin(key, suggestion, lot_size)
        if heuristic is not None:
            margin_per_lot = heuristic
            margin_source = MarginSource.HEURISTIC.value

    # Tier 4: Static config
    if margin_per_lot == 0:
        margin_estimates = cfg.get("margin_estimates", {})
        margin_per_lot = margin_estimates.get(key, 0)
        if margin_per_lot > 0:
            margin_source = MarginSource.STATIC.value

    # Tier 5: Net premium per lot (debit strategies)
    if margin_per_lot == 0:
        per_lot_premium = abs(
            sum(leg.ltp * (1 if leg.action == "BUY" else -1) for leg in suggestion.legs)
        ) * lot_size
        margin_per_lot = per_lot_premium if per_lot_premium > 0 else 1  # safety
        margin_source = MarginSource.PREMIUM.value

    usable_capital = available_capital * utilization
    lots = max(1, min(max_lots, math.floor(usable_capital / margin_per_lot)))

    # --- Pass 2: Accurate SPAN margin for actual lot count ---
    total_margin = margin_per_lot * lots  # default: linear estimate
    if expiry and lots > 0:
        kite_margin_n = get_kite_margin(legs_dicts, expiry, lots=lots, lot_size=lot_size)
        if kite_margin_n is not None:
            total_margin = kite_margin_n
            margin_source = MarginSource.KITE.value
            # Update cache with N-lot per-lot value
            update_cached_margin(key, total_margin / lots)

    return lots, total_margin, margin_source


def compute_execution_cost(
    num_legs: int,
    suggestion: TradeSuggestion,
    lot_size: int,
    lots: int,
    expiry: str | None = None,
) -> float:
    """Compute estimated execution cost for a round-trip trade.

    Tries Kite virtual contract note first; falls back to static formula
    that includes brokerage, STT, exchange txn charges, SEBI fee, GST, and stamp duty.
    """
    # Try Kite charges API first
    if expiry:
        legs_dicts = [
            {"action": leg.action, "strike": leg.strike, "option_type": leg.option_type, "ltp": leg.ltp}
            for leg in suggestion.legs
        ]
        kite_charges = get_kite_charges(legs_dicts, expiry, lots, lot_size)
        if kite_charges is not None:
            return kite_charges

    # Static fallback with all charge components
    cfg = _pt_cfg()
    cost_cfg = cfg.get("execution_cost", {})
    brokerage_per_order = cost_cfg.get("brokerage_per_order", 20)
    stt_sell_pct = cost_cfg.get("stt_sell_pct", 0.001)
    exchange_txn_pct = cost_cfg.get("exchange_txn_pct", 0.00035)
    sebi_turnover_pct = cost_cfg.get("sebi_turnover_pct", 0.000001)
    gst_pct = cost_cfg.get("gst_pct", 0.18)
    stamp_duty_buy_pct = cost_cfg.get("stamp_duty_buy_pct", 0.00003)

    # Brokerage: each leg has an entry and exit order
    brokerage = num_legs * 2 * brokerage_per_order

    # Turnover by side
    sell_turnover = sum(
        leg.ltp * lots * lot_size
        for leg in suggestion.legs
        if leg.action == "SELL"
    )
    buy_turnover = sum(
        leg.ltp * lots * lot_size
        for leg in suggestion.legs
        if leg.action == "BUY"
    )
    # Round-trip: entry sells + exit sells (BUY legs close via sell at exit)
    total_sell_turnover = sell_turnover + buy_turnover
    total_turnover = (sell_turnover + buy_turnover) * 2  # entry + exit

    # STT on all sell transactions (entry SELL legs + exit of BUY legs)
    stt = total_sell_turnover * stt_sell_pct

    # Exchange transaction charges on total round-trip turnover
    exchange_txn = total_turnover * exchange_txn_pct

    # SEBI turnover fee (₹10/crore on total turnover)
    sebi_fee = total_turnover * sebi_turnover_pct

    # GST on brokerage + exchange charges + SEBI fee
    gst = (brokerage + exchange_txn + sebi_fee) * gst_pct

    # Stamp duty on buy-side turnover (entry BUY legs + exit of SELL legs)
    total_buy_turnover = sell_turnover + buy_turnover  # mirrors sell side
    stamp_duty = total_buy_turnover * stamp_duty_buy_pct

    return brokerage + stt + exchange_txn + sebi_fee + gst + stamp_duty


# ---------------------------------------------------------------------------
# Position lifecycle
# ---------------------------------------------------------------------------

def open_position(
    suggestion: TradeSuggestion,
    lot_size: int,
    available_capital: float | None = None,
    expiry: str | None = None,
    technicals: TechnicalIndicators | None = None,
    analytics: OptionsAnalytics | None = None,
    chain: OptionChainData | None = None,
) -> PaperPosition:
    """Create a new paper position from a trade suggestion."""
    strategy_type = classify_strategy(suggestion.strategy)

    # Position sizing
    if available_capital is not None:
        lots, total_margin, margin_source = compute_lots(
            suggestion.strategy, available_capital, lot_size, suggestion, expiry=expiry,
        )
    else:
        lots = 1
        total_margin = 0.0
        margin_source = MarginSource.STATIC.value

    net_premium, sl, pt = compute_risk_params(suggestion, lot_size, lots)

    # Execution cost
    num_legs = len(suggestion.legs)
    exec_cost = compute_execution_cost(num_legs, suggestion, lot_size, lots, expiry=expiry)

    # Slippage model: BUY legs pay more, SELL legs receive less
    slippage = _pt_cfg().get("slippage_per_leg", 0.0)

    legs = [
        PositionLeg(
            action=leg.action,
            instrument=leg.instrument,
            strike=leg.strike,
            option_type=leg.option_type,
            lots=lots,
            lot_size=lot_size,
            entry_ltp=leg.ltp + (slippage if leg.action == "BUY" else -slippage),
            current_ltp=leg.ltp + (slippage if leg.action == "BUY" else -slippage),
        )
        for leg in suggestion.legs
    ]

    # Build entry context snapshot
    entry_context_dict = None
    if technicals or analytics:
        leg_ctxs = None
        if chain:
            leg_dicts = [
                {"strike": leg.strike, "option_type": leg.option_type, "ltp": leg.ltp}
                for leg in suggestion.legs
            ]
            leg_ctxs = build_leg_contexts(chain, leg_dicts)
        ctx = MarketContextSnapshot.from_snapshot_data(
            technicals=technicals,
            analytics=analytics,
            leg_contexts=leg_ctxs,
            score=suggestion.score,
            reasoning=list(suggestion.reasoning),
        )
        entry_context_dict = ctx.model_dump(mode="json")

    return PaperPosition(
        strategy=suggestion.strategy.value,
        strategy_type=strategy_type,
        direction_bias=suggestion.direction_bias,
        confidence=suggestion.confidence,
        score=suggestion.score,
        legs=legs,
        lots=lots,
        net_premium=net_premium,
        stop_loss_amount=sl,
        profit_target_amount=pt,
        execution_cost=exec_cost,
        margin_required=total_margin,
        margin_source=margin_source,
        entry_context=entry_context_dict,
    )


def _build_ltp_lookup(chain: OptionChainData) -> dict[tuple[float, str], float]:
    """Build O(1) lookup: (strike, option_type) -> LTP from option chain."""
    lookup: dict[tuple[float, str], float] = {}
    for strike_data in chain.strikes:
        if strike_data.ce_ltp > 0:
            lookup[(strike_data.strike_price, "CE")] = strike_data.ce_ltp
        if strike_data.pe_ltp > 0:
            lookup[(strike_data.strike_price, "PE")] = strike_data.pe_ltp
    return lookup


def update_position_ltp(
    position: PaperPosition, chain: OptionChainData,
) -> PaperPosition:
    """Update all legs' current_ltp from the latest option chain.

    Falls back to last known LTP if a strike is missing.
    """
    lookup = _build_ltp_lookup(chain)
    updated_legs = []
    for leg in position.legs:
        new_ltp = lookup.get((leg.strike, leg.option_type))
        updated_legs.append(
            leg.model_copy(update={"current_ltp": new_ltp if new_ltp is not None else leg.current_ltp})
        )
    return position.model_copy(update={"legs": updated_legs})


def check_exit_conditions(position: PaperPosition) -> PositionStatus | None:
    """Check if position should be auto-closed.

    Returns PositionStatus if exit triggered, None otherwise.
    """
    cfg = _pt_cfg()
    pnl = position.total_unrealized_pnl

    if position.stop_loss_amount > 0 and pnl <= -position.stop_loss_amount:
        return PositionStatus.CLOSED_STOP_LOSS

    if position.profit_target_amount > 0 and pnl >= position.profit_target_amount:
        return PositionStatus.CLOSED_PROFIT_TARGET

    # Time-based exit for debit positions
    debit_max_hold = cfg.get("debit_max_hold_minutes", 0)
    if debit_max_hold > 0 and position.strategy_type == StrategyType.DEBIT:
        now = _now_ist()
        hold_minutes = (now - position.entry_time).total_seconds() / 60
        if hold_minutes >= debit_max_hold:
            return PositionStatus.CLOSED_TIME_LIMIT

    # EOD auto-close
    eod_close_time = cfg.get("eod_close_time", "15:20")
    now = _now_ist()
    h, m = (int(x) for x in eod_close_time.split(":"))
    if now.time() >= time(h, m):
        return PositionStatus.CLOSED_EOD

    return None


def close_position(
    position: PaperPosition,
    reason: PositionStatus,
    technicals: TechnicalIndicators | None = None,
    analytics: OptionsAnalytics | None = None,
    chain: OptionChainData | None = None,
) -> tuple[PaperPosition, TradeRecord]:
    """Close a position and create an immutable trade record."""
    now = _now_ist()
    closed_position = position.model_copy(
        update={"status": reason, "exit_time": now},
    )

    # Build exit context snapshot
    exit_context_dict = None
    if technicals or analytics:
        leg_ctxs = None
        if chain:
            leg_dicts = [
                {"strike": leg.strike, "option_type": leg.option_type, "ltp": leg.current_ltp}
                for leg in position.legs
            ]
            leg_ctxs = build_leg_contexts(chain, leg_dicts)
        ctx = MarketContextSnapshot.from_snapshot_data(
            technicals=technicals, analytics=analytics, leg_contexts=leg_ctxs,
        )
        exit_context_dict = ctx.model_dump(mode="json")

    # Extract spot prices from contexts
    spot_at_entry = 0.0
    if position.entry_context and isinstance(position.entry_context, dict):
        spot_at_entry = position.entry_context.get("spot", 0.0)
    spot_at_exit = technicals.spot if technicals else 0.0

    gross_pnl = position.total_unrealized_pnl
    record = TradeRecord(
        id=position.id,
        strategy=position.strategy,
        strategy_type=position.strategy_type,
        direction_bias=position.direction_bias,
        confidence=position.confidence,
        score=position.score,
        legs_summary=[
            {
                "instrument": leg.instrument,
                "action": leg.action,
                "entry_ltp": leg.entry_ltp,
                "exit_ltp": leg.current_ltp,
                "pnl": leg.unrealized_pnl,
            }
            for leg in position.legs
        ],
        lots=position.lots,
        entry_time=position.entry_time,
        exit_time=now,
        exit_reason=reason,
        realized_pnl=gross_pnl,
        execution_cost=position.execution_cost,
        net_pnl=gross_pnl - position.execution_cost,
        margin_required=position.margin_required,
        margin_source=position.margin_source,
        net_premium=position.net_premium,
        stop_loss_amount=position.stop_loss_amount,
        profit_target_amount=position.profit_target_amount,
        entry_context=position.entry_context,
        exit_context=exit_context_dict,
        spot_at_entry=spot_at_entry,
        spot_at_exit=spot_at_exit,
        max_drawdown=abs(position.trough_pnl),
        max_favorable=position.peak_pnl,
    )

    return closed_position, record


# ---------------------------------------------------------------------------
# Expiry-day guards
# ---------------------------------------------------------------------------

_IST = timezone(timedelta(hours=5, minutes=30))
_MARKET_CLOSE = time(15, 30)


def _is_blocked_by_expiry_guards(
    suggestion: TradeSuggestion,
    chain: OptionChainData,
    lot_size: int,
) -> str | None:
    """Return block reason if suggestion should be skipped, else None."""
    cfg = _pt_cfg().get("expiry_guards", {})
    if not cfg:
        return None

    now = datetime.now(_IST)

    # --- Time cutoff on expiry day ---
    cutoff_minutes = cfg.get("no_new_after_minutes_before_close", 0)
    if cutoff_minutes and chain.expiry:
        try:
            expiry_date = datetime.strptime(chain.expiry, "%d-%b-%Y").date()
        except ValueError:
            expiry_date = None

        if expiry_date and now.date() == expiry_date:
            cutoff_time = (
                datetime.combine(now.date(), _MARKET_CLOSE) - timedelta(minutes=cutoff_minutes)
            ).time()
            if now.time() >= cutoff_time:
                return (
                    f"expiry-day time cutoff: {now.strftime('%H:%M')} IST >= "
                    f"{cutoff_time.strftime('%H:%M')} (close minus {cutoff_minutes}min)"
                )

    # --- Minimum leg premium ---
    min_leg_premium = cfg.get("min_leg_premium", 0)
    if min_leg_premium:
        for leg in suggestion.legs:
            if leg.ltp < min_leg_premium:
                return (
                    f"leg {leg.instrument} has LTP ₹{leg.ltp} < min ₹{min_leg_premium}"
                )

    # --- Minimum net premium per lot ---
    min_net = cfg.get("min_net_premium_per_lot", 0)
    if min_net:
        net_per_lot = sum(
            leg.ltp * (1 if leg.action == "SELL" else -1)
            for leg in suggestion.legs
        ) * lot_size
        if abs(net_per_lot) < min_net:
            return (
                f"|net premium/lot| ₹{abs(net_per_lot):.1f} < min ₹{min_net}"
            )

    return None


# ---------------------------------------------------------------------------
# Main loop — called every refresh cycle
# ---------------------------------------------------------------------------

def evaluate_and_manage(
    state: PaperTradingState,
    suggestions: list[TradeSuggestion] | None,
    chain: OptionChainData | None,
    technicals: TechnicalIndicators | None = None,
    analytics: OptionsAnalytics | None = None,
    lot_size: int | None = None,
    refresh_ts: float = 0.0,
) -> PaperTradingState:
    """Main paper trading loop. Pure function: state in, state out.

    Called every 60s when Options Desk refreshes:
    Phase 1: Manage existing positions — update LTPs, check SL/PT exits
    Phase 2: Open new positions — fill remaining slots from suggestions
    """
    if lot_size is None:
        lot_size = _lot_size()

    cfg = _pt_cfg()
    min_score = cfg.get("min_score_to_trade", 30)

    # --- Phase 1: Manage existing positions ---
    still_open: list[PaperPosition] = []
    new_records: list[TradeRecord] = []
    added_realized = 0.0
    added_costs = 0.0
    new_pending_critiques: list[str] = []

    for position in state.open_positions:
        if position.status != PositionStatus.OPEN:
            continue

        # Update LTPs if chain available
        if chain:
            position = update_position_ltp(position, chain)

        # Track peak/trough PnL
        pnl = position.total_unrealized_pnl
        position = position.model_copy(update={
            "peak_pnl": max(position.peak_pnl, pnl),
            "trough_pnl": min(position.trough_pnl, pnl),
        })

        # Check exit conditions
        exit_reason = check_exit_conditions(position)
        if exit_reason:
            _closed_pos, record = close_position(
                position, exit_reason,
                technicals=technicals, analytics=analytics, chain=chain,
            )
            logger.info(
                "Paper trade closed: %s | reason=%s | pnl=%.2f | cost=%.2f | net=%.2f",
                record.strategy, exit_reason.value, record.realized_pnl,
                record.execution_cost, record.net_pnl,
            )
            new_records.append(record)
            added_realized += record.realized_pnl
            added_costs += record.execution_cost
            # Queue for critique if entry context was captured
            if record.entry_context:
                new_pending_critiques.append(record.id)
        else:
            still_open.append(position)

    # Build intermediate state after Phase 1
    state = state.model_copy(
        update={
            "open_positions": still_open,
            "trade_log": state.trade_log + new_records,
            "total_realized_pnl": state.total_realized_pnl + added_realized,
            "total_execution_costs": state.total_execution_costs + added_costs,
            "last_open_refresh_ts": state.last_open_refresh_ts,
            "pending_critiques": state.pending_critiques + new_pending_critiques,
        },
    )

    # --- Initialize session_start_capital on first cycle of the day ---
    if state.session_start_capital == 0.0:
        state = state.model_copy(update={
            "session_start_capital": state.initial_capital + state.net_realized_pnl,
        })

    # --- Data staleness guard ---
    max_staleness = cfg.get("max_staleness_minutes", 20)
    if technicals and technicals.data_staleness_minutes > max_staleness:
        logger.warning(
            "Data stale: %.1f min since last candle (limit: %d min). Blocking new trades.",
            technicals.data_staleness_minutes, max_staleness,
        )
        return state

    # --- Phase 2: Open new positions ---
    # Daily loss circuit breaker
    daily_loss_limit_pct = cfg.get("daily_loss_limit_pct", 3.0)
    session_pnl = (state.initial_capital + state.net_realized_pnl) - state.session_start_capital
    daily_loss_limit = state.session_start_capital * (daily_loss_limit_pct / 100)
    if session_pnl <= -daily_loss_limit:
        logger.warning(
            "CIRCUIT BREAKER: Daily loss %.2f exceeds %.1f%% limit (%.2f). Blocking new trades.",
            abs(session_pnl), daily_loss_limit_pct, daily_loss_limit,
        )
        return state

    if (
        state.is_auto_trading
        and suggestions
        and chain
    ):
        # 60-second cooldown between successive trade opens
        cooldown = cfg.get("trade_cooldown_seconds", 60)
        now_ts = _time.time()
        if state.last_trade_opened_ts > 0 and (now_ts - state.last_trade_opened_ts) < cooldown:
            logger.info(
                "Trade cooldown active — %.0fs remaining",
                cooldown - (now_ts - state.last_trade_opened_ts),
            )
            return state

        # Track held strategy types to avoid duplicates
        held_strategies = {p.strategy for p in state.open_positions}

        # Portfolio concentration limits
        max_open = cfg.get("max_open_positions", 3)
        max_per_dir = cfg.get("max_positions_per_direction", 2)
        dir_counts = Counter(p.direction_bias for p in state.open_positions if p.status == PositionStatus.OPEN)

        for suggestion in suggestions:
            # (e) Portfolio concentration limits
            open_count = len([p for p in state.open_positions if p.status == PositionStatus.OPEN])
            if open_count >= max_open:
                logger.info("Max open positions (%d) reached — blocking new trades", max_open)
                break
            if dir_counts.get(suggestion.direction_bias, 0) >= max_per_dir:
                logger.info(
                    "Max %s positions (%d) reached — skipping %s",
                    suggestion.direction_bias, max_per_dir, suggestion.strategy.value,
                )
                continue

            if suggestion.score < min_score:
                logger.debug("Skipping %s — score %.1f < min %d", suggestion.strategy.value, suggestion.score, min_score)
                continue
            # (d) High-confidence-only gate
            if suggestion.confidence != "High":
                logger.info("Skipping %s — confidence %s (require High)", suggestion.strategy.value, suggestion.confidence)
                continue
            block_reason = _is_blocked_by_expiry_guards(suggestion, chain, lot_size)
            if block_reason:
                logger.info("Suggestion blocked: %s — %s", suggestion.strategy.value, block_reason)
                continue
            if suggestion.strategy.value in held_strategies:
                continue
            if state.capital_remaining <= 0:
                break

            position = open_position(
                suggestion, lot_size, state.capital_remaining,
                expiry=chain.expiry,
                technicals=technicals, analytics=analytics, chain=chain,
            )
            logger.info(
                "Paper trade opened: %s | bias=%s | conf=%s | score=%.1f | lots=%d | margin=%.0f (%s) | premium=%.2f | cost=%.2f",
                position.strategy, position.direction_bias, position.confidence,
                position.score, position.lots, position.margin_required,
                position.margin_source, position.net_premium, position.execution_cost,
            )
            held_strategies.add(position.strategy)
            # Update state: add position + record trade timestamp
            state = state.model_copy(
                update={
                    "open_positions": state.open_positions + [position],
                    "last_trade_opened_ts": _time.time(),
                },
            )
            # (c) Max 1 trade per cycle
            break

    return state


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

_STATE_FILE = Path(__file__).resolve().parent.parent / "data" / "paper_trading_state.json"


def save_state(state: PaperTradingState) -> None:
    """Persist paper trading state to disk (atomic write)."""
    _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = _STATE_FILE.with_suffix(".tmp")
    tmp.write_text(state.model_dump_json(indent=2))
    tmp.replace(_STATE_FILE)  # atomic on POSIX


def load_state() -> PaperTradingState | None:
    """Load paper trading state from disk. Returns None if no saved state."""
    if _STATE_FILE.exists():
        return PaperTradingState.model_validate_json(_STATE_FILE.read_text())
    return None
