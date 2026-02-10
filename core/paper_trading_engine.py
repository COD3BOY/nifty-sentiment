"""Pure business logic for paper trading simulation.

No Streamlit imports — all functions take data in and return data out.
Designed for future reuse in algorithmic trading.
"""

from __future__ import annotations

import logging
import math
import re
from datetime import datetime
from pathlib import Path

from core.config import load_config
from core.kite_margins import get_kite_charges, get_kite_margin
from core.options_models import (
    OptionChainData,
    StrategyName,
    TradeSuggestion,
)
from core.paper_trading_models import (
    PaperPosition,
    PaperTradingState,
    PositionLeg,
    PositionStatus,
    StrategyType,
    TradeRecord,
)

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


def compute_lots(
    strategy: str | StrategyName,
    available_capital: float,
    lot_size: int,
    suggestion: TradeSuggestion,
    expiry: str | None = None,
) -> tuple[int, float]:
    """Compute how many lots to trade based on capital and margin requirements.

    Two-pass approach:
      Pass 1: Use 1-lot margin (Kite or static) to estimate lot count.
      Pass 2: Call Kite with actual lots for true SPAN margin (hedge benefits).

    Returns (lots, total_margin).
    """
    cfg = _pt_cfg()
    max_lots = cfg.get("max_lots", 4)
    utilization = cfg.get("capital_utilization", 0.70)

    margin_per_lot: float = 0
    legs_dicts = [
        {"action": leg.action, "strike": leg.strike, "option_type": leg.option_type, "ltp": leg.ltp}
        for leg in suggestion.legs
    ]

    # Pass 1 — sizing estimate using 1-lot margin
    if expiry:
        kite_margin_1 = get_kite_margin(legs_dicts, expiry, lots=1, lot_size=lot_size)
        if kite_margin_1 is not None:
            margin_per_lot = kite_margin_1

    # Fallback to static estimates
    if margin_per_lot == 0:
        margin_estimates = cfg.get("margin_estimates", {})
        key = _strategy_margin_key(strategy)
        margin_per_lot = margin_estimates.get(key, 0)

    if margin_per_lot == 0:
        # Debit strategy — use premium paid per lot as "margin"
        per_lot_premium = sum(leg.ltp for leg in suggestion.legs if leg.action == "BUY") * lot_size
        margin_per_lot = per_lot_premium if per_lot_premium > 0 else 1  # safety

    usable_capital = available_capital * utilization
    lots = max(1, min(max_lots, math.floor(usable_capital / margin_per_lot)))

    # Pass 2 — accurate SPAN margin for actual lot count
    total_margin = margin_per_lot * lots  # default: linear estimate
    if expiry and lots > 0:
        kite_margin_n = get_kite_margin(legs_dicts, expiry, lots=lots, lot_size=lot_size)
        if kite_margin_n is not None:
            total_margin = kite_margin_n

    return lots, total_margin


def compute_execution_cost(
    num_legs: int,
    suggestion: TradeSuggestion,
    lot_size: int,
    lots: int,
    expiry: str | None = None,
) -> float:
    """Compute estimated execution cost for a round-trip trade.

    Tries Kite virtual contract note first; falls back to static formula
    that includes brokerage, STT, exchange txn charges, GST, and stamp duty.
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

    # GST on brokerage + exchange charges
    gst = (brokerage + exchange_txn) * gst_pct

    # Stamp duty on buy-side turnover (entry BUY legs + exit of SELL legs)
    total_buy_turnover = sell_turnover + buy_turnover  # mirrors sell side
    stamp_duty = total_buy_turnover * stamp_duty_buy_pct

    return brokerage + stt + exchange_txn + gst + stamp_duty


# ---------------------------------------------------------------------------
# Position lifecycle
# ---------------------------------------------------------------------------

def open_position(
    suggestion: TradeSuggestion,
    lot_size: int,
    available_capital: float | None = None,
    expiry: str | None = None,
) -> PaperPosition:
    """Create a new paper position from a trade suggestion."""
    strategy_type = classify_strategy(suggestion.strategy)

    # Position sizing
    if available_capital is not None:
        lots, total_margin = compute_lots(
            suggestion.strategy, available_capital, lot_size, suggestion, expiry=expiry,
        )
    else:
        lots = 1
        total_margin = 0.0

    net_premium, sl, pt = compute_risk_params(suggestion, lot_size, lots)

    # Execution cost
    num_legs = len(suggestion.legs)
    exec_cost = compute_execution_cost(num_legs, suggestion, lot_size, lots, expiry=expiry)

    legs = [
        PositionLeg(
            action=leg.action,
            instrument=leg.instrument,
            strike=leg.strike,
            option_type=leg.option_type,
            lots=lots,
            lot_size=lot_size,
            entry_ltp=leg.ltp,
            current_ltp=leg.ltp,
        )
        for leg in suggestion.legs
    ]

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
    pnl = position.total_unrealized_pnl

    if position.stop_loss_amount > 0 and pnl <= -position.stop_loss_amount:
        return PositionStatus.CLOSED_STOP_LOSS

    if position.profit_target_amount > 0 and pnl >= position.profit_target_amount:
        return PositionStatus.CLOSED_PROFIT_TARGET

    return None


def close_position(
    position: PaperPosition, reason: PositionStatus,
) -> tuple[PaperPosition, TradeRecord]:
    """Close a position and create an immutable trade record."""
    now = datetime.utcnow()
    closed_position = position.model_copy(
        update={"status": reason, "exit_time": now},
    )

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
        net_premium=position.net_premium,
        stop_loss_amount=position.stop_loss_amount,
        profit_target_amount=position.profit_target_amount,
    )

    return closed_position, record


# ---------------------------------------------------------------------------
# Main loop — called every refresh cycle
# ---------------------------------------------------------------------------

def evaluate_and_manage(
    state: PaperTradingState,
    suggestions: list[TradeSuggestion] | None,
    chain: OptionChainData | None,
    lot_size: int | None = None,
    refresh_ts: float = 0.0,
) -> PaperTradingState:
    """Main paper trading loop. Pure function: state in, state out.

    Called every 60s when Options Desk refreshes:
    Phase 1: Manage existing positions — update LTPs, check SL/PT exits
    Phase 2: Open new positions — fill remaining slots from suggestions

    refresh_ts gates the "open new position" path — prevents re-opening
    on the same refresh cycle after a close (manual, SL, or PT).
    """
    if lot_size is None:
        lot_size = _lot_size()

    cfg = _pt_cfg()
    min_score = cfg.get("min_score_to_trade", 30)
    max_positions = cfg.get("max_positions", 5)

    # --- Phase 1: Manage existing positions ---
    still_open: list[PaperPosition] = []
    new_records: list[TradeRecord] = []
    added_realized = 0.0
    added_costs = 0.0

    for position in state.open_positions:
        if position.status != PositionStatus.OPEN:
            continue

        # Update LTPs if chain available
        if chain:
            position = update_position_ltp(position, chain)

        # Check exit conditions
        exit_reason = check_exit_conditions(position)
        if exit_reason:
            _closed_pos, record = close_position(position, exit_reason)
            logger.info(
                "Paper trade closed: %s | reason=%s | pnl=%.2f | cost=%.2f | net=%.2f",
                record.strategy, exit_reason.value, record.realized_pnl,
                record.execution_cost, record.net_pnl,
            )
            new_records.append(record)
            added_realized += record.realized_pnl
            added_costs += record.execution_cost
        else:
            still_open.append(position)

    # Build intermediate state after Phase 1
    state = state.model_copy(
        update={
            "open_positions": still_open,
            "trade_log": state.trade_log + new_records,
            "total_realized_pnl": state.total_realized_pnl + added_realized,
            "total_execution_costs": state.total_execution_costs + added_costs,
            # If any positions were closed, gate re-open to next cycle
            "last_open_refresh_ts": refresh_ts if new_records else state.last_open_refresh_ts,
        },
    )

    # --- Phase 2: Open new positions ---
    if (
        state.is_auto_trading
        and suggestions
        and chain
        and refresh_ts > state.last_open_refresh_ts
        and len(state.open_positions) < max_positions
    ):
        # Track held strategy types to avoid duplicates
        held_strategies = {p.strategy for p in state.open_positions}
        new_positions: list[PaperPosition] = []

        for suggestion in suggestions:
            if len(state.open_positions) + len(new_positions) >= max_positions:
                break
            if suggestion.score < min_score:
                continue
            if suggestion.strategy.value in held_strategies:
                continue
            if state.capital_remaining <= 0:
                break

            position = open_position(
                suggestion, lot_size, state.capital_remaining,
                expiry=chain.expiry,
            )
            logger.info(
                "Paper trade opened: %s | bias=%s | score=%.1f | lots=%d | margin=%.0f | premium=%.2f | cost=%.2f",
                position.strategy, position.direction_bias,
                position.score, position.lots, position.margin_required,
                position.net_premium, position.execution_cost,
            )
            new_positions.append(position)
            held_strategies.add(position.strategy)
            # Update state so capital_remaining reflects newly locked margin
            state = state.model_copy(
                update={"open_positions": state.open_positions + [position]},
            )

        if new_positions:
            state = state.model_copy(update={"last_open_refresh_ts": refresh_ts})

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
