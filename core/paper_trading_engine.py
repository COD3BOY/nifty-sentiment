"""Pure business logic for paper trading simulation.

No Streamlit imports — all functions take data in and return data out.
Designed for future reuse in algorithmic trading.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime

from core.config import load_config
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

def compute_net_premium(suggestion: TradeSuggestion, lot_size: int) -> float:
    """Compute net premium from legs. SELL adds, BUY subtracts."""
    total = 0.0
    for leg in suggestion.legs:
        if leg.action == "SELL":
            total += leg.ltp * leg.lots * lot_size
        else:
            total -= leg.ltp * leg.lots * lot_size
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
    suggestion: TradeSuggestion, lot_size: int,
) -> tuple[float, float, float]:
    """Compute (net_premium, stop_loss_amount, profit_target_amount).

    Credit strategies: SL = parsed max_loss (or 1.5x premium fallback), PT = 50% of premium
    Debit strategies: SL = 50% of premium paid, PT = 100% of premium paid
    """
    cfg = _pt_cfg()
    net_premium = compute_net_premium(suggestion, lot_size)
    strategy_type = classify_strategy(suggestion.strategy)

    if strategy_type == StrategyType.CREDIT:
        # Stop loss
        parsed_loss = _parse_currency(suggestion.max_loss)
        credit_fallback = cfg.get("stop_loss", {}).get("credit_fallback_multiplier", 1.5)
        if parsed_loss and parsed_loss > 0:
            sl = parsed_loss
        else:
            sl = abs(net_premium) * credit_fallback

        # Profit target: take 50% of premium received
        credit_pt_pct = cfg.get("profit_target", {}).get("credit_pct", 0.50)
        pt = abs(net_premium) * credit_pt_pct
    else:
        # Debit: premium paid is negative net_premium
        premium_paid = abs(net_premium)
        debit_sl_pct = cfg.get("stop_loss", {}).get("debit_premium_pct", 0.50)
        sl = premium_paid * debit_sl_pct

        debit_pt_pct = cfg.get("profit_target", {}).get("debit_pct", 1.00)
        pt = premium_paid * debit_pt_pct

    return net_premium, sl, pt


# ---------------------------------------------------------------------------
# Position lifecycle
# ---------------------------------------------------------------------------

def open_position(suggestion: TradeSuggestion, lot_size: int) -> PaperPosition:
    """Create a new paper position from a trade suggestion."""
    strategy_type = classify_strategy(suggestion.strategy)
    net_premium, sl, pt = compute_risk_params(suggestion, lot_size)

    legs = [
        PositionLeg(
            action=leg.action,
            instrument=leg.instrument,
            strike=leg.strike,
            option_type=leg.option_type,
            lots=leg.lots,
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
        net_premium=net_premium,
        stop_loss_amount=sl,
        profit_target_amount=pt,
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
        entry_time=position.entry_time,
        exit_time=now,
        exit_reason=reason,
        realized_pnl=position.total_unrealized_pnl,
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
    1. If open position -> update LTPs -> check exits
    2. If no position + auto_trading + suggestions exist -> open best

    refresh_ts gates the "open new position" path — prevents re-opening
    on the same refresh cycle after a close (manual, SL, or PT).
    """
    if lot_size is None:
        lot_size = _lot_size()

    cfg = _pt_cfg()
    min_score = cfg.get("min_score_to_trade", 30)

    # --- Manage existing position ---
    if state.current_position and state.current_position.status == PositionStatus.OPEN:
        position = state.current_position

        # Update LTPs if chain available
        if chain:
            position = update_position_ltp(position, chain)

        # Check exit conditions
        exit_reason = check_exit_conditions(position)
        if exit_reason:
            closed_pos, record = close_position(position, exit_reason)
            logger.info(
                "Paper trade closed: %s | reason=%s | pnl=%.2f",
                record.strategy, exit_reason.value, record.realized_pnl,
            )
            return state.model_copy(
                update={
                    "current_position": None,
                    "trade_log": state.trade_log + [record],
                    "total_realized_pnl": state.total_realized_pnl + record.realized_pnl,
                    "last_open_refresh_ts": refresh_ts,
                },
            )

        # Position still open, just update LTPs
        return state.model_copy(update={"current_position": position})

    # --- Open new position (only on fresh data) ---
    if (
        state.is_auto_trading
        and state.current_position is None
        and suggestions
        and chain
        and refresh_ts > state.last_open_refresh_ts
    ):
        # Pick the best suggestion that meets the minimum score
        for suggestion in suggestions:
            if suggestion.score >= min_score:
                position = open_position(suggestion, lot_size)
                logger.info(
                    "Paper trade opened: %s | bias=%s | score=%.1f | premium=%.2f",
                    position.strategy, position.direction_bias,
                    position.score, position.net_premium,
                )
                return state.model_copy(
                    update={
                        "current_position": position,
                        "last_open_refresh_ts": refresh_ts,
                    },
                )

    return state
