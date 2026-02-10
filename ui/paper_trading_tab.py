"""Streamlit UI for the Paper Trading simulation tab."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from core.config import load_config
from core.options_models import OptionChainData, TradeSuggestion
from core.paper_trading_engine import close_position, evaluate_and_manage
from core.paper_trading_models import (
    PaperPosition,
    PaperTradingState,
    PositionStatus,
)


def _get_state() -> PaperTradingState:
    """Get or initialize paper trading state from session_state."""
    if "paper_trading_state" not in st.session_state:
        cfg = load_config().get("paper_trading", {})
        st.session_state.paper_trading_state = PaperTradingState(
            initial_capital=cfg.get("initial_capital", 100_000),
            is_auto_trading=cfg.get("auto_execute", True),
        )
    return st.session_state.paper_trading_state


def _set_state(state: PaperTradingState) -> None:
    st.session_state.paper_trading_state = state


def _pnl_color(pnl: float) -> str:
    if pnl > 0:
        return "green"
    elif pnl < 0:
        return "red"
    return "gray"


def _fmt_pnl(pnl: float) -> str:
    sign = "+" if pnl > 0 else ""
    return f"{sign}\u20b9{pnl:,.0f}"


def _exit_reason_label(status: PositionStatus) -> str:
    return {
        PositionStatus.CLOSED_STOP_LOSS: "Stop Loss",
        PositionStatus.CLOSED_PROFIT_TARGET: "Profit Target",
        PositionStatus.CLOSED_MANUAL: "Manual Close",
        PositionStatus.CLOSED_EOD: "End of Day",
    }.get(status, status.value)


def render_paper_trading_tab(
    suggestions: list[TradeSuggestion] | None,
    chain: OptionChainData | None,
) -> None:
    """Main entry point for the Paper Trading tab."""
    state = _get_state()

    # --- Run engine logic (evaluate_and_manage) ---
    cfg = load_config().get("paper_trading", {})
    lot_size = cfg.get("lot_size", 25)
    refresh_ts = st.session_state.get("options_last_refresh", 0.0)

    new_state = evaluate_and_manage(state, suggestions, chain, lot_size, refresh_ts=refresh_ts)
    _set_state(new_state)
    state = new_state

    # --- Header row ---
    h1, h2 = st.columns([4, 1.5])
    with h1:
        st.title("Paper Trading")
        st.caption("Simulated trading using Options Desk suggestions with auto stop-loss and profit targets")
    with h2:
        auto_trading = st.toggle(
            "Auto-Trading",
            value=state.is_auto_trading,
            key="paper_auto_trading_toggle",
        )
        if auto_trading != state.is_auto_trading:
            state = state.model_copy(update={"is_auto_trading": auto_trading})
            _set_state(state)

    st.divider()

    # --- Capital metrics ---
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Initial Capital", f"\u20b9{state.initial_capital:,.0f}")
    with c2:
        st.metric(
            "Realized P&L",
            _fmt_pnl(state.total_realized_pnl),
            delta=_fmt_pnl(state.total_realized_pnl) if state.total_realized_pnl != 0 else None,
            delta_color="normal" if state.total_realized_pnl >= 0 else "inverse",
        )
    with c3:
        unr = state.unrealized_pnl
        st.metric(
            "Unrealized P&L",
            _fmt_pnl(unr),
            delta=_fmt_pnl(unr) if unr != 0 else None,
            delta_color="normal" if unr >= 0 else "inverse",
        )
    with c4:
        net_cap = state.capital_remaining + state.unrealized_pnl
        st.metric("Net Capital", f"\u20b9{net_cap:,.0f}")
    with c5:
        trades = len(state.trade_log)
        wr = state.win_rate
        st.metric("Trades / Win Rate", f"{trades} / {wr:.0f}%")

    st.divider()

    # --- Open Position Card ---
    if state.current_position and state.current_position.status == PositionStatus.OPEN:
        _render_open_position(state)
    else:
        if state.is_auto_trading:
            st.info("No open position. Auto-trading is ON — a new position will open on the next Options Desk refresh.")
        else:
            st.info("No open position. Enable **Auto-Trading** to automatically execute trade suggestions.")

    # --- Trade History ---
    st.divider()
    st.subheader("Trade History")
    if state.trade_log:
        _render_trade_history(state)
    else:
        st.caption("No trades yet. History will appear here after positions are closed.")


def _render_open_position(state: PaperTradingState) -> None:
    """Render the open position card."""
    pos = state.current_position
    st.subheader("Open Position")

    # Strategy info row
    info1, info2, info3, info4 = st.columns(4)
    with info1:
        st.markdown(f"**Strategy:** {pos.strategy}")
    with info2:
        st.markdown(f"**Bias:** {pos.direction_bias}")
    with info3:
        st.markdown(f"**Confidence:** {pos.confidence}")
    with info4:
        st.markdown(f"**Entry:** {pos.entry_time.strftime('%H:%M:%S')}")

    # Legs table
    leg_rows = []
    for leg in pos.legs:
        leg_rows.append({
            "Action": leg.action,
            "Instrument": leg.instrument,
            "Entry LTP": f"\u20b9{leg.entry_ltp:.1f}",
            "Current LTP": f"\u20b9{leg.current_ltp:.1f}",
            "Leg P&L": _fmt_pnl(leg.unrealized_pnl),
        })
    st.dataframe(pd.DataFrame(leg_rows), use_container_width=True, hide_index=True)

    # P&L and risk params
    pnl_col, sl_col, pt_col, btn_col = st.columns([2, 1, 1, 1])
    with pnl_col:
        pnl = pos.total_unrealized_pnl
        color = _pnl_color(pnl)
        st.markdown(
            f'<div style="font-size:1.8em; font-weight:700; color:{color};">'
            f'Unrealized P&L: {_fmt_pnl(pnl)}</div>',
            unsafe_allow_html=True,
        )
    with sl_col:
        st.metric("Stop Loss", f"\u20b9{pos.stop_loss_amount:,.0f}")
    with pt_col:
        st.metric("Profit Target", f"\u20b9{pos.profit_target_amount:,.0f}")
    with btn_col:
        if st.button("Close Position", key="manual_close_btn", type="secondary", use_container_width=True):
            closed_pos, record = close_position(pos, PositionStatus.CLOSED_MANUAL)
            new_state = state.model_copy(
                update={
                    "current_position": None,
                    "trade_log": state.trade_log + [record],
                    "total_realized_pnl": state.total_realized_pnl + record.realized_pnl,
                    "last_open_refresh_ts": st.session_state.get("options_last_refresh", 0.0),
                },
            )
            _set_state(new_state)
            st.rerun()


def _render_trade_history(state: PaperTradingState) -> None:
    """Render closed trade history as a table."""
    rows = []
    for i, t in enumerate(reversed(state.trade_log), 1):
        rows.append({
            "#": i,
            "Strategy": t.strategy,
            "Bias": t.direction_bias,
            "Entry": t.entry_time.strftime("%H:%M:%S"),
            "Exit": t.exit_time.strftime("%H:%M:%S"),
            "Exit Reason": _exit_reason_label(t.exit_reason),
            "P&L": _fmt_pnl(t.realized_pnl),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
