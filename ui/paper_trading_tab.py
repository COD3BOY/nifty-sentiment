"""Streamlit UI for the Paper Trading simulation tab."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from core.config import load_config
from core.options_models import OptionChainData, TradeSuggestion
from core.paper_trading_engine import close_position, evaluate_and_manage, load_state, save_state
from core.paper_trading_models import (
    PaperPosition,
    PaperTradingState,
    PositionStatus,
)


def _get_state() -> PaperTradingState:
    """Get or initialize paper trading state from session_state."""
    if "paper_trading_state" not in st.session_state:
        saved = load_state()
        if saved is not None:
            st.session_state.paper_trading_state = saved
        else:
            cfg = load_config().get("paper_trading", {})
            st.session_state.paper_trading_state = PaperTradingState(
                initial_capital=cfg.get("initial_capital", 100_000),
                is_auto_trading=cfg.get("auto_execute", True),
            )
    return st.session_state.paper_trading_state


def _set_state(state: PaperTradingState) -> None:
    st.session_state.paper_trading_state = state
    save_state(state)


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
        if st.button("Reset Session", key="paper_reset_btn", type="secondary"):
            cfg = load_config().get("paper_trading", {})
            fresh = PaperTradingState(
                initial_capital=cfg.get("initial_capital", 100_000),
                is_auto_trading=cfg.get("auto_execute", True),
            )
            _set_state(fresh)
            st.rerun()

    st.divider()

    # --- Capital metrics ---
    margin_in_use = state.margin_in_use
    cap_cols = st.columns(7)
    with cap_cols[0]:
        st.metric("Initial Capital", f"\u20b9{state.initial_capital:,.0f}")
    with cap_cols[1]:
        net_rpnl = state.net_realized_pnl
        st.metric(
            "Net Realized P&L",
            _fmt_pnl(net_rpnl),
            delta=_fmt_pnl(net_rpnl) if net_rpnl != 0 else None,
            delta_color="normal" if net_rpnl >= 0 else "inverse",
        )
    with cap_cols[2]:
        unr = state.unrealized_pnl
        st.metric(
            "Unrealized P&L",
            _fmt_pnl(unr),
            delta=_fmt_pnl(unr) if unr != 0 else None,
            delta_color="normal" if unr >= 0 else "inverse",
        )
    with cap_cols[3]:
        net_cap = state.capital_remaining + state.unrealized_pnl
        st.metric("Net Capital", f"\u20b9{net_cap:,.0f}")
    with cap_cols[4]:
        st.metric("Margin in Use", f"\u20b9{margin_in_use:,.0f}")
    with cap_cols[5]:
        st.metric("Total Costs", f"\u20b9{state.total_execution_costs:,.0f}")
    with cap_cols[6]:
        trades = len(state.trade_log)
        wr = state.win_rate
        st.metric("Trades / Win Rate", f"{trades} / {wr:.0f}%")

    st.divider()

    # --- Open Positions ---
    open_pos = [p for p in state.open_positions if p.status == PositionStatus.OPEN]
    if open_pos:
        hdr_col, btn_col = st.columns([5, 1])
        with hdr_col:
            st.subheader(f"Open Positions ({len(open_pos)})")
        with btn_col:
            if len(open_pos) > 1 and st.button("Close All", key="close_all_btn", type="secondary", use_container_width=True):
                new_records = []
                added_realized = 0.0
                added_costs = 0.0
                for p in open_pos:
                    _, record = close_position(p, PositionStatus.CLOSED_MANUAL)
                    new_records.append(record)
                    added_realized += record.realized_pnl
                    added_costs += record.execution_cost
                new_state = state.model_copy(
                    update={
                        "open_positions": [],
                        "trade_log": state.trade_log + new_records,
                        "total_realized_pnl": state.total_realized_pnl + added_realized,
                        "total_execution_costs": state.total_execution_costs + added_costs,
                        "last_open_refresh_ts": st.session_state.get("options_last_refresh", 0.0),
                    },
                )
                _set_state(new_state)
                st.rerun()
        for pos in open_pos:
            _render_open_position_card(state, pos)
    else:
        if state.is_auto_trading:
            st.info("No open positions. Auto-trading is ON — new positions will open on the next Options Desk refresh.")
        else:
            st.info("No open positions. Enable **Auto-Trading** to automatically execute trade suggestions.")

    # --- Trade History ---
    st.divider()
    st.subheader("Trade History")
    if state.trade_log:
        _render_trade_history(state)
    else:
        st.caption("No trades yet. History will appear here after positions are closed.")


def _render_open_position_card(state: PaperTradingState, pos: PaperPosition) -> None:
    """Render a single open position as an expander card."""
    pnl = pos.total_unrealized_pnl
    label = f"{pos.strategy} | {pos.direction_bias} | {pos.lots} lots | {_fmt_pnl(pnl)}"
    with st.expander(label, expanded=True):
        # Strategy info row
        info1, info2, info3, info4, info5 = st.columns(5)
        with info1:
            st.markdown(f"**Strategy:** {pos.strategy}")
        with info2:
            st.markdown(f"**Bias:** {pos.direction_bias}")
        with info3:
            st.markdown(f"**Lots:** {pos.lots}")
        with info4:
            st.markdown(f"**Confidence:** {pos.confidence}")
        with info5:
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

        # P&L row
        pnl_col, btn_col = st.columns([5, 1])
        with pnl_col:
            color = _pnl_color(pnl)
            st.markdown(
                f'<div style="font-size:1.8em; font-weight:700; color:{color};">'
                f'Unrealized P&L: {_fmt_pnl(pnl)}</div>',
                unsafe_allow_html=True,
            )
        with btn_col:
            if st.button("Close", key=f"close_{pos.id}", type="secondary", use_container_width=True):
                _, record = close_position(pos, PositionStatus.CLOSED_MANUAL)
                remaining = [p for p in state.open_positions if p.id != pos.id]
                new_state = state.model_copy(
                    update={
                        "open_positions": remaining,
                        "trade_log": state.trade_log + [record],
                        "total_realized_pnl": state.total_realized_pnl + record.realized_pnl,
                        "total_execution_costs": state.total_execution_costs + record.execution_cost,
                        "last_open_refresh_ts": st.session_state.get("options_last_refresh", 0.0),
                    },
                )
                _set_state(new_state)
                st.rerun()

        # Trade economics row
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.metric("Net Premium", f"\u20b9{pos.net_premium:,.0f}")
        with m2:
            st.metric("Margin Required", f"\u20b9{pos.margin_required:,.0f}")
        with m3:
            st.metric("Est. Cost", f"\u20b9{pos.execution_cost:,.0f}")
        with m4:
            st.metric("Stop Loss", f"\u20b9{pos.stop_loss_amount:,.0f}")
        with m5:
            total_cap = state.initial_capital + state.total_realized_pnl - state.total_execution_costs
            util_pct = (pos.margin_required / total_cap * 100) if total_cap > 0 else 0.0
            st.metric("Capital Utilization", f"{util_pct:.1f}%")


def _render_trade_history(state: PaperTradingState) -> None:
    """Render closed trade history as a table."""
    rows = []
    for i, t in enumerate(reversed(state.trade_log), 1):
        rows.append({
            "#": i,
            "Strategy": t.strategy,
            "Lots": t.lots,
            "Bias": t.direction_bias,
            "Entry": t.entry_time.strftime("%H:%M:%S"),
            "Exit": t.exit_time.strftime("%H:%M:%S"),
            "Exit Reason": _exit_reason_label(t.exit_reason),
            "Premium": _fmt_pnl(t.net_premium),
            "Margin": f"\u20b9{t.margin_required:,.0f}",
            "Gross P&L": _fmt_pnl(t.realized_pnl),
            "Cost": f"\u20b9{t.execution_cost:,.0f}",
            "Net P&L": _fmt_pnl(t.net_pnl),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
