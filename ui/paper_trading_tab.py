"""Streamlit UI for the Paper Trading simulation tab."""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta

import pandas as pd
import streamlit as st

from core.config import load_config
from core.options_models import OptionChainData, OptionsAnalytics, TechnicalIndicators, TradeSuggestion
from core.paper_trading_engine import close_position, evaluate_and_manage, load_state, save_state
from core.paper_trading_models import (
    PaperPosition,
    PaperTradingState,
    PositionStatus,
)

logger = logging.getLogger(__name__)

_IST = timezone(timedelta(hours=5, minutes=30))


def _to_ist(dt: datetime) -> datetime:
    """Convert a datetime to IST for display.

    Handles both naive-UTC (old data) and timezone-aware datetimes.
    """
    if dt.tzinfo is None:
        # Naive datetime — assume UTC (legacy data)
        return dt.replace(tzinfo=timezone.utc).astimezone(_IST)
    return dt.astimezone(_IST)


def _fmt_time(dt: datetime) -> str:
    """Format a datetime as IST HH:MM:SS."""
    return _to_ist(dt).strftime("%H:%M:%S")


def _state_key(algo_name: str) -> str:
    return f"paper_trading_state_{algo_name}"


def _get_state(algo_name: str = "sentinel") -> PaperTradingState:
    """Get or initialize paper trading state from session_state."""
    key = _state_key(algo_name)
    if key not in st.session_state:
        saved = load_state(algo_name=algo_name)
        if saved is not None:
            st.session_state[key] = saved
        else:
            cfg = load_config().get("paper_trading", {})
            st.session_state[key] = PaperTradingState(
                initial_capital=cfg.get("initial_capital", 100_000),
                is_auto_trading=cfg.get("auto_execute", True),
            )
    return st.session_state[key]


def _set_state(state: PaperTradingState, algo_name: str = "sentinel") -> None:
    st.session_state[_state_key(algo_name)] = state
    save_state(state, algo_name=algo_name)


def _format_context_indicators(ctx: dict) -> list[str]:
    """Extract key indicator strings from an entry/exit context dict."""
    indicators: list[str] = []
    if ctx.get("rsi"):
        indicators.append(f"RSI {ctx['rsi']:.1f}")
    if ctx.get("atm_iv"):
        indicators.append(f"IV {ctx['atm_iv']:.1f}%")
    if ctx.get("pcr"):
        indicators.append(f"PCR {ctx['pcr']:.2f}")
    if ctx.get("spot") and ctx.get("vwap"):
        rel = "above" if ctx["spot"] > ctx["vwap"] else "below"
        indicators.append(f"Spot {rel} VWAP")
    if ctx.get("supertrend_direction"):
        st_dir = "Bullish" if ctx["supertrend_direction"] == 1 else "Bearish"
        indicators.append(f"Supertrend {st_dir}")
    return indicators


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
    technicals: TechnicalIndicators | None = None,
    analytics: OptionsAnalytics | None = None,
    algo_name: str = "sentinel",
    algo_display_name: str = "Paper Trading",
    evaluate_fn=None,
) -> None:
    """Main entry point for the Paper Trading tab.

    Parameters
    ----------
    algo_name : unique algorithm identifier (used for state persistence and widget keys)
    algo_display_name : UI label shown in header
    evaluate_fn : optional override for evaluate_and_manage (algorithm's own implementation)
    """
    state = _get_state(algo_name)

    # --- Header row (render before engine so reset takes effect first) ---
    h1, h2 = st.columns([4, 1.5])
    with h1:
        st.title(algo_display_name)
        st.caption("Simulated trading using Options Desk suggestions with auto stop-loss and profit targets")
    with h2:
        auto_trading = st.toggle(
            "Auto-Trading",
            value=state.is_auto_trading,
            key=f"paper_auto_trading_toggle_{algo_name}",
        )
        if auto_trading != state.is_auto_trading:
            state = state.model_copy(update={"is_auto_trading": auto_trading})
            _set_state(state, algo_name)
        if st.button("Reset Session", key=f"paper_reset_btn_{algo_name}", type="secondary"):
            cfg = load_config().get("paper_trading", {})
            fresh = PaperTradingState(
                initial_capital=cfg.get("initial_capital", 100_000),
                is_auto_trading=cfg.get("auto_execute", True),
            )
            _set_state(fresh, algo_name)
            _clear_critiques_db()
            st.rerun()

    # --- Run engine logic (evaluate_and_manage) ---
    cfg = load_config().get("paper_trading", {})
    lot_size = cfg.get("lot_size", 25)
    refresh_ts = st.session_state.get("options_last_refresh", 0.0)

    _eval_fn = evaluate_fn or evaluate_and_manage
    new_state = _eval_fn(
        state, suggestions, chain,
        technicals=technicals, analytics=analytics,
        lot_size=lot_size, refresh_ts=refresh_ts,
    )
    _set_state(new_state, algo_name)
    state = new_state

    # --- Process one pending critique per cycle (non-blocking) ---
    state = _process_pending_critique(state, algo_name)

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
            if len(open_pos) > 1 and st.button("Close All", key=f"close_all_btn_{algo_name}", type="secondary", use_container_width=True):
                new_records = []
                added_realized = 0.0
                added_costs = 0.0
                pending_ids = []
                for p in open_pos:
                    _, record = close_position(
                        p, PositionStatus.CLOSED_MANUAL,
                        technicals=technicals, analytics=analytics, chain=chain,
                    )
                    new_records.append(record)
                    added_realized += record.realized_pnl
                    added_costs += record.execution_cost
                    if record.entry_context:
                        pending_ids.append(record.id)
                new_state = state.model_copy(
                    update={
                        "open_positions": [],
                        "trade_log": state.trade_log + new_records,
                        "total_realized_pnl": state.total_realized_pnl + added_realized,
                        "total_execution_costs": state.total_execution_costs + added_costs,
                        "last_open_refresh_ts": st.session_state.get("options_last_refresh", 0.0),
                        "pending_critiques": state.pending_critiques + pending_ids,
                    },
                )
                _set_state(new_state, algo_name)
                st.rerun()
        for pos in open_pos:
            _render_open_position_card(state, pos, technicals=technicals, analytics=analytics, chain=chain, algo_name=algo_name)
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

        # --- EOD Report Downloads ---
        st.divider()
        st.subheader("EOD Report")
        _render_eod_downloads(state)
    else:
        st.caption("No trades yet. History will appear here after positions are closed.")


def _render_eod_downloads(state: PaperTradingState) -> None:
    """Render download buttons for the end-of-day trade report."""
    from analyzers.eod_report import generate_eod_report, render_json, render_markdown

    report = generate_eod_report(state)
    date_str = report["date"]

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="Download Markdown Report",
            data=render_markdown(report),
            file_name=f"nifty_trades_{date_str}.md",
            mime="text/markdown",
        )
    with col2:
        st.download_button(
            label="Download JSON Report",
            data=render_json(report),
            file_name=f"nifty_trades_{date_str}.json",
            mime="application/json",
        )


_MARGIN_BADGE = {
    "kite": ("Kite SPAN", "#22c55e"),      # green
    "cached": ("Cached SPAN", "#3b82f6"),   # blue
    "heuristic": ("Heuristic", "#f59e0b"),  # amber
    "static": ("Static", "#94a3b8"),        # gray
    "premium": ("Premium", "#94a3b8"),      # gray
}


def _render_margin_badge(source: str, prefix: str = "") -> None:
    """Render a color-coded badge indicating the margin source."""
    label, color = _MARGIN_BADGE.get(source, ("Static", "#94a3b8"))
    st.caption(
        f'{prefix}<span style="background:{color};color:white;padding:1px 6px;'
        f'border-radius:3px;font-size:0.75em">{label}</span>',
        unsafe_allow_html=True,
    )


def _render_open_position_card(
    state: PaperTradingState,
    pos: PaperPosition,
    technicals: TechnicalIndicators | None = None,
    analytics: OptionsAnalytics | None = None,
    chain: OptionChainData | None = None,
    algo_name: str = "sentinel",
) -> None:
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
            st.markdown(f"**Entry:** {_fmt_time(pos.entry_time)}")

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

        # Entry reasoning
        if pos.entry_context and isinstance(pos.entry_context, dict):
            reasons = pos.entry_context.get("strategy_reasoning", [])
            if reasons:
                st.markdown("**Why this trade:**")
                st.markdown("  \n".join(f"- {r}" for r in reasons))
            indicators = _format_context_indicators(pos.entry_context)
            if indicators:
                st.caption("Entry: " + " | ".join(indicators))

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
            if st.button("Close", key=f"close_{algo_name}_{pos.id}", type="secondary", use_container_width=True):
                _, record = close_position(
                    pos, PositionStatus.CLOSED_MANUAL,
                    technicals=technicals, analytics=analytics, chain=chain,
                )
                remaining = [p for p in state.open_positions if p.id != pos.id]
                pending_ids = [record.id] if record.entry_context else []
                new_state = state.model_copy(
                    update={
                        "open_positions": remaining,
                        "trade_log": state.trade_log + [record],
                        "total_realized_pnl": state.total_realized_pnl + record.realized_pnl,
                        "total_execution_costs": state.total_execution_costs + record.execution_cost,
                        "last_open_refresh_ts": st.session_state.get("options_last_refresh", 0.0),
                        "pending_critiques": state.pending_critiques + pending_ids,
                    },
                )
                _set_state(new_state, algo_name)
                st.rerun()

        # Trade economics row
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.metric("Net Premium", f"\u20b9{pos.net_premium:,.0f}")
        with m2:
            st.metric("Margin Required", f"\u20b9{pos.margin_required:,.0f}")
            _render_margin_badge(pos.margin_source)
        with m3:
            st.metric("Est. Cost", f"\u20b9{pos.execution_cost:,.0f}")
        with m4:
            st.metric("Stop Loss", f"\u20b9{pos.stop_loss_amount:,.0f}")
        with m5:
            total_cap = state.initial_capital + state.total_realized_pnl - state.total_execution_costs
            util_pct = (pos.margin_required / total_cap * 100) if total_cap > 0 else 0.0
            st.metric("Capital Utilization", f"{util_pct:.1f}%")


def _render_trade_history(state: PaperTradingState) -> None:
    """Render closed trade history as expandable cards with inline critiques."""
    # Load all critiques from DB, keyed by trade_id
    critique_map: dict[str, dict] = {}
    try:
        from core.database import SentimentDatabase
        db = SentimentDatabase()
        for c in db.get_all_critiques(limit=100):
            critique_map[c["trade_id"]] = c
    except Exception:
        pass

    for i, t in enumerate(reversed(state.trade_log), 1):
        critique = critique_map.get(t.id)
        grade_tag = ""
        if critique:
            grade = critique["overall_grade"]
            color = _GRADE_COLORS.get(grade, "#94a3b8")
            grade_tag = f" | {grade.upper()}"

        pnl_sign = "+" if t.net_pnl > 0 else ""
        label = (
            f"#{i} {t.strategy} | {t.direction_bias} | {t.lots}L | "
            f"{_exit_reason_label(t.exit_reason)} | "
            f"{pnl_sign}\u20b9{t.net_pnl:,.0f}{grade_tag}"
        )
        with st.expander(label, expanded=False):
            # Trade summary metrics
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            with c1:
                st.metric("Entry", _fmt_time(t.entry_time))
            with c2:
                st.metric("Exit", _fmt_time(t.exit_time))
            with c3:
                st.metric("Premium", _fmt_pnl(t.net_premium))
            with c4:
                st.metric("Gross P&L", _fmt_pnl(t.realized_pnl))
            with c5:
                st.metric("Cost", f"\u20b9{t.execution_cost:,.0f}")
            with c6:
                st.metric("Net P&L", _fmt_pnl(t.net_pnl))

            # Drawdown / favorable if available
            if t.max_favorable > 0 or t.max_drawdown > 0:
                d1, d2, d3, d4 = st.columns(4)
                with d1:
                    st.metric("Max Favorable", _fmt_pnl(t.max_favorable))
                with d2:
                    st.metric("Max Drawdown", f"\u20b9{t.max_drawdown:,.0f}")
                with d3:
                    if t.spot_at_entry > 0:
                        st.metric("Spot at Entry", f"{t.spot_at_entry:,.1f}")
                with d4:
                    if t.spot_at_exit > 0:
                        move = t.spot_at_exit - t.spot_at_entry
                        st.metric("Spot at Exit", f"{t.spot_at_exit:,.1f}", delta=f"{move:+.1f}")

            # Margin source badge
            _render_margin_badge(t.margin_source, prefix="Margin: ")

            # Legs detail
            leg_rows = []
            for leg in t.legs_summary:
                leg_rows.append({
                    "Action": leg.get("action", ""),
                    "Instrument": leg.get("instrument", ""),
                    "Entry": f"\u20b9{leg.get('entry_ltp', 0):.1f}",
                    "Exit": f"\u20b9{leg.get('exit_ltp', 0):.1f}",
                    "P&L": _fmt_pnl(leg.get("pnl", 0)),
                })
            if leg_rows:
                st.dataframe(pd.DataFrame(leg_rows), use_container_width=True, hide_index=True)

            # Entry & exit context
            if t.entry_context and isinstance(t.entry_context, dict):
                reasons = t.entry_context.get("strategy_reasoning", [])
                if reasons:
                    st.markdown("**Entry reasoning:** " + " / ".join(reasons))
                entry_indicators = _format_context_indicators(t.entry_context)
                if entry_indicators:
                    st.caption("Entry: " + " | ".join(entry_indicators))

            if t.exit_context and isinstance(t.exit_context, dict):
                exit_indicators = _format_context_indicators(t.exit_context)
                if exit_indicators:
                    st.caption("Exit: " + " | ".join(exit_indicators))

            # Inline critique
            if critique:
                st.divider()
                grade = critique["overall_grade"]
                color = _GRADE_COLORS.get(grade, "#94a3b8")
                st.markdown(
                    f'**Critique:** <span style="background:{color}; color:white; padding:2px 10px; '
                    f'border-radius:4px; font-weight:700;">{grade.upper()}</span>',
                    unsafe_allow_html=True,
                )
                summary = critique.get("summary", "")
                if summary:
                    st.markdown(summary)

                # Signal analysis in columns
                signals = critique.get("entry_signal_analysis", {})
                has_signals = any(signals.get(k) for k in ("signals_that_worked", "signals_that_failed", "signals_missed"))
                if has_signals:
                    sc1, sc2, sc3 = st.columns(3)
                    with sc1:
                        worked = signals.get("signals_that_worked", [])
                        if worked:
                            st.markdown("**Worked:**")
                            for s in worked:
                                st.markdown(f"- {s}")
                    with sc2:
                        failed = signals.get("signals_that_failed", [])
                        if failed:
                            st.markdown("**Failed:**")
                            for s in failed:
                                st.markdown(f"- {s}")
                    with sc3:
                        missed = signals.get("signals_missed", [])
                        if missed:
                            st.markdown("**Missed:**")
                            for s in missed:
                                st.markdown(f"- {s}")

                # Strategy fitness
                fitness = critique.get("strategy_fitness", {})
                if fitness:
                    right = fitness.get("was_right_strategy", True)
                    better = fitness.get("better_strategy", "none")
                    regime = fitness.get("market_regime_match", "")
                    parts = [f"{'Correct strategy' if right else 'Suboptimal strategy'}"]
                    if better and better != "none":
                        parts.append(f"better: {better}")
                    if regime:
                        parts.append(f"regime: {regime}")
                    st.markdown("**Fitness:** " + " | ".join(parts))

                # Parameter recommendations
                recs = critique.get("parameter_recommendations", [])
                if recs:
                    st.markdown("**Recommendations:**")
                    for rec in recs:
                        conf = rec.get("confidence", 0)
                        st.markdown(
                            f"- `{rec.get('parameter_name')}`: "
                            f"{rec.get('current_value')} -> {rec.get('recommended_value')} "
                            f"(conf: {conf:.0%}) — {rec.get('reasoning', '')}"
                        )

                # Risk management
                rm = critique.get("risk_management_notes", "")
                if rm:
                    st.markdown(f"**Risk notes:** {rm}")


# ---------------------------------------------------------------------------
# Critique processing & display
# ---------------------------------------------------------------------------

def _clear_critiques_db() -> None:
    """Wipe all trade critiques and parameter adjustments from the database."""
    try:
        from core.database import SentimentDatabase
        db = SentimentDatabase()
        db.clear_critiques()
    except Exception:
        logger.warning("Failed to clear critiques from DB", exc_info=True)


_GRADE_COLORS = {
    "excellent": "#22c55e",
    "good": "#4ade80",
    "acceptable": "#facc15",
    "poor": "#f87171",
    "terrible": "#ef4444",
}


def _process_pending_critique(state: PaperTradingState, algo_name: str = "sentinel") -> PaperTradingState:
    """Process one pending critique per refresh cycle. Non-blocking."""
    crit_cfg = load_config().get("criticizer", {})
    if not crit_cfg.get("enabled", False) or not crit_cfg.get("auto_critique", True):
        return state
    if not state.pending_critiques:
        return state

    trade_id = state.pending_critiques[0]
    record = next((t for t in state.trade_log if t.id == trade_id), None)
    if not record or not record.entry_context:
        # Skip — no context to critique
        state = state.model_copy(update={"pending_critiques": state.pending_critiques[1:]})
        _set_state(state, algo_name)
        return state

    try:
        from analyzers.trade_criticizer import criticize_trade
        from core.database import SentimentDatabase

        # Get recent performance for this strategy
        db = SentimentDatabase()
        recent = db.get_critiques_for_strategy(record.strategy, limit=10)

        critique = criticize_trade(record, recent_performance=recent)
        db.save_critique(critique, trade_record_dict=record.model_dump(mode="json"))
        logger.info("Critique saved for trade %s: grade=%s", trade_id, critique.overall_grade)
    except Exception as exc:
        logger.warning("Failed to critique trade %s", trade_id, exc_info=True)
        st.error(f"Critique failed for trade {trade_id[:8]}: {exc}")

    # Pop from queue regardless of success/failure
    state = state.model_copy(update={"pending_critiques": state.pending_critiques[1:]})
    _set_state(state, algo_name)
    return state


