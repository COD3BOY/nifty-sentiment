"""Side-by-side algorithm performance comparison view."""

from __future__ import annotations

import streamlit as st

from core.paper_trading_engine import load_state
from core.paper_trading_models import PaperTradingState


def _fmt_pnl(pnl: float) -> str:
    sign = "+" if pnl > 0 else ""
    return f"{sign}\u20b9{pnl:,.0f}"


def render_algorithm_comparison(algo_names: list[str], algo_display_names: dict[str, str]) -> None:
    """Render a comparison table across all enabled algorithms."""
    if len(algo_names) < 2:
        return

    st.subheader("Algorithm Comparison")

    states: dict[str, PaperTradingState] = {}
    for name in algo_names:
        s = load_state(algo_name=name)
        if s is None:
            s = PaperTradingState()
        states[name] = s

    # Header row
    cols = st.columns(len(algo_names) + 1)
    with cols[0]:
        st.markdown("**Metric**")
    for i, name in enumerate(algo_names):
        with cols[i + 1]:
            st.markdown(f"**{algo_display_names.get(name, name)}**")

    st.divider()

    # Metrics
    metrics = [
        ("Trades", lambda s: str(len(s.trade_log))),
        ("Win Rate", lambda s: f"{s.win_rate:.0f}%"),
        ("Avg RR", lambda s: f"1:{s.avg_risk_reward:.1f}" if s.avg_risk_reward > 0 else "N/A"),
        ("Max Drawdown", lambda s: f"{s.max_drawdown_pct:.1f}%"),
        ("Sharpe Ratio", lambda s: f"{s.sharpe_ratio:.2f}"),
        ("Total P&L", lambda s: _fmt_pnl(s.total_pnl)),
        ("Net Realized", lambda s: _fmt_pnl(s.net_realized_pnl)),
        ("Total Costs", lambda s: f"\u20b9{s.total_execution_costs:,.0f}"),
        ("Open Positions", lambda s: str(len([p for p in s.open_positions if p.status.value == "open"]))),
        ("Margin in Use", lambda s: f"\u20b9{s.margin_in_use:,.0f}"),
    ]

    for label, fn in metrics:
        cols = st.columns(len(algo_names) + 1)
        with cols[0]:
            st.markdown(f"_{label}_")
        for i, name in enumerate(algo_names):
            with cols[i + 1]:
                st.markdown(fn(states[name]))
