"""Streamlit UI for the Intraday Options Desk tab."""

import time
from datetime import datetime, timedelta, timezone

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from core.indicators import (
    compute_bollinger_bands,
    compute_ema,
    compute_rsi,
    compute_supertrend,
    compute_vwap,
)
from core.options_engine import OptionsDeskEngine
from core.options_models import SignalDirection

IST = timezone(timedelta(hours=5, minutes=30))

_SIGNAL_COLORS = {
    SignalDirection.BULLISH: ("#1b5e20", "#e8f5e9"),
    SignalDirection.BEARISH: ("#b71c1c", "#ffebee"),
    SignalDirection.NEUTRAL: ("#616161", "#f5f5f5"),
}

_SIGNAL_ICONS = {
    SignalDirection.BULLISH: "**B**",
    SignalDirection.BEARISH: "**B**",
    SignalDirection.NEUTRAL: "**N**",
}


def _get_engine() -> OptionsDeskEngine:
    if "options_engine" not in st.session_state:
        st.session_state.options_engine = OptionsDeskEngine()
    return st.session_state.options_engine


def _is_market_hours() -> bool:
    now = datetime.now(IST)
    start = now.replace(hour=9, minute=15, second=0, microsecond=0)
    end = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return start <= now <= end


def _last_refresh_text(ts: float) -> str:
    if ts <= 0:
        return "Never"
    elapsed = int(time.time() - ts)
    if elapsed < 60:
        return f"{elapsed}s ago"
    return f"{elapsed // 60}m {elapsed % 60}s ago"


def render_options_desk_tab() -> None:
    """Main entry point — renders the full Options Desk tab."""
    engine = _get_engine()

    # --- Refresh logic ---
    if "options_snapshot" not in st.session_state:
        st.session_state.options_snapshot = None
    if "options_last_refresh" not in st.session_state:
        st.session_state.options_last_refresh = 0.0

    # --- Standardized header ---
    _h1, _h2 = st.columns([4, 1.5])
    with _h1:
        st.title("⚡ Intraday Options Desk")
        auto_status = "ON (60s)" if _is_market_hours() else "OFF (outside market hours)"
        st.caption(f"Option chain analytics, technical indicators & aggregated signals | Auto-refresh: {auto_status}")
    with _h2:
        manual_refresh = st.button("Refresh Data", key="options_refresh", type="primary", use_container_width=True)
        st.caption(f"Last refresh: {_last_refresh_text(st.session_state.options_last_refresh)}")

    if manual_refresh:
        with st.spinner("Fetching options desk data..."):
            st.session_state.options_snapshot = engine.fetch_snapshot()
            st.session_state.options_last_refresh = time.time()
        st.rerun()

    # Auto-refresh during market hours
    if (
        _is_market_hours()
        and st.session_state.options_last_refresh > 0
        and time.time() - st.session_state.options_last_refresh > 60
    ):
        st.session_state.options_snapshot = engine.fetch_snapshot()
        st.session_state.options_last_refresh = time.time()
        st.rerun()

    snap = st.session_state.options_snapshot
    if snap is None:
        st.info("Click **Refresh Data** to fetch the latest options desk snapshot.")
        return

    # Show errors as warnings
    for err in snap.errors:
        st.warning(err)

    # ================================================================
    # MARKET PULSE (top bar)
    # ================================================================
    st.divider()
    tech = snap.technicals
    anl = snap.analytics

    pulse_cols = st.columns(4)
    with pulse_cols[0]:
        if tech:
            arrow = "+" if tech.spot_change_pct >= 0 else ""
            st.metric("NIFTY Spot", f"{tech.spot:,.1f}", delta=f"{arrow}{tech.spot_change_pct:.2f}%")
        else:
            st.metric("NIFTY Spot", "N/A")

    with pulse_cols[1]:
        if tech:
            dist = ((tech.spot - tech.vwap) / tech.vwap) * 100 if tech.vwap else 0
            st.metric("VWAP", f"{tech.vwap:,.1f}", delta=f"{dist:+.2f}% from spot")
        else:
            st.metric("VWAP", "N/A")

    with pulse_cols[2]:
        if anl:
            st.metric("ATM IV", f"{anl.atm_iv:.1f}%")
        else:
            st.metric("ATM IV", "N/A")

    with pulse_cols[3]:
        if anl:
            st.metric("PCR", f"{anl.pcr:.3f}", delta=anl.pcr_label)
        else:
            st.metric("PCR", "N/A")

    # ================================================================
    # SIGNAL CARDS (row 1)
    # ================================================================
    st.divider()
    st.subheader("Signal Cards")

    if snap.signals:
        sig_cols = st.columns(len(snap.signals))
        for i, sig in enumerate(snap.signals):
            fg, bg = _SIGNAL_COLORS[sig.direction]
            with sig_cols[i]:
                st.markdown(
                    f'<div style="background:{bg}; border-left:4px solid {fg}; '
                    f'padding:12px; border-radius:6px; margin-bottom:8px;">'
                    f'<strong style="color:{fg}; font-size:1.1em;">{sig.label}</strong><br>'
                    f'<span style="color:{fg}; font-size:1.3em;">'
                    f'{sig.direction.value.upper()}</span></div>',
                    unsafe_allow_html=True,
                )
                with st.expander("Reasoning"):
                    for r in sig.reasoning:
                        st.markdown(f"- {r}")

    # ================================================================
    # OPTION CHAIN SNAPSHOT (row 2)
    # ================================================================
    st.divider()
    st.subheader("Option Chain Snapshot")

    col_chart, col_levels = st.columns([2, 1])

    with col_chart:
        if snap.chain and snap.chain.strikes and anl:
            _render_oi_chart(snap, anl)
        else:
            st.caption("Option chain data unavailable")

    with col_levels:
        if anl:
            st.markdown("**Key Levels**")
            st.metric("Max Pain", f"{anl.max_pain:,.0f}")
            st.metric("Support (Max Put OI)", f"{anl.support_strike:,.0f}", delta=f"OI: {anl.support_oi:,.0f}")
            st.metric("Resistance (Max Call OI)", f"{anl.resistance_strike:,.0f}", delta=f"OI: {anl.resistance_oi:,.0f}")
            if tech:
                st.metric("Spot vs Max Pain", f"{tech.spot - anl.max_pain:+,.0f} pts")
        else:
            st.caption("Analytics unavailable")

    # ================================================================
    # TECHNICAL CHART (row 3)
    # ================================================================
    st.divider()
    st.subheader("Technical Chart (5-min)")

    df = engine.get_candle_dataframe()
    if df is not None and not df.empty:
        _render_technical_chart(df, tech)
    else:
        st.caption("Candle data unavailable")

    # ================================================================
    # EXPANDABLE DETAILS (row 4)
    # ================================================================
    st.divider()
    _render_expandable_details(snap)


def _render_oi_chart(snap, anl) -> None:
    """Horizontal OI bar chart — Call OI left, Put OI right, ~15 strikes around ATM."""
    chain = snap.chain
    atm = anl.atm_strike
    sorted_strikes = sorted(chain.strikes, key=lambda s: s.strike_price)

    # Find ATM index and take ~7 on each side
    atm_idx = 0
    for i, s in enumerate(sorted_strikes):
        if s.strike_price == atm:
            atm_idx = i
            break

    start = max(0, atm_idx - 7)
    end = min(len(sorted_strikes), atm_idx + 8)
    subset = sorted_strikes[start:end]

    strike_labels = [str(int(s.strike_price)) for s in subset]
    ce_oi = [-s.ce_oi for s in subset]  # negative for left side
    pe_oi = [s.pe_oi for s in subset]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=strike_labels, x=ce_oi, orientation="h",
        name="Call OI", marker_color="#ef5350",
        text=[f"{abs(v):,.0f}" for v in ce_oi], textposition="outside",
    ))
    fig.add_trace(go.Bar(
        y=strike_labels, x=pe_oi, orientation="h",
        name="Put OI", marker_color="#66bb6a",
        text=[f"{v:,.0f}" for v in pe_oi], textposition="outside",
    ))
    fig.update_layout(
        barmode="overlay",
        height=400,
        margin=dict(t=30, b=30, l=60, r=60),
        xaxis_title="Open Interest",
        yaxis_title="Strike",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_technical_chart(df, tech) -> None:
    """Plotly candlestick with EMAs, Supertrend, Bollinger Bands + RSI subplot."""
    close = df["Close"]

    ema9 = compute_ema(close, 9)
    ema21 = compute_ema(close, 21)
    ema50 = compute_ema(close, 50)
    st_vals, st_dirs = compute_supertrend(df)
    bb_upper, bb_mid, bb_lower = compute_bollinger_bands(close)
    rsi = compute_rsi(close)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.75, 0.25], vertical_spacing=0.03,
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="NIFTY",
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
    ), row=1, col=1)

    # EMAs
    for ema, name, color in [
        (ema9, "EMA 9", "#ff9800"),
        (ema21, "EMA 21", "#2196f3"),
        (ema50, "EMA 50", "#9c27b0"),
    ]:
        fig.add_trace(go.Scatter(
            x=df.index, y=ema, name=name, line=dict(color=color, width=1),
        ), row=1, col=1)

    # Supertrend (colored by direction) — replace zeros with NaN to avoid Y-axis distortion
    import numpy as np
    st_vals = st_vals.replace(0, np.nan)
    bull_st = st_vals.where(st_dirs == 1)
    bear_st = st_vals.where(st_dirs == -1)
    fig.add_trace(go.Scatter(
        x=df.index, y=bull_st, name="Supertrend Bull",
        line=dict(color="#4caf50", width=1.5, dash="dot"),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=bear_st, name="Supertrend Bear",
        line=dict(color="#f44336", width=1.5, dash="dot"),
    ), row=1, col=1)

    # Bollinger Bands
    fig.add_trace(go.Scatter(
        x=df.index, y=bb_upper, name="BB Upper",
        line=dict(color="#90a4ae", width=0.8, dash="dash"),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=bb_lower, name="BB Lower",
        line=dict(color="#90a4ae", width=0.8, dash="dash"),
        fill="tonexty", fillcolor="rgba(144,164,174,0.08)",
    ), row=1, col=1)

    # RSI subplot
    fig.add_trace(go.Scatter(
        x=df.index, y=rsi, name="RSI",
        line=dict(color="#7e57c2", width=1.5),
    ), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="#ef5350", opacity=0.5, row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#4caf50", opacity=0.5, row=2, col=1)
    fig.add_hrect(y0=30, y1=70, fillcolor="rgba(158,158,158,0.05)", line_width=0, row=2, col=1)

    fig.update_layout(
        height=600,
        margin=dict(t=30, b=30, l=50, r=30),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font_size=10),
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)


def _render_expandable_details(snap) -> None:
    """Expandable sections: full chain table, Change-in-OI analysis."""
    chain = snap.chain
    anl = snap.analytics

    # Full option chain table
    if chain and chain.strikes:
        with st.expander("Full Option Chain Table"):
            import pandas as pd

            rows = []
            for s in sorted(chain.strikes, key=lambda x: x.strike_price):
                rows.append({
                    "CE OI": f"{s.ce_oi:,.0f}",
                    "CE ChgOI": f"{s.ce_change_in_oi:+,.0f}",
                    "CE Vol": f"{s.ce_volume:,}",
                    "CE IV": f"{s.ce_iv:.1f}" if s.ce_iv else "-",
                    "CE LTP": f"{s.ce_ltp:.1f}" if s.ce_ltp else "-",
                    "Strike": int(s.strike_price),
                    "PE LTP": f"{s.pe_ltp:.1f}" if s.pe_ltp else "-",
                    "PE IV": f"{s.pe_iv:.1f}" if s.pe_iv else "-",
                    "PE Vol": f"{s.pe_volume:,}",
                    "PE ChgOI": f"{s.pe_change_in_oi:+,.0f}",
                    "PE OI": f"{s.pe_oi:,.0f}",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Change-in-OI buildup analysis
    if chain and chain.strikes:
        with st.expander("Change in OI Analysis"):
            sorted_strikes = sorted(chain.strikes, key=lambda s: s.strike_price)
            atm = anl.atm_strike if anl else 0

            st.markdown("**Call OI Buildup (top 5 by Change in OI):**")
            ce_buildup = sorted(sorted_strikes, key=lambda s: s.ce_change_in_oi, reverse=True)[:5]
            for s in ce_buildup:
                tag = " (ATM)" if s.strike_price == atm else ""
                if s.ce_change_in_oi > 0:
                    st.markdown(f"- Strike {s.strike_price:.0f}{tag}: +{s.ce_change_in_oi:,.0f} (Short buildup / Resistance)")
                elif s.ce_change_in_oi < 0:
                    st.markdown(f"- Strike {s.strike_price:.0f}{tag}: {s.ce_change_in_oi:,.0f} (Short covering / Bullish)")

            st.markdown("**Put OI Buildup (top 5 by Change in OI):**")
            pe_buildup = sorted(sorted_strikes, key=lambda s: s.pe_change_in_oi, reverse=True)[:5]
            for s in pe_buildup:
                tag = " (ATM)" if s.strike_price == atm else ""
                if s.pe_change_in_oi > 0:
                    st.markdown(f"- Strike {s.strike_price:.0f}{tag}: +{s.pe_change_in_oi:,.0f} (Put writing / Support)")
                elif s.pe_change_in_oi < 0:
                    st.markdown(f"- Strike {s.strike_price:.0f}{tag}: {s.pe_change_in_oi:,.0f} (Long unwinding / Bearish)")

    # Signal reasoning summary
    if snap.signals:
        with st.expander("Signal Reasoning Details"):
            for sig in snap.signals:
                fg, _ = _SIGNAL_COLORS[sig.direction]
                st.markdown(f"**{sig.label}** — :{sig.direction.value}[{sig.direction.value.upper()}]")
                for r in sig.reasoning:
                    st.markdown(f"  - {r}")
                st.divider()
