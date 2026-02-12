"""Streamlit UI for the Intraday Options Desk tab."""

import time
from datetime import datetime, time as dt_time

import streamlit as st
from streamlit_autorefresh import st_autorefresh
from core.config import load_config
from core.market_hours import IST, is_market_open
from core.options_engine import OptionsDeskEngine
from core.options_models import SignalDirection, TradeSuggestion

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
    return is_market_open()


def _last_refresh_text(ts: float) -> str:
    if ts <= 0:
        return "Never"
    elapsed = int(time.time() - ts)
    if elapsed < 60:
        return f"{elapsed}s ago"
    return f"{elapsed // 60}m {elapsed % 60}s ago"


def _render_market_status_banner() -> None:
    now = datetime.now(IST)
    is_open = _is_market_hours()
    last_ts = st.session_state.options_last_refresh

    if is_open:
        refresh_text = _last_refresh_text(last_ts) if last_ts > 0 else "fetching..."
        st.success(f"Market Open  |  Auto-refreshing every 60s  |  Last updated: {refresh_text}")
    else:
        if now.weekday() >= 5:
            context = "Weekend"
        elif now.hour < 9 or (now.hour == 9 and now.minute < 15):
            context = "Pre-market"
        else:
            context = "Post-market"

        # Show time elapsed since market close (3:30 PM IST)
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
        if now >= market_close:
            elapsed = int((now - market_close).total_seconds())
            h, rem = divmod(elapsed, 3600)
            m, _ = divmod(rem, 60)
            since_close = f"{h}h {m}m ago" if h else f"{m}m ago"
            st.info(f"Market Closed ({context})  |  Showing last available data  |  Closed: {since_close}")
        else:
            st.info(f"Market Closed ({context})  |  Showing last available data")


def render_options_desk_tab() -> None:
    """Main entry point — renders the full Options Desk tab."""
    engine = _get_engine()

    # --- Refresh logic ---
    if "options_snapshot" not in st.session_state:
        st.session_state.options_snapshot = None
    if "options_last_refresh" not in st.session_state:
        st.session_state.options_last_refresh = 0.0

    # --- Header ---
    st.title("Intraday Options Desk")
    st.caption("Option chain analytics, technical indicators & aggregated signals")

    # --- Auto-refresh during market hours ---
    if _is_market_hours():
        st_autorefresh(interval=60_000, key="options_desk_autorefresh")

    # --- Fetch: auto on first load, re-fetch every ~60s during market hours ---
    need_fetch = False
    if st.session_state.options_snapshot is None:
        need_fetch = True  # first load — always fetch regardless of market hours
    elif _is_market_hours() and time.time() - st.session_state.options_last_refresh > 55:
        need_fetch = True  # 55s threshold avoids edge-case with 60s timer

    if need_fetch:
        with st.spinner("Fetching options data..."):
            st.session_state.options_snapshot = engine.fetch_snapshot()
            st.session_state.options_last_refresh = time.time()
            # Populate session state timestamps for System Health Data Freshness
            snap_meta = st.session_state.options_snapshot
            chain_m = getattr(snap_meta, "chain_meta", None) if snap_meta else None
            candle_m = getattr(snap_meta, "candle_meta", None) if snap_meta else None
            if chain_m:
                st.session_state.last_chain_fetch_ts = chain_m.fetch_ts
            if candle_m:
                st.session_state.last_candle_fetch_ts = candle_m.fetch_ts

    # --- Market status banner (after fetch so timestamp is current) ---
    _render_market_status_banner()

    snap = st.session_state.options_snapshot
    if snap is None:
        st.info("Unable to fetch options data. NSE may be unreachable.")
        return

    # Show errors as warnings (deduplicate)
    for err in dict.fromkeys(snap.errors):
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
        if anl and anl.atm_iv > 0:
            st.metric("ATM IV", f"{anl.atm_iv:.1f}%")
        else:
            st.metric("ATM IV", "N/A")

    with pulse_cols[3]:
        if anl and anl.pcr > 0:
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
    # SUGGESTED TRADES (row 1.5)
    # ================================================================
    st.divider()
    st.subheader("Suggested Trades")

    is_warmup, entry_time, secs_remaining = _is_warmup_period()
    if is_warmup:
        _render_warmup_view(snap, secs_remaining, entry_time)
    else:
        # Post-warmup: show compact morning context if observation is complete
        obs = snap.observation
        if obs and obs.is_complete and obs.bars_collected > 0:
            with st.expander("Morning Context (9:15-10:00 observation)", expanded=False):
                mc_cols = st.columns(5)
                with mc_cols[0]:
                    or_str = f"{obs.opening_range.range_points:.0f} pts ({obs.opening_range.range_pct:.2f}%)"
                    st.metric("Opening Range", or_str)
                with mc_cols[1]:
                    gap_str = f"{obs.gap.gap_pct:+.2f}%" if obs.gap.direction != "flat" else "Flat"
                    st.metric("Gap", gap_str)
                with mc_cols[2]:
                    st.metric("Trend", f"{obs.initial_trend.direction.replace('_', ' ').title()} ({obs.initial_trend.strength})")
                with mc_cols[3]:
                    st.metric("Volume", f"{obs.volume.relative_volume:.1f}x ({obs.volume.classification})")
                with mc_cols[4]:
                    bias_color = {"bullish": "green", "bearish": "red", "neutral": "gray"}.get(obs.bias, "gray")
                    st.metric("Bias", obs.bias.upper())
                if obs.opening_range.breakout_direction:
                    st.caption(f"OR breakout: {obs.opening_range.breakout_direction} ({obs.opening_range.breakout_distance_pct:.2f}%)")

        # Get top high-confidence suggestion per algorithm from session state
        from algorithms import get_algorithm_registry
        registry = get_algorithm_registry()
        algo_suggestions = st.session_state.get("algo_suggestions", {})

        top_trades: list[tuple[str, TradeSuggestion]] = []
        for algo_name, suggestions in algo_suggestions.items():
            high = [s for s in suggestions if s.confidence == "High" and not s.rejection_reason]
            if high:
                display_name = registry[algo_name].display_name if algo_name in registry else algo_name
                top_trades.append((display_name, high[0]))

        if top_trades:
            _render_algo_trade_suggestions(top_trades)
        else:
            st.info("No high-confidence trades from any algorithm at this time.")

    # ================================================================
    # EXPANDABLE DETAILS (row 4)
    # ================================================================
    st.divider()
    _render_expandable_details(snap)


_BIAS_COLORS = {
    "Bullish": ("#1b5e20", "#e8f5e9"),
    "Bearish": ("#b71c1c", "#ffebee"),
    "Neutral": ("#616161", "#f5f5f5"),
}

_CONFIDENCE_BADGE = {
    "High": "\u2b50 High",
    "Medium": "\U0001f7e1 Medium",
    "Low": "\u26aa Low",
}


def _is_warmup_period() -> tuple[bool, str, int]:
    """Check whether we are in the warmup window (market open to entry_start_time).

    Returns (is_warmup, entry_time_str, seconds_remaining).
    is_warmup is False when the market is closed.
    """
    cfg = load_config().get("paper_trading", {})
    entry_time_str = cfg.get("entry_start_time", "10:00")
    h, m = (int(x) for x in entry_time_str.split(":"))
    entry_time = dt_time(h, m)

    now = datetime.now(IST)
    # Not a trading day or outside market hours → not warmup
    if now.weekday() >= 5 or now.time() > dt_time(15, 30) or now.time() < dt_time(9, 15):
        return False, entry_time_str, 0

    if now.time() < entry_time:
        target = now.replace(hour=h, minute=m, second=0, microsecond=0)
        secs = int((target - now).total_seconds())
        return True, entry_time_str, max(secs, 0)

    return False, entry_time_str, 0


def _render_warmup_view(snap, seconds_remaining: int, entry_time: str) -> None:
    """Show warmup UI during the observation period (9:15 to entry_start_time)."""
    mins, secs = divmod(seconds_remaining, 60)
    st.info(f"Warming up — suggestions will appear at **{entry_time} IST** (in {mins}m {secs}s)")

    # Observation data (accumulated during warmup)
    obs = snap.observation
    if obs and obs.bars_collected > 0:
        st.markdown("**Observation Period Data**")
        obs_cols = st.columns(4)
        with obs_cols[0]:
            or_label = f"{obs.opening_range.high:,.0f} - {obs.opening_range.low:,.0f}" if obs.opening_range.high > 0 else "—"
            or_detail = f"{obs.opening_range.range_points:.0f} pts ({obs.opening_range.range_pct:.2f}%)" if obs.opening_range.range_points > 0 else ""
            st.metric("Opening Range", or_label)
            if or_detail:
                st.caption(or_detail)
            st.caption(f"{obs.bars_collected} bars collected")
        with obs_cols[1]:
            gap_label = f"{obs.gap.gap_pct:+.2f}%" if obs.gap.direction != "flat" else "Flat"
            st.metric("Opening Gap", gap_label)
            if obs.gap.direction != "flat":
                fill_str = f"Fill: {obs.gap.gap_fill_pct:.0f}%"
                st.caption(f"{obs.gap.direction.replace('_', ' ').title()} | {fill_str}")
        with obs_cols[2]:
            trend_label = f"{obs.initial_trend.direction.replace('_', ' ').title()}"
            trend_detail = f"{obs.initial_trend.move_pct:+.2f}% ({obs.initial_trend.strength})"
            st.metric("Initial Trend", trend_label)
            st.caption(trend_detail)
        with obs_cols[3]:
            vol_label = f"{obs.volume.relative_volume:.1f}x"
            st.metric("Relative Volume", vol_label)
            vwap_str = obs.vwap_context.relationship.replace("_", " ").title()
            st.caption(f"VWAP: {vwap_str} ({obs.vwap_context.pct_above:.0f}%)")

        # Bias summary
        bias_color = {"bullish": "green", "bearish": "red", "neutral": "gray"}.get(obs.bias, "gray")
        st.markdown(f"**Observation Bias:** :{bias_color}[{obs.bias.upper()}]")
        if obs.bias_reasons:
            for reason in obs.bias_reasons[:5]:
                st.caption(f"  - {reason}")

    # Live indicator snapshot
    tech = snap.technicals
    anl = snap.analytics
    if tech:
        st.markdown("**Live Indicators**")
        ind_cols = st.columns(5)
        with ind_cols[0]:
            st.metric("RSI", f"{tech.rsi:.1f}")
        with ind_cols[1]:
            st_dir = "Bullish" if tech.supertrend_direction == 1 else "Bearish"
            st.metric("Supertrend", st_dir)
        with ind_cols[2]:
            dist = ((tech.spot - tech.vwap) / tech.vwap * 100) if tech.vwap else 0
            st.metric("Spot vs VWAP", f"{dist:+.2f}%")
        with ind_cols[3]:
            bb_w = ((tech.bb_upper - tech.bb_lower) / tech.bb_middle * 100) if tech.bb_middle else 0
            st.metric("BB Width", f"{bb_w:.2f}%")
        with ind_cols[4]:
            if tech.ema_9 > tech.ema_21 > tech.ema_50:
                trend = "Bullish"
            elif tech.ema_9 < tech.ema_21 < tech.ema_50:
                trend = "Bearish"
            else:
                trend = "Mixed"
            st.metric("EMA Trend", trend)

    # Per-algorithm candidate preview
    algo_suggestions = st.session_state.get("algo_suggestions", {})
    if algo_suggestions:
        from algorithms import get_algorithm_registry
        registry = get_algorithm_registry()

        st.markdown("**Algorithm Candidates (preview)**")
        preview_cols = st.columns(len(algo_suggestions))
        for idx, (algo_name, suggestions) in enumerate(algo_suggestions.items()):
            with preview_cols[idx]:
                display = registry[algo_name].display_name if algo_name in registry else algo_name
                high = sum(1 for s in suggestions if s.confidence == "High" and not s.rejection_reason)
                med = sum(1 for s in suggestions if s.confidence == "Medium" and not s.rejection_reason)
                low = sum(1 for s in suggestions if s.confidence == "Low" and not s.rejection_reason)
                st.markdown(f"**{display}**")
                st.caption(f"{len(suggestions)} candidates: {high}H / {med}M / {low}L")
                # Show leading strategy preview
                viable = [s for s in suggestions if not s.rejection_reason]
                if viable:
                    best = max(viable, key=lambda s: s.score)
                    st.caption(f"Leading: {best.strategy.value} ({best.direction_bias}, score {best.score:.0f})")


def _render_algo_trade_suggestions(top_trades: list[tuple[str, TradeSuggestion]]) -> None:
    """Render one high-confidence trade card per algorithm."""
    for row_start in range(0, len(top_trades), 2):
        cols = st.columns(2)
        for col_idx, t_idx in enumerate(range(row_start, min(row_start + 2, len(top_trades)))):
            display_name, s = top_trades[t_idx]
            fg, bg = _BIAS_COLORS.get(s.direction_bias, ("#616161", "#f5f5f5"))
            confidence = _CONFIDENCE_BADGE.get(s.confidence, s.confidence)

            with cols[col_idx]:
                # Card header with algorithm badge
                st.markdown(
                    f'<div style="background:{bg}; border-left:4px solid {fg}; '
                    f'padding:14px; border-radius:8px; margin-bottom:4px; position:relative;">'
                    f'<span style="position:absolute; top:8px; right:12px; background:{fg}; '
                    f'color:white; padding:2px 8px; border-radius:4px; font-size:0.75em;">'
                    f'{display_name}</span>'
                    f'<strong style="color:{fg}; font-size:1.15em;">{s.strategy.value}</strong>'
                    f'<br><span style="color:{fg}; font-size:0.9em;">'
                    f'{s.direction_bias} &bull; {confidence} &bull; Score {s.score:.0f}</span></div>',
                    unsafe_allow_html=True,
                )

                # Legs table
                leg_rows = ""
                for leg in s.legs:
                    action_color = "#b71c1c" if leg.action == "SELL" else "#1b5e20"
                    leg_rows += (
                        f"<tr><td style='color:{action_color}; font-weight:600;'>{leg.action}</td>"
                        f"<td>{leg.instrument}</td>"
                        f"<td style='text-align:right;'>\u20b9{leg.ltp:.1f}</td></tr>"
                    )
                st.markdown(
                    f"<table style='width:100%; font-size:0.85em; margin:6px 0;'>"
                    f"<tr style='border-bottom:1px solid #ddd;'>"
                    f"<th style='text-align:left;'>Action</th>"
                    f"<th style='text-align:left;'>Instrument</th>"
                    f"<th style='text-align:right;'>LTP</th></tr>"
                    f"{leg_rows}</table>",
                    unsafe_allow_html=True,
                )

                # Key details
                st.markdown(f"**Entry:** {s.entry_timing}")
                st.markdown(f"**Expected:** {s.expected_outcome}")

                detail_cols = st.columns(3)
                with detail_cols[0]:
                    st.markdown(f"**Max Profit**  \n{s.max_profit}")
                with detail_cols[1]:
                    st.markdown(f"**Max Loss**  \n{s.max_loss}")
                with detail_cols[2]:
                    st.markdown(f"**Size**  \n{s.position_size}")

                st.markdown(f"**Stop Loss:** {s.stop_loss}")

                # Expandable reasoning
                with st.expander("Reasoning & Checks"):
                    st.markdown("**Why this trade:**")
                    for r in s.reasoning:
                        st.markdown(f"- {r}")
                    st.markdown("**Technicals to watch:**")
                    for c in s.technicals_to_check:
                        st.markdown(f"- {c}")


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
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    # Change-in-OI buildup analysis
    if chain and chain.strikes:
        has_chg_oi = any(s.ce_change_in_oi != 0 or s.pe_change_in_oi != 0 for s in chain.strikes)
        with st.expander("Change in OI Analysis"):
            if not has_chg_oi:
                st.info("Change-in-OI data not available from Kite. This section requires NSE as data source.")
            else:
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
