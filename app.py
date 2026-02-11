"""NIFTY Sentiment Dashboard - Streamlit app."""

import asyncio
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Ensure project root is on sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from core.config import load_config
from core.engine import SentimentEngine
from core.models import SentimentLevel
from ui.options_desk_tab import render_options_desk_tab

st.set_page_config(
    page_title="NIFTY Sentiment Dashboard",
    page_icon="📊",
    layout="wide",
)

config = load_config()

# --- Session State Init ---
if "engine" not in st.session_state:
    st.session_state.engine = SentimentEngine()
if "pre_market_sentiment" not in st.session_state:
    st.session_state.pre_market_sentiment = None
if "pm_last_refresh" not in st.session_state:
    st.session_state.pm_last_refresh = 0.0
engine: SentimentEngine = st.session_state.engine

IST = timezone(timedelta(hours=5, minutes=30))

LEVEL_COLORS = {
    SentimentLevel.STRONGLY_BEARISH: "#d32f2f",
    SentimentLevel.BEARISH: "#f44336",
    SentimentLevel.SLIGHTLY_BEARISH: "#ff9800",
    SentimentLevel.NEUTRAL: "#fdd835",
    SentimentLevel.SLIGHTLY_BULLISH: "#8bc34a",
    SentimentLevel.BULLISH: "#4caf50",
    SentimentLevel.STRONGLY_BULLISH: "#1b5e20",
}


def run_async(coro):
    """Run an async coroutine from sync Streamlit context."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _get_source_data(sentiment, source_name: str) -> dict | None:
    """Extract raw_data for a specific source from aggregated sentiment."""
    if not sentiment or not sentiment.source_scores:
        return None
    for ss in sentiment.source_scores:
        if ss.source_name == source_name:
            return {"raw_data": ss.raw_data, "score": ss.score, "confidence": ss.confidence,
                    "explanation": ss.explanation, "bullish_factors": ss.bullish_factors,
                    "bearish_factors": ss.bearish_factors}
    return None


def _sentiment_color(score: float) -> str:
    if score > 0.05:
        return "green"
    elif score < -0.05:
        return "red"
    return "gray"


def _arrow(val: float) -> str:
    if val > 0:
        return "▲"
    elif val < 0:
        return "▼"
    return "—"


def _last_refresh_text(ts: float) -> str:
    """Return human-readable last-refresh string."""
    if ts <= 0:
        return "Never"
    elapsed = int(time.time() - ts)
    if elapsed < 60:
        return f"{elapsed}s ago"
    return f"{elapsed // 60}m {elapsed % 60}s ago"


def _live_clock_html() -> str:
    """Return a JS-powered live clock displaying IST."""
    return """
    <div id="live-clock" style="font-family: monospace; font-size: 1.4rem; font-weight: 700;
         color: #e0e0e0; line-height: 1.3;"></div>
    <div id="live-date" style="font-family: sans-serif; font-size: 0.8rem; color: #999; margin-top: 2px;"></div>
    <script>
    function updateClock() {
        const now = new Date();
        const ist = new Date(now.getTime() + (5.5 * 60 * 60 * 1000) + (now.getTimezoneOffset() * 60 * 1000));
        const h = String(ist.getHours()).padStart(2, '0');
        const m = String(ist.getMinutes()).padStart(2, '0');
        const s = String(ist.getSeconds()).padStart(2, '0');
        document.getElementById('live-clock').textContent = h + ':' + m + ':' + s + ' IST';
        const days = ['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'];
        const months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
        document.getElementById('live-date').textContent =
            days[ist.getDay()] + ', ' + ist.getDate() + ' ' + months[ist.getMonth()] + ' ' + ist.getFullYear();
    }
    updateClock();
    setInterval(updateClock, 1000);
    </script>
    """


# ============================================================
# Algorithm discovery
# ============================================================
from algorithms import discover_algorithms, get_algorithm_registry

discover_algorithms()
_algo_registry = get_algorithm_registry()
_algo_cfg = config.get("algorithms", {})
# Filter to enabled algorithms, preserving registry order
_enabled_algos = [
    name for name in _algo_registry
    if _algo_cfg.get(name, {}).get("enabled", True)
]

# ============================================================
# TABS
# ============================================================
_algo_tab_labels = [f"📈 {_algo_registry[n].display_name}" for n in _enabled_algos]
_all_tab_labels = ["🌅 Pre-Market Analysis", "⚡ Options Desk"] + _algo_tab_labels
if len(_enabled_algos) >= 2:
    _all_tab_labels.append("📊 Comparison")
_all_tabs = st.tabs(_all_tab_labels)
tab_premarket, tab_options = _all_tabs[0], _all_tabs[1]
algo_tabs = _all_tabs[2:2 + len(_enabled_algos)]
tab_comparison = _all_tabs[-1] if len(_enabled_algos) >= 2 else None


# ============================================================
# TAB 1: PRE-MARKET ANALYSIS
# ============================================================
with tab_premarket:
    # --- Standardized header ---
    _h1, _h2, _h3 = st.columns([3, 1.5, 1.5])
    with _h1:
        st.title("🌅 Pre-Market Analysis")
        st.caption("GIFT Nifty gap, global cues, FII/DII flows, VIX, macro & news sentiment")
    with _h2:
        components.html(_live_clock_html(), height=55)
    with _h3:
        if st.button("Refresh Pre-Market", key="refresh_premarket", type="primary", use_container_width=True):
            with st.spinner("Fetching pre-market sentiment..."):
                st.session_state.pre_market_sentiment = run_async(engine.compute_sentiment(mode="pre_market"))
                engine.update_market_actuals()
                st.session_state.pm_last_refresh = time.time()
            st.rerun()
        st.caption(f"Last refresh: {_last_refresh_text(st.session_state.pm_last_refresh)}")

    # --- Market status bar ---
    now_ist = datetime.now(IST)
    market_open = now_ist.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now_ist.replace(hour=15, minute=30, second=0, microsecond=0)

    if now_ist < market_open:
        time_to_open = market_open - now_ist
        hours, remainder = divmod(int(time_to_open.total_seconds()), 3600)
        minutes, _ = divmod(remainder, 60)
        st.info(f"⏳ Market opens in **{hours}h {minutes}m** | Pre-market is the best time to analyse")
    elif now_ist <= market_close:
        st.success("🟢 **Market is OPEN**")
    else:
        st.warning("🔴 **Market is CLOSED** for today")

    st.divider()

    pm = st.session_state.pre_market_sentiment

    if pm is None:
        st.info("Click **Refresh Pre-Market** to fetch the latest pre-market analysis.")
    else:
        # --- GIFT Nifty Gap ---
        gift = _get_source_data(pm, "gift_nifty")
        st.subheader("🎯 GIFT Nifty Gap")
        if gift and gift["raw_data"]:
            rd = gift["raw_data"]
            gap_pct = rd.get("gap_pct", 0)
            ltp = rd.get("gift_nifty_ltp", 0)
            prev_close = rd.get("nifty_prev_close", 0)
            arrow = _arrow(gap_pct)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Gap", f"{gap_pct:+.2f}%", delta=f"{arrow} {'Up' if gap_pct > 0 else 'Down' if gap_pct < 0 else 'Flat'}")
            with c2:
                st.metric("GIFT Nifty LTP", f"{ltp:,.0f}")
            with c3:
                st.metric("NIFTY Prev Close", f"{prev_close:,.0f}")
            st.caption(gift["explanation"])
        else:
            st.caption("GIFT Nifty data unavailable (Kite Connect credentials needed)")

        st.divider()

        # --- Overnight Global Cues ---
        gm = _get_source_data(pm, "global_markets")
        st.subheader("🌍 Overnight Global Cues")
        if gm and gm["raw_data"]:
            indices = gm["raw_data"].get("indices", [])
            if indices:
                cols = st.columns(len(indices))
                for i, idx in enumerate(indices):
                    with cols[i]:
                        change = idx.get("change_pct", 0)
                        delta_color = "normal" if change >= 0 else "inverse"
                        st.metric(
                            idx.get("name", idx.get("ticker", "?")),
                            f"{idx.get('current_price', 0):,.0f}",
                            delta=f"{change:+.2f}%",
                            delta_color=delta_color,
                        )
            st.caption(gm["explanation"])
        else:
            st.caption("Global markets data unavailable")

        st.divider()

        # --- FII/DII Positioning ---
        fii = _get_source_data(pm, "fii_dii")
        st.subheader("🏦 FII/DII Positioning")
        if fii and fii["raw_data"]:
            rd = fii["raw_data"]
            fii_net = rd.get("fii_net", 0)
            dii_net = rd.get("dii_net", 0)
            combined = fii_net + dii_net

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("FII Net", f"₹{fii_net:,.0f} Cr",
                          delta=f"{'Buying' if fii_net > 0 else 'Selling'}",
                          delta_color="normal" if fii_net >= 0 else "inverse")
            with c2:
                st.metric("DII Net", f"₹{dii_net:,.0f} Cr",
                          delta=f"{'Buying' if dii_net > 0 else 'Selling'}",
                          delta_color="normal" if dii_net >= 0 else "inverse")
            with c3:
                st.metric("Combined Net", f"₹{combined:,.0f} Cr",
                          delta=f"{'Net Inflow' if combined > 0 else 'Net Outflow'}",
                          delta_color="normal" if combined >= 0 else "inverse")
            st.caption(fii["explanation"])
        else:
            st.caption("FII/DII data unavailable (nsepython needed)")

        st.divider()

        # --- Fear Gauge (VIX) ---
        vix_data = _get_source_data(pm, "vix")
        st.subheader("😰 Fear Gauge — India VIX")
        if vix_data and vix_data["raw_data"]:
            rd = vix_data["raw_data"]
            current_vix = rd.get("current_vix", 0)
            day_change = rd.get("day_change_pct", 0)
            spike = rd.get("spike_detected", False)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("VIX Level", f"{current_vix:.1f}",
                          delta=f"{day_change:+.1f}%",
                          delta_color="inverse" if day_change > 0 else "normal")
            with c2:
                if current_vix < 15:
                    zone = "🟢 Low Fear"
                elif current_vix < 20:
                    zone = "🟡 Moderate"
                elif current_vix < 25:
                    zone = "🟠 Elevated"
                else:
                    zone = "🔴 High Fear"
                st.metric("Fear Zone", zone)
            with c3:
                st.metric("Spike Alert", "⚠️ YES" if spike else "✅ No")
            st.caption(vix_data["explanation"])
        else:
            st.caption("VIX data unavailable")

        st.divider()

        # --- Macro Pulse (Crude + USD/INR) ---
        macro = _get_source_data(pm, "crude_oil")
        st.subheader("🛢️ Macro Pulse")
        if macro and macro["raw_data"]:
            rd = macro["raw_data"]
            c1, c2 = st.columns(2)
            with c1:
                crude_price = rd.get("crude_price")
                crude_change = rd.get("crude_change_pct")
                if crude_price is not None:
                    st.metric("Brent Crude", f"${crude_price:.1f}",
                              delta=f"{crude_change:+.1f}%",
                              delta_color="inverse" if crude_change > 0 else "normal")
                else:
                    st.caption("Crude oil data unavailable")
            with c2:
                usdinr = rd.get("usdinr")
                usdinr_change = rd.get("usdinr_change_pct")
                if usdinr is not None:
                    st.metric("USD/INR", f"₹{usdinr:.2f}",
                              delta=f"{usdinr_change:+.2f}%",
                              delta_color="inverse" if usdinr_change > 0 else "normal")
                else:
                    st.caption("USD/INR data unavailable")
            st.caption(macro["explanation"])
        else:
            st.caption("Macro data unavailable")

        st.divider()

        # --- News Sentiment ---
        news = _get_source_data(pm, "zerodha_pulse")
        st.subheader("📰 News Sentiment")
        if news and news["raw_data"]:
            rd = news["raw_data"]
            headline_sentiments = rd.get("headline_sentiments", [])
            if headline_sentiments:
                for hs in headline_sentiments:
                    score = hs.get("score", 0)
                    color = _sentiment_color(score)
                    st.markdown(
                        f":{color}[{_arrow(score)} **{score:+.2f}**] — {hs.get('headline', '')}"
                    )
                    st.caption(f"  _{hs.get('reasoning', '')}_")
            else:
                headlines = rd.get("headlines", [])
                for h in headlines:
                    st.markdown(f"- {h}")
            st.caption(f"Overall news score: **{news['score']:+.3f}** | {news['explanation']}")
        else:
            st.caption("News data unavailable (requires ANTHROPIC_API_KEY)")

        st.divider()

        # --- Pre-Market Verdict ---
        st.subheader("🎯 Pre-Market Verdict")

        col_gauge, col_factors = st.columns([1, 1])

        with col_gauge:
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=pm.overall_score,
                title={"text": pm.level.value.replace("_", " ").title()},
                number={"suffix": "", "valueformat": ".2f"},
                gauge={
                    "axis": {"range": [-1, 1], "tickwidth": 1},
                    "bar": {"color": LEVEL_COLORS.get(pm.level, "#888")},
                    "steps": [
                        {"range": [-1.0, -0.6], "color": "#ffcdd2"},
                        {"range": [-0.6, -0.3], "color": "#ffe0b2"},
                        {"range": [-0.3, -0.1], "color": "#fff9c4"},
                        {"range": [-0.1, 0.1], "color": "#f5f5f5"},
                        {"range": [0.1, 0.3], "color": "#dcedc8"},
                        {"range": [0.3, 0.6], "color": "#c8e6c9"},
                        {"range": [0.6, 1.0], "color": "#a5d6a7"},
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 4},
                        "thickness": 0.75,
                        "value": pm.overall_score,
                    },
                },
            ))
            fig.update_layout(height=300, margin=dict(t=60, b=20, l=30, r=30))
            st.plotly_chart(fig, use_container_width=True)

        with col_factors:
            st.markdown("**Bullish Factors**")
            if pm.bullish_factors:
                for f in pm.bullish_factors[:5]:
                    st.markdown(f"- 🟢 {f}")
            else:
                st.caption("No bullish factors identified")

            st.markdown("**Bearish Factors**")
            if pm.bearish_factors:
                for f in pm.bearish_factors[:5]:
                    st.markdown(f"- 🔴 {f}")
            else:
                st.caption("No bearish factors identified")

        st.metric("Confidence", f"{pm.confidence:.1%}")
        st.caption(f"Sources: {pm.sources_used} active / {pm.sources_failed} failed | "
                   f"Updated: {pm.timestamp.strftime('%Y-%m-%d %H:%M UTC')}")


# ============================================================
# TAB 2: OPTIONS DESK
# ============================================================
with tab_options:
    render_options_desk_tab()

# ============================================================
# ALGORITHM TABS (dynamic)
# ============================================================
from ui.paper_trading_tab import render_paper_trading_tab

snap = st.session_state.get("options_snapshot")

for algo_tab, algo_name in zip(algo_tabs, _enabled_algos):
    with algo_tab:
        algo_cls = _algo_registry[algo_name]
        algo_instance = algo_cls(config=_algo_cfg.get(algo_name, {}))

        # Generate algorithm-specific suggestions if data is available
        algo_suggestions = None
        if snap and snap.chain and snap.technicals and snap.analytics:
            try:
                algo_suggestions = algo_instance.generate_suggestions(
                    snap.chain, snap.technicals, snap.analytics,
                )
            except Exception as _exc:
                st.warning(f"Suggestion generation failed for {algo_cls.display_name}: {_exc}")

        render_paper_trading_tab(
            suggestions=algo_suggestions,
            chain=snap.chain if snap else None,
            technicals=snap.technicals if snap else None,
            analytics=snap.analytics if snap else None,
            algo_name=algo_name,
            algo_display_name=algo_cls.display_name,
            evaluate_fn=algo_instance.evaluate_and_manage,
        )

# ============================================================
# COMPARISON TAB
# ============================================================
if tab_comparison is not None:
    with tab_comparison:
        from ui.algorithm_comparison import render_algorithm_comparison
        render_algorithm_comparison(
            _enabled_algos,
            {n: _algo_registry[n].display_name for n in _enabled_algos},
        )
