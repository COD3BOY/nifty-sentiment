"""System Health monitoring tab for the Streamlit dashboard.

Displays startup checks, API health (circuit breakers), data freshness,
and the trade audit trail.
"""

import logging
import time
from datetime import datetime, timedelta, timezone

import streamlit as st

logger = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))


def _staleness_indicator(ts: float | None, label: str) -> None:
    """Display a metric with green/yellow/red staleness based on age."""
    if ts is None or ts <= 0:
        st.metric(label, "No data", delta="unknown")
        return

    age_seconds = time.time() - ts
    if age_seconds < 300:  # <5 min
        color = "normal"
        status = f"{int(age_seconds)}s ago"
    elif age_seconds < 1800:  # <30 min
        color = "off"
        status = f"{int(age_seconds // 60)}m ago"
    else:
        color = "inverse"
        status = f"{int(age_seconds // 60)}m ago"

    st.metric(label, status, delta="Fresh" if age_seconds < 300 else "Stale", delta_color=color)


def render_system_health_tab() -> None:
    """Render the System Health monitoring tab."""

    st.title("System Health")
    st.caption("Startup checks, API health, data freshness, and trade audit trail")

    # ----------------------------------------------------------------
    # Section 1: Startup Checks
    # ----------------------------------------------------------------
    st.subheader("Startup Checks")

    startup_checks = st.session_state.get("startup_checks", [])
    if not startup_checks:
        st.info("No startup check results available.")
    else:
        cols = st.columns(len(startup_checks))
        for i, check in enumerate(startup_checks):
            with cols[i]:
                icon = "PASS" if check["ok"] else "FAIL"
                if check["ok"]:
                    st.success(f"**{check['name']}**: {icon}")
                elif check.get("critical"):
                    st.error(f"**{check['name']}**: {icon}")
                else:
                    st.warning(f"**{check['name']}**: {icon}")
                st.caption(check["message"])

    st.divider()

    # ----------------------------------------------------------------
    # Section 2: API Health (Circuit Breakers)
    # ----------------------------------------------------------------
    st.subheader("API Health")

    col_kite, col_cb = st.columns(2)

    with col_kite:
        st.markdown("**Kite Token Status**")
        if st.button("Test Kite Token", key="test_kite_token"):
            with st.spinner("Testing Kite token..."):
                try:
                    from core.nse_fetcher import is_kite_token_valid
                    valid = is_kite_token_valid()
                    if valid:
                        st.success("Kite token is valid")
                    else:
                        st.error("Kite token is invalid or credentials not set")
                except Exception as e:
                    st.error(f"Kite token check failed: {e}")
        else:
            st.caption("Click to test Kite API connectivity")

    with col_cb:
        st.markdown("**Circuit Breaker States**")
        try:
            from core.circuit_breaker import circuit_breaker_registry
            states = circuit_breaker_registry.get_all_states()
            if not states:
                st.caption("No circuit breakers registered yet (they initialize on first API call)")
            else:
                for name, info in states.items():
                    state = info["state"]
                    failures = info["failure_count"]
                    if state == "CLOSED":
                        st.success(f"**{name}**: CLOSED (failures: {failures})")
                    elif state == "HALF_OPEN":
                        st.warning(f"**{name}**: HALF_OPEN (failures: {failures})")
                    else:
                        st.error(f"**{name}**: OPEN (failures: {failures})")
        except Exception as e:
            st.error(f"Failed to read circuit breaker states: {e}")

    st.divider()

    # ----------------------------------------------------------------
    # Section 3: Data Freshness
    # ----------------------------------------------------------------
    st.subheader("Data Freshness")

    cols = st.columns(4)

    with cols[0]:
        chain_ts = st.session_state.get("last_chain_fetch_ts")
        _staleness_indicator(chain_ts, "Option Chain")

    with cols[1]:
        candle_ts = st.session_state.get("last_candle_fetch_ts")
        _staleness_indicator(candle_ts, "Candle Data")

    with cols[2]:
        # Check vol distribution freshness from cache
        vol_ts = None
        try:
            from core.vol_distribution import _mem_cache
            if "vol_dist" in _mem_cache:
                _, vol_ts = _mem_cache["vol_dist"]
        except Exception:
            pass
        _staleness_indicator(vol_ts, "Vol Distribution")

    with cols[3]:
        pm_ts = st.session_state.get("pm_last_refresh", 0.0)
        _staleness_indicator(pm_ts if pm_ts > 0 else None, "Sentiment")

    st.divider()

    # ----------------------------------------------------------------
    # Section 4: Trade Audit Trail
    # ----------------------------------------------------------------
    st.subheader("Trade Audit Trail")
    st.caption("Last 10 trade decisions across all algorithms")

    try:
        from core.trade_audit import get_recent_decisions
        decisions = get_recent_decisions(10)
        if not decisions:
            st.info("No trade decisions logged yet.")
        else:
            for d in reversed(decisions):
                action = d.get("action", "?")
                if action == "OPEN":
                    icon = "+"
                elif action == "CLOSE":
                    icon = "-"
                else:
                    icon = "x"

                algo = d.get("algorithm", "?")
                strategy = d.get("strategy", "?")
                ts = d.get("timestamp", "")
                score = d.get("score")
                reason = d.get("reject_reason") or d.get("exit_reason") or ""
                pnl = d.get("pnl")

                header = f"[{icon}] {action} | {algo} | {strategy}"
                if score is not None:
                    header += f" | score={score:.2f}"
                if pnl is not None:
                    header += f" | PnL={pnl:+,.0f}"

                with st.expander(header, expanded=False):
                    st.text(f"Timestamp: {ts}")
                    st.text(f"Algorithm: {algo}")
                    st.text(f"Action: {action}")
                    st.text(f"Strategy: {strategy}")
                    if d.get("vol_regime"):
                        st.text(f"Vol Regime: {d['vol_regime']}")
                    if reason:
                        st.text(f"Reason: {reason}")
                    if d.get("gate_checks"):
                        st.json(d["gate_checks"])
                    if d.get("dynamic_params"):
                        st.json(d["dynamic_params"])
    except Exception as e:
        st.error(f"Failed to load audit trail: {e}")
