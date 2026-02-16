"""Streamlit dashboard for simulation visualization.

Launch with::

    streamlit run simulation/sim_dashboard.py
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Simulation Dashboard", layout="wide")

RESULTS_DIR = Path("data/sim_results")


def main():
    st.title("Market Simulation Dashboard")

    tab1, tab2, tab3 = st.tabs([
        "Run Simulation",
        "View Results",
        "Compare Algorithms",
    ])

    with tab1:
        _render_run_tab()

    with tab2:
        _render_results_tab()

    with tab3:
        _render_compare_tab()


# ---------------------------------------------------------------------------
# Tab 1: Run Simulation
# ---------------------------------------------------------------------------

def _render_run_tab():
    from simulation.scenario_models import list_scenarios

    st.header("Run Simulation")

    scenarios = list_scenarios()
    if not scenarios:
        st.warning("No scenarios found in simulation/scenarios/")
        return

    col1, col2, col3 = st.columns(3)

    with col1:
        algo = st.selectbox(
            "Algorithm",
            ["sentinel", "jarvis", "optimus", "atlas"],
            key="run_algo",
        )

    with col2:
        scenario_names = [s["name"] for s in scenarios]
        selected = st.selectbox("Scenario", scenario_names, key="run_scenario")

    with col3:
        seed = st.number_input("Seed", value=42, min_value=1, key="run_seed")

    # Show scenario description
    for sc in scenarios:
        if sc["name"] == selected:
            st.info(sc["description"].strip()[:200] if sc["description"] else "No description")
            break

    sim_date = st.date_input("Simulation Date", value=date.today(), key="run_date")

    if st.button("Run Simulation", type="primary", key="run_btn"):
        _execute_simulation(algo, selected, int(seed), sim_date)


def _execute_simulation(algo: str, scenario_path: str, seed: int, sim_date: date):
    from simulation.runner import run_day, run_multi_day
    from simulation.results import SimulationResult
    from simulation.scenario_models import load_scenario

    sc = load_scenario(scenario_path)

    progress_bar = st.progress(0, text="Initializing...")
    status_text = st.empty()

    # Track progress via tick count
    total_ticks = 375 * sc.num_days
    ticks_done = [0]

    def tick_callback(tick_idx, state, spot, vix, sim_time):
        ticks_done[0] += 1
        pct = min(ticks_done[0] / total_ticks, 1.0)
        progress_bar.progress(pct, text=f"Day {sim_time.date()} | {sim_time.strftime('%H:%M')} | Spot: {spot:,.0f} | VIX: {vix:.1f}")

    try:
        if sc.num_days > 1:
            day_results = run_multi_day(sc, algo, sim_date, seed, speed=1e9, callback=tick_callback)
        else:
            result = run_day(sc, algo, sim_date, seed, speed=1e9, callback=tick_callback)
            day_results = [result]

        progress_bar.progress(1.0, text="Complete!")

        sim_result = SimulationResult(
            algo_name=algo,
            scenario_name=sc.name,
            seed=seed,
            day_results=day_results,
        )
        path = sim_result.save()

        metrics = sim_result.compute_metrics()
        _show_result_summary(sim_result, metrics)

        st.success(f"Results saved to {path}")

    except Exception as e:
        st.error(f"Simulation failed: {e}")
        import traceback
        st.code(traceback.format_exc())


# ---------------------------------------------------------------------------
# Tab 2: View Results
# ---------------------------------------------------------------------------

def _render_results_tab():
    st.header("Simulation Results")

    if not RESULTS_DIR.exists():
        st.info("No results yet. Run a simulation first.")
        return

    result_files = sorted(RESULTS_DIR.glob("*.json"), reverse=True)
    if not result_files:
        st.info("No result files found.")
        return

    selected_file = st.selectbox(
        "Select Result",
        result_files,
        format_func=lambda p: p.stem,
        key="view_result",
    )

    if selected_file:
        with open(selected_file) as f:
            data = json.load(f)

        _show_loaded_result(data)


def _show_loaded_result(data: dict):
    """Display a loaded result from JSON."""
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Algorithm", data["algo_name"])
    with col2:
        st.metric("Scenario", data["scenario_name"])
    with col3:
        st.metric("Seed", data.get("seed", "N/A"))

    metrics = data.get("metrics", {})
    _show_metrics_row(metrics)

    # Trades table
    trades = data.get("trades", [])
    if trades:
        st.subheader("Trade Log")
        df = pd.DataFrame(trades)
        display_cols = [
            c for c in ["id", "strategy", "strategy_type", "direction_bias",
                         "entry_time", "exit_time", "exit_reason",
                         "realized_pnl", "net_pnl", "max_drawdown", "max_favorable"]
            if c in df.columns
        ]
        if display_cols:
            trade_df = df[display_cols].copy()
            if "net_pnl" in trade_df.columns:
                trade_df["net_pnl"] = trade_df["net_pnl"].round(2)
            st.dataframe(trade_df, width="stretch")
    else:
        st.info("No trades were generated in this simulation.")

    # Day summaries
    day_summaries = data.get("day_summaries", [])
    if day_summaries:
        st.subheader("Daily Summary")
        ds_df = pd.DataFrame(day_summaries)
        st.dataframe(ds_df, width="stretch")


def _show_metrics_row(metrics: dict):
    """Display performance metrics as Streamlit metrics."""
    col1, col2, col3, col4 = st.columns(4)

    pnl = metrics.get("total_pnl", 0)
    with col1:
        st.metric("Total P&L", f"{pnl:+,.0f}", delta_color="normal" if pnl >= 0 else "inverse")
    with col2:
        st.metric("Trades", metrics.get("num_trades", 0))
    with col3:
        st.metric("Win Rate", f"{metrics.get('win_rate', 0):.1f}%")
    with col4:
        st.metric("Sharpe", f"{metrics.get('sharpe_ratio', 0):.3f}")

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.metric("Avg Win", f"{metrics.get('avg_win', 0):+,.0f}")
    with col6:
        st.metric("Avg Loss", f"{metrics.get('avg_loss', 0):+,.0f}")
    with col7:
        st.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0):,.0f}")
    with col8:
        st.metric("Profit Factor", f"{metrics.get('profit_factor', 0):.2f}")


# ---------------------------------------------------------------------------
# Tab 3: Compare Algorithms
# ---------------------------------------------------------------------------

def _render_compare_tab():
    st.header("Algorithm Comparison")

    if not RESULTS_DIR.exists():
        st.info("No results yet. Run simulations first.")
        return

    result_files = sorted(RESULTS_DIR.glob("*.json"))
    if not result_files:
        st.info("No result files found.")
        return

    # Group by scenario
    results_by_scenario = {}
    for f in result_files:
        try:
            with open(f) as fh:
                data = json.load(fh)
            scenario = data.get("scenario_name", "unknown")
            algo = data.get("algo_name", "unknown")
            if scenario not in results_by_scenario:
                results_by_scenario[scenario] = {}
            results_by_scenario[scenario][algo] = data
        except Exception:
            continue

    if not results_by_scenario:
        st.info("No valid result files found.")
        return

    selected_scenario = st.selectbox(
        "Select Scenario",
        list(results_by_scenario.keys()),
        key="compare_scenario",
    )

    if selected_scenario:
        algo_results = results_by_scenario[selected_scenario]

        if len(algo_results) < 2:
            st.info(f"Only {len(algo_results)} algorithm(s) have results for this scenario. Need at least 2 for comparison.")

        # Comparison table
        rows = []
        for algo, data in sorted(algo_results.items()):
            m = data.get("metrics", {})
            rows.append({
                "Algorithm": algo,
                "Trades": m.get("num_trades", 0),
                "Total P&L": m.get("total_pnl", 0),
                "Win Rate %": round(m.get("win_rate", 0), 1),
                "Avg Win": round(m.get("avg_win", 0)),
                "Avg Loss": round(m.get("avg_loss", 0)),
                "Max Drawdown": round(m.get("max_drawdown", 0)),
                "Sharpe": round(m.get("sharpe_ratio", 0), 3),
                "Profit Factor": round(m.get("profit_factor", 0), 2),
            })

        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(df, width="stretch")

        # Side-by-side P&L
        st.subheader("P&L Comparison")
        pnl_data = {
            algo: data["metrics"].get("total_pnl", 0)
            for algo, data in algo_results.items()
        }
        st.bar_chart(pd.Series(pnl_data, name="Total P&L"))


def _show_result_summary(sim_result, metrics):
    """Show results summary for a just-completed simulation."""
    st.subheader("Results")
    _show_metrics_row(metrics.to_dict())

    trades = sim_result.all_trades
    if trades:
        st.subheader(f"Trade Log ({len(trades)} trades)")
        trade_data = []
        for t in trades:
            trade_data.append({
                "ID": t.id[:8],
                "Strategy": t.strategy,
                "Type": t.strategy_type,
                "Bias": t.direction_bias,
                "Entry": t.entry_time.strftime("%H:%M") if t.entry_time else "",
                "Exit": t.exit_time.strftime("%H:%M") if t.exit_time else "",
                "Reason": t.exit_reason,
                "P&L": round(t.net_pnl, 2),
                "Max DD": round(t.max_drawdown, 2),
                "Max Fav": round(t.max_favorable, 2),
            })
        st.dataframe(pd.DataFrame(trade_data), width="stretch")

    # Tick-level price chart
    for dr in sim_result.day_results:
        st.subheader(f"Price Path — {dr.sim_date}")
        price_df = pd.DataFrame({
            "Close": dr.candles["Close"].values,
        }, index=dr.candles.index)
        st.line_chart(price_df)

        if dr.vix_path is not None:
            vix_df = pd.DataFrame({
                "VIX": dr.vix_path,
            }, index=dr.candles.index[:len(dr.vix_path)])
            st.line_chart(vix_df)


if __name__ == "__main__":
    main()
