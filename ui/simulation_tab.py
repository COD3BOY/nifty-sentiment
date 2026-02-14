"""Simulation tab for the main NIFTY dashboard.

Provides a UI for running simulations, viewing results, and comparing
algorithms — integrated into the main Streamlit app.

Two-phase execution pattern:
  Button click → store params in session_state + set sim_running=True + st.rerun()
  Next rerun  → sim_running is True BEFORE options desk renders (skips st_autorefresh)
              → simulation tab detects pending params and executes the simulation
This prevents the options desk's 60s autorefresh from killing in-progress simulations.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

RESULTS_DIR = Path("data/sim_results")


def render_simulation_tab():
    """Render the full simulation tab content."""
    st.header("Market Simulation & Stress Testing")
    st.caption("Run algorithms against synthetic market scenarios for offline testing")

    # --- Phase 2: execute pending simulation (set up on previous rerun) ---
    _check_and_run_pending_sim()
    _check_and_run_pending_run_all()

    sub_tab1, sub_tab2, sub_tab3, sub_tab4 = st.tabs([
        "Run Simulation",
        "View Results",
        "Compare Algorithms",
        "Run All",
    ])

    with sub_tab1:
        _render_run_section()

    with sub_tab2:
        _render_results_section()

    with sub_tab3:
        _render_compare_section()

    with sub_tab4:
        _render_run_all_section()


# ---------------------------------------------------------------------------
# Two-phase simulation execution
# ---------------------------------------------------------------------------

def _check_and_run_pending_sim():
    """Phase 2: if a single simulation was requested on the previous rerun, execute it now."""
    params = st.session_state.get("_sim_pending_params")
    if params is None:
        return

    # Clear immediately so it doesn't re-trigger on the next rerun
    del st.session_state["_sim_pending_params"]

    _execute_simulation(params["algo"], params["scenario"], params["seed"], params["sim_date"])


def _check_and_run_pending_run_all():
    """Phase 2: if a batch run was requested on the previous rerun, execute it now."""
    params = st.session_state.get("_sim_pending_run_all")
    if params is None:
        return

    del st.session_state["_sim_pending_run_all"]

    algos = params.get("algos", _ALGOS)
    _execute_run_all(algos, params["scenario_names"], params["seed"], params["sim_date"])


# ---------------------------------------------------------------------------
# Run Simulation
# ---------------------------------------------------------------------------

def _render_run_section():
    from simulation.scenario_models import list_scenarios

    scenarios = list_scenarios()
    if not scenarios:
        st.warning("No scenarios found in `simulation/scenarios/`")
        return

    col1, col2 = st.columns(2)

    with col1:
        algo = st.selectbox(
            "Algorithm",
            ["sentinel", "jarvis", "optimus", "atlas"],
            key="sim_run_algo",
        )
        seed = st.number_input(
            "Seed", value=42, min_value=1, key="sim_run_seed",
            help="Random seed for reproducibility. Same seed + same scenario = identical results every time. Change the seed to see different random market paths.",
        )

    with col2:
        # Group scenarios by category
        categories = sorted(set(s["category"] for s in scenarios))
        selected_cat = st.selectbox("Category", categories, key="sim_run_cat")
        filtered = [s for s in scenarios if s["category"] == selected_cat]
        scenario_names = [s["name"] for s in filtered]
        selected_scenario = st.selectbox("Scenario", scenario_names, key="sim_run_scenario")

    # Show description
    for sc in scenarios:
        if sc["name"] == selected_scenario:
            desc = sc["description"].strip()
            if desc:
                st.info(desc[:300])
            break

    col_date, col_btn = st.columns([1, 1])
    with col_date:
        sim_date = st.date_input(
            "Simulation Date",
            value=_next_weekday(),
            key="sim_run_date",
        )
    with col_btn:
        st.write("")  # spacer
        run_clicked = st.button(
            "Run Simulation",
            type="primary",
            key="sim_run_btn",
            use_container_width=True,
        )

    if run_clicked:
        # Phase 1: store params, set flag, rerun.
        # The simulation will execute on the NEXT rerun (phase 2),
        # after st_autorefresh has been skipped.
        st.session_state.sim_running = True
        st.session_state["_sim_pending_params"] = {
            "algo": algo,
            "scenario": selected_scenario,
            "seed": int(seed),
            "sim_date": sim_date,
        }
        st.rerun()


def _execute_simulation(algo: str, scenario_path: str, seed: int, sim_date: date):
    from simulation.runner import run_day, run_multi_day
    from simulation.results import SimulationResult
    from simulation.scenario_models import load_scenario

    try:
        sc = load_scenario(scenario_path)
    except Exception as e:
        st.session_state.sim_running = False
        st.error(f"Failed to load scenario: {e}")
        return

    total_ticks = 375 * sc.num_days
    ticks_done = [0]
    progress_bar = st.progress(0, text="Starting simulation...")

    def tick_callback(tick_idx, state, spot, vix, sim_time):
        ticks_done[0] += 1
        pct = min(ticks_done[0] / total_ticks, 1.0)
        progress_bar.progress(
            pct,
            text=(
                f"Day {sim_time.date()} | {sim_time.strftime('%H:%M')} | "
                f"Spot: {spot:,.0f} | VIX: {vix:.1f} | "
                f"Positions: {len(state.open_positions)} | "
                f"P&L: {state.total_realized_pnl:+,.0f}"
            ),
        )

    try:
        if sc.num_days > 1:
            day_results = run_multi_day(
                sc, algo, sim_date, seed, speed=1e9, callback=tick_callback,
            )
        else:
            result = run_day(
                sc, algo, sim_date, seed, speed=1e9, callback=tick_callback,
            )
            day_results = [result]

        progress_bar.progress(1.0, text="Simulation complete!")

        sim_result = SimulationResult(
            algo_name=algo,
            scenario_name=sc.name,
            seed=seed,
            day_results=day_results,
        )
        path = sim_result.save()
        metrics = sim_result.compute_metrics()

        st.success(f"Saved to `{path}`")
        _show_metrics(metrics)
        _show_trades(sim_result.all_trades)
        _show_price_charts(day_results)

    except Exception as e:
        st.error(f"Simulation failed: {e}")
        import traceback
        st.code(traceback.format_exc())
    finally:
        st.session_state.sim_running = False


# ---------------------------------------------------------------------------
# View Results
# ---------------------------------------------------------------------------

def _render_results_section():
    if not RESULTS_DIR.exists():
        st.info("No results yet. Run a simulation first.")
        return

    result_files = sorted(RESULTS_DIR.glob("*.json"), reverse=True)
    if not result_files:
        st.info("No result files found in `data/sim_results/`.")
        return

    selected = st.selectbox(
        "Select Result",
        result_files,
        format_func=lambda p: p.stem,
        key="sim_view_result",
    )

    if selected:
        with open(selected) as f:
            data = json.load(f)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Algorithm", data.get("algo_name", "?"))
        with col2:
            st.metric("Scenario", data.get("scenario_name", "?"))
        with col3:
            st.metric("Seed", data.get("seed", "?"))
        with col4:
            st.metric("Days", data.get("num_days", 1))

        metrics = data.get("metrics", {})
        _show_metrics_dict(metrics)

        # Trades
        trades = data.get("trades", [])
        if trades:
            st.subheader(f"Trades ({len(trades)})")
            df = pd.DataFrame(trades)
            cols = [c for c in [
                "id", "strategy", "strategy_type", "direction_bias",
                "entry_time", "exit_time", "exit_reason",
                "realized_pnl", "net_pnl",
            ] if c in df.columns]
            if cols:
                display_df = df[cols].copy()
                for c in ["realized_pnl", "net_pnl"]:
                    if c in display_df.columns:
                        display_df[c] = display_df[c].round(2)
                st.dataframe(display_df, use_container_width=True)
        else:
            st.info("No trades generated.")

        # Day summaries
        day_summaries = data.get("day_summaries", [])
        if day_summaries and len(day_summaries) > 1:
            st.subheader("Daily Breakdown")
            st.dataframe(pd.DataFrame(day_summaries), use_container_width=True)

        # Delete button
        if st.button("Delete this result", key="sim_delete_btn"):
            selected.unlink()
            st.rerun()


# ---------------------------------------------------------------------------
# Compare Algorithms
# ---------------------------------------------------------------------------

def _render_compare_section():
    if not RESULTS_DIR.exists():
        st.info("No results yet.")
        return

    result_files = sorted(RESULTS_DIR.glob("*.json"))
    if not result_files:
        st.info("No result files found.")
        return

    # Group by scenario
    by_scenario: dict[str, dict[str, dict]] = {}
    for f in result_files:
        try:
            with open(f) as fh:
                data = json.load(fh)
            sc = data.get("scenario_name", "?")
            algo = data.get("algo_name", "?")
            by_scenario.setdefault(sc, {})[algo] = data
        except Exception:
            continue

    if not by_scenario:
        st.info("No valid results to compare.")
        return

    selected_sc = st.selectbox(
        "Scenario",
        sorted(by_scenario.keys()),
        key="sim_compare_scenario",
    )

    if selected_sc:
        algo_results = by_scenario[selected_sc]

        if len(algo_results) < 2:
            st.info(
                f"Only **{list(algo_results.keys())[0]}** has results for "
                f"this scenario. Run more algorithms to compare."
            )

        # Table
        rows = []
        for algo, data in sorted(algo_results.items()):
            m = data.get("metrics", {})
            rows.append({
                "Algorithm": algo,
                "Trades": m.get("num_trades", 0),
                "Total P&L": round(m.get("total_pnl", 0)),
                "Win Rate %": round(m.get("win_rate", 0), 1),
                "Avg Win": round(m.get("avg_win", 0)),
                "Avg Loss": round(m.get("avg_loss", 0)),
                "Max DD": round(m.get("max_drawdown", 0)),
                "Sharpe": round(m.get("sharpe_ratio", 0), 3),
                "Profit Factor": round(m.get("profit_factor", 0), 2),
            })

        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

            # P&L bar chart
            pnl_series = pd.Series(
                {r["Algorithm"]: r["Total P&L"] for r in rows},
                name="Total P&L",
            )
            st.bar_chart(pnl_series)


# ---------------------------------------------------------------------------
# Run All Simulations
# ---------------------------------------------------------------------------

_ALGOS = ["sentinel", "jarvis", "optimus", "atlas"]


def _render_run_all_section():
    """Batch-run algorithms across all scenarios."""
    from simulation.scenario_models import list_scenarios

    scenarios = list_scenarios()
    if not scenarios:
        st.warning("No scenarios found in `simulation/scenarios/`")
        return

    st.subheader("Batch Simulation")
    st.caption(
        "Run one or all algorithms against every scenario. "
        "Results are saved to `data/sim_results/` for later comparison."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        algo_options = ["All Algorithms"] + _ALGOS
        selected_algo = st.selectbox(
            "Algorithm",
            algo_options,
            key="sim_run_all_algo",
        )
    with col2:
        seed = st.number_input(
            "Seed", value=42, min_value=1, key="sim_run_all_seed",
            help="Random seed for reproducibility. Same seed + same scenario = identical results every time.",
        )
    with col3:
        sim_date = st.date_input(
            "Simulation Date",
            value=_next_weekday(),
            key="sim_run_all_date",
        )

    scenario_names = [s["name"] for s in scenarios]
    algos_to_run = _ALGOS if selected_algo == "All Algorithms" else [selected_algo]
    total_runs = len(algos_to_run) * len(scenario_names)
    st.info(f"Will run **{len(algos_to_run)}** algorithm(s) x **{len(scenario_names)}** scenarios = **{total_runs}** simulations")

    run_clicked = st.button(
        "Run All Simulations",
        type="primary",
        key="sim_run_all_btn",
        use_container_width=True,
    )

    if not run_clicked:
        return

    # Phase 1: store params, set flag, rerun
    st.session_state.sim_running = True
    st.session_state["_sim_pending_run_all"] = {
        "algos": algos_to_run,
        "scenario_names": scenario_names,
        "seed": int(seed),
        "sim_date": sim_date,
    }
    st.rerun()


def _execute_run_all(algos: list[str], scenario_names: list[str], seed: int, sim_date: date):
    """Execute algo x scenario combinations and display summary."""
    from simulation.runner import run_day, run_multi_day
    from simulation.results import SimulationResult
    from simulation.scenario_models import load_scenario

    total_runs = len(algos) * len(scenario_names)
    progress_bar = st.progress(0, text="Starting batch...")
    summary_rows = []
    errors = []

    try:
        run_idx = 0
        for algo in algos:
            for sc_name in scenario_names:
                run_idx += 1
                progress_bar.progress(
                    run_idx / total_runs,
                    text=f"Running **{algo}** on **{sc_name}**... ({run_idx}/{total_runs})",
                )

                try:
                    sc = load_scenario(sc_name)
                    if sc.num_days > 1:
                        day_results = run_multi_day(sc, algo, sim_date, seed, speed=1e9)
                    else:
                        result = run_day(sc, algo, sim_date, seed, speed=1e9)
                        day_results = [result]

                    sim_result = SimulationResult(
                        algo_name=algo,
                        scenario_name=sc_name,
                        seed=seed,
                        day_results=day_results,
                    )
                    sim_result.save()
                    metrics = sim_result.compute_metrics()

                    summary_rows.append({
                        "Algorithm": algo,
                        "Scenario": sc_name,
                        "Trades": metrics.num_trades,
                        "Total P&L": round(metrics.total_pnl),
                        "Win Rate %": round(metrics.win_rate, 1),
                        "Avg Win": round(metrics.avg_win),
                        "Avg Loss": round(metrics.avg_loss),
                        "Max DD": round(metrics.max_drawdown),
                        "Sharpe": round(metrics.sharpe_ratio, 3),
                        "Profit Factor": round(metrics.profit_factor, 2),
                    })
                except Exception as e:
                    errors.append(f"{algo}/{sc_name}: {e}")
                    summary_rows.append({
                        "Algorithm": algo,
                        "Scenario": sc_name,
                        "Trades": 0,
                        "Total P&L": 0,
                        "Win Rate %": 0,
                        "Avg Win": 0,
                        "Avg Loss": 0,
                        "Max DD": 0,
                        "Sharpe": 0,
                        "Profit Factor": 0,
                    })
    finally:
        st.session_state.sim_running = False

    progress_bar.progress(1.0, text=f"Complete! {total_runs} simulations finished.")

    if errors:
        with st.expander(f"{len(errors)} errors", expanded=False):
            for err in errors:
                st.text(err)

    if summary_rows:
        _show_run_all_summary(summary_rows)


def _show_run_all_summary(summary_rows: list[dict]):
    """Display batch simulation summary table and heatmap."""
    df = pd.DataFrame(summary_rows)
    st.subheader("Summary")

    styled = df.style.applymap(
        _pnl_color, subset=["Total P&L"],
    ).format({
        "Total P&L": "{:+,}",
        "Avg Win": "{:+,}",
        "Avg Loss": "{:+,}",
        "Max DD": "{:,}",
        "Sharpe": "{:.3f}",
        "Profit Factor": "{:.2f}",
        "Win Rate %": "{:.1f}",
    })
    st.dataframe(styled, use_container_width=True, height=min(len(df) * 35 + 40, 800))

    st.subheader("P&L Heatmap")
    pivot = df.pivot_table(
        index="Algorithm", columns="Scenario", values="Total P&L", aggfunc="sum",
    )
    styled_pivot = pivot.style.applymap(_pnl_color).format("{:+,.0f}")
    st.dataframe(styled_pivot, use_container_width=True)


def _pnl_color(val) -> str:
    """Return CSS color string: green for profit, red for loss, neutral for zero."""
    if not isinstance(val, (int, float)):
        return ""
    if val > 0:
        return "color: #2e7d32"  # green
    if val < 0:
        return "color: #c62828"  # red
    return ""


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _next_weekday() -> date:
    d = date.today()
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def _show_metrics(metrics):
    """Show PerformanceMetrics object."""
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        pnl = metrics.total_pnl
        st.metric(
            "Total P&L",
            f"{pnl:+,.0f}",
            delta_color="normal" if pnl >= 0 else "inverse",
        )
    with col2:
        st.metric("Trades", metrics.num_trades)
    with col3:
        st.metric("Win Rate", f"{metrics.win_rate:.1f}%")
    with col4:
        st.metric("Sharpe", f"{metrics.sharpe_ratio:.3f}")

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.metric("Avg Win", f"{metrics.avg_win:+,.0f}")
    with col6:
        st.metric("Avg Loss", f"{metrics.avg_loss:+,.0f}")
    with col7:
        st.metric("Max Drawdown", f"{metrics.max_drawdown:,.0f}")
    with col8:
        st.metric("Profit Factor", f"{metrics.profit_factor:.2f}")


def _show_metrics_dict(m: dict):
    """Show metrics from a dict (loaded from JSON)."""
    col1, col2, col3, col4 = st.columns(4)
    pnl = m.get("total_pnl", 0)
    with col1:
        st.metric(
            "Total P&L",
            f"{pnl:+,.0f}",
            delta_color="normal" if pnl >= 0 else "inverse",
        )
    with col2:
        st.metric("Trades", m.get("num_trades", 0))
    with col3:
        st.metric("Win Rate", f"{m.get('win_rate', 0):.1f}%")
    with col4:
        st.metric("Sharpe", f"{m.get('sharpe_ratio', 0):.3f}")

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.metric("Avg Win", f"{m.get('avg_win', 0):+,.0f}")
    with col6:
        st.metric("Avg Loss", f"{m.get('avg_loss', 0):+,.0f}")
    with col7:
        st.metric("Max Drawdown", f"{m.get('max_drawdown', 0):,.0f}")
    with col8:
        st.metric("Profit Factor", f"{m.get('profit_factor', 0):.2f}")


def _show_trades(trades):
    """Show trade records from a simulation."""
    if not trades:
        st.info("No trades generated.")
        return

    st.subheader(f"Trades ({len(trades)})")
    rows = []
    for t in trades:
        rows.append({
            "ID": t.id[:8],
            "Strategy": t.strategy,
            "Type": t.strategy_type,
            "Bias": t.direction_bias,
            "Entry": t.entry_time.strftime("%H:%M") if t.entry_time else "",
            "Exit": t.exit_time.strftime("%H:%M") if t.exit_time else "",
            "Reason": t.exit_reason,
            "P&L": round(t.net_pnl, 2),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


def _show_price_charts(day_results):
    """Show price + VIX charts for each day."""
    for dr in day_results:
        st.subheader(f"Price Path — {dr.sim_date}")
        price_df = pd.DataFrame(
            {"Close": dr.candles["Close"].values},
            index=dr.candles.index,
        )
        st.line_chart(price_df)

        if dr.vix_path is not None and len(dr.vix_path) > 0:
            vix_df = pd.DataFrame(
                {"VIX": dr.vix_path},
                index=dr.candles.index[:len(dr.vix_path)],
            )
            st.line_chart(vix_df)
