"""CLI entry point for the market simulation system.

Usage::

    # Run single scenario
    python -m simulation.cli run --algo atlas --scenario crisis/flash_crash --seed 42

    # Run all scenarios in a category
    python -m simulation.cli run-all --algo jarvis --category crisis --seeds 42,43,44

    # Compare algorithms on the same scenario
    python -m simulation.cli compare --algos jarvis,optimus,atlas --scenario crisis/flash_crash

    # Feed results into self-improvement pipeline
    python -m simulation.cli review --result data/sim_results/atlas_crisis_flash_crash_42.json

    # List all available scenarios
    python -m simulation.cli list-scenarios
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

import click

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("simulation")


@click.group()
def cli():
    """Market Simulation & Stress Testing System."""
    pass


@cli.command()
@click.option("--algo", required=True, help="Algorithm name (sentinel/jarvis/optimus/atlas)")
@click.option("--scenario", required=True, help="Scenario path (e.g. crisis/flash_crash)")
@click.option("--seed", default=42, type=int, help="RNG seed for reproducibility")
@click.option("--date", "sim_date", default=None, help="Simulation date (YYYY-MM-DD)")
@click.option("--speed", default=6.0, type=float, help="Clock speed multiplier")
@click.option("--no-save", is_flag=True, help="Don't save results to disk")
def run(algo: str, scenario: str, seed: int, sim_date: str | None, speed: float, no_save: bool):
    """Run a single scenario against an algorithm."""
    from simulation.runner import run_day, run_multi_day
    from simulation.results import SimulationResult
    from simulation.scenario_models import load_scenario

    sc = load_scenario(scenario)
    d = _parse_date(sim_date)

    click.echo(f"Scenario: {sc.name} ({sc.category})")
    click.echo(f"Algorithm: {algo}")
    click.echo(f"Seed: {seed}")
    click.echo(f"Date: {d}")
    click.echo(f"Days: {sc.num_days}")
    click.echo()

    if sc.num_days > 1:
        day_results = run_multi_day(sc, algo, d, seed, speed=speed)
    else:
        result = run_day(sc, algo, d, seed, speed=speed)
        day_results = [result]

    sim_result = SimulationResult(
        algo_name=algo,
        scenario_name=sc.name,
        seed=seed,
        day_results=day_results,
    )

    metrics = sim_result.compute_metrics()
    _print_metrics(metrics, sim_result)

    if not no_save:
        path = sim_result.save()
        click.echo(f"\nResults saved to: {path}")


@cli.command("run-all")
@click.option("--algo", required=True, help="Algorithm name")
@click.option("--category", default=None, help="Filter by scenario category")
@click.option("--seeds", default="42", help="Comma-separated seeds")
@click.option("--date", "sim_date", default=None, help="Simulation date")
@click.option("--speed", default=6.0, type=float, help="Clock speed")
def run_all(algo: str, category: str | None, seeds: str, sim_date: str | None, speed: float):
    """Run all scenarios (optionally filtered by category) for an algorithm."""
    from simulation.runner import run_day, run_multi_day
    from simulation.results import SimulationResult
    from simulation.scenario_models import list_scenarios, load_scenario

    seed_list = [int(s.strip()) for s in seeds.split(",")]
    d = _parse_date(sim_date)

    scenarios = list_scenarios()
    if category:
        scenarios = [s for s in scenarios if s["category"] == category]

    if not scenarios:
        click.echo("No matching scenarios found.")
        return

    click.echo(f"Running {len(scenarios)} scenarios × {len(seed_list)} seeds for {algo}")
    click.echo()

    for sc_info in scenarios:
        sc = load_scenario(sc_info["name"])
        for seed in seed_list:
            click.echo(f"  {sc_info['name']} (seed={seed})...", nl=False)

            try:
                if sc.num_days > 1:
                    day_results = run_multi_day(sc, algo, d, seed, speed=speed)
                else:
                    result = run_day(sc, algo, d, seed, speed=speed)
                    day_results = [result]

                sim_result = SimulationResult(
                    algo_name=algo,
                    scenario_name=sc.name,
                    seed=seed,
                    day_results=day_results,
                )
                metrics = sim_result.compute_metrics()
                path = sim_result.save()

                pnl = metrics.total_pnl
                color = "green" if pnl >= 0 else "red"
                click.echo(click.style(
                    f" {metrics.num_trades} trades, P&L: {pnl:+,.0f}", fg=color,
                ))
            except Exception as e:
                click.echo(click.style(f" ERROR: {e}", fg="red"))


@cli.command()
@click.option("--algos", required=True, help="Comma-separated algorithm names")
@click.option("--scenario", required=True, help="Scenario path")
@click.option("--seed", default=42, type=int, help="RNG seed")
@click.option("--date", "sim_date", default=None, help="Simulation date")
@click.option("--speed", default=6.0, type=float, help="Clock speed")
def compare(algos: str, scenario: str, seed: int, sim_date: str | None, speed: float):
    """Compare multiple algorithms on the same scenario."""
    from simulation.runner import run_day, run_multi_day
    from simulation.results import SimulationResult
    from simulation.scenario_models import load_scenario

    algo_list = [a.strip() for a in algos.split(",")]
    sc = load_scenario(scenario)
    d = _parse_date(sim_date)

    click.echo(f"Scenario: {sc.name}")
    click.echo(f"Algorithms: {', '.join(algo_list)}")
    click.echo(f"Seed: {seed}")
    click.echo()

    results = {}
    for algo in algo_list:
        click.echo(f"Running {algo}...", nl=False)
        try:
            if sc.num_days > 1:
                day_results = run_multi_day(sc, algo, d, seed, speed=speed)
            else:
                result = run_day(sc, algo, d, seed, speed=speed)
                day_results = [result]

            sim_result = SimulationResult(
                algo_name=algo,
                scenario_name=sc.name,
                seed=seed,
                day_results=day_results,
            )
            sim_result.save()
            results[algo] = sim_result
            click.echo(" done")
        except Exception as e:
            click.echo(click.style(f" ERROR: {e}", fg="red"))

    # Print comparison table
    if results:
        click.echo()
        click.echo(f"{'Algorithm':<12} {'Trades':>7} {'P&L':>10} {'Win%':>7} {'MaxDD':>10} {'Sharpe':>8}")
        click.echo("-" * 60)
        for algo, sim_result in results.items():
            m = sim_result.compute_metrics()
            color = "green" if m.total_pnl >= 0 else "red"
            click.echo(
                f"{algo:<12} {m.num_trades:>7} "
                + click.style(f"{m.total_pnl:>+10,.0f}", fg=color)
                + f" {m.win_rate:>6.1f}% {m.max_drawdown:>+10,.0f} {m.sharpe_ratio:>8.2f}"
            )


@cli.command()
@click.option("--result", "result_path", required=True, help="Path to simulation result JSON")
def review(result_path: str):
    """Feed simulation results into the self-improvement pipeline."""
    from simulation.results import SimulationResult

    path = Path(result_path)
    if not path.exists():
        click.echo(f"File not found: {path}")
        sys.exit(1)

    summary = SimulationResult.load_summary(path)
    click.echo(f"Algorithm: {summary['algo_name']}")
    click.echo(f"Scenario: {summary['scenario_name']}")
    click.echo(f"Trades: {summary['metrics']['num_trades']}")

    # To run the full review, we need the actual trade records
    # The saved JSON has serialized trades that we can reconstruct
    trades_data = summary.get("trades", [])
    if not trades_data:
        click.echo("No trades to review.")
        return

    # Reconstruct minimal state for review
    from core.paper_trading_models import PaperTradingState, TradeRecord
    trade_records = [TradeRecord.model_validate(t) for t in trades_data]

    state = PaperTradingState(
        initial_capital=2_500_000,
        open_positions=[],
        trade_log=trade_records,
        total_realized_pnl=sum(t.realized_pnl for t in trade_records),
        total_execution_costs=sum(t.execution_cost for t in trade_records),
        is_auto_trading=True,
        last_open_refresh_ts=0.0,
        last_trade_opened_ts=0.0,
        pending_critiques=[],
        session_start_capital=2_500_000,
        consecutive_losses=0,
        peak_capital=2_500_000,
        trading_halted=False,
        weekly_start_capital=2_500_000,
        week_start_date="",
        daily_start_capital=2_500_000,
        session_date="",
        allocation_tracker={},
        vol_regime="sell_premium",
        vol_snapshot_ts=0.0,
        vol_dynamic_params={},
        trade_status_notes=[],
    )

    from analyzers.daily_review import run_daily_review
    session = run_daily_review(state, summary["algo_name"])

    click.echo(f"\nClassification Summary: {session.classification_summary}")
    click.echo(f"Signal Reliability entries: {len(session.signal_reliability)}")
    click.echo(f"Parameter Calibrations: {len(session.parameter_calibrations)}")

    for note in session.notes:
        click.echo(f"  - {note}")

    if session.classifications:
        click.echo("\nTrade Classifications:")
        for tc in session.classifications:
            color = {"A": "green", "B": "yellow", "C": "cyan", "D": "red"}.get(tc.category, "white")
            click.echo(click.style(
                f"  [{tc.category}] {tc.strategy} — entry_quality={tc.entry_quality_score:.1f}", fg=color,
            ))


@cli.command("list-scenarios")
def list_scenarios_cmd():
    """List all available scenarios."""
    from simulation.scenario_models import list_scenarios

    scenarios = list_scenarios()
    if not scenarios:
        click.echo("No scenarios found.")
        return

    current_cat = None
    for sc in scenarios:
        if sc["category"] != current_cat:
            current_cat = sc["category"]
            click.echo(f"\n{current_cat.upper()}")
            click.echo("-" * 40)
        desc = sc["description"].strip().split("\n")[0][:60] if sc["description"] else ""
        click.echo(f"  {sc['name']:<30} {desc}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_date(sim_date: str | None) -> date:
    """Parse date string or return next weekday."""
    if sim_date:
        return date.fromisoformat(sim_date)
    # Default: next Monday (to avoid weekend issues)
    d = date.today()
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def _print_metrics(metrics, sim_result):
    """Print a summary of simulation metrics."""
    click.echo("=" * 50)
    click.echo("SIMULATION RESULTS")
    click.echo("=" * 50)

    color = "green" if metrics.total_pnl >= 0 else "red"
    click.echo(f"Total P&L:      " + click.style(f"{metrics.total_pnl:+,.2f}", fg=color))
    click.echo(f"Trades:         {metrics.num_trades}")
    click.echo(f"Win Rate:       {metrics.win_rate:.1f}%")
    click.echo(f"Avg Win:        {metrics.avg_win:+,.2f}")
    click.echo(f"Avg Loss:       {metrics.avg_loss:+,.2f}")
    click.echo(f"Max Drawdown:   {metrics.max_drawdown:,.2f} ({metrics.max_drawdown_pct:.2f}%)")
    click.echo(f"Sharpe Ratio:   {metrics.sharpe_ratio:.3f}")
    click.echo(f"Profit Factor:  {metrics.profit_factor:.2f}")
    click.echo(f"Avg Hold Time:  {metrics.avg_hold_minutes:.0f} min")

    for dr in sim_result.day_results:
        click.echo(f"\n  Day {dr.sim_date}: {len(dr.trades)} trades, "
                    f"wall-clock {dr.wall_clock_seconds:.1f}s")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli()
