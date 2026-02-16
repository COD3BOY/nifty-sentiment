"""Simulation runner — orchestrates tick-by-tick algo execution.

Generates synthetic market data for a full trading day, then feeds it
tick-by-tick through the algorithm's ``generate_suggestions()`` and
``evaluate_and_manage()`` methods with the virtual clock advancing.

The runner is agnostic to which algorithm is being tested — it talks
through the ``TradingAlgorithm`` ABC.
"""

from __future__ import annotations

import copy
import logging
import time as wall_time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import TYPE_CHECKING
from unittest.mock import patch as mock_patch

import numpy as np
import pandas as pd

from algorithms import discover_algorithms, get_algorithm_registry
from algorithms.base import TradingAlgorithm
from core.options_models import OptionChainData, TechnicalIndicators, OptionsAnalytics
from core.paper_trading_models import PaperTradingState, TradeRecord
from simulation.chain_synthesizer import build_chain, update_chain
from simulation.clock import VirtualClock
from simulation.data_assembler import (
    AssembledTick,
    build_vol_snapshot,
    compute_analytics,
    compute_observation,
    compute_technicals,
)
from simulation.price_engine import (
    compute_log_returns,
    generate_price_path,
    generate_vix_path,
    generate_warmup_days,
)
from simulation.adversarial import AdversarialPerturber
from simulation.context_assembler import SimContextBuilder
from simulation.scenario_models import Scenario

if TYPE_CHECKING:
    from core.observation import ObservationSnapshot

logger = logging.getLogger(__name__)

_IST = timezone(timedelta(hours=5, minutes=30))


@dataclass
class TickSnapshot:
    """Recorded state at a single tick for post-simulation analysis."""
    tick_index: int
    sim_datetime: datetime
    spot: float
    vix: float
    num_open_positions: int
    unrealized_pnl: float
    realized_pnl: float
    trade_status_notes: list[str] = field(default_factory=list)


@dataclass
class DayResult:
    """Result of a single simulated trading day."""
    sim_date: date
    algo_name: str
    seed: int
    scenario_name: str
    final_state: PaperTradingState
    trades: list[TradeRecord]
    tick_snapshots: list[TickSnapshot]
    candles: pd.DataFrame  # full day OHLCV
    vix_path: np.ndarray
    wall_clock_seconds: float


def get_algorithm(algo_name: str, config: dict | None = None) -> TradingAlgorithm:
    """Instantiate an algorithm by name.

    Parameters
    ----------
    algo_name : str
        e.g. "sentinel", "jarvis", "optimus", "atlas"
    config : dict, optional
        Algorithm config. If None, loads from config.yaml.
    """
    discover_algorithms()
    registry = get_algorithm_registry()

    if algo_name not in registry:
        raise ValueError(
            f"Unknown algorithm '{algo_name}'. Available: {list(registry.keys())}"
        )

    if config is None:
        from core.config import load_config
        full_cfg = load_config()
        config = full_cfg.get("algorithms", {}).get(algo_name, {})

    algo_cls = registry[algo_name]
    return algo_cls(config)


def _make_initial_state(
    scenario: Scenario,
    algo_name: str,
) -> PaperTradingState:
    """Create a fresh PaperTradingState for simulation."""
    from core.config import load_config
    cfg = load_config().get("paper_trading", {})
    initial_capital = cfg.get("initial_capital", 2_500_000)

    return PaperTradingState(
        initial_capital=initial_capital,
        open_positions=[],
        trade_log=[],
        total_realized_pnl=0.0,
        total_execution_costs=0.0,
        is_auto_trading=True,
        last_open_refresh_ts=0.0,
        last_trade_opened_ts=0.0,
        pending_critiques=[],
        session_start_capital=initial_capital,
        consecutive_losses=0,
        peak_capital=initial_capital,
        trading_halted=False,
        weekly_start_capital=initial_capital,
        week_start_date="",
        daily_start_capital=initial_capital,
        session_date="",
        allocation_tracker={},
        vol_regime="sell_premium",
        vol_snapshot_ts=0.0,
        vol_dynamic_params={},
        trade_status_notes=[],
    )


def run_day(
    scenario: Scenario,
    algo_name: str,
    sim_date: date,
    seed: int,
    initial_state: PaperTradingState | None = None,
    prev_close: float | None = None,
    warmup_df: pd.DataFrame | None = None,
    speed: float = 6.0,
    callback=None,
    context_builder: SimContextBuilder | None = None,
) -> DayResult:
    """Run a single simulated trading day.

    Parameters
    ----------
    scenario : Scenario
        Market scenario configuration.
    algo_name : str
        Algorithm to test (e.g. "atlas").
    sim_date : date
        Simulated date.
    seed : int
        RNG seed for reproducibility.
    initial_state : PaperTradingState, optional
        Starting state.  If None, creates fresh.
    prev_close : float, optional
        Previous day's close (for gap calculation).
    warmup_df : DataFrame, optional
        Prior days' candles for indicator warmup.  If None, generates them.
    speed : float
        Clock speed (only affects sleep, not tick size).
    callback : callable, optional
        Called after each tick with ``(tick_index, assembled_tick, state)``.

    Returns
    -------
    DayResult with final state, trades, tick snapshots, and candle data.
    """
    wall_start = wall_time.time()

    # --- Instantiate algorithm ---
    algo = get_algorithm(algo_name)

    # --- Generate full day's candles (reproducible from seed) ---
    day_candles = generate_price_path(scenario, sim_date, seed, prev_close)
    closes = day_candles["Close"].values
    log_returns = compute_log_returns(closes)
    vix_path = generate_vix_path(scenario, log_returns, seed)

    # --- Generate warmup candles if needed ---
    if warmup_df is None:
        warmup_df = generate_warmup_days(scenario, sim_date, seed, scenario.num_warmup_days)

    # --- Initialize context builder (if not provided by multi-day caller) ---
    if context_builder is None:
        context_builder = SimContextBuilder(scenario, sim_date, seed, warmup_df)

    # --- Resolve expiry date ---
    expiry_date = sim_date + timedelta(days=scenario.chain.expiry_dte)

    # --- Initialize state ---
    state = initial_state or _make_initial_state(scenario, algo_name)

    # --- Initialize virtual clock ---
    clock = VirtualClock(sim_date, tick_interval=60, speed=speed)

    # --- Config ---
    from core.config import load_config
    pt_cfg = load_config().get("paper_trading", {})
    lot_size = pt_cfg.get("lot_size", 65)

    # --- Prepare vol_snapshot for Atlas patching ---
    vol_snapshot = build_vol_snapshot(
        scenario.vol_regime,
        closes[0],
        vix_path[0],
        sim_date.isoformat(),
        dte=scenario.chain.expiry_dte,
    )

    # --- Build initial chain ---
    chain = build_chain(
        closes[0], clock.now, scenario.chain, seed, expiry_date,
    )

    # --- Initialize adversarial engine (if configured) ---
    perturber = None
    if scenario.adversarial.mode:
        perturber = AdversarialPerturber(
            mode=scenario.adversarial.mode,
            intensity=scenario.adversarial.intensity,
            seed=seed,
        )

    open_price = closes[0]

    tick_snapshots: list[TickSnapshot] = []
    trades_before = len(state.trade_log)

    # --- Run simulation with clock patching ---
    with clock.patch():
        # Also patch get_today_vol_snapshot for Atlas
        with mock_patch(
            "algorithms.atlas.get_today_vol_snapshot",
            return_value=vol_snapshot,
        ):
            tick_idx = 0
            while not clock.is_day_complete() and tick_idx < len(closes):
                spot = closes[tick_idx]
                vix = vix_path[tick_idx]

                # --- Apply adversarial perturbation ---
                adv_iv_bump = 0.0
                adv_spread_mult = 1.0
                if perturber is not None:
                    perturb_result = perturber.perturb(
                        spot, tick_idx, state, scenario.chain,
                    )
                    spot = perturb_result.modified_close
                    adv_iv_bump = perturb_result.iv_bump
                    adv_spread_mult = perturb_result.spread_multiplier

                # Update vol_snapshot with current VIX
                vol_snapshot = build_vol_snapshot(
                    scenario.vol_regime, spot, vix,
                    sim_date.isoformat(), dte=scenario.chain.expiry_dte,
                )

                # --- Progressive candle reveal ---
                revealed_candles = day_candles.iloc[:tick_idx + 1]
                if warmup_df is not None and len(warmup_df) > 0:
                    full_df = pd.concat([warmup_df, revealed_candles])
                else:
                    full_df = revealed_candles

                # --- Update chain from spot ---
                if tick_idx == 0:
                    # Already built
                    pass
                else:
                    chain = update_chain(
                        chain, spot, clock.now, scenario.chain,
                        seed + tick_idx, expiry_date,
                        open_price=open_price,
                        current_vix=vix,
                        iv_bump=adv_iv_bump,
                        extra_spread_mult=adv_spread_mult,
                    )

                # --- Compute technicals from progressively revealed candles ---
                technicals = compute_technicals(full_df, clock.now)

                # --- Compute analytics from chain ---
                analytics = compute_analytics(chain, scenario.chain.iv_percentile)

                # --- Compute observation (9:15-10:00 window) ---
                observation = compute_observation(full_df)

                # --- Build market context ---
                context = context_builder.build_context(full_df, technicals)

                # --- Generate suggestions ---
                try:
                    suggestions = algo.generate_suggestions(
                        chain, technicals, analytics, observation,
                        context=context,
                    )
                except Exception as e:
                    logger.warning("generate_suggestions error at tick %d: %s", tick_idx, e)
                    suggestions = []

                # --- Evaluate and manage ---
                refresh_ts = clock.now.timestamp()
                try:
                    state = algo.evaluate_and_manage(
                        state, suggestions, chain,
                        technicals=technicals, analytics=analytics,
                        lot_size=lot_size, refresh_ts=refresh_ts,
                        observation=observation, context=context,
                    )
                except Exception as e:
                    logger.warning("evaluate_and_manage error at tick %d: %s", tick_idx, e)

                # --- Record tick snapshot ---
                tick_snapshots.append(TickSnapshot(
                    tick_index=tick_idx,
                    sim_datetime=clock.now,
                    spot=spot,
                    vix=vix,
                    num_open_positions=len(state.open_positions),
                    unrealized_pnl=state.unrealized_pnl,
                    realized_pnl=state.total_realized_pnl,
                    trade_status_notes=list(state.trade_status_notes),
                ))

                # --- Callback for live display ---
                if callback:
                    try:
                        callback(tick_idx, state, spot, vix, clock.now)
                    except Exception:
                        pass

                # --- Advance clock ---
                clock.tick()
                tick_idx += 1

            # --- Force EOD close of remaining positions ---
            state = _force_eod_close(state, chain, technicals, analytics)

    # --- Collect new trades ---
    new_trades = state.trade_log[trades_before:]

    wall_elapsed = wall_time.time() - wall_start

    return DayResult(
        sim_date=sim_date,
        algo_name=algo_name,
        seed=seed,
        scenario_name=scenario.name,
        final_state=state,
        trades=new_trades,
        tick_snapshots=tick_snapshots,
        candles=day_candles,
        vix_path=vix_path,
        wall_clock_seconds=wall_elapsed,
    )


def _force_eod_close(
    state: PaperTradingState,
    chain: OptionChainData | None,
    technicals: TechnicalIndicators | None,
    analytics: OptionsAnalytics | None,
) -> PaperTradingState:
    """Close all open positions at end of day."""
    from core.paper_trading_engine import close_position
    from core.paper_trading_models import PositionStatus

    if not state.open_positions:
        return state

    state = state.model_copy(deep=True)
    closed = []
    new_trades = list(state.trade_log)

    for pos in state.open_positions:
        try:
            closed_pos, trade_record = close_position(
                pos, PositionStatus.CLOSED_EOD,
                technicals=technicals, analytics=analytics, chain=chain,
            )
            new_trades.append(trade_record)
            closed.append(closed_pos)
        except Exception as e:
            logger.warning("EOD close failed for %s: %s", pos.id, e)

    state.open_positions = []
    state.trade_log = new_trades
    state.total_realized_pnl = sum(t.realized_pnl for t in new_trades)
    state.total_execution_costs = sum(t.execution_cost for t in new_trades)

    return state


def run_multi_day(
    scenario: Scenario,
    algo_name: str,
    start_date: date,
    seed: int,
    speed: float = 6.0,
    callback=None,
) -> list[DayResult]:
    """Run a multi-day simulation.

    Chains state between days, carries prev_close for gap calculation,
    and maintains a rolling candle lookback window.

    Parameters
    ----------
    scenario : Scenario
        Scenario config (``days`` field controls duration).
    algo_name : str
        Algorithm to test.
    start_date : date
        First simulation date.
    seed : int
        Base seed (incremented per day).
    speed : float
        Clock speed factor.
    callback : callable, optional
        Per-tick callback.

    Returns
    -------
    List of DayResult, one per simulated day.
    """
    results = []
    state = None
    prev_close = None
    warmup_df = None
    current_date = start_date

    # Create context builder once for the full multi-day run
    ctx_builder = SimContextBuilder(scenario, start_date, seed)

    for day_idx in range(scenario.num_days):
        # Skip weekends
        while current_date.weekday() >= 5:
            current_date += timedelta(days=1)

        day_seed = seed + day_idx * 10000

        # Get per-day scenario (multi-day uses day_configs list)
        day_scenario = scenario.get_day_scenario(day_idx)

        result = run_day(
            scenario=day_scenario,
            algo_name=algo_name,
            sim_date=current_date,
            seed=day_seed,
            initial_state=state,
            prev_close=prev_close,
            warmup_df=warmup_df,
            speed=speed,
            callback=callback,
            context_builder=ctx_builder,
        )

        results.append(result)

        # Carry state forward
        state = result.final_state
        prev_close = result.candles["Close"].iloc[-1]

        # Build rolling warmup from completed days' candles
        completed_candles = [r.candles for r in results]
        warmup_df = pd.concat(completed_candles[-5:])  # Keep last 5 days

        # Update context builder for next day
        next_day_idx = day_idx + 1
        next_vol = None
        if next_day_idx < scenario.num_days:
            next_scenario = scenario.get_day_scenario(next_day_idx)
            next_vol = next_scenario.vol_regime
        next_date = current_date + timedelta(days=1)
        # Skip weekends for next_date
        while next_date.weekday() >= 5:
            next_date += timedelta(days=1)
        ctx_builder.end_of_day(result.candles, next_vol, next_date)

        current_date += timedelta(days=1)

    return results


