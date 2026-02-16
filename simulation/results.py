"""Simulation results — models and self-improvement bridge.

``SimulationResult`` aggregates ``DayResult`` instances and provides
methods for serialization, performance metrics, and bridging to the
self-improvement pipeline (``daily_review.run_daily_review()``).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from core.paper_trading_models import PaperTradingState, TradeRecord
from simulation.runner import DayResult, TickSnapshot

logger = logging.getLogger(__name__)

_IST = timezone(timedelta(hours=5, minutes=30))
_RESULTS_DIR = Path("data/sim_results")


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, date):
            return obj.isoformat()
        return super().default(obj)


@dataclass
class PerformanceMetrics:
    """Aggregated performance statistics."""
    total_pnl: float = 0.0
    num_trades: int = 0
    num_wins: int = 0
    num_losses: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    profit_factor: float = 0.0
    avg_hold_minutes: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_pnl": round(self.total_pnl, 2),
            "num_trades": self.num_trades,
            "num_wins": self.num_wins,
            "num_losses": self.num_losses,
            "win_rate": round(self.win_rate, 2),
            "avg_win": round(self.avg_win, 2),
            "avg_loss": round(self.avg_loss, 2),
            "max_drawdown": round(self.max_drawdown, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 3),
            "profit_factor": round(self.profit_factor, 2),
            "avg_hold_minutes": round(self.avg_hold_minutes, 1),
        }


@dataclass
class SimulationResult:
    """Top-level result container for one or more simulated days."""

    algo_name: str
    scenario_name: str
    seed: int
    day_results: list[DayResult] = field(default_factory=list)

    @property
    def all_trades(self) -> list[TradeRecord]:
        """All trades across all simulated days."""
        trades = []
        for dr in self.day_results:
            trades.extend(dr.trades)
        return trades

    @property
    def all_tick_snapshots(self) -> list[TickSnapshot]:
        """All tick snapshots across all days."""
        snapshots = []
        for dr in self.day_results:
            snapshots.extend(dr.tick_snapshots)
        return snapshots

    @property
    def final_state(self) -> PaperTradingState | None:
        if self.day_results:
            return self.day_results[-1].final_state
        return None

    def compute_metrics(self) -> PerformanceMetrics:
        """Compute aggregated performance metrics."""
        trades = self.all_trades
        if not trades:
            return PerformanceMetrics()

        pnls = [t.net_pnl for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        # Max drawdown from cumulative P&L
        cum_pnl = np.cumsum(pnls)
        peak = np.maximum.accumulate(cum_pnl)
        drawdown = peak - cum_pnl
        max_dd = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0

        # Sharpe ratio (annualized from per-trade returns)
        if len(pnls) > 1 and np.std(pnls) > 0:
            sharpe = (np.mean(pnls) / np.std(pnls)) * np.sqrt(252)
        else:
            sharpe = 0.0

        # Profit factor
        gross_profit = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Average hold time
        hold_minutes = []
        for t in trades:
            if t.entry_time and t.exit_time:
                delta = (t.exit_time - t.entry_time).total_seconds() / 60
                hold_minutes.append(delta)

        initial_capital = 2_500_000
        if self.final_state:
            initial_capital = self.final_state.initial_capital

        return PerformanceMetrics(
            total_pnl=sum(pnls),
            num_trades=len(trades),
            num_wins=len(wins),
            num_losses=len(losses),
            win_rate=len(wins) / len(trades) * 100 if trades else 0.0,
            avg_win=np.mean(wins) if wins else 0.0,
            avg_loss=np.mean(losses) if losses else 0.0,
            max_drawdown=max_dd,
            max_drawdown_pct=max_dd / initial_capital * 100 if initial_capital > 0 else 0.0,
            sharpe_ratio=sharpe,
            profit_factor=profit_factor,
            avg_hold_minutes=np.mean(hold_minutes) if hold_minutes else 0.0,
        )

    def to_review_session(self):
        """Bridge to the self-improvement pipeline.

        Builds a PaperTradingState from simulation trade records and
        calls ``daily_review.run_daily_review()`` — unchanged.

        Returns
        -------
        ReviewSession with trade classifications (A/B/C/D), signal
        reliability, and parameter calibrations.
        """
        from analyzers.daily_review import run_daily_review

        state = self.final_state
        if state is None:
            logger.warning("No final state — cannot run review")
            return None

        return run_daily_review(
            state=state,
            algo_name=self.algo_name,
        )

    def save(self, path: str | Path | None = None) -> Path:
        """Save results to JSON.

        Parameters
        ----------
        path : str or Path, optional
            Output path.  If None, uses default naming under
            ``data/sim_results/``.

        Returns
        -------
        Path to saved file.
        """
        if path is None:
            _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            filename = f"{self.algo_name}_{self.scenario_name}_{self.seed}.json"
            filename = filename.replace("/", "_")
            path = _RESULTS_DIR / filename

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "algo_name": self.algo_name,
            "scenario_name": self.scenario_name,
            "seed": self.seed,
            "metrics": self.compute_metrics().to_dict(),
            "num_days": len(self.day_results),
            "trades": [
                t.model_dump(mode="json") for t in self.all_trades
            ],
            "day_summaries": [
                {
                    "date": dr.sim_date.isoformat(),
                    "num_trades": len(dr.trades),
                    "wall_clock_seconds": round(dr.wall_clock_seconds, 1),
                    "final_pnl": dr.final_state.total_realized_pnl,
                }
                for dr in self.day_results
            ],
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2, cls=_NumpyEncoder)

        logger.info("Saved simulation result to %s", path)
        return path

    @staticmethod
    def load_summary(path: str | Path) -> dict[str, Any]:
        """Load a results summary from JSON (without full state)."""
        with open(path) as f:
            return json.load(f)
