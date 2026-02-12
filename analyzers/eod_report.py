"""End-of-day trade report generator.

Collects all trades and critiques for a trading day and produces
human-readable (Markdown) and machine-readable (JSON) reports.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

from core.database import SentimentDatabase
from core.paper_trading_models import PaperTradingState, TradeRecord

logger = logging.getLogger(__name__)

_IST = timezone(timedelta(hours=5, minutes=30))


def generate_eod_report(
    state: PaperTradingState,
    date: datetime | None = None,
) -> dict[str, Any]:
    """Collect all data for a trading day into a structured report dict.

    Parameters
    ----------
    state : PaperTradingState
        The current paper-trading session state (contains trade_log).
    date : datetime, optional
        The date to report on (IST). Defaults to today.
    """
    if date is None:
        date = datetime.now(_IST)
    report_date = date.date()

    # Filter trades for the given date (using entry_time in IST)
    day_trades: list[TradeRecord] = [
        t for t in state.trade_log
        if t.entry_time.astimezone(_IST).date() == report_date
    ]

    # Load critiques from DB keyed by trade_id
    critiques_by_id: dict[str, dict[str, Any]] = {}
    try:
        db = SentimentDatabase()
        all_critiques = db.get_all_critiques(limit=200)
        critiques_by_id = {c["trade_id"]: c for c in all_critiques}
    except Exception:
        logger.warning("Could not load critiques from DB", exc_info=True)

    # --- Summary stats ---
    total_trades = len(day_trades)
    wins = sum(1 for t in day_trades if t.net_pnl > 0)
    losses = sum(1 for t in day_trades if t.net_pnl <= 0)
    net_pnl = sum(t.net_pnl for t in day_trades)
    total_costs = sum(t.execution_cost for t in day_trades)
    win_rate = (wins / total_trades * 100) if total_trades else 0.0

    # --- Per-strategy breakdown ---
    strategy_stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"trades": 0, "net_pnl": 0.0, "grades": []}
    )
    for t in day_trades:
        s = strategy_stats[t.strategy]
        s["trades"] += 1
        s["net_pnl"] += t.net_pnl
        critique = critiques_by_id.get(t.id)
        if critique:
            s["grades"].append(critique["overall_grade"])

    per_strategy = []
    for name, s in sorted(strategy_stats.items()):
        avg_grade = _avg_grade(s["grades"]) if s["grades"] else "N/A"
        per_strategy.append({
            "strategy": name,
            "trades": s["trades"],
            "net_pnl": s["net_pnl"],
            "avg_grade": avg_grade,
        })

    # --- Trade details with matched critiques ---
    trade_details = []
    for t in sorted(day_trades, key=lambda x: x.entry_time):
        critique = critiques_by_id.get(t.id)
        detail: dict[str, Any] = {
            "id": t.id,
            "strategy": t.strategy,
            "direction": t.direction_bias,
            "lots": t.lots,
            "entry_time": t.entry_time.astimezone(_IST).isoformat(),
            "exit_time": t.exit_time.astimezone(_IST).isoformat(),
            "exit_reason": str(t.exit_reason),
            "realized_pnl": t.realized_pnl,
            "execution_cost": t.execution_cost,
            "net_pnl": t.net_pnl,
            "spot_at_entry": t.spot_at_entry,
            "spot_at_exit": t.spot_at_exit,
            "max_drawdown": t.max_drawdown,
            "max_favorable": t.max_favorable,
        }
        if critique:
            detail["grade"] = critique["overall_grade"]
            detail["summary"] = critique["summary"]
            detail["signals_worked"] = critique.get("entry_signal_analysis", {}).get("signals_that_worked", [])
            detail["signals_failed"] = critique.get("entry_signal_analysis", {}).get("signals_that_failed", [])
            detail["patterns"] = critique.get("patterns_observed", [])
        trade_details.append(detail)

    # --- Aggregate parameter recommendations ---
    param_agg: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "current_values": [],
            "recommended_values": [],
            "confidences": [],
            "reasonings": [],
            "trade_ids": [],
        }
    )
    for t in day_trades:
        critique = critiques_by_id.get(t.id)
        if not critique:
            continue
        for rec in critique.get("parameter_recommendations", []):
            key = (rec["strategy_name"], rec["parameter_name"])
            agg = param_agg[key]
            agg["current_values"].append(rec["current_value"])
            agg["recommended_values"].append(rec["recommended_value"])
            agg["confidences"].append(rec["confidence"])
            agg["reasonings"].append(rec["reasoning"])
            agg["trade_ids"].append(t.id)

    aggregated_recs = []
    for (strat, param), agg in sorted(param_agg.items()):
        aggregated_recs.append({
            "strategy": strat,
            "parameter": param,
            "current_value": agg["current_values"][0],
            "avg_recommended": sum(agg["recommended_values"]) / len(agg["recommended_values"]),
            "avg_confidence": sum(agg["confidences"]) / len(agg["confidences"]),
            "frequency": len(agg["trade_ids"]),
            "trade_ids": agg["trade_ids"],
            "reasonings": agg["reasonings"],
        })

    # --- Deduplicated patterns ---
    all_patterns: list[str] = []
    for t in day_trades:
        critique = critiques_by_id.get(t.id)
        if critique:
            all_patterns.extend(critique.get("patterns_observed", []))
    unique_patterns = list(dict.fromkeys(all_patterns))  # preserves order

    return {
        "date": str(report_date),
        "summary": {
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "win_rate": round(win_rate, 1),
            "net_pnl": round(net_pnl, 2),
            "total_costs": round(total_costs, 2),
        },
        "per_strategy": per_strategy,
        "trades": trade_details,
        "aggregated_recommendations": aggregated_recs,
        "patterns_observed": unique_patterns,
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Render the report dict as human-readable Markdown."""
    lines: list[str] = []
    s = report["summary"]
    lines.append(f"# NIFTY Paper Trading Report — {report['date']}\n")

    # Session summary
    lines.append("## Session Summary\n")
    lines.append(
        f"- **Trades:** {s['total_trades']} | "
        f"**Win rate:** {s['win_rate']}% | "
        f"**Net P&L:** {_fmt_inr(s['net_pnl'])}"
    )
    lines.append(
        f"- **Wins:** {s['wins']} | **Losses:** {s['losses']} | "
        f"**Execution costs:** {_fmt_inr(s['total_costs'])}"
    )
    lines.append("")

    # Per-strategy breakdown
    if report["per_strategy"]:
        lines.append("## Per-Strategy Breakdown\n")
        lines.append("| Strategy | Trades | Net P&L | Avg Grade |")
        lines.append("|---|---|---|---|")
        for ps in report["per_strategy"]:
            lines.append(
                f"| {ps['strategy']} | {ps['trades']} | "
                f"{_fmt_inr(ps['net_pnl'])} | {ps['avg_grade']} |"
            )
        lines.append("")

    # Trade log
    if report["trades"]:
        lines.append("## Trade Log\n")
        for i, t in enumerate(report["trades"], 1):
            grade = t.get("grade", "N/A")
            entry = t["entry_time"][:16].replace("T", " ")
            exit_ = t["exit_time"][:16].replace("T", " ")
            lines.append(
                f"### #{i} {t['strategy']} ({entry} → {exit_}) — {grade.upper()}\n"
            )
            lines.append(
                f"- **Direction:** {t['direction']} | **Lots:** {t['lots']} | "
                f"**Exit:** {t['exit_reason']}"
            )
            lines.append(
                f"- **P&L:** {_fmt_inr(t['net_pnl'])} "
                f"(gross {_fmt_inr(t['realized_pnl'])}, costs {_fmt_inr(t['execution_cost'])})"
            )
            lines.append(
                f"- **Spot:** {t['spot_at_entry']:.0f} → {t['spot_at_exit']:.0f} | "
                f"**Max DD:** {_fmt_inr(t['max_drawdown'])} | "
                f"**Max Fav:** {_fmt_inr(t['max_favorable'])}"
            )
            if t.get("summary"):
                lines.append(f"- **Critique:** {t['summary']}")
            if t.get("signals_worked"):
                lines.append(f"- **Signals worked:** {', '.join(t['signals_worked'])}")
            if t.get("signals_failed"):
                lines.append(f"- **Signals failed:** {', '.join(t['signals_failed'])}")
            lines.append("")

    # Aggregated parameter recommendations
    if report["aggregated_recommendations"]:
        lines.append("## Aggregate Parameter Recommendations\n")
        lines.append("| Strategy | Parameter | Current | Suggested | Confidence | Trades |")
        lines.append("|---|---|---|---|---|---|")
        for r in report["aggregated_recommendations"]:
            lines.append(
                f"| {r['strategy']} | {r['parameter']} | "
                f"{r['current_value']:.4g} | {r['avg_recommended']:.4g} | "
                f"{r['avg_confidence']:.0%} | {r['frequency']} |"
            )
        lines.append("")

    # Patterns observed
    if report["patterns_observed"]:
        lines.append("## Patterns Observed\n")
        for p in report["patterns_observed"]:
            lines.append(f"- {p}")
        lines.append("")

    return "\n".join(lines)


def render_json(report: dict[str, Any]) -> str:
    """Render the report dict as machine-readable JSON."""
    return json.dumps(report, indent=2, default=str)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_GRADE_MAP = {
    "excellent": 5,
    "good": 4,
    "acceptable": 3,
    "poor": 2,
    "terrible": 1,
}
_GRADE_REVERSE = {v: k for k, v in _GRADE_MAP.items()}


def _avg_grade(grades: list[str]) -> str:
    nums = [_GRADE_MAP.get(g.lower(), 3) for g in grades]
    avg = sum(nums) / len(nums)
    return _GRADE_REVERSE.get(round(avg), "acceptable")


def _fmt_inr(value: float) -> str:
    sign = "-" if value < 0 else ""
    return f"{sign}₹{abs(value):,.0f}"
