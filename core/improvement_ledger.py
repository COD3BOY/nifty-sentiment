"""Improvement ledger — lifecycle management for parameter changes.

Wraps the raw ``SentimentDatabase`` ledger tables with business logic for
proposing, approving, applying, reverting, and monitoring parameter changes.
All state transitions flow through this module.
"""

from __future__ import annotations

import logging
from datetime import datetime

from core.database import SentimentDatabase
from core.improvement_models import ImprovementProposal, ReviewSession
from core.parameter_bounds import (
    COOLING_PERIOD_TRADES,
    MAX_CHANGES_PER_SESSION,
    REVERSION_TRIGGER_LOSSES,
    validate_proposed_value,
)
from core.strategy_rules import STRATEGY_RULES

logger = logging.getLogger(__name__)


def _get_default(strategy_name: str, parameter_name: str) -> float | None:
    """Look up the original default for a param_key from strategy_rules."""
    rules = STRATEGY_RULES.get(strategy_name, {})
    for rule in rules.get("scoring_rules", []):
        if rule.get("param_key") == parameter_name:
            return rule.get("default")
    return None


# -------------------------------------------------------------------------
# Proposal management
# -------------------------------------------------------------------------

def propose_change(
    db: SentimentDatabase,
    proposal: ImprovementProposal,
    algorithm: str,
) -> tuple[bool, str]:
    """Validate and record a new improvement proposal.

    Returns ``(success, message)``.
    """
    # Safety rail: validate proposed value
    allowed, reason = validate_proposed_value(
        proposal.parameter_name,
        proposal.default_value,
        proposal.current_value,
        proposal.proposed_value,
    )
    if not allowed:
        return False, f"Safety rail blocked: {reason}"

    # Safety rail: check cooling period
    active = db.get_active_ledger_for_param(
        proposal.strategy_name, proposal.parameter_name
    )
    if active and active["status"] in ("applied", "monitoring"):
        trades_since = active.get("trades_since_applied", 0)
        if trades_since < COOLING_PERIOD_TRADES:
            remaining = COOLING_PERIOD_TRADES - trades_since
            return False, (
                f"Cooling period: {remaining} trades remaining before "
                f"{proposal.parameter_name} can be changed again"
            )

    # Safety rail: check session change budget
    today = proposal.created_at.strftime("%Y-%m-%d")
    recent = db.get_ledger_entries(algorithm=algorithm, status="proposed")
    today_proposals = [e for e in recent if e["date"] == today]
    if len(today_proposals) >= MAX_CHANGES_PER_SESSION:
        return False, (
            f"Change budget exhausted: {MAX_CHANGES_PER_SESSION} proposals "
            f"already created today"
        )

    # Record the proposal
    entry = {
        "id": proposal.id,
        "date": today,
        "algorithm": algorithm,
        "strategy_name": proposal.strategy_name,
        "parameter_name": proposal.parameter_name,
        "default_value": proposal.default_value,
        "old_value": proposal.current_value,
        "new_value": proposal.proposed_value,
        "change_pct": proposal.change_pct,
        "evidence_trade_count": proposal.evidence_trade_count,
        "evidence": {
            "summary": proposal.evidence_summary,
            "expected_impact": proposal.expected_impact,
            "reversion_criteria": proposal.reversion_criteria,
            "classification": proposal.classification,
        },
        "confidence": proposal.confidence,
        "status": "proposed",
        "pre_metrics": {},
        "created_at": proposal.created_at,
    }
    db.save_ledger_entry(entry)
    logger.info(
        "Proposed change: %s.%s %.2f → %.2f (%+.1f%%)",
        proposal.strategy_name,
        proposal.parameter_name,
        proposal.current_value,
        proposal.proposed_value,
        proposal.change_pct * 100,
    )
    return True, f"Proposal {proposal.id} recorded"


# -------------------------------------------------------------------------
# Approval / rejection
# -------------------------------------------------------------------------

def approve_change(db: SentimentDatabase, entry_id: str) -> bool:
    """Mark a proposal as approved (still not applied)."""
    return db.update_ledger_status(entry_id, "approved")


def reject_change(db: SentimentDatabase, entry_id: str) -> bool:
    """Mark a proposal as rejected."""
    return db.update_ledger_status(entry_id, "rejected")


def defer_change(db: SentimentDatabase, entry_id: str) -> bool:
    """Mark a proposal as deferred (needs more data)."""
    return db.update_ledger_status(entry_id, "deferred")


# -------------------------------------------------------------------------
# Apply / revert
# -------------------------------------------------------------------------

def apply_change(db: SentimentDatabase, entry_id: str) -> bool:
    """Apply an approved change — writes override to ParameterAdjustmentRow.

    This creates a new ``ParameterAdjustmentRow`` with ``applied=1`` so that
    ``get_active_overrides()`` picks it up on the next trading cycle.
    """
    entries = db.get_ledger_entries()
    entry = next((e for e in entries if e["id"] == entry_id), None)
    if not entry:
        logger.error("Ledger entry %s not found", entry_id)
        return False
    if entry["status"] not in ("approved",):
        logger.error("Cannot apply entry %s with status %s", entry_id, entry["status"])
        return False

    # Write the override via ParameterAdjustmentRow
    from core.database import ParameterAdjustmentRow

    with db.SessionLocal() as session:
        adj = ParameterAdjustmentRow(
            trade_id=f"ledger_{entry_id}",
            strategy_name=entry["strategy_name"],
            parameter_name=entry["parameter_name"],
            current_value=entry["old_value"],
            recommended_value=entry["new_value"],
            confidence=entry.get("confidence", 0.0),
            reasoning=f"Improvement ledger entry {entry_id}",
            condition="",
            applied=1,
        )
        session.add(adj)
        session.commit()

    # Supersede any previous applied entry for the same param
    existing = db.get_active_ledger_for_param(
        entry["strategy_name"], entry["parameter_name"]
    )
    if existing and existing["id"] != entry_id:
        db.update_ledger_status(existing["id"], "superseded")

    # Update ledger entry
    now = datetime.utcnow()
    db.update_ledger_status(
        entry_id, "applied", applied_at=now, trades_since_applied=0
    )
    logger.info(
        "Applied change %s: %s.%s → %.2f",
        entry_id,
        entry["strategy_name"],
        entry["parameter_name"],
        entry["new_value"],
    )
    return True


def revert_change(db: SentimentDatabase, entry_id: str) -> bool:
    """Revert a previously applied change.

    Sets ``applied=0`` on the override row so ``get_active_overrides()``
    no longer returns it. The original default resumes.
    """
    entries = db.get_ledger_entries()
    entry = next((e for e in entries if e["id"] == entry_id), None)
    if not entry:
        return False
    if entry["status"] not in ("applied", "monitoring"):
        logger.error("Cannot revert entry %s with status %s", entry_id, entry["status"])
        return False

    # Find and deactivate the override row
    from core.database import ParameterAdjustmentRow

    with db.SessionLocal() as session:
        rows = (
            session.query(ParameterAdjustmentRow)
            .filter(
                ParameterAdjustmentRow.trade_id == f"ledger_{entry_id}",
                ParameterAdjustmentRow.applied == 1,
            )
            .all()
        )
        for row in rows:
            row.applied = 0
        session.commit()

    now = datetime.utcnow()
    db.update_ledger_status(entry_id, "reverted", reverted_at=now)
    logger.info("Reverted change %s", entry_id)
    return True


# -------------------------------------------------------------------------
# Monitoring helpers
# -------------------------------------------------------------------------

def update_trade_counts(
    db: SentimentDatabase,
    strategy_name: str,
    trade_won: bool,
) -> list[str]:
    """Called after a trade closes to update cooling counters and check triggers.

    Returns a list of entry IDs that tripped the reversion trigger.
    """
    triggered: list[str] = []
    entries = db.get_ledger_entries(status="applied") + db.get_ledger_entries(
        status="monitoring"
    )

    for entry in entries:
        if entry["strategy_name"] != strategy_name:
            continue

        trades_since = entry.get("trades_since_applied", 0) + 1
        consec = entry.get("consecutive_losses", 0)

        if trade_won:
            consec = 0
        else:
            consec += 1

        new_status = entry["status"]
        if trades_since >= COOLING_PERIOD_TRADES and entry["status"] == "applied":
            new_status = "monitoring"

        db.update_ledger_status(
            entry["id"],
            new_status,
            trades_since_applied=trades_since,
            consecutive_losses=consec,
        )

        if consec >= REVERSION_TRIGGER_LOSSES:
            triggered.append(entry["id"])
            logger.warning(
                "Reversion trigger: %s has %d consecutive losses",
                entry["id"],
                consec,
            )

    return triggered


def check_cooling_periods(
    db: SentimentDatabase,
    algorithm: str | None = None,
) -> dict[str, int]:
    """Return parameters currently in cooling with trades remaining.

    Returns ``{param_key: trades_remaining}``.
    """
    entries = db.get_ledger_entries(algorithm=algorithm, status="applied")
    result: dict[str, int] = {}
    for entry in entries:
        trades_since = entry.get("trades_since_applied", 0)
        if trades_since < COOLING_PERIOD_TRADES:
            key = f"{entry['strategy_name']}.{entry['parameter_name']}"
            result[key] = COOLING_PERIOD_TRADES - trades_since
    return result


def save_review(db: SentimentDatabase, session: ReviewSession) -> str:
    """Persist a review session and its proposals."""
    # Save proposals first
    for proposal in session.proposals:
        propose_change(db, proposal, session.algorithm)

    # Save session summary
    session_data = {
        "id": session.id,
        "date": session.date,
        "algorithm": session.algorithm,
        "trades_reviewed": session.trades_reviewed,
        "summary": {
            "classification_summary": session.classification_summary,
            "signal_count": len(session.signal_reliability),
            "calibration_count": len(session.parameter_calibrations),
            "proposal_count": len(session.proposals),
            "notes": session.notes,
        },
        "created_at": session.created_at,
    }
    return db.save_review_session(session_data)
