"""Pydantic models for the self-improvement / daily review system.

These models represent the analysis pipeline output — from individual trade
classification through signal reliability analysis to improvement proposals.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from core.paper_trading_models import _now_ist


# -------------------------------------------------------------------------
# Trade classification
# -------------------------------------------------------------------------

class TradeClassification(BaseModel):
    """Classification of a single closed trade on the entry-quality / outcome matrix.

    Categories:
        A — Process Win (sound entry, profitable)
        B — Lucky (flawed entry, profitable)
        C — Bad Luck (sound entry, unprofitable)
        D — Process Failure (flawed entry, unprofitable)
    """

    trade_id: str
    strategy: str
    category: Literal["A", "B", "C", "D"]
    profitable: bool
    signal_alignment_count: int
    signal_opposition_count: int
    strategy_regime_fit: bool
    sizing_compliant: bool
    entry_quality_score: float = Field(ge=0.0, le=100.0)
    signals_present: list[str] = Field(default_factory=list)
    signals_aligned: list[str] = Field(default_factory=list)
    signals_opposed: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


# -------------------------------------------------------------------------
# Signal reliability
# -------------------------------------------------------------------------

class SignalReliability(BaseModel):
    """Rolling reliability metrics for a single signal across trades."""

    signal_name: str
    total_trades: int
    confirming_wins: int = 0
    confirming_losses: int = 0
    opposing_wins: int = 0
    opposing_losses: int = 0

    @property
    def predictive_accuracy(self) -> float:
        """Fraction of confirming-signal trades that won."""
        total = self.confirming_wins + self.confirming_losses
        if total == 0:
            return 0.0
        return self.confirming_wins / total

    @property
    def information_coefficient(self) -> float:
        """Simple correlation proxy: (correct - incorrect) / total.

        Ranges from -1 (perfectly wrong) to +1 (perfectly right).
        """
        correct = self.confirming_wins + self.opposing_losses
        incorrect = self.confirming_losses + self.opposing_wins
        total = correct + incorrect
        if total == 0:
            return 0.0
        return (correct - incorrect) / total


# -------------------------------------------------------------------------
# Parameter calibration
# -------------------------------------------------------------------------

class ParameterCalibration(BaseModel):
    """MAE/MFE-based calibration analysis for a strategy parameter."""

    parameter_name: str
    strategy_name: str
    current_value: float
    default_value: float
    evidence_trades: int
    # Stop-loss analysis
    sl_hit_then_reversed_pct: float = 0.0
    # Profit-target analysis
    pt_exceeded_pct: float = 0.0
    # MAE/MFE distributions
    mae_median_winners: float = 0.0
    mae_median_losers: float = 0.0
    mfe_median: float = 0.0
    # Suggestion (None if no suggestion)
    suggested_value: float | None = None
    suggestion_reason: str = ""


# -------------------------------------------------------------------------
# Improvement proposal
# -------------------------------------------------------------------------

class ImprovementProposal(BaseModel):
    """A single proposed parameter change with full evidence chain."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    strategy_name: str
    parameter_name: str
    current_value: float
    proposed_value: float
    default_value: float
    change_pct: float
    evidence_trade_count: int
    evidence_summary: str
    expected_impact: str
    reversion_criteria: str
    classification: Literal["structural", "calibration", "rule_addition"]
    confidence: float = Field(ge=0.0, le=1.0)
    status: Literal[
        "proposed", "approved", "applied", "monitoring",
        "confirmed", "reverted", "rejected", "deferred", "superseded",
    ] = "proposed"
    created_at: datetime = Field(default_factory=_now_ist)


# -------------------------------------------------------------------------
# Review session
# -------------------------------------------------------------------------

class ReviewSession(BaseModel):
    """Complete output of a daily trade review session."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    date: str
    algorithm: str
    trades_reviewed: int
    classification_summary: dict[str, int] = Field(default_factory=dict)
    classifications: list[TradeClassification] = Field(default_factory=list)
    signal_reliability: list[SignalReliability] = Field(default_factory=list)
    parameter_calibrations: list[ParameterCalibration] = Field(default_factory=list)
    proposals: list[ImprovementProposal] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_now_ist)
