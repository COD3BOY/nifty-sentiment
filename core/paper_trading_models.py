"""Pydantic models for the Paper Trading simulation."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone, timedelta
from enum import Enum

_IST = timezone(timedelta(hours=5, minutes=30))


def _now_ist() -> datetime:
    return datetime.now(_IST)

from pydantic import BaseModel, Field, computed_field, model_validator


class StrategyType(str, Enum):
    CREDIT = "credit"
    DEBIT = "debit"


class PositionStatus(str, Enum):
    OPEN = "open"
    CLOSED_STOP_LOSS = "closed_stop_loss"
    CLOSED_PROFIT_TARGET = "closed_profit_target"
    CLOSED_MANUAL = "closed_manual"
    CLOSED_EOD = "closed_eod"


class PositionLeg(BaseModel):
    """Single leg of a paper-traded position."""

    action: str  # "SELL" or "BUY"
    instrument: str  # e.g. "NIFTY 23500 CE"
    strike: float
    option_type: str  # "CE" or "PE"
    lots: int = 1
    lot_size: int = 65
    entry_ltp: float
    current_ltp: float

    @computed_field  # type: ignore[prop-decorator]
    @property
    def unrealized_pnl(self) -> float:
        if self.action == "BUY":
            return (self.current_ltp - self.entry_ltp) * self.lots * self.lot_size
        else:  # SELL
            return (self.entry_ltp - self.current_ltp) * self.lots * self.lot_size


class PaperPosition(BaseModel):
    """Multi-leg paper position."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    strategy: str
    strategy_type: StrategyType
    direction_bias: str
    confidence: str
    score: float
    legs: list[PositionLeg]
    lots: int = 1
    entry_time: datetime = Field(default_factory=_now_ist)
    exit_time: datetime | None = None
    status: PositionStatus = PositionStatus.OPEN
    net_premium: float = 0.0
    stop_loss_amount: float = 0.0
    profit_target_amount: float = 0.0
    execution_cost: float = 0.0
    margin_required: float = 0.0
    entry_context: dict | None = None  # serialized MarketContextSnapshot
    peak_pnl: float = 0.0      # best unrealized PnL during hold
    trough_pnl: float = 0.0    # worst unrealized PnL during hold

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_unrealized_pnl(self) -> float:
        return sum(leg.unrealized_pnl for leg in self.legs)


class TradeRecord(BaseModel):
    """Immutable record of a closed trade."""

    id: str
    strategy: str
    strategy_type: StrategyType
    direction_bias: str
    confidence: str
    score: float
    legs_summary: list[dict]
    lots: int = 1
    entry_time: datetime
    exit_time: datetime
    exit_reason: PositionStatus
    realized_pnl: float
    execution_cost: float = 0.0
    net_pnl: float = 0.0
    margin_required: float = 0.0
    net_premium: float
    stop_loss_amount: float
    profit_target_amount: float
    entry_context: dict | None = None   # serialized MarketContextSnapshot
    exit_context: dict | None = None
    spot_at_entry: float = 0.0
    spot_at_exit: float = 0.0
    max_drawdown: float = 0.0          # abs(trough_pnl)
    max_favorable: float = 0.0         # peak_pnl


class PaperTradingState(BaseModel):
    """Root container for paper trading session state."""

    initial_capital: float = 2_500_000.0
    open_positions: list[PaperPosition] = Field(default_factory=list)
    trade_log: list[TradeRecord] = Field(default_factory=list)
    total_realized_pnl: float = 0.0
    total_execution_costs: float = 0.0
    is_auto_trading: bool = True
    last_open_refresh_ts: float = 0.0  # prevents re-open on same refresh cycle
    last_trade_opened_ts: float = 0.0  # cooldown between successive trade opens
    pending_critiques: list[str] = Field(default_factory=list)  # trade IDs awaiting critique

    @model_validator(mode="before")
    @classmethod
    def _migrate_single_position(cls, data):
        """Migrate old single-position JSON format to open_positions list."""
        if isinstance(data, dict) and "current_position" in data:
            old_pos = data.pop("current_position")
            if "open_positions" not in data:
                data["open_positions"] = [old_pos] if old_pos else []
        return data

    @computed_field  # type: ignore[prop-decorator]
    @property
    def net_realized_pnl(self) -> float:
        return self.total_realized_pnl - self.total_execution_costs

    @computed_field  # type: ignore[prop-decorator]
    @property
    def margin_in_use(self) -> float:
        return sum(p.margin_required for p in self.open_positions if p.status == PositionStatus.OPEN)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def capital_remaining(self) -> float:
        return self.initial_capital + self.total_realized_pnl - self.total_execution_costs - self.margin_in_use

    @computed_field  # type: ignore[prop-decorator]
    @property
    def unrealized_pnl(self) -> float:
        return sum(p.total_unrealized_pnl for p in self.open_positions if p.status == PositionStatus.OPEN)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_pnl(self) -> float:
        return self.total_realized_pnl + self.unrealized_pnl

    @computed_field  # type: ignore[prop-decorator]
    @property
    def win_rate(self) -> float:
        if not self.trade_log:
            return 0.0
        wins = sum(1 for t in self.trade_log if t.realized_pnl > 0)
        return (wins / len(self.trade_log)) * 100
