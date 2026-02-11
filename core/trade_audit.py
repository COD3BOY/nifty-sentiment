"""Trade decision audit trail — logs every OPEN / CLOSE / REJECT with full context."""

from __future__ import annotations

import json
import logging
import threading
from collections import deque
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from core.paper_trading_models import _now_ist

logger = logging.getLogger("core.trade_audit")

# ---------------------------------------------------------------------------
# In-memory ring buffer (thread-safe)
# ---------------------------------------------------------------------------

_BUFFER_SIZE = 200
_buffer: deque[dict] = deque(maxlen=_BUFFER_SIZE)
_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class TradeDecisionLog(BaseModel):
    """A single auditable trade decision."""

    timestamp: datetime = Field(default_factory=_now_ist)
    algorithm: str
    action: str  # "OPEN", "CLOSE", "REJECT"
    strategy: str  # e.g. "Iron Condor"
    position_id: str | None = None
    vol_regime: str | None = None
    gate_checks: list[dict[str, Any]] = Field(default_factory=list)
    dynamic_params: dict[str, Any] = Field(default_factory=dict)
    score: float | None = None
    lots: int | None = None
    reject_reason: str | None = None
    exit_reason: str | None = None
    pnl: float | None = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def log_trade_decision(decision: TradeDecisionLog) -> None:
    """Log *decision* as JSON to the ``core.trade_audit`` logger and store in
    the in-memory ring buffer."""

    payload = decision.model_dump(mode="json")
    logger.info(json.dumps(payload, default=str))

    with _lock:
        _buffer.append(payload)


def get_recent_decisions(n: int = 10) -> list[dict]:
    """Return the last *n* decisions from the ring buffer (thread-safe)."""

    with _lock:
        # deque doesn't support negative slicing; convert to list tail
        items = list(_buffer)
    return items[-n:]
