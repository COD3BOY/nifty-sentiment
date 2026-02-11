"""IV history storage and percentile computation."""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

_IV_STORAGE_FLOOR = 0.5
_IV_STORAGE_CAP = 200.0
_DEDUP_MINUTES = 5


def save_iv_reading(db, symbol: str, atm_iv: float, vix: float | None = None) -> None:
    """Save a single IV reading to the iv_history table.

    Rejects invalid IVs and deduplicates within 5-minute windows.
    """
    # Validate IV bounds
    if not isinstance(atm_iv, (int, float)) or not math.isfinite(atm_iv):
        logger.warning("IV rejected (not finite): %s", atm_iv)
        return
    if not (_IV_STORAGE_FLOOR <= atm_iv <= _IV_STORAGE_CAP):
        logger.warning("IV rejected (out of bounds [%.1f,%.1f]): %.2f", _IV_STORAGE_FLOOR, _IV_STORAGE_CAP, atm_iv)
        return

    from core.database import IVHistoryRow

    now = datetime.utcnow()

    with db.SessionLocal() as session:
        # Dedup: skip if a reading exists within the last DEDUP_MINUTES
        cutoff = now - timedelta(minutes=_DEDUP_MINUTES)
        existing = (
            session.query(IVHistoryRow)
            .filter(
                IVHistoryRow.symbol == symbol,
                IVHistoryRow.timestamp >= cutoff,
            )
            .first()
        )
        if existing:
            return

        row = IVHistoryRow(
            timestamp=now,
            symbol=symbol,
            atm_iv=atm_iv,
            vix=vix,
        )
        session.add(row)
        session.commit()


def get_iv_percentile(
    db,
    symbol: str,
    current_iv: float,
    lookback_days: int = 252,
) -> float:
    """Compute the percentile rank of current_iv over the lookback period.

    Returns a value 0-100 indicating what percentage of historical readings
    were below the current IV.
    """
    if current_iv <= 0:
        logger.warning("IV percentile fallback: current_iv=%.2f <= 0 for %s, returning 50.0", current_iv, symbol)
        return 50.0

    from core.database import IVHistoryRow
    cutoff = datetime.utcnow() - timedelta(days=lookback_days)

    with db.SessionLocal() as session:
        rows = (
            session.query(IVHistoryRow.atm_iv)
            .filter(
                IVHistoryRow.symbol == symbol,
                IVHistoryRow.timestamp >= cutoff,
            )
            .all()
        )

    if not rows:
        logger.warning("IV percentile fallback: no history for %s in last %d days, returning 50.0", symbol, lookback_days)
        return 50.0

    historical_ivs = [r[0] for r in rows]
    below = sum(1 for iv in historical_ivs if iv < current_iv)
    return round((below / len(historical_ivs)) * 100, 1)
