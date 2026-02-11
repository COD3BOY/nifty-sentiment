"""IV history storage and percentile computation."""

from __future__ import annotations

from datetime import datetime, timedelta


def save_iv_reading(db, symbol: str, atm_iv: float, vix: float | None = None) -> None:
    """Save a single IV reading to the iv_history table."""
    if atm_iv <= 0:
        return
    from core.database import IVHistoryRow
    with db.SessionLocal() as session:
        row = IVHistoryRow(
            timestamp=datetime.utcnow(),
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
        return 50.0  # no history yet, assume mid-range

    historical_ivs = [r[0] for r in rows]
    below = sum(1 for iv in historical_ivs if iv < current_iv)
    return round((below / len(historical_ivs)) * 100, 1)
