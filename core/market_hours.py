"""NSE market hours utilities.

Centralised weekday + time-of-day checks so every component uses the same
definition of "market open".  No NSE holiday calendar yet — only weekday
filtering (Mon-Fri, 9:15-15:30 IST).
"""

from __future__ import annotations

from datetime import datetime, time, timedelta, timezone

IST = timezone(timedelta(hours=5, minutes=30))


def is_trading_day() -> bool:
    """True if today is a weekday (Mon-Fri)."""
    return datetime.now(IST).weekday() < 5


def is_market_open() -> bool:
    """True if right now is within NSE trading hours (Mon-Fri, 9:15-15:30 IST)."""
    now = datetime.now(IST)
    if now.weekday() >= 5:
        return False
    return time(9, 15) <= now.time() <= time(15, 30)
