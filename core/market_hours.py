"""NSE market hours utilities.

Centralised weekday + time-of-day + holiday checks so every component uses the
same definition of "market open".  Includes NSE holiday calendar for 2025-2026.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone

IST = timezone(timedelta(hours=5, minutes=30))

# NSE holidays for 2025 and 2026 (excludes weekends).
# Source: NSE circulars. Update annually.
_NSE_HOLIDAYS: set[date] = {
    # 2025
    date(2025, 2, 26),   # Mahashivratri
    date(2025, 3, 14),   # Holi
    date(2025, 3, 31),   # Id-Ul-Fitr (Ramadan)
    date(2025, 4, 10),   # Shri Mahavir Jayanti
    date(2025, 4, 14),   # Dr. Baba Saheb Ambedkar Jayanti
    date(2025, 4, 18),   # Good Friday
    date(2025, 5, 1),    # Maharashtra Day
    date(2025, 6, 7),    # Id-Ul-Adha (Bakri Id)
    date(2025, 8, 15),   # Independence Day
    date(2025, 8, 16),   # Janmashtami
    date(2025, 8, 27),   # Milad-Un-Nabi
    date(2025, 10, 2),   # Mahatma Gandhi Jayanti
    date(2025, 10, 21),  # Dussehra
    date(2025, 10, 22),  # Dussehra (additional)
    date(2025, 11, 5),   # Prakash Gurpurab Sri Guru Nanak Dev
    date(2025, 11, 12),  # Diwali (Laxmi Pujan)
    date(2025, 12, 25),  # Christmas
    # 2026
    date(2026, 1, 26),   # Republic Day
    date(2026, 2, 17),   # Mahashivratri
    date(2026, 3, 3),    # Holi
    date(2026, 3, 20),   # Id-Ul-Fitr (Ramadan)
    date(2026, 3, 25),   # Shri Mahavir Jayanti
    date(2026, 4, 3),    # Good Friday
    date(2026, 4, 14),   # Dr. Baba Saheb Ambedkar Jayanti
    date(2026, 5, 1),    # Maharashtra Day
    date(2026, 5, 28),   # Id-Ul-Adha (Bakri Id)
    date(2026, 8, 15),   # Independence Day
    date(2026, 8, 25),   # Milad-Un-Nabi
    date(2026, 10, 2),   # Mahatma Gandhi Jayanti
    date(2026, 10, 9),   # Dussehra
    date(2026, 10, 26),  # Diwali (Laxmi Pujan)
    date(2026, 11, 16),  # Prakash Gurpurab Sri Guru Nanak Dev
    date(2026, 11, 25),  # Guru Tegh Bahadur Martyrdom Day
    date(2026, 12, 25),  # Christmas
}


def is_nse_holiday(d: date | None = None) -> bool:
    """True if the given date (default: today IST) is an NSE holiday."""
    if d is None:
        d = datetime.now(IST).date()
    return d in _NSE_HOLIDAYS


def is_trading_day(d: date | None = None) -> bool:
    """True if the given date is a weekday and not an NSE holiday."""
    if d is None:
        d = datetime.now(IST).date()
    return d.weekday() < 5 and d not in _NSE_HOLIDAYS


def is_market_open() -> bool:
    """True if right now is within NSE trading hours (Mon-Fri, 9:15-15:30 IST, not a holiday)."""
    now = datetime.now(IST)
    d = now.date()
    if d.weekday() >= 5 or d in _NSE_HOLIDAYS:
        return False
    return time(9, 15) <= now.time() <= time(15, 30)


def next_trading_day(d: date | None = None) -> date:
    """Return the next trading day after the given date (default: today IST)."""
    if d is None:
        d = datetime.now(IST).date()
    candidate = d + timedelta(days=1)
    while candidate.weekday() >= 5 or candidate in _NSE_HOLIDAYS:
        candidate += timedelta(days=1)
    return candidate
