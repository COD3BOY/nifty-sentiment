"""Static event calendar for NIFTY options trading.

Known high-impact events where implied volatility typically expands.
Block short straddles/strangles on event days; boost long straddle scores.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

_IST = timezone(timedelta(hours=5, minutes=30))

# -----------------------------------------------------------------
# Known event dates for 2025-2026 (update periodically)
# -----------------------------------------------------------------

_EVENTS: list[dict] = [
    # RBI Monetary Policy (bi-monthly)
    {"date": "2025-04-09", "type": "rbi_policy", "label": "RBI MPC Decision"},
    {"date": "2025-06-06", "type": "rbi_policy", "label": "RBI MPC Decision"},
    {"date": "2025-08-08", "type": "rbi_policy", "label": "RBI MPC Decision"},
    {"date": "2025-10-08", "type": "rbi_policy", "label": "RBI MPC Decision"},
    {"date": "2025-12-05", "type": "rbi_policy", "label": "RBI MPC Decision"},
    {"date": "2026-02-06", "type": "rbi_policy", "label": "RBI MPC Decision"},
    {"date": "2026-04-08", "type": "rbi_policy", "label": "RBI MPC Decision"},

    # Union Budget
    {"date": "2025-07-01", "type": "budget", "label": "Union Budget"},
    {"date": "2026-02-01", "type": "budget", "label": "Union Budget"},

    # US Fed FOMC
    {"date": "2025-03-19", "type": "fomc", "label": "US Fed FOMC Decision"},
    {"date": "2025-05-07", "type": "fomc", "label": "US Fed FOMC Decision"},
    {"date": "2025-06-18", "type": "fomc", "label": "US Fed FOMC Decision"},
    {"date": "2025-07-30", "type": "fomc", "label": "US Fed FOMC Decision"},
    {"date": "2025-09-17", "type": "fomc", "label": "US Fed FOMC Decision"},
    {"date": "2025-11-05", "type": "fomc", "label": "US Fed FOMC Decision"},
    {"date": "2025-12-17", "type": "fomc", "label": "US Fed FOMC Decision"},
    {"date": "2026-01-28", "type": "fomc", "label": "US Fed FOMC Decision"},
    {"date": "2026-03-18", "type": "fomc", "label": "US Fed FOMC Decision"},
]


def _parse_date(d: str) -> date:
    return datetime.strptime(d, "%Y-%m-%d").date()


def is_event_day(check_date: date | None = None) -> tuple[bool, str | None]:
    """Check if *check_date* (default today IST) is a known high-impact event day.

    Returns (is_event, event_label).
    """
    if check_date is None:
        check_date = datetime.now(_IST).date()

    for ev in _EVENTS:
        if _parse_date(ev["date"]) == check_date:
            return True, ev["label"]
    return False, None


def is_near_event(check_date: date | None = None, days_ahead: int = 1) -> tuple[bool, str | None]:
    """Check if an event is within *days_ahead* days from *check_date*.

    Returns (is_near, event_label).
    """
    if check_date is None:
        check_date = datetime.now(_IST).date()

    for ev in _EVENTS:
        ev_date = _parse_date(ev["date"])
        diff = (ev_date - check_date).days
        if 0 <= diff <= days_ahead:
            return True, ev["label"]
    return False, None
