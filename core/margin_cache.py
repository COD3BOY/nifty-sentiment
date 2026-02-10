"""JSON disk cache for margin-per-lot values from successful Kite SPAN calls.

Stores per-strategy margin so that when Kite is unavailable, the system can
fall back to a recent (< 24 h) cached value instead of a static estimate.

Cache file: data/margin_cache.json
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_IST = timezone(timedelta(hours=5, minutes=30))
_CACHE_FILE = Path(__file__).resolve().parent.parent / "data" / "margin_cache.json"
_TTL = timedelta(hours=24)


def _load_cache() -> dict:
    """Load cache from disk. Returns empty structure on missing/corrupt file."""
    if not _CACHE_FILE.exists():
        return {"version": 1, "entries": {}}
    try:
        data = json.loads(_CACHE_FILE.read_text())
        if not isinstance(data, dict) or "entries" not in data:
            raise ValueError("unexpected cache structure")
        return data
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("Corrupt margin cache, starting fresh: %s", exc)
        return {"version": 1, "entries": {}}


def get_cached_margin(strategy_key: str) -> float | None:
    """Return cached margin_per_lot if entry exists and is < 24 h old."""
    cache = _load_cache()
    entry = cache["entries"].get(strategy_key)
    if entry is None:
        return None
    try:
        ts = datetime.fromisoformat(entry["timestamp"])
    except (KeyError, ValueError):
        return None
    if datetime.now(_IST) - ts > _TTL:
        return None
    return entry.get("margin_per_lot")


def update_cached_margin(strategy_key: str, margin_per_lot: float) -> None:
    """Write/update a cache entry. Atomic write via tmp + replace."""
    cache = _load_cache()
    cache["entries"][strategy_key] = {
        "margin_per_lot": margin_per_lot,
        "timestamp": datetime.now(_IST).isoformat(),
    }
    _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = _CACHE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache, indent=2))
    tmp.replace(_CACHE_FILE)  # atomic on POSIX
