"""Kite Connect helpers for real-time SPAN margins and trade charges.

Both public functions return ``None`` on any failure (missing creds, token
expired, API error) so callers can fall back to static estimates.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-memory cache (5-minute TTL)
# ---------------------------------------------------------------------------

_cache: dict[str, tuple[float, float]] = {}  # key -> (timestamp, value)
_CACHE_TTL = 300  # seconds


def _cache_get(key: str) -> float | None:
    entry = _cache.get(key)
    if entry and (time.time() - entry[0]) < _CACHE_TTL:
        return entry[1]
    return None


def _cache_set(key: str, value: float) -> None:
    _cache[key] = (time.time(), value)


# ---------------------------------------------------------------------------
# Kite client factory
# ---------------------------------------------------------------------------

def _get_kite():
    """Create a fresh KiteConnect instance, re-reading .env each call."""
    from dotenv import load_dotenv

    load_dotenv(override=True)

    api_key = os.environ.get("KITE_API_KEY", "")
    access_token = os.environ.get("KITE_ACCESS_TOKEN", "")
    if not api_key or not access_token:
        return None

    from kiteconnect import KiteConnect

    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    return kite


# ---------------------------------------------------------------------------
# Trading symbol builder
# ---------------------------------------------------------------------------

def _build_tradingsymbol(
    symbol: str, expiry: str, strike: float, option_type: str,
) -> str:
    """Convert leg params to Kite-format trading symbol.

    Example: ("NIFTY", "06-Feb-2026", 23000.0, "CE") -> "NIFTY26020623000CE"
    """
    dt = datetime.strptime(expiry, "%d-%b-%Y")
    yy = dt.strftime("%y")
    mm = dt.strftime("%m")
    dd = dt.strftime("%d")
    strike_int = int(strike)
    return f"{symbol}{yy}{mm}{dd}{strike_int}{option_type}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_kite_margin(
    legs: list[dict],
    expiry: str,
    lots: int,
    lot_size: int,
) -> float | None:
    """Fetch SPAN margin (with hedge benefit) via basket order margins.

    Parameters
    ----------
    legs : list[dict]
        Each dict has keys: action, strike, option_type, ltp
    expiry : str
        Expiry in ``dd-Mon-YYYY`` format (e.g. "06-Feb-2026")
    lots : int
        Number of lots per leg.
    lot_size : int
        Units per lot.

    Returns ``None`` on any failure so the caller can fall back to static estimates.
    """
    cache_key = f"margin:{expiry}:{lots}:{_legs_key(legs)}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    kite = _get_kite()
    if kite is None:
        return None

    try:
        orders = [
            {
                "exchange": "NFO",
                "tradingsymbol": _build_tradingsymbol("NIFTY", expiry, leg["strike"], leg["option_type"]),
                "transaction_type": "SELL" if leg["action"] == "SELL" else "BUY",
                "variety": "regular",
                "product": "NRML",
                "order_type": "MARKET",
                "quantity": lots * lot_size,
            }
            for leg in legs
        ]
        result = kite.basket_order_margins(orders)
        total_margin = result["final"]["total"]
        if total_margin <= 0:
            return None  # API returned zero — likely expired token
        _cache_set(cache_key, total_margin)
        return total_margin
    except Exception:
        logger.warning("Kite basket_order_margins failed, falling back to static", exc_info=True)
        return None


def get_kite_charges(
    legs: list[dict],
    expiry: str,
    lots: int,
    lot_size: int,
) -> float | None:
    """Fetch round-trip charges via virtual contract note.

    Returns total charges summed across entry + exit legs, or ``None`` on failure.
    """
    cache_key = f"charges:{expiry}:{lots}:{_legs_key(legs)}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    kite = _get_kite()
    if kite is None:
        return None

    try:
        # Build entry + exit orders (round-trip)
        orders = []
        for leg in legs:
            tsym = _build_tradingsymbol("NIFTY", expiry, leg["strike"], leg["option_type"])
            qty = lots * lot_size
            # Entry order
            orders.append({
                "exchange": "NFO",
                "tradingsymbol": tsym,
                "transaction_type": "SELL" if leg["action"] == "SELL" else "BUY",
                "variety": "regular",
                "product": "NRML",
                "order_type": "MARKET",
                "quantity": qty,
                "average_price": leg["ltp"],
            })
            # Exit order (reverse)
            orders.append({
                "exchange": "NFO",
                "tradingsymbol": tsym,
                "transaction_type": "BUY" if leg["action"] == "SELL" else "SELL",
                "variety": "regular",
                "product": "NRML",
                "order_type": "MARKET",
                "quantity": qty,
                "average_price": leg["ltp"],
            })
        result = kite.get_virtual_contract_note(orders)
        total_charges = sum(item.get("charges", {}).get("total", 0) for item in result)
        _cache_set(cache_key, total_charges)
        return total_charges
    except Exception:
        logger.warning("Kite get_virtual_contract_note failed, falling back to static", exc_info=True)
        return None


def _legs_key(legs: list[dict]) -> str:
    """Stable cache key fragment from leg dicts."""
    parts = []
    for leg in sorted(legs, key=lambda l: (l["strike"], l["option_type"])):
        parts.append(f"{leg['action']}{leg['strike']}{leg['option_type']}")
    return "|".join(parts)
