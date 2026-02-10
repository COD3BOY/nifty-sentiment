"""NSE Option Chain API client with session/cookie management."""

import logging
import time
from datetime import datetime

import requests

from core.config import get_env
from core.options_models import OptionChainData, StrikeData

logger = logging.getLogger(__name__)

_BASE_URL = "https://www.nseindia.com"
_CHAIN_URL = f"{_BASE_URL}/api/option-chain-indices"
_COOKIE_URL = f"{_BASE_URL}/option-chain"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": f"{_BASE_URL}/option-chain",
}
_COOKIE_MAX_AGE = 120  # seconds


class NseOptionChainFetcher:
    """Fetches NIFTY option chain data from NSE India."""

    def __init__(self) -> None:
        self._session = requests.Session()
        self._session.headers.update(_HEADERS)
        self._cookie_ts: float = 0.0

    def _ensure_session(self) -> None:
        """Hit the option-chain page to get/refresh cookies."""
        if time.time() - self._cookie_ts < _COOKIE_MAX_AGE:
            return
        try:
            resp = self._session.get(_COOKIE_URL, timeout=10)
            resp.raise_for_status()
            self._cookie_ts = time.time()
        except requests.RequestException as exc:
            logger.warning("NSE cookie refresh failed: %s", exc)
            raise

    def fetch(self, symbol: str = "NIFTY") -> OptionChainData:
        """Fetch the option chain for *symbol* and return parsed data."""
        self._ensure_session()
        try:
            resp = self._session.get(
                _CHAIN_URL, params={"symbol": symbol}, timeout=10
            )
            if resp.status_code in (401, 403):
                # Cookie likely expired — force refresh and retry once
                self._cookie_ts = 0.0
                self._ensure_session()
                resp = self._session.get(
                    _CHAIN_URL, params={"symbol": symbol}, timeout=10
                )
            resp.raise_for_status()
            return self._parse(resp.json(), symbol)
        except requests.RequestException as exc:
            logger.error("NSE option chain fetch failed: %s", exc)
            raise

    def _parse(self, raw: dict, symbol: str) -> OptionChainData:
        """Extract nearest-expiry strikes from NSE JSON response."""
        records = raw.get("records", {})
        filtered = raw.get("filtered", {})

        underlying = records.get("underlyingValue", 0.0)
        expiry_dates = records.get("expiryDates", [])
        nearest_expiry = expiry_dates[0] if expiry_dates else ""

        data_rows = records.get("data", [])
        strikes: list[StrikeData] = []
        total_ce_oi = 0.0
        total_pe_oi = 0.0

        for row in data_rows:
            if row.get("expiryDate") != nearest_expiry:
                continue

            ce = row.get("CE", {})
            pe = row.get("PE", {})
            strike_price = row.get("strikePrice", 0.0)

            ce_oi = ce.get("openInterest", 0.0)
            pe_oi = pe.get("openInterest", 0.0)

            strikes.append(
                StrikeData(
                    strike_price=strike_price,
                    ce_oi=ce_oi,
                    ce_change_in_oi=ce.get("changeinOpenInterest", 0.0),
                    ce_volume=ce.get("totalTradedVolume", 0),
                    ce_iv=ce.get("impliedVolatility", 0.0),
                    ce_ltp=ce.get("lastPrice", 0.0),
                    ce_bid=ce.get("bidprice", 0.0),
                    ce_ask=ce.get("askPrice", 0.0),
                    pe_oi=pe_oi,
                    pe_change_in_oi=pe.get("changeinOpenInterest", 0.0),
                    pe_volume=pe.get("totalTradedVolume", 0),
                    pe_iv=pe.get("impliedVolatility", 0.0),
                    pe_ltp=pe.get("lastPrice", 0.0),
                    pe_bid=pe.get("bidprice", 0.0),
                    pe_ask=pe.get("askPrice", 0.0),
                )
            )
            total_ce_oi += ce_oi
            total_pe_oi += pe_oi

        # Use filtered totals from NSE if available (more accurate)
        if filtered:
            ce_total = filtered.get("CE", {}).get("totOI", total_ce_oi)
            pe_total = filtered.get("PE", {}).get("totOI", total_pe_oi)
        else:
            ce_total = total_ce_oi
            pe_total = total_pe_oi

        return OptionChainData(
            symbol=symbol,
            underlying_value=underlying,
            expiry=nearest_expiry,
            strikes=strikes,
            total_ce_oi=ce_total,
            total_pe_oi=pe_total,
        )


class KiteOptionChainFetcher:
    """Fetches NIFTY option chain data via Kite Connect API.

    Used as a fallback when NSE direct access is blocked by bot protection.
    Note: Kite quote() does not return implied volatility, so IV fields are 0.
    """

    _QUOTE_BATCH_SIZE = 500  # Kite max instruments per quote() call

    def __init__(self) -> None:
        self._kite = None

    def _get_kite(self):
        """Lazy-init KiteConnect; returns instance or None if creds missing."""
        if self._kite is not None:
            return self._kite

        api_key = get_env("KITE_API_KEY")
        access_token = get_env("KITE_ACCESS_TOKEN")
        if not api_key or not access_token:
            return None

        from kiteconnect import KiteConnect

        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(access_token)
        self._kite = kite
        return kite

    def fetch(self, symbol: str = "NIFTY") -> OptionChainData:
        """Fetch option chain for *symbol* from Kite Connect."""
        kite = self._get_kite()
        if kite is None:
            raise ValueError("Kite Connect credentials not configured (KITE_API_KEY / KITE_ACCESS_TOKEN)")

        # 1. Get all NFO instruments and filter for this symbol
        all_instruments = kite.instruments("NFO")
        option_instruments = [
            inst for inst in all_instruments
            if inst["name"] == symbol
            and inst["instrument_type"] in ("CE", "PE")
        ]
        if not option_instruments:
            raise ValueError(f"No NFO option instruments found for {symbol}")

        # 2. Find nearest expiry
        expiries = sorted({inst["expiry"] for inst in option_instruments})
        today = datetime.now().date()
        future_expiries = [e for e in expiries if e >= today]
        nearest_expiry = future_expiries[0] if future_expiries else expiries[-1]

        # Filter to nearest expiry only
        expiry_instruments = [
            inst for inst in option_instruments
            if inst["expiry"] == nearest_expiry
        ]

        # 3. Get underlying value
        index_symbol = "NSE:NIFTY 50" if symbol == "NIFTY" else f"NSE:{symbol}"
        try:
            index_quote = kite.quote([index_symbol])
            underlying = index_quote[index_symbol]["last_price"]
        except Exception:
            underlying = 0.0

        # 4. Batch quote all option instruments
        trading_symbols = [f"NFO:{inst['tradingsymbol']}" for inst in expiry_instruments]
        quotes: dict = {}
        for i in range(0, len(trading_symbols), self._QUOTE_BATCH_SIZE):
            batch = trading_symbols[i : i + self._QUOTE_BATCH_SIZE]
            quotes.update(kite.quote(batch))

        # 5. Build a lookup: strike -> {CE: quote, PE: quote}
        strike_map: dict[float, dict[str, dict]] = {}
        inst_lookup = {inst["tradingsymbol"]: inst for inst in expiry_instruments}

        for ts_key, q in quotes.items():
            # ts_key is "NFO:TRADINGSYMBOL"
            ts = ts_key.split(":", 1)[1] if ":" in ts_key else ts_key
            inst = inst_lookup.get(ts)
            if not inst:
                continue
            strike = float(inst["strike"])
            opt_type = inst["instrument_type"]  # "CE" or "PE"
            strike_map.setdefault(strike, {})[opt_type] = q

        # 6. Build StrikeData list
        strikes: list[StrikeData] = []
        total_ce_oi = 0.0
        total_pe_oi = 0.0

        for strike_price in sorted(strike_map.keys()):
            data = strike_map[strike_price]
            ce = data.get("CE", {})
            pe = data.get("PE", {})

            ce_oi = float(ce.get("oi", 0))
            pe_oi = float(pe.get("oi", 0))

            ce_depth = ce.get("depth", {})
            pe_depth = pe.get("depth", {})
            ce_buy = ce_depth.get("buy", [{}])
            ce_sell = ce_depth.get("sell", [{}])
            pe_buy = pe_depth.get("buy", [{}])
            pe_sell = pe_depth.get("sell", [{}])

            strikes.append(
                StrikeData(
                    strike_price=strike_price,
                    ce_oi=ce_oi,
                    ce_change_in_oi=0.0,  # Kite doesn't provide change in OI
                    ce_volume=int(ce.get("volume", 0)),
                    ce_iv=0.0,  # Kite doesn't provide IV
                    ce_ltp=float(ce.get("last_price", 0)),
                    ce_bid=float(ce_buy[0].get("price", 0)) if ce_buy else 0.0,
                    ce_ask=float(ce_sell[0].get("price", 0)) if ce_sell else 0.0,
                    pe_oi=pe_oi,
                    pe_change_in_oi=0.0,
                    pe_volume=int(pe.get("volume", 0)),
                    pe_iv=0.0,
                    pe_ltp=float(pe.get("last_price", 0)),
                    pe_bid=float(pe_buy[0].get("price", 0)) if pe_buy else 0.0,
                    pe_ask=float(pe_sell[0].get("price", 0)) if pe_sell else 0.0,
                )
            )
            total_ce_oi += ce_oi
            total_pe_oi += pe_oi

        expiry_str = nearest_expiry.strftime("%d-%b-%Y") if hasattr(nearest_expiry, "strftime") else str(nearest_expiry)

        return OptionChainData(
            symbol=symbol,
            underlying_value=underlying,
            expiry=expiry_str,
            strikes=strikes,
            total_ce_oi=total_ce_oi,
            total_pe_oi=total_pe_oi,
        )
