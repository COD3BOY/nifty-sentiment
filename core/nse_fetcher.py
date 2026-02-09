"""NSE Option Chain API client with session/cookie management."""

import logging
import time

import requests

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
