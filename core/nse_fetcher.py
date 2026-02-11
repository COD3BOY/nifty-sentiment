"""Kite Connect option chain fetcher with Black-Scholes IV computation."""

import logging
import os
from datetime import datetime

from core.config import load_config
from core.error_types import AuthenticationError, DataFetchError
from core.iv_calculator import compute_iv_for_chain
from core.options_models import OptionChainData, StrikeData

logger = logging.getLogger(__name__)


class KiteOptionChainFetcher:
    """Fetches NIFTY option chain data via Kite Connect API.

    IV is computed post-fetch via Black-Scholes since Kite quote()
    does not return implied volatility.
    """

    _QUOTE_BATCH_SIZE = 500  # Kite max instruments per quote() call

    def __init__(self) -> None:
        pass

    def _get_kite(self):
        """Create a fresh KiteConnect instance each time.

        Re-reads env vars on every call so that a refreshed access token
        is picked up without restarting the process.
        """
        from dotenv import load_dotenv
        load_dotenv(override=True)  # re-read .env to pick up token refreshes

        api_key = os.environ.get("KITE_API_KEY", "")
        access_token = os.environ.get("KITE_ACCESS_TOKEN", "")
        if not api_key or not access_token:
            return None

        from kiteconnect import KiteConnect

        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(access_token)
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
        except Exception as e:
            _check_token_error(e)
            logger.error("Failed to fetch underlying quote for %s: %s", index_symbol, e)
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
                    ce_iv=0.0,  # IV computed post-loop via Black-Scholes
                    ce_ltp=float(ce.get("last_price", 0)),
                    ce_bid=float(ce_buy[0].get("price", 0)) if ce_buy else 0.0,
                    ce_ask=float(ce_sell[0].get("price", 0)) if ce_sell else 0.0,
                    pe_oi=pe_oi,
                    pe_change_in_oi=0.0,
                    pe_volume=int(pe.get("volume", 0)),
                    pe_iv=0.0,  # IV computed post-loop via Black-Scholes
                    pe_ltp=float(pe.get("last_price", 0)),
                    pe_bid=float(pe_buy[0].get("price", 0)) if pe_buy else 0.0,
                    pe_ask=float(pe_sell[0].get("price", 0)) if pe_sell else 0.0,
                )
            )
            total_ce_oi += ce_oi
            total_pe_oi += pe_oi

        # Compute implied volatility from Black-Scholes (Kite doesn't provide IV)
        if underlying > 0 and strikes:
            try:
                iv_cfg = load_config().get("options_desk", {}).get("iv_calculator", {})
                strikes = compute_iv_for_chain(
                    strikes, underlying, nearest_expiry,
                    risk_free_rate=iv_cfg.get("risk_free_rate", 0.065),
                    atm_range=iv_cfg.get("atm_range", 15),
                )
            except Exception as exc:
                logger.warning("IV computation failed, proceeding without IV: %s", exc)

        expiry_str = nearest_expiry.strftime("%d-%b-%Y") if hasattr(nearest_expiry, "strftime") else str(nearest_expiry)

        return OptionChainData(
            symbol=symbol,
            underlying_value=underlying,
            expiry=expiry_str,
            strikes=strikes,
            total_ce_oi=total_ce_oi,
            total_pe_oi=total_pe_oi,
        )


def _check_token_error(exc: Exception) -> None:
    """Raise AuthenticationError if the exception looks like a Kite token issue."""
    try:
        from kiteconnect.exceptions import TokenException
        if isinstance(exc, TokenException):
            raise AuthenticationError(
                "Kite access token expired or invalid. "
                "Regenerate via utils/kite_auth.py and update KITE_ACCESS_TOKEN."
            ) from exc
    except ImportError:
        pass
    exc_str = str(exc).lower()
    if "token" in exc_str and ("invalid" in exc_str or "expired" in exc_str):
        raise AuthenticationError(
            "Kite access token appears invalid. "
            "Regenerate via utils/kite_auth.py and update KITE_ACCESS_TOKEN."
        ) from exc


def is_kite_token_valid() -> bool:
    """Quick health check — attempt a lightweight Kite API call."""
    try:
        fetcher = KiteOptionChainFetcher()
        kite = fetcher._get_kite()
        if kite is None:
            return False
        kite.profile()
        return True
    except Exception:
        return False
