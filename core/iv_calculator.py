"""Black-Scholes implied volatility calculator.

Uses Newton-Raphson with bisection fallback to solve for IV from
market prices.  Pure Python — only dependency is ``math`` stdlib.
"""

import math
from datetime import date

from core.options_models import StrikeData

# ---------------------------------------------------------------------------
# Normal distribution helpers (Abramowitz & Stegun approximation)
# ---------------------------------------------------------------------------

_A1 = 0.254829592
_A2 = -0.284496736
_A3 = 1.421413741
_A4 = -1.453152027
_A5 = 1.061405429
_P = 0.3275911
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def _norm_cdf(x: float) -> float:
    """Cumulative standard-normal distribution (max error ~1.5e-7)."""
    sign = 1.0 if x >= 0 else -1.0
    x = abs(x)
    t = 1.0 / (1.0 + _P * x)
    y = 1.0 - (
        ((((_A5 * t + _A4) * t + _A3) * t + _A2) * t + _A1)
    ) * t * math.exp(-x * x / 2.0)
    return 0.5 * (1.0 + sign * y)


def _norm_pdf(x: float) -> float:
    """Standard-normal probability density function."""
    return _INV_SQRT_2PI * math.exp(-0.5 * x * x)


# ---------------------------------------------------------------------------
# Black-Scholes pricing & vega
# ---------------------------------------------------------------------------


def bs_price(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str
) -> float:
    """European option price under Black-Scholes.

    Parameters
    ----------
    S : underlying price
    K : strike price
    T : time to expiry in years
    r : risk-free rate (annual, decimal)
    sigma : volatility (annual, decimal — e.g. 0.15 for 15%)
    option_type : ``"CE"`` for call, ``"PE"`` for put
    """
    if S <= 0 or K <= 0:
        return 0.0

    if T <= 0 or sigma <= 0:
        # Return intrinsic value
        if option_type == "CE":
            return max(S - K, 0.0)
        return max(K - S, 0.0)

    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    if option_type == "CE":
        return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    else:
        return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def bs_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes vega (dPrice/dSigma)."""
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return 0.0

    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    return S * sqrt_T * _norm_pdf(d1)


# ---------------------------------------------------------------------------
# Newton-Raphson IV solver with bisection fallback
# ---------------------------------------------------------------------------

_MAX_ITER = 50
_TOLERANCE = 1e-5
_SIGMA_MIN = 0.001
_SIGMA_MAX = 5.0
_INITIAL_GUESS = 0.20


def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str,
) -> float | None:
    """Solve for implied volatility via Newton-Raphson.

    Returns IV as a **percentage** (e.g. 14.5 for 14.5%) to match the
    convention used by NSE, or ``None`` if IV cannot be computed.
    """
    import logging

    _logger = logging.getLogger(__name__)

    # --- Pre-checks ---
    if S <= 0 or K <= 0:
        return None

    if market_price <= 0 or T <= 0:
        return None

    intrinsic = max(S - K, 0.0) if option_type == "CE" else max(K - S, 0.0)
    if market_price < intrinsic:
        return None

    # --- Newton-Raphson ---
    sigma = _INITIAL_GUESS

    for _ in range(_MAX_ITER):
        price = bs_price(S, K, T, r, sigma, option_type)
        vega = bs_vega(S, K, T, r, sigma)

        diff = price - market_price

        if abs(diff) < _TOLERANCE:
            return sigma * 100.0

        if vega < 1e-10:
            # Vega too small — switch to bisection
            break

        sigma -= diff / vega
        sigma = max(_SIGMA_MIN, min(sigma, _SIGMA_MAX))
    else:
        # Loop completed without convergence or early break
        final_price = bs_price(S, K, T, r, sigma, option_type)
        if abs(final_price - market_price) < _TOLERANCE * 10:
            return sigma * 100.0
        _logger.debug(
            "Newton-Raphson did not converge: S=%.1f K=%.1f T=%.4f diff=%.4f",
            S, K, T, final_price - market_price,
        )
        return None

    # --- Bisection fallback ---
    lo, hi = _SIGMA_MIN, _SIGMA_MAX

    for _ in range(_MAX_ITER):
        mid = (lo + hi) / 2.0
        price = bs_price(S, K, T, r, mid, option_type)
        if abs(price - market_price) < _TOLERANCE:
            return mid * 100.0
        if price < market_price:
            lo = mid
        else:
            hi = mid

    # Bisection finished without meeting tolerance
    result = ((lo + hi) / 2.0) * 100.0
    final_price = bs_price(S, K, T, r, (lo + hi) / 2.0, option_type)
    if abs(final_price - market_price) < _TOLERANCE * 100:
        _logger.debug("Bisection returning approximate IV=%.2f%%", result)
        return result

    _logger.debug("Bisection did not converge: S=%.1f K=%.1f T=%.4f", S, K, T)
    return None


# ---------------------------------------------------------------------------
# Batch IV computation for an option chain
# ---------------------------------------------------------------------------

_IV_FLOOR = 0.5    # discard IV below 0.5%
_IV_CAP = 200.0    # discard IV above 200%
_EXPIRY_DAY_T = 0.5 / 365.0  # half-day for expiry-day options


def compute_iv_for_chain(
    strikes: list[StrikeData],
    underlying: float,
    expiry_date: date,
    risk_free_rate: float = 0.065,
    atm_range: int = 15,
) -> list[StrikeData]:
    """Compute Black-Scholes IV for near-ATM strikes.

    Parameters
    ----------
    strikes : list of StrikeData from the option chain
    underlying : current spot / underlying price
    expiry_date : expiry as a ``datetime.date``
    risk_free_rate : annual risk-free rate (decimal)
    atm_range : compute IV only for ± this many strikes around ATM

    Returns a **new** list with ``ce_iv`` / ``pe_iv`` populated.
    """
    if not strikes or underlying <= 0:
        return strikes

    # Find ATM index
    atm_idx = min(
        range(len(strikes)),
        key=lambda i: abs(strikes[i].strike_price - underlying),
    )
    lo = max(0, atm_idx - atm_range)
    hi = min(len(strikes), atm_idx + atm_range + 1)

    # Time to expiry
    today = date.today()
    calendar_days = (expiry_date - today).days
    T = max(calendar_days / 365.0, _EXPIRY_DAY_T)

    result: list[StrikeData] = []

    for i, strike in enumerate(strikes):
        if i < lo or i >= hi:
            result.append(strike)
            continue

        S = underlying
        K = strike.strike_price
        r = risk_free_rate

        # Compute CE IV
        ce_iv = 0.0
        if strike.ce_ltp > 0 and not (isinstance(strike.ce_ltp, float) and math.isnan(strike.ce_ltp)):
            raw = implied_volatility(strike.ce_ltp, S, K, T, r, "CE")
            if raw is not None and _IV_FLOOR <= raw <= _IV_CAP:
                ce_iv = round(raw, 2)

        # Compute PE IV
        pe_iv = 0.0
        if strike.pe_ltp > 0 and not (isinstance(strike.pe_ltp, float) and math.isnan(strike.pe_ltp)):
            raw = implied_volatility(strike.pe_ltp, S, K, T, r, "PE")
            if raw is not None and _IV_FLOOR <= raw <= _IV_CAP:
                pe_iv = round(raw, 2)

        result.append(strike.model_copy(update={"ce_iv": ce_iv, "pe_iv": pe_iv}))

    return result
