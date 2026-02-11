"""Black-Scholes Greeks and probability-of-profit calculations.

Reuses _norm_cdf from iv_calculator to avoid duplicating the Abramowitz & Stegun
approximation.
"""

from __future__ import annotations

import math

from core.iv_calculator import _norm_cdf
from core.options_models import OptionChainData, StrikeData


def bs_delta(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str,
) -> float:
    """Black-Scholes delta for a European option.

    Parameters
    ----------
    S : underlying price
    K : strike price
    T : time to expiry in years
    r : risk-free rate (annual, decimal)
    sigma : volatility (annual, decimal — e.g. 0.15 for 15%)
    option_type : "CE" for call, "PE" for put

    Returns delta in [-1, 1].
    """
    if T <= 0 or sigma <= 0:
        if option_type == "CE":
            return 1.0 if S > K else 0.0
        return -1.0 if S < K else 0.0

    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)

    if option_type == "CE":
        return _norm_cdf(d1)
    else:
        return _norm_cdf(d1) - 1.0


def compute_chain_deltas(
    chain: OptionChainData,
    T: float,
    risk_free_rate: float = 0.065,
) -> list[StrikeData]:
    """Compute Black-Scholes delta for all strikes with non-zero IV.

    Returns a new list of StrikeData with ce_delta and pe_delta populated.
    """
    S = chain.underlying_value
    if S <= 0 or T <= 0:
        return list(chain.strikes)

    result: list[StrikeData] = []
    for strike in chain.strikes:
        K = strike.strike_price
        updates: dict = {}

        # CE delta
        if strike.ce_iv > 0:
            sigma = strike.ce_iv / 100.0  # IV stored as percentage
            updates["ce_delta"] = round(bs_delta(S, K, T, risk_free_rate, sigma, "CE"), 4)

        # PE delta
        if strike.pe_iv > 0:
            sigma = strike.pe_iv / 100.0
            updates["pe_delta"] = round(bs_delta(S, K, T, risk_free_rate, sigma, "PE"), 4)

        result.append(strike.model_copy(update=updates) if updates else strike)

    return result


def compute_pop(
    strategy_type: str,
    short_strike_delta: float = 0.0,
    breakeven: float = 0.0,
    spot: float = 0.0,
    atm_iv: float = 0.0,
    T: float = 0.0,
) -> float:
    """Compute probability of profit (0-100).

    Credit spread POP ≈ 1 - |delta of short strike|
    Debit spread POP from breakeven probability using log-normal model.
    """
    if strategy_type == "credit":
        # POP = probability that the short strike stays OTM
        return round((1.0 - abs(short_strike_delta)) * 100, 1)

    # Debit: probability of reaching breakeven
    if spot > 0 and breakeven > 0 and atm_iv > 0 and T > 0:
        sigma = atm_iv / 100.0
        # P(S_T > breakeven) for long call, P(S_T < breakeven) for long put
        d2 = (math.log(spot / breakeven) + (0.0 - 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
        # For a long call (breakeven > spot typically): P(S_T > breakeven)
        if breakeven > spot:
            return round(_norm_cdf(d2) * 100, 1)
        else:
            return round((1.0 - _norm_cdf(d2)) * 100, 1)

    return 0.0
