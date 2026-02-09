"""Pure functions for option chain analytics."""

from core.options_models import OptionChainData, OptionsAnalytics


def compute_pcr(chain: OptionChainData) -> float:
    """Total Put OI / Total Call OI."""
    if chain.total_ce_oi == 0:
        return 0.0
    return chain.total_pe_oi / chain.total_ce_oi


def compute_max_pain(chain: OptionChainData) -> float:
    """Find strike where total buyer pain (intrinsic value * OI) is minimized."""
    if not chain.strikes:
        return 0.0

    min_pain = float("inf")
    max_pain_strike = chain.strikes[0].strike_price

    for candidate in chain.strikes:
        total_pain = 0.0
        sp = candidate.strike_price
        for s in chain.strikes:
            # CE buyers pain: max(0, strike - candidate) * CE OI
            ce_pain = max(0.0, s.strike_price - sp) * s.ce_oi
            # PE buyers pain: max(0, candidate - strike) * PE OI
            pe_pain = max(0.0, sp - s.strike_price) * s.pe_oi
            total_pain += ce_pain + pe_pain

        if total_pain < min_pain:
            min_pain = total_pain
            max_pain_strike = sp

    return max_pain_strike


def compute_oi_levels(
    chain: OptionChainData,
) -> tuple[float, float, float, float]:
    """Find highest Put OI (support) and highest Call OI (resistance).

    Returns (support_strike, support_oi, resistance_strike, resistance_oi).
    """
    if not chain.strikes:
        return 0.0, 0.0, 0.0, 0.0

    max_pe_oi = 0.0
    support_strike = 0.0
    max_ce_oi = 0.0
    resistance_strike = 0.0

    for s in chain.strikes:
        if s.pe_oi > max_pe_oi:
            max_pe_oi = s.pe_oi
            support_strike = s.strike_price
        if s.ce_oi > max_ce_oi:
            max_ce_oi = s.ce_oi
            resistance_strike = s.strike_price

    return support_strike, max_pe_oi, resistance_strike, max_ce_oi


def find_atm_strike(chain: OptionChainData) -> float:
    """Find the strike closest to underlying value."""
    if not chain.strikes:
        return 0.0
    return min(chain.strikes, key=lambda s: abs(s.strike_price - chain.underlying_value)).strike_price


def compute_atm_iv(chain: OptionChainData) -> float:
    """Average of CE IV and PE IV at the ATM strike."""
    atm = find_atm_strike(chain)
    for s in chain.strikes:
        if s.strike_price == atm:
            ivs = [v for v in (s.ce_iv, s.pe_iv) if v > 0]
            return sum(ivs) / len(ivs) if ivs else 0.0
    return 0.0


def compute_iv_skew(chain: OptionChainData, otm_offset: int = 3) -> float:
    """IV skew = avg OTM put IV - avg OTM call IV.

    Positive skew = puts are more expensive (fear premium).
    """
    atm = find_atm_strike(chain)
    if not chain.strikes or atm == 0:
        return 0.0

    # Sort strikes by strike price
    sorted_strikes = sorted(chain.strikes, key=lambda s: s.strike_price)
    atm_idx = None
    for i, s in enumerate(sorted_strikes):
        if s.strike_price == atm:
            atm_idx = i
            break

    if atm_idx is None:
        return 0.0

    # OTM puts: strikes below ATM
    otm_put_ivs = []
    for i in range(max(0, atm_idx - otm_offset), atm_idx):
        iv = sorted_strikes[i].pe_iv
        if iv > 0:
            otm_put_ivs.append(iv)

    # OTM calls: strikes above ATM
    otm_call_ivs = []
    for i in range(atm_idx + 1, min(len(sorted_strikes), atm_idx + 1 + otm_offset)):
        iv = sorted_strikes[i].ce_iv
        if iv > 0:
            otm_call_ivs.append(iv)

    avg_put_iv = sum(otm_put_ivs) / len(otm_put_ivs) if otm_put_ivs else 0.0
    avg_call_iv = sum(otm_call_ivs) / len(otm_call_ivs) if otm_call_ivs else 0.0

    if avg_put_iv == 0 and avg_call_iv == 0:
        return 0.0
    return avg_put_iv - avg_call_iv


def build_analytics(chain: OptionChainData) -> OptionsAnalytics:
    """Run all option chain analytics and return an OptionsAnalytics model."""
    pcr = compute_pcr(chain)

    if pcr > 1.2:
        pcr_label = "Bullish (heavy put writing)"
    elif pcr > 0.8:
        pcr_label = "Neutral"
    elif pcr > 0.5:
        pcr_label = "Bearish"
    else:
        pcr_label = "Strongly Bearish"

    support_strike, support_oi, resistance_strike, resistance_oi = compute_oi_levels(chain)

    return OptionsAnalytics(
        pcr=round(pcr, 3),
        pcr_label=pcr_label,
        max_pain=compute_max_pain(chain),
        support_strike=support_strike,
        support_oi=support_oi,
        resistance_strike=resistance_strike,
        resistance_oi=resistance_oi,
        atm_strike=find_atm_strike(chain),
        atm_iv=round(compute_atm_iv(chain), 2),
        iv_skew=round(compute_iv_skew(chain), 2),
    )
