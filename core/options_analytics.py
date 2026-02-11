"""Pure functions for option chain analytics."""

from core.options_models import OptionChainData, OptionsAnalytics


def compute_pcr(chain: OptionChainData) -> float:
    """Total Put OI / Total Call OI.

    Returns -1.0 sentinel when CE OI is 0 (data error / unavailable).
    """
    if chain.total_ce_oi == 0:
        return -1.0
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
    spot: float = 0.0,
    proximity_strikes: int = 10,
) -> tuple[float, float, float, float]:
    """Find highest Put OI (support) and highest Call OI (resistance).

    When *spot* is provided, only considers strikes within ±proximity_strikes
    of ATM and ensures support < spot and resistance > spot.

    Returns (support_strike, support_oi, resistance_strike, resistance_oi).
    """
    if not chain.strikes:
        return 0.0, 0.0, 0.0, 0.0

    sorted_strikes = sorted(chain.strikes, key=lambda s: s.strike_price)

    # Filter to strikes near spot if spot is provided
    if spot > 0:
        atm_strike = min(sorted_strikes, key=lambda s: abs(s.strike_price - spot)).strike_price
        atm_idx = next(i for i, s in enumerate(sorted_strikes) if s.strike_price == atm_strike)
        lo = max(0, atm_idx - proximity_strikes)
        hi = min(len(sorted_strikes), atm_idx + proximity_strikes + 1)
        nearby = sorted_strikes[lo:hi]
    else:
        nearby = sorted_strikes

    max_pe_oi = 0.0
    support_strike = 0.0
    max_ce_oi = 0.0
    resistance_strike = 0.0

    for s in nearby:
        # Support: highest PE OI below spot (or all if no spot)
        if (spot <= 0 or s.strike_price < spot) and s.pe_oi > max_pe_oi:
            max_pe_oi = s.pe_oi
            support_strike = s.strike_price
        # Resistance: highest CE OI above spot (or all if no spot)
        if (spot <= 0 or s.strike_price > spot) and s.ce_oi > max_ce_oi:
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


def compute_liquidity_score(
    chain: OptionChainData,
    strikes: list[float],
    min_oi: float = 10_000,
    min_volume: int = 1_000,
    max_ba_pct: float = 5.0,
) -> tuple[float, str | None]:
    """Compute a liquidity score (0-100) for the given strikes.

    Returns (score, rejection_reason or None).
    A rejection_reason is set if any strike fails minimum thresholds.
    """
    if not strikes:
        return 0.0, "no strikes provided"

    strike_lookup = {s.strike_price: s for s in chain.strikes}
    total_score = 0.0
    checked = 0

    for sp in strikes:
        sd = strike_lookup.get(sp)
        if sd is None:
            return 0.0, f"strike {sp} not found in chain"

        # Check both CE and PE sides
        for side, oi, vol, bid, ask in [
            ("CE", sd.ce_oi, sd.ce_volume, sd.ce_bid, sd.ce_ask),
            ("PE", sd.pe_oi, sd.pe_volume, sd.pe_bid, sd.pe_ask),
        ]:
            if oi <= 0 and vol <= 0:
                continue  # skip side with no data
            checked += 1

            # OI score (0-40)
            if oi < min_oi:
                return 0.0, f"{side} OI {oi:.0f} at {sp} < min {min_oi:.0f}"
            oi_score = min(40.0, (oi / min_oi) * 10)

            # Volume score (0-30)
            if vol < min_volume:
                return 0.0, f"{side} volume {vol} at {sp} < min {min_volume}"
            vol_score = min(30.0, (vol / min_volume) * 10)

            # Bid-ask spread score (0-30)
            if bid > 0 and ask > 0:
                mid = (bid + ask) / 2.0
                ba_pct = ((ask - bid) / mid) * 100 if mid > 0 else 100.0
                if ba_pct > max_ba_pct:
                    return 0.0, f"{side} bid-ask {ba_pct:.1f}% at {sp} > max {max_ba_pct}%"
                ba_score = max(0.0, 30.0 * (1.0 - ba_pct / max_ba_pct))
            else:
                ba_score = 0.0

            total_score += oi_score + vol_score + ba_score

    if checked == 0:
        return 0.0, "no liquidity data available"

    return round(min(100.0, total_score / checked), 1), None


def build_analytics(chain: OptionChainData) -> OptionsAnalytics:
    """Run all option chain analytics and return an OptionsAnalytics model."""
    pcr = compute_pcr(chain)

    if pcr < 0:
        pcr_label = "Data Error"
    elif pcr > 1.2:
        pcr_label = "Bullish (heavy put writing)"
    elif pcr > 0.8:
        pcr_label = "Neutral"
    elif pcr > 0.5:
        pcr_label = "Bearish"
    else:
        pcr_label = "Strongly Bearish"

    support_strike, support_oi, resistance_strike, resistance_oi = compute_oi_levels(
        chain, spot=chain.underlying_value,
    )

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
