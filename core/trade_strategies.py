"""Evaluate option strategies and return the best 4 trade suggestions."""

from __future__ import annotations

import logging
from collections import Counter

from core.config import load_config
from core.event_calendar import is_event_day
from core.options_models import (
    OptionChainData,
    OptionsAnalytics,
    StrikeData,
    StrategyName,
    TechnicalIndicators,
    TradeLeg,
    TradeSuggestion,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _trade_cfg() -> dict:
    return load_config().get("options_desk", {}).get("trade_suggestions", {})


def _lot_size() -> int:
    return _trade_cfg().get("lot_size", 25)


def _otm_offset() -> int:
    return _trade_cfg().get("otm_offset", 2)


def _spread_width() -> int:
    return _trade_cfg().get("spread_width", 2)


def _min_score() -> float:
    return _trade_cfg().get("min_score", 20)


# ---------------------------------------------------------------------------
# Parameter override helper
# ---------------------------------------------------------------------------

def _p(overrides: dict[str, float], name: str, default: float) -> float:
    """Look up a parameter value from overrides, falling back to *default*."""
    return overrides.get(name, default)


# ---------------------------------------------------------------------------
# Confidence normalization
# ---------------------------------------------------------------------------

def _confidence_from_score(score: float) -> str:
    """Uniform confidence mapping across all strategies."""
    if score >= 55:
        return "High"
    if score >= 35:
        return "Medium"
    return "Low"


# ---------------------------------------------------------------------------
# Strike selection helpers
# ---------------------------------------------------------------------------

def _sorted_strikes(chain: OptionChainData) -> list[StrikeData]:
    return sorted(chain.strikes, key=lambda s: s.strike_price)


def _get_atm_strike_data(chain: OptionChainData, atm: float) -> StrikeData | None:
    for s in chain.strikes:
        if s.strike_price == atm:
            return s
    return None


def _get_strike_by_offset(
    chain: OptionChainData, atm: float, offset: int, side: str,
) -> StrikeData | None:
    """Get OTM strike `offset` steps away from ATM.

    side="CE" → higher strikes (OTM calls), side="PE" → lower strikes (OTM puts).
    """
    strikes = _sorted_strikes(chain)
    atm_idx = None
    for i, s in enumerate(strikes):
        if s.strike_price == atm:
            atm_idx = i
            break
    if atm_idx is None:
        return None
    if side == "CE":
        target = atm_idx + offset
    else:
        target = atm_idx - offset
    if 0 <= target < len(strikes):
        return strikes[target]
    return None


def _instrument(strike: float, opt_type: str) -> str:
    return f"NIFTY {int(strike)} {opt_type}"


def _fmt_currency(amount: float) -> str:
    if abs(amount) >= 100_000:
        return f"\u20b9{amount:,.0f}"
    return f"\u20b9{amount:,.0f}"


def _bid_ask_penalty(
    chain: OptionChainData,
    strikes: list[tuple[float, str]],
    threshold_pct: float = 5.0,
) -> tuple[float, str | None]:
    """Check bid-ask spread for selected strikes.

    Returns (penalty, reason). Penalty is -10 per wide-spread strike.
    strikes: list of (strike_price, "CE"/"PE") tuples.
    """
    lookup: dict[float, StrikeData] = {s.strike_price: s for s in chain.strikes}
    penalty = 0.0
    wide_strikes: list[str] = []
    for strike_price, opt_type in strikes:
        sd = lookup.get(strike_price)
        if not sd:
            continue
        if opt_type == "CE":
            bid, ask, ltp = sd.ce_bid, sd.ce_ask, sd.ce_ltp
        else:
            bid, ask, ltp = sd.pe_bid, sd.pe_ask, sd.pe_ltp
        if ltp > 0 and bid > 0 and ask > 0:
            spread_pct = ((ask - bid) / ltp) * 100
            if spread_pct > threshold_pct:
                penalty -= 10
                wide_strikes.append(f"{int(strike_price)} {opt_type} ({spread_pct:.0f}%)")
    reason = None
    if wide_strikes:
        reason = f"Wide bid-ask spread: {', '.join(wide_strikes)}"
    return penalty, reason


def _bb_width_pct(tech: TechnicalIndicators) -> float:
    if tech.bb_middle > 0:
        width = ((tech.bb_upper - tech.bb_lower) / tech.bb_middle) * 100
        if width < 0.1:
            return 99.0  # BB data invalid (collapsed to spot) — treat as wide
        return width
    return 99.0  # no data — treat as wide (non-squeeze)


def _ema_bullish(tech: TechnicalIndicators) -> bool:
    return tech.ema_9 > tech.ema_21 > tech.ema_50


def _ema_bearish(tech: TechnicalIndicators) -> bool:
    return tech.ema_9 < tech.ema_21 < tech.ema_50


def _trend_confirmed_bullish(tech: TechnicalIndicators) -> bool:
    """EMA + Supertrend both confirm bullish trend (for RSI penalty exemption)."""
    return _ema_bullish(tech) and tech.supertrend_direction == 1


def _trend_confirmed_bearish(tech: TechnicalIndicators) -> bool:
    """EMA + Supertrend both confirm bearish trend (for RSI penalty exemption)."""
    return _ema_bearish(tech) and tech.supertrend_direction == -1


def _all_bullish(tech: TechnicalIndicators) -> bool:
    if tech.supertrend_direction == 0:
        return False  # unknown supertrend — can't confirm full alignment
    return (
        _ema_bullish(tech)
        and tech.supertrend_direction == 1
        and tech.spot > tech.vwap
    )


def _all_bearish(tech: TechnicalIndicators) -> bool:
    if tech.supertrend_direction == 0:
        return False  # unknown supertrend — can't confirm full alignment
    return (
        _ema_bearish(tech)
        and tech.supertrend_direction == -1
        and tech.spot < tech.vwap
    )


# ---------------------------------------------------------------------------
# Individual strategy evaluators
# ---------------------------------------------------------------------------

def _eval_short_straddle(
    analytics: OptionsAnalytics,
    tech: TechnicalIndicators,
    chain: OptionChainData,
    overrides: dict[str, float] | None = None,
) -> TradeSuggestion | None:
    # Block on high-impact event days — vol expansion expected
    is_event, event_label = is_event_day()
    if is_event:
        logger.info("Short Straddle blocked — event day: %s", event_label)
        return None

    atm = _get_atm_strike_data(chain, analytics.atm_strike)
    if not atm or atm.ce_ltp <= 0 or atm.pe_ltp <= 0:
        return None

    ov = overrides or {}
    score = 0.0
    reasons: list[str] = []
    checks: list[str] = []

    iv_floor = _p(ov, "iv_floor", 8.0)
    iv_low = _p(ov, "iv_low_threshold", 12)
    iv_sweet = _p(ov, "iv_sweet_threshold", 18)
    if analytics.atm_iv > 0:
        if analytics.atm_iv < iv_floor:
            logger.info("Short Straddle blocked — IV %.1f%% below floor %.1f%%", analytics.atm_iv, iv_floor)
            return None
        elif analytics.atm_iv < iv_low:
            score += 10
            reasons.append(f"ATM IV at {analytics.atm_iv:.1f}% — low premium, marginal")
        elif analytics.atm_iv < iv_sweet:
            score += 25
            reasons.append(f"ATM IV at {analytics.atm_iv:.1f}% — sweet spot for selling")
        else:
            score -= 10
    else:
        score -= 5
        reasons.append("IV data unavailable — cannot assess volatility")

    rsi_n_lo = _p(ov, "rsi_neutral_low", 40)
    rsi_n_hi = _p(ov, "rsi_neutral_high", 60)
    rsi_m_lo = _p(ov, "rsi_moderate_low", 30)
    rsi_m_hi = _p(ov, "rsi_moderate_high", 70)
    if rsi_n_lo <= tech.rsi <= rsi_n_hi:
        score += 20
        reasons.append(f"RSI {tech.rsi:.1f} in neutral zone — no strong directional bias")
    elif rsi_m_lo <= tech.rsi <= rsi_m_hi:
        score += 5
    else:
        score -= 15

    # Spot near max pain (reduced weight — max pain is EOD concept, not intraday)
    if analytics.max_pain > 0:
        dist_pct = abs(tech.spot - analytics.max_pain) / analytics.max_pain * 100
        mp_close = _p(ov, "max_pain_close_pct", 0.3)
        mp_near = _p(ov, "max_pain_near_pct", 0.8)
        if dist_pct < mp_close:
            score += 10
            reasons.append(f"Spot within {mp_close}% of Max Pain ({analytics.max_pain:.0f})")
        elif dist_pct < mp_near:
            score += 5
        else:
            score -= 5

    # Spot not near heavy OI walls (support or resistance)
    if analytics.support_strike > 0 and analytics.resistance_strike > 0:
        range_width = analytics.resistance_strike - analytics.support_strike
        if range_width > 0:
            mid = (analytics.support_strike + analytics.resistance_strike) / 2
            dist_from_mid_pct = abs(tech.spot - mid) / range_width * 100
            if dist_from_mid_pct < 20:
                score += 10
                reasons.append(f"Spot centered in OI range ({analytics.support_strike:.0f}-{analytics.resistance_strike:.0f})")
            elif dist_from_mid_pct > 40:
                score -= 10
                reasons.append("Spot near edge of OI range — directional risk")

    # PCR near neutral
    pcr_lo = _p(ov, "pcr_neutral_low", 0.8)
    pcr_hi = _p(ov, "pcr_neutral_high", 1.2)
    if analytics.pcr >= 0:
        if pcr_lo <= analytics.pcr <= pcr_hi:
            score += 15
            reasons.append(f"PCR {analytics.pcr:.2f} — balanced options flow")
        else:
            score -= 5

    # Tight BB = range-bound
    bw = _bb_width_pct(tech)
    bb_tight = _p(ov, "bb_width_tight_pct", 1.0)
    if bw < bb_tight:
        score += 10
        reasons.append(f"Bollinger Bands tight ({bw:.1f}%) — range-bound market")

    # Regime penalties — credit is vulnerable when IV is high or expanding
    iv_pct_thresh = _p(ov, "iv_percentile_high_threshold", 75)
    if analytics.iv_percentile > iv_pct_thresh:
        score -= 10
        reasons.append(f"IV percentile {analytics.iv_percentile:.0f}% > {iv_pct_thresh:.0f}% — elevated vega risk")
    iv_stress = _p(ov, "iv_stress_threshold", 20)
    if analytics.atm_iv > iv_stress:
        score -= 10
        reasons.append(f"ATM IV {analytics.atm_iv:.1f}% > {iv_stress:.0f}% — stress regime")

    checks.append("Ensure no major event/news expected")
    checks.append(f"RSI stays in {rsi_n_lo:.0f}-{rsi_n_hi:.0f} range (currently {tech.rsi:.1f})")
    checks.append(f"Spot stays near {analytics.max_pain:.0f} max pain")

    if score < 0:
        return None

    lot = _lot_size()
    premium_collected = (atm.ce_ltp + atm.pe_ltp) * lot
    min_premium_per_lot = _p(ov, "min_premium_per_lot", 150)
    if premium_collected < min_premium_per_lot:
        logger.info("Short Straddle blocked — premium/lot ₹%.0f < min ₹%.0f", premium_collected, min_premium_per_lot)
        return None

    return TradeSuggestion(
        strategy=StrategyName.SHORT_STRADDLE,
        legs=[
            TradeLeg(action="SELL", instrument=_instrument(atm.strike_price, "CE"),
                     strike=atm.strike_price, option_type="CE", ltp=atm.ce_ltp),
            TradeLeg(action="SELL", instrument=_instrument(atm.strike_price, "PE"),
                     strike=atm.strike_price, option_type="PE", ltp=atm.pe_ltp),
        ],
        direction_bias="Neutral",
        confidence=_confidence_from_score(score),
        score=score,
        entry_timing="Enter now — neutral conditions confirmed" if score >= 50 else "Wait for RSI to settle in 45-55 range",
        technicals_to_check=checks,
        expected_outcome=f"Max profit {_fmt_currency(premium_collected)} if NIFTY expires at {int(atm.strike_price)}",
        max_profit=_fmt_currency(premium_collected),
        max_loss="Unlimited (hedge with wider strikes if needed)",
        stop_loss=f"Exit if NIFTY moves beyond {int(atm.strike_price - atm.ce_ltp - atm.pe_ltp)}-{int(atm.strike_price + atm.ce_ltp + atm.pe_ltp)} range",
        position_size=f"1 lot ({lot} qty) \u2248 {_fmt_currency(premium_collected)} margin",
        reasoning=reasons,
    )


def _eval_short_strangle(
    analytics: OptionsAnalytics,
    tech: TechnicalIndicators,
    chain: OptionChainData,
    overrides: dict[str, float] | None = None,
) -> TradeSuggestion | None:
    # Block on high-impact event days — vol expansion expected
    is_event, event_label = is_event_day()
    if is_event:
        logger.info("Short Strangle blocked — event day: %s", event_label)
        return None

    offset = _otm_offset()
    otm_ce = _get_strike_by_offset(chain, analytics.atm_strike, offset, "CE")
    otm_pe = _get_strike_by_offset(chain, analytics.atm_strike, offset, "PE")
    if not otm_ce or not otm_pe or otm_ce.ce_ltp <= 0 or otm_pe.pe_ltp <= 0:
        return None

    ov = overrides or {}
    score = 0.0
    reasons: list[str] = []
    checks: list[str] = []

    iv_floor = _p(ov, "iv_floor", 8.0)
    iv_low = _p(ov, "iv_low_threshold", 14)
    iv_sweet = _p(ov, "iv_sweet_threshold", 22)
    if analytics.atm_iv > 0:
        if analytics.atm_iv < iv_floor:
            logger.info("Short Strangle blocked — IV %.1f%% below floor %.1f%%", analytics.atm_iv, iv_floor)
            return None
        elif analytics.atm_iv < iv_low:
            score += 10
            reasons.append(f"ATM IV {analytics.atm_iv:.1f}% — low premium, marginal")
        elif analytics.atm_iv < iv_sweet:
            score += 20
            reasons.append(f"ATM IV {analytics.atm_iv:.1f}% — sweet spot for selling")
        else:
            score -= 10
    else:
        score -= 5
        reasons.append("IV data unavailable — cannot assess volatility")

    # Range-bound: strong support + resistance
    if analytics.support_oi > 0 and analytics.resistance_oi > 0:
        score += 15
        reasons.append(f"Strong support at {analytics.support_strike:.0f} and resistance at {analytics.resistance_strike:.0f}")

    rsi_lo = _p(ov, "rsi_neutral_low", 35)
    rsi_hi = _p(ov, "rsi_neutral_high", 65)
    if rsi_lo <= tech.rsi <= rsi_hi:
        score += 15
        reasons.append(f"RSI {tech.rsi:.1f} — no extreme momentum")

    # Spot between support and resistance
    if analytics.support_strike < tech.spot < analytics.resistance_strike:
        score += 15
        reasons.append("Spot within support-resistance band")

    bw = _bb_width_pct(tech)
    bb_tight = _p(ov, "bb_width_tight_pct", 1.2)
    if bw < bb_tight:
        score += 10
        reasons.append(f"BB width {bw:.1f}% — compressed, range-bound")

    # Regime penalties — credit is vulnerable when IV is high or expanding
    iv_pct_thresh = _p(ov, "iv_percentile_high_threshold", 75)
    if analytics.iv_percentile > iv_pct_thresh:
        score -= 10
        reasons.append(f"IV percentile {analytics.iv_percentile:.0f}% > {iv_pct_thresh:.0f}% — elevated vega risk")
    iv_stress = _p(ov, "iv_stress_threshold", 20)
    if analytics.atm_iv > iv_stress:
        score -= 10
        reasons.append(f"ATM IV {analytics.atm_iv:.1f}% > {iv_stress:.0f}% — stress regime")

    checks.append(f"Support holds at {analytics.support_strike:.0f}")
    checks.append(f"Resistance holds at {analytics.resistance_strike:.0f}")
    checks.append("No breakout signals (Supertrend stable)")

    # Bid-ask liquidity check
    ba_pen, ba_reason = _bid_ask_penalty(chain, [
        (otm_ce.strike_price, "CE"), (otm_pe.strike_price, "PE"),
    ])
    score += ba_pen
    if ba_reason:
        reasons.append(ba_reason)

    if score < 0:
        return None

    lot = _lot_size()
    premium = (otm_ce.ce_ltp + otm_pe.pe_ltp) * lot
    min_premium_per_lot = _p(ov, "min_premium_per_lot", 150)
    if premium < min_premium_per_lot:
        logger.info("Short Strangle blocked — premium/lot ₹%.0f < min ₹%.0f", premium, min_premium_per_lot)
        return None

    return TradeSuggestion(
        strategy=StrategyName.SHORT_STRANGLE,
        legs=[
            TradeLeg(action="SELL", instrument=_instrument(otm_ce.strike_price, "CE"),
                     strike=otm_ce.strike_price, option_type="CE", ltp=otm_ce.ce_ltp),
            TradeLeg(action="SELL", instrument=_instrument(otm_pe.strike_price, "PE"),
                     strike=otm_pe.strike_price, option_type="PE", ltp=otm_pe.pe_ltp),
        ],
        direction_bias="Neutral",
        confidence=_confidence_from_score(score),
        score=score,
        entry_timing="Enter now — range-bound setup confirmed" if score >= 45 else "Wait for spot to return to mid-range",
        technicals_to_check=checks,
        expected_outcome=f"Max profit {_fmt_currency(premium)} if NIFTY stays in {int(otm_pe.strike_price)}-{int(otm_ce.strike_price)} range",
        max_profit=_fmt_currency(premium),
        max_loss="Unlimited beyond breakeven strikes",
        stop_loss=f"Exit if NIFTY breaks {int(otm_pe.strike_price)} or {int(otm_ce.strike_price)}",
        position_size=f"1 lot ({lot} qty) \u2248 {_fmt_currency(premium)} margin",
        reasoning=reasons,
    )


def _eval_long_straddle(
    analytics: OptionsAnalytics,
    tech: TechnicalIndicators,
    chain: OptionChainData,
    overrides: dict[str, float] | None = None,
) -> TradeSuggestion | None:
    atm = _get_atm_strike_data(chain, analytics.atm_strike)
    if not atm or atm.ce_ltp <= 0 or atm.pe_ltp <= 0:
        return None

    ov = overrides or {}
    score = 0.0
    reasons: list[str] = []
    checks: list[str] = []

    # Event day boost — vol expansion expected
    is_event, event_label = is_event_day()
    if is_event:
        score += 15
        reasons.append(f"Event day ({event_label}) — volatility expansion likely")

    bw = _bb_width_pct(tech)
    bb_squeeze = _p(ov, "bb_width_squeeze_pct", 0.8)
    bb_mod = _p(ov, "bb_width_moderate_pct", 1.0)
    if bw < bb_squeeze:
        score += 25
        reasons.append(f"Bollinger Band squeeze ({bw:.1f}%) — breakout expected")
    elif bw < bb_mod:
        score += 15

    iv_skew_thresh = _p(ov, "iv_skew_threshold", 3)
    if abs(analytics.iv_skew) > iv_skew_thresh:
        score += 20
        reasons.append(f"IV skew {analytics.iv_skew:+.1f} — significant imbalance, move expected")

    iv_cheap = _p(ov, "iv_cheap_threshold", 14)
    iv_mod = _p(ov, "iv_moderate_threshold", 18)
    if analytics.atm_iv > 0:
        if analytics.atm_iv < iv_cheap:
            score += 20
            reasons.append(f"ATM IV {analytics.atm_iv:.1f}% — cheap premiums, ideal for buying straddle")
        elif analytics.atm_iv < iv_mod:
            score += 10
            reasons.append(f"ATM IV {analytics.atm_iv:.1f}% — moderate, acceptable entry")
        else:
            score -= 10
            reasons.append(f"ATM IV {analytics.atm_iv:.1f}% — expensive, poor entry for long straddle")
    else:
        score -= 5
        reasons.append("IV data unavailable — cannot assess volatility")

    rsi_coil_lo = _p(ov, "rsi_coiled_low", 45)
    rsi_coil_hi = _p(ov, "rsi_coiled_high", 55)
    if rsi_coil_lo <= tech.rsi <= rsi_coil_hi:
        score += 10
        reasons.append(f"RSI {tech.rsi:.1f} — coiled, ready to break")

    checks.append("Confirm BB squeeze is still active")
    checks.append("Watch for volume surge as breakout trigger")
    checks.append(f"IV skew: {analytics.iv_skew:+.1f}")

    if score < 0:
        return None

    lot = _lot_size()
    premium_paid = (atm.ce_ltp + atm.pe_ltp) * lot
    be_up = atm.strike_price + atm.ce_ltp + atm.pe_ltp
    be_down = atm.strike_price - atm.ce_ltp - atm.pe_ltp
    return TradeSuggestion(
        strategy=StrategyName.LONG_STRADDLE,
        legs=[
            TradeLeg(action="BUY", instrument=_instrument(atm.strike_price, "CE"),
                     strike=atm.strike_price, option_type="CE", ltp=atm.ce_ltp),
            TradeLeg(action="BUY", instrument=_instrument(atm.strike_price, "PE"),
                     strike=atm.strike_price, option_type="PE", ltp=atm.pe_ltp),
        ],
        direction_bias="Neutral",
        confidence=_confidence_from_score(score),
        score=score,
        entry_timing="Enter now — squeeze detected, breakout imminent" if bw < bb_squeeze else "Enter before major event/news",
        technicals_to_check=checks,
        expected_outcome=f"Profitable if NIFTY moves beyond {int(be_down)}-{int(be_up)} range",
        max_profit="Unlimited (both directions)",
        max_loss=_fmt_currency(premium_paid),
        stop_loss=f"Exit if premium decays 50% without move ({_fmt_currency(premium_paid * 0.5)})",
        position_size=f"1 lot ({lot} qty) = {_fmt_currency(premium_paid)} cost",
        reasoning=reasons,
    )


def _eval_long_strangle(
    analytics: OptionsAnalytics,
    tech: TechnicalIndicators,
    chain: OptionChainData,
    overrides: dict[str, float] | None = None,
) -> TradeSuggestion | None:
    offset = _otm_offset()
    otm_ce = _get_strike_by_offset(chain, analytics.atm_strike, offset, "CE")
    otm_pe = _get_strike_by_offset(chain, analytics.atm_strike, offset, "PE")
    if not otm_ce or not otm_pe or otm_ce.ce_ltp <= 0 or otm_pe.pe_ltp <= 0:
        return None

    ov = overrides or {}
    score = 0.0
    reasons: list[str] = []
    checks: list[str] = []

    bw = _bb_width_pct(tech)
    bb_tight = _p(ov, "bb_width_tight_pct", 1.0)
    if bw < bb_tight:
        score += 20
        reasons.append(f"BB width {bw:.1f}% — breakout setup")

    iv_cheap = _p(ov, "iv_cheap_threshold", 16)
    iv_mod = _p(ov, "iv_moderate_threshold", 20)
    if analytics.atm_iv > 0:
        if analytics.atm_iv < iv_cheap:
            score += 20
            reasons.append(f"IV {analytics.atm_iv:.1f}% — cheap premiums, good entry for longs")
        elif analytics.atm_iv < iv_mod:
            score += 10
    else:
        score -= 5
        reasons.append("IV data unavailable — cannot assess volatility")

    iv_skew_thresh = _p(ov, "iv_skew_threshold", 2)
    if abs(analytics.iv_skew) > iv_skew_thresh:
        score += 10
        reasons.append(f"IV skew {analytics.iv_skew:+.1f} — directional pressure building")

    checks.append("Volume surge confirmation needed for breakout")
    checks.append(f"BB width currently {bw:.1f}%")

    # Bid-ask liquidity check
    ba_pen, ba_reason = _bid_ask_penalty(chain, [
        (otm_ce.strike_price, "CE"), (otm_pe.strike_price, "PE"),
    ])
    score += ba_pen
    if ba_reason:
        reasons.append(ba_reason)

    if score < 0:
        return None

    lot = _lot_size()
    premium_paid = (otm_ce.ce_ltp + otm_pe.pe_ltp) * lot
    return TradeSuggestion(
        strategy=StrategyName.LONG_STRANGLE,
        legs=[
            TradeLeg(action="BUY", instrument=_instrument(otm_ce.strike_price, "CE"),
                     strike=otm_ce.strike_price, option_type="CE", ltp=otm_ce.ce_ltp),
            TradeLeg(action="BUY", instrument=_instrument(otm_pe.strike_price, "PE"),
                     strike=otm_pe.strike_price, option_type="PE", ltp=otm_pe.pe_ltp),
        ],
        direction_bias="Neutral",
        confidence=_confidence_from_score(score),
        score=score,
        entry_timing="Enter now — cheap OTM premiums with squeeze" if score >= 40 else "Wait for IV to drop further",
        technicals_to_check=checks,
        expected_outcome=f"Profitable on big move beyond {int(otm_pe.strike_price)}/{int(otm_ce.strike_price)}",
        max_profit="Unlimited (both directions)",
        max_loss=_fmt_currency(premium_paid),
        stop_loss=f"Exit if premium decays 50% ({_fmt_currency(premium_paid * 0.5)})",
        position_size=f"1 lot ({lot} qty) = {_fmt_currency(premium_paid)} cost",
        reasoning=reasons,
    )


def _eval_bull_put_spread(
    analytics: OptionsAnalytics,
    tech: TechnicalIndicators,
    chain: OptionChainData,
    overrides: dict[str, float] | None = None,
) -> TradeSuggestion | None:
    # Sell PE at support, buy PE further OTM
    width = _spread_width()
    sell_pe = _get_strike_by_offset(chain, analytics.atm_strike, 1, "PE")  # 1 step below ATM
    buy_pe = _get_strike_by_offset(chain, analytics.atm_strike, 1 + width, "PE")
    if not sell_pe or not buy_pe or sell_pe.pe_ltp <= 0 or buy_pe.pe_ltp <= 0:
        return None
    if sell_pe.pe_ltp <= buy_pe.pe_ltp:
        return None  # no credit

    ov = overrides or {}
    score = 0.0
    reasons: list[str] = []
    checks: list[str] = []

    # Bullish trend
    if tech.supertrend_direction == 1:
        score += 15
        reasons.append("Supertrend bullish")
    if tech.spot > tech.vwap:
        score += 15
        reasons.append(f"Spot above VWAP ({tech.vwap:.0f})")
    if _ema_bullish(tech):
        score += 10
        reasons.append("EMA alignment bullish (9 > 21 > 50)")

    pcr_bull = _p(ov, "pcr_bullish_threshold", 1.2)
    pcr_mod = _p(ov, "pcr_moderate_threshold", 1.0)
    if analytics.pcr >= 0:
        if analytics.pcr > pcr_bull:
            score += 15
            reasons.append(f"PCR {analytics.pcr:.2f} — heavy put writing (bullish)")
        elif analytics.pcr > pcr_mod:
            score += 5

    rsi_ob = _p(ov, "rsi_overbought_threshold", 70)
    if tech.rsi < rsi_ob:
        score += 5
    else:
        score -= 10
        checks.append(f"RSI {tech.rsi:.1f} — overbought, wait for pullback")

    # Regime penalties — credit is vulnerable when IV is high or expanding
    iv_pct_thresh = _p(ov, "iv_percentile_high_threshold", 75)
    if analytics.iv_percentile > iv_pct_thresh:
        score -= 10
        reasons.append(f"IV percentile {analytics.iv_percentile:.0f}% > {iv_pct_thresh:.0f}% — elevated vega risk")
    iv_stress = _p(ov, "iv_stress_threshold", 20)
    if analytics.atm_iv > iv_stress:
        score -= 10
        reasons.append(f"ATM IV {analytics.atm_iv:.1f}% > {iv_stress:.0f}% — stress regime")

    checks.append(f"Supertrend direction stays bullish")
    checks.append(f"Support at {analytics.support_strike:.0f} holds")
    checks.append(f"RSI stays below {rsi_ob:.0f} (currently {tech.rsi:.1f})")

    if score < 0:
        return None

    lot = _lot_size()
    credit = (sell_pe.pe_ltp - buy_pe.pe_ltp) * lot
    strike_diff = abs(sell_pe.strike_price - buy_pe.strike_price)
    max_loss_val = (strike_diff - (sell_pe.pe_ltp - buy_pe.pe_ltp)) * lot
    return TradeSuggestion(
        strategy=StrategyName.BULL_PUT_SPREAD,
        legs=[
            TradeLeg(action="SELL", instrument=_instrument(sell_pe.strike_price, "PE"),
                     strike=sell_pe.strike_price, option_type="PE", ltp=sell_pe.pe_ltp),
            TradeLeg(action="BUY", instrument=_instrument(buy_pe.strike_price, "PE"),
                     strike=buy_pe.strike_price, option_type="PE", ltp=buy_pe.pe_ltp),
        ],
        direction_bias="Bullish",
        confidence=_confidence_from_score(score),
        score=score,
        entry_timing="Enter now — bullish trend confirmed" if score >= 45 else "Wait for pullback to VWAP",
        technicals_to_check=checks,
        expected_outcome=f"Max profit {_fmt_currency(credit)} if NIFTY stays above {int(sell_pe.strike_price)}",
        max_profit=_fmt_currency(credit),
        max_loss=_fmt_currency(max_loss_val),
        stop_loss=f"Exit if NIFTY breaks below {int(analytics.support_strike)} (support)",
        position_size=f"1 lot ({lot} qty) \u2248 {_fmt_currency(max_loss_val)} margin",
        reasoning=reasons,
    )


def _eval_bear_call_spread(
    analytics: OptionsAnalytics,
    tech: TechnicalIndicators,
    chain: OptionChainData,
    overrides: dict[str, float] | None = None,
) -> TradeSuggestion | None:
    width = _spread_width()
    sell_ce = _get_strike_by_offset(chain, analytics.atm_strike, 1, "CE")
    buy_ce = _get_strike_by_offset(chain, analytics.atm_strike, 1 + width, "CE")
    if not sell_ce or not buy_ce or sell_ce.ce_ltp <= 0 or buy_ce.ce_ltp <= 0:
        return None
    if sell_ce.ce_ltp <= buy_ce.ce_ltp:
        return None

    ov = overrides or {}
    score = 0.0
    reasons: list[str] = []
    checks: list[str] = []

    if tech.supertrend_direction == -1:
        score += 15
        reasons.append("Supertrend bearish")
    if tech.spot < tech.vwap:
        score += 15
        reasons.append(f"Spot below VWAP ({tech.vwap:.0f})")
    if _ema_bearish(tech):
        score += 10
        reasons.append("EMA alignment bearish (9 < 21 < 50)")

    pcr_bear = _p(ov, "pcr_bearish_threshold", 0.7)
    pcr_mod = _p(ov, "pcr_moderate_threshold", 1.0)
    if analytics.pcr >= 0:
        if analytics.pcr < pcr_bear:
            score += 15
            reasons.append(f"PCR {analytics.pcr:.2f} — low put writing (bearish)")
        elif analytics.pcr < pcr_mod:
            score += 5

    rsi_os = _p(ov, "rsi_oversold_threshold", 30)
    if tech.rsi > rsi_os:
        score += 5
    else:
        score -= 10
        checks.append(f"RSI {tech.rsi:.1f} — oversold, wait for bounce")

    # Regime penalties — credit is vulnerable when IV is high or expanding
    iv_pct_thresh = _p(ov, "iv_percentile_high_threshold", 75)
    if analytics.iv_percentile > iv_pct_thresh:
        score -= 10
        reasons.append(f"IV percentile {analytics.iv_percentile:.0f}% > {iv_pct_thresh:.0f}% — elevated vega risk")
    iv_stress = _p(ov, "iv_stress_threshold", 20)
    if analytics.atm_iv > iv_stress:
        score -= 10
        reasons.append(f"ATM IV {analytics.atm_iv:.1f}% > {iv_stress:.0f}% — stress regime")

    checks.append("Supertrend direction stays bearish")
    checks.append(f"Resistance at {analytics.resistance_strike:.0f} holds")
    checks.append(f"RSI stays above {rsi_os:.0f} (currently {tech.rsi:.1f})")

    if score < 0:
        return None

    lot = _lot_size()
    credit = (sell_ce.ce_ltp - buy_ce.ce_ltp) * lot
    strike_diff = abs(buy_ce.strike_price - sell_ce.strike_price)
    max_loss_val = (strike_diff - (sell_ce.ce_ltp - buy_ce.ce_ltp)) * lot
    return TradeSuggestion(
        strategy=StrategyName.BEAR_CALL_SPREAD,
        legs=[
            TradeLeg(action="SELL", instrument=_instrument(sell_ce.strike_price, "CE"),
                     strike=sell_ce.strike_price, option_type="CE", ltp=sell_ce.ce_ltp),
            TradeLeg(action="BUY", instrument=_instrument(buy_ce.strike_price, "CE"),
                     strike=buy_ce.strike_price, option_type="CE", ltp=buy_ce.ce_ltp),
        ],
        direction_bias="Bearish",
        confidence=_confidence_from_score(score),
        score=score,
        entry_timing="Enter now — bearish trend confirmed" if score >= 45 else "Wait for bounce to VWAP",
        technicals_to_check=checks,
        expected_outcome=f"Max profit {_fmt_currency(credit)} if NIFTY stays below {int(sell_ce.strike_price)}",
        max_profit=_fmt_currency(credit),
        max_loss=_fmt_currency(max_loss_val),
        stop_loss=f"Exit if NIFTY breaks above {int(analytics.resistance_strike)} (resistance)",
        position_size=f"1 lot ({lot} qty) \u2248 {_fmt_currency(max_loss_val)} margin",
        reasoning=reasons,
    )


def _eval_bull_call_spread(
    analytics: OptionsAnalytics,
    tech: TechnicalIndicators,
    chain: OptionChainData,
    overrides: dict[str, float] | None = None,
) -> TradeSuggestion | None:
    width = _spread_width()
    buy_ce = _get_atm_strike_data(chain, analytics.atm_strike)
    sell_ce = _get_strike_by_offset(chain, analytics.atm_strike, width, "CE")
    if not buy_ce or not sell_ce or buy_ce.ce_ltp <= 0 or sell_ce.ce_ltp <= 0:
        return None
    if buy_ce.ce_ltp <= sell_ce.ce_ltp:
        return None

    ov = overrides or {}
    score = 0.0
    reasons: list[str] = []
    checks: list[str] = []

    if _ema_bullish(tech):
        score += 20
        reasons.append("EMA alignment bullish (9 > 21 > 50)")
    if tech.supertrend_direction == 1:
        score += 15
        reasons.append("Supertrend bullish")
    if tech.spot > tech.vwap:
        score += 10
        reasons.append(f"Spot above VWAP ({tech.vwap:.0f})")

    rsi_bull_lo = _p(ov, "rsi_bullish_low", 50)
    rsi_ob = _p(ov, "rsi_overbought_threshold", 65)
    if rsi_bull_lo < tech.rsi < rsi_ob:
        score += 15
        reasons.append(f"RSI {tech.rsi:.1f} — bullish momentum, not overbought")
    elif tech.rsi >= rsi_ob:
        if _trend_confirmed_bullish(tech):
            reasons.append(f"RSI {tech.rsi:.1f} overbought but trend confirmed — no penalty")
        else:
            score -= 10

    # IV penalty — debit spread buys expensive premium when IV is high
    iv_pen_thresh = _p(ov, "iv_high_penalty_threshold", 18)
    if analytics.atm_iv > 0 and analytics.atm_iv >= iv_pen_thresh:
        score -= 15
        reasons.append(f"IV {analytics.atm_iv:.1f}% >= {iv_pen_thresh:.0f}% — buying expensive premium")

    # BB width penalty — wide BB means breakout already happened
    bw = _bb_width_pct(tech)
    bb_exp = _p(ov, "bb_width_expanded_pct", 1.5)
    if bw >= bb_exp:
        score -= 15
        reasons.append(f"BB width {bw:.1f}% >= {bb_exp:.1f}% — momentum may be exhausted")

    # Pre-breakout tight BB bonus — compression before expansion
    bb_tight = _p(ov, "bb_width_tight_pct", 1.0)
    if bw < bb_tight and bw > 0.1:
        score += 5
        reasons.append(f"BB tight ({bw:.1f}%) — pre-breakout compression")

    checks.append("EMA bullish alignment intact")
    checks.append(f"RSI below {rsi_ob:.0f} (currently {tech.rsi:.1f})")
    checks.append("Supertrend stays bullish")

    if score < 0:
        return None

    lot = _lot_size()
    debit = (buy_ce.ce_ltp - sell_ce.ce_ltp) * lot
    strike_diff = abs(sell_ce.strike_price - buy_ce.strike_price)
    max_profit_val = (strike_diff - (buy_ce.ce_ltp - sell_ce.ce_ltp)) * lot
    return TradeSuggestion(
        strategy=StrategyName.BULL_CALL_SPREAD,
        legs=[
            TradeLeg(action="BUY", instrument=_instrument(buy_ce.strike_price, "CE"),
                     strike=buy_ce.strike_price, option_type="CE", ltp=buy_ce.ce_ltp),
            TradeLeg(action="SELL", instrument=_instrument(sell_ce.strike_price, "CE"),
                     strike=sell_ce.strike_price, option_type="CE", ltp=sell_ce.ce_ltp),
        ],
        direction_bias="Bullish",
        confidence=_confidence_from_score(score),
        score=score,
        entry_timing="Enter now — bullish momentum confirmed" if score >= 45 else "Wait for pullback to EMA 9",
        technicals_to_check=checks,
        expected_outcome=f"Max profit {_fmt_currency(max_profit_val)} if NIFTY reaches {int(sell_ce.strike_price)}",
        max_profit=_fmt_currency(max_profit_val),
        max_loss=_fmt_currency(debit),
        stop_loss=f"Exit if NIFTY breaks below {int(tech.ema_21)} (EMA 21)",
        position_size=f"1 lot ({lot} qty) = {_fmt_currency(debit)} cost",
        reasoning=reasons,
    )


def _eval_bear_put_spread(
    analytics: OptionsAnalytics,
    tech: TechnicalIndicators,
    chain: OptionChainData,
    overrides: dict[str, float] | None = None,
) -> TradeSuggestion | None:
    width = _spread_width()
    buy_pe = _get_atm_strike_data(chain, analytics.atm_strike)
    sell_pe = _get_strike_by_offset(chain, analytics.atm_strike, width, "PE")
    if not buy_pe or not sell_pe or buy_pe.pe_ltp <= 0 or sell_pe.pe_ltp <= 0:
        return None
    if buy_pe.pe_ltp <= sell_pe.pe_ltp:
        return None

    ov = overrides or {}
    score = 0.0
    reasons: list[str] = []
    checks: list[str] = []

    if _ema_bearish(tech):
        score += 20
        reasons.append("EMA alignment bearish (9 < 21 < 50)")
    if tech.supertrend_direction == -1:
        score += 15
        reasons.append("Supertrend bearish")
    if tech.spot < tech.vwap:
        score += 10
        reasons.append(f"Spot below VWAP ({tech.vwap:.0f})")

    rsi_bear_lo = _p(ov, "rsi_bearish_low", 35)
    rsi_bear_hi = _p(ov, "rsi_bearish_high", 50)
    rsi_os = _p(ov, "rsi_oversold_threshold", 35)
    if rsi_bear_lo < tech.rsi < rsi_bear_hi:
        score += 15
        reasons.append(f"RSI {tech.rsi:.1f} — bearish momentum, not oversold")
    elif tech.rsi <= rsi_os:
        if _trend_confirmed_bearish(tech):
            reasons.append(f"RSI {tech.rsi:.1f} oversold but trend confirmed — no penalty")
        else:
            score -= 10

    # IV penalty — debit spread buys expensive premium when IV is high
    iv_pen_thresh = _p(ov, "iv_high_penalty_threshold", 18)
    if analytics.atm_iv > 0 and analytics.atm_iv >= iv_pen_thresh:
        score -= 15
        reasons.append(f"IV {analytics.atm_iv:.1f}% >= {iv_pen_thresh:.0f}% — buying expensive premium")

    # BB width penalty — wide BB means breakout already happened
    bw = _bb_width_pct(tech)
    bb_exp = _p(ov, "bb_width_expanded_pct", 1.5)
    if bw >= bb_exp:
        score -= 15
        reasons.append(f"BB width {bw:.1f}% >= {bb_exp:.1f}% — breakout already happened")

    # Pre-breakout tight BB bonus — compression before expansion
    bb_tight = _p(ov, "bb_width_tight_pct", 1.0)
    if bw < bb_tight and bw > 0.1:
        score += 5
        reasons.append(f"BB tight ({bw:.1f}%) — pre-breakout compression")

    checks.append("EMA bearish alignment intact")
    checks.append(f"RSI above {rsi_os:.0f} (currently {tech.rsi:.1f})")
    checks.append("Supertrend stays bearish")

    if score < 0:
        return None

    lot = _lot_size()
    debit = (buy_pe.pe_ltp - sell_pe.pe_ltp) * lot
    strike_diff = abs(buy_pe.strike_price - sell_pe.strike_price)
    max_profit_val = (strike_diff - (buy_pe.pe_ltp - sell_pe.pe_ltp)) * lot
    return TradeSuggestion(
        strategy=StrategyName.BEAR_PUT_SPREAD,
        legs=[
            TradeLeg(action="BUY", instrument=_instrument(buy_pe.strike_price, "PE"),
                     strike=buy_pe.strike_price, option_type="PE", ltp=buy_pe.pe_ltp),
            TradeLeg(action="SELL", instrument=_instrument(sell_pe.strike_price, "PE"),
                     strike=sell_pe.strike_price, option_type="PE", ltp=sell_pe.pe_ltp),
        ],
        direction_bias="Bearish",
        confidence=_confidence_from_score(score),
        score=score,
        entry_timing="Enter now — bearish momentum confirmed" if score >= 45 else "Wait for bounce to EMA 9",
        technicals_to_check=checks,
        expected_outcome=f"Max profit {_fmt_currency(max_profit_val)} if NIFTY reaches {int(sell_pe.strike_price)}",
        max_profit=_fmt_currency(max_profit_val),
        max_loss=_fmt_currency(debit),
        stop_loss=f"Exit if NIFTY breaks above {int(tech.ema_21)} (EMA 21)",
        position_size=f"1 lot ({lot} qty) = {_fmt_currency(debit)} cost",
        reasoning=reasons,
    )


def _eval_iron_condor(
    analytics: OptionsAnalytics,
    tech: TechnicalIndicators,
    chain: OptionChainData,
    overrides: dict[str, float] | None = None,
) -> TradeSuggestion | None:
    offset = _otm_offset()
    width = _spread_width()
    sell_ce = _get_strike_by_offset(chain, analytics.atm_strike, offset, "CE")
    buy_ce = _get_strike_by_offset(chain, analytics.atm_strike, offset + width, "CE")
    sell_pe = _get_strike_by_offset(chain, analytics.atm_strike, offset, "PE")
    buy_pe = _get_strike_by_offset(chain, analytics.atm_strike, offset + width, "PE")
    if not all([sell_ce, buy_ce, sell_pe, buy_pe]):
        return None
    if sell_ce.ce_ltp <= 0 or buy_ce.ce_ltp <= 0 or sell_pe.pe_ltp <= 0 or buy_pe.pe_ltp <= 0:
        return None

    ov = overrides or {}
    score = 0.0
    reasons: list[str] = []
    checks: list[str] = []

    rsi_n_lo = _p(ov, "rsi_neutral_low", 40)
    rsi_n_hi = _p(ov, "rsi_neutral_high", 60)
    rsi_m_lo = _p(ov, "rsi_moderate_low", 35)
    rsi_m_hi = _p(ov, "rsi_moderate_high", 65)
    if rsi_n_lo <= tech.rsi <= rsi_n_hi:
        score += 20
        reasons.append(f"RSI {tech.rsi:.1f} — perfectly neutral")
    elif rsi_m_lo <= tech.rsi <= rsi_m_hi:
        score += 10

    bw = _bb_width_pct(tech)
    bb_tight = _p(ov, "bb_width_tight_pct", 1.0)
    if bw < bb_tight:
        score += 15
        reasons.append(f"BB width {bw:.1f}% — tight range")

    iv_floor = _p(ov, "iv_floor", 8.0)
    iv_low = _p(ov, "iv_low_threshold", 12)
    iv_sweet = _p(ov, "iv_sweet_threshold", 18)
    if analytics.atm_iv > 0:
        if analytics.atm_iv < iv_floor:
            logger.info("Iron Condor blocked — IV %.1f%% below floor %.1f%%", analytics.atm_iv, iv_floor)
            return None
        elif analytics.atm_iv < iv_low:
            score += 5
            reasons.append(f"IV {analytics.atm_iv:.1f}% — low premium, marginal for Iron Condor")
        elif analytics.atm_iv < iv_sweet:
            score += 15
            reasons.append(f"IV {analytics.atm_iv:.1f}% — sweet spot for selling")
        else:
            score -= 10
    else:
        score -= 5
        reasons.append("IV data unavailable — cannot assess volatility")

    # Regime penalties — credit is vulnerable when IV is high or expanding
    iv_pct_thresh = _p(ov, "iv_percentile_high_threshold", 75)
    if analytics.iv_percentile > iv_pct_thresh:
        score -= 10
        reasons.append(f"IV percentile {analytics.iv_percentile:.0f}% > {iv_pct_thresh:.0f}% — elevated vega risk")
    iv_stress = _p(ov, "iv_stress_threshold", 20)
    if analytics.atm_iv > iv_stress:
        score -= 10
        reasons.append(f"ATM IV {analytics.atm_iv:.1f}% > {iv_stress:.0f}% — stress regime")

    pcr_bal_lo = _p(ov, "pcr_balanced_low", 0.85)
    pcr_bal_hi = _p(ov, "pcr_balanced_high", 1.15)
    if analytics.pcr >= 0 and pcr_bal_lo <= analytics.pcr <= pcr_bal_hi:
        score += 10
        reasons.append(f"PCR {analytics.pcr:.2f} — balanced")

    if analytics.support_strike < tech.spot < analytics.resistance_strike:
        score += 10
        reasons.append("Spot between support and resistance")

    checks.append(f"Range {int(sell_pe.strike_price)}-{int(sell_ce.strike_price)} holds")
    checks.append("No breakout triggers (news/events)")
    checks.append(f"PCR stays near {analytics.pcr:.2f}")

    # Bid-ask liquidity check (4 legs)
    ba_pen, ba_reason = _bid_ask_penalty(chain, [
        (sell_ce.strike_price, "CE"), (buy_ce.strike_price, "CE"),
        (sell_pe.strike_price, "PE"), (buy_pe.strike_price, "PE"),
    ])
    score += ba_pen
    if ba_reason:
        reasons.append(ba_reason)

    if score < 0:
        return None

    lot = _lot_size()
    ce_credit = (sell_ce.ce_ltp - buy_ce.ce_ltp)
    pe_credit = (sell_pe.pe_ltp - buy_pe.pe_ltp)
    total_credit = (ce_credit + pe_credit) * lot
    strike_width = abs(buy_ce.strike_price - sell_ce.strike_price)
    max_loss_val = (strike_width - ce_credit - pe_credit) * lot
    if total_credit <= 0:
        return None
    min_premium_per_lot = _p(ov, "min_premium_per_lot", 150)
    if total_credit < min_premium_per_lot:
        logger.info("Iron Condor blocked — premium/lot ₹%.0f < min ₹%.0f", total_credit, min_premium_per_lot)
        return None

    return TradeSuggestion(
        strategy=StrategyName.IRON_CONDOR,
        legs=[
            TradeLeg(action="SELL", instrument=_instrument(sell_ce.strike_price, "CE"),
                     strike=sell_ce.strike_price, option_type="CE", ltp=sell_ce.ce_ltp),
            TradeLeg(action="BUY", instrument=_instrument(buy_ce.strike_price, "CE"),
                     strike=buy_ce.strike_price, option_type="CE", ltp=buy_ce.ce_ltp),
            TradeLeg(action="SELL", instrument=_instrument(sell_pe.strike_price, "PE"),
                     strike=sell_pe.strike_price, option_type="PE", ltp=sell_pe.pe_ltp),
            TradeLeg(action="BUY", instrument=_instrument(buy_pe.strike_price, "PE"),
                     strike=buy_pe.strike_price, option_type="PE", ltp=buy_pe.pe_ltp),
        ],
        direction_bias="Neutral",
        confidence=_confidence_from_score(score),
        score=score,
        entry_timing="Enter now — iron condor conditions ideal" if score >= 50 else "Wait for RSI to settle near 50",
        technicals_to_check=checks,
        expected_outcome=f"Max profit {_fmt_currency(total_credit)} if NIFTY stays in {int(sell_pe.strike_price)}-{int(sell_ce.strike_price)}",
        max_profit=_fmt_currency(total_credit),
        max_loss=_fmt_currency(max_loss_val),
        stop_loss=f"Exit if NIFTY breaks {int(sell_pe.strike_price)} or {int(sell_ce.strike_price)}",
        position_size=f"1 lot ({lot} qty) \u2248 {_fmt_currency(max_loss_val)} margin",
        reasoning=reasons,
    )


def _eval_long_ce(
    analytics: OptionsAnalytics,
    tech: TechnicalIndicators,
    chain: OptionChainData,
    overrides: dict[str, float] | None = None,
) -> TradeSuggestion | None:
    atm = _get_atm_strike_data(chain, analytics.atm_strike)
    if not atm or atm.ce_ltp <= 0:
        return None

    ov = overrides or {}
    score = 0.0
    reasons: list[str] = []
    checks: list[str] = []

    if _all_bullish(tech):
        score += 30
        reasons.append("All signals aligned bullish (EMA, Supertrend, VWAP)")
    else:
        if _ema_bullish(tech):
            score += 10
            reasons.append("EMA alignment bullish")
        if tech.supertrend_direction == 1:
            score += 10
            reasons.append("Supertrend bullish")
        if tech.spot > tech.vwap:
            score += 5
            reasons.append(f"Spot above VWAP ({tech.vwap:.0f})")

    rsi_ob = _p(ov, "rsi_overbought_threshold", 70)
    if tech.rsi < rsi_ob:
        score += 10
        reasons.append(f"RSI {tech.rsi:.1f} — room to run higher")
    else:
        if _trend_confirmed_bullish(tech):
            reasons.append(f"RSI {tech.rsi:.1f} overbought but trend confirmed — no penalty")
        else:
            score -= 15

    pcr_bull = _p(ov, "pcr_bullish_threshold", 1.2)
    if analytics.pcr >= 0 and analytics.pcr > pcr_bull:
        score += 10
        reasons.append(f"PCR {analytics.pcr:.2f} — bullish options flow")

    # IV penalty — naked long is highly IV-sensitive
    iv_pen_thresh = _p(ov, "iv_high_penalty_threshold", 18)
    if analytics.atm_iv > 0 and analytics.atm_iv >= iv_pen_thresh:
        score -= 15
        reasons.append(f"IV {analytics.atm_iv:.1f}% >= {iv_pen_thresh:.0f}% — buying expensive premium")

    checks.append(f"RSI not overbought (currently {tech.rsi:.1f})")
    checks.append("Supertrend still bullish")
    checks.append(f"Spot holds above VWAP ({tech.vwap:.0f})")

    if score < 10:
        return None

    lot = _lot_size()
    cost = atm.ce_ltp * lot
    return TradeSuggestion(
        strategy=StrategyName.LONG_CE,
        legs=[
            TradeLeg(action="BUY", instrument=_instrument(atm.strike_price, "CE"),
                     strike=atm.strike_price, option_type="CE", ltp=atm.ce_ltp),
        ],
        direction_bias="Bullish",
        confidence=_confidence_from_score(score),
        score=score,
        entry_timing="Enter now — strong bullish alignment" if _all_bullish(tech) else "Wait for all signals to align bullish",
        technicals_to_check=checks,
        expected_outcome=f"Unlimited upside; breakeven at {int(atm.strike_price + atm.ce_ltp)}",
        max_profit="Unlimited",
        max_loss=_fmt_currency(cost),
        stop_loss=f"Exit if Supertrend flips bearish or spot breaks below {int(tech.bb_lower)} (BB lower)",
        position_size=f"1 lot ({lot} qty) = {_fmt_currency(cost)} cost",
        reasoning=reasons,
    )


def _eval_long_pe(
    analytics: OptionsAnalytics,
    tech: TechnicalIndicators,
    chain: OptionChainData,
    overrides: dict[str, float] | None = None,
) -> TradeSuggestion | None:
    atm = _get_atm_strike_data(chain, analytics.atm_strike)
    if not atm or atm.pe_ltp <= 0:
        return None

    ov = overrides or {}
    score = 0.0
    reasons: list[str] = []
    checks: list[str] = []

    if _all_bearish(tech):
        score += 30
        reasons.append("All signals aligned bearish (EMA, Supertrend, VWAP)")
    else:
        if _ema_bearish(tech):
            score += 10
            reasons.append("EMA alignment bearish")
        if tech.supertrend_direction == -1:
            score += 10
            reasons.append("Supertrend bearish")
        if tech.spot < tech.vwap:
            score += 5
            reasons.append(f"Spot below VWAP ({tech.vwap:.0f})")

    rsi_os = _p(ov, "rsi_oversold_threshold", 30)
    if tech.rsi > rsi_os:
        score += 10
        reasons.append(f"RSI {tech.rsi:.1f} — room to fall further")
    else:
        if _trend_confirmed_bearish(tech):
            reasons.append(f"RSI {tech.rsi:.1f} oversold but trend confirmed — no penalty")
        else:
            score -= 15

    pcr_bear = _p(ov, "pcr_bearish_threshold", 0.7)
    if analytics.pcr >= 0 and analytics.pcr < pcr_bear:
        score += 10
        reasons.append(f"PCR {analytics.pcr:.2f} — bearish options flow")

    # IV penalty — naked long is highly IV-sensitive
    iv_pen_thresh = _p(ov, "iv_high_penalty_threshold", 18)
    if analytics.atm_iv > 0 and analytics.atm_iv >= iv_pen_thresh:
        score -= 15
        reasons.append(f"IV {analytics.atm_iv:.1f}% >= {iv_pen_thresh:.0f}% — buying expensive premium")

    checks.append(f"RSI not oversold (currently {tech.rsi:.1f})")
    checks.append("Supertrend still bearish")
    checks.append(f"Spot stays below VWAP ({tech.vwap:.0f})")

    if score < 10:
        return None

    lot = _lot_size()
    cost = atm.pe_ltp * lot
    return TradeSuggestion(
        strategy=StrategyName.LONG_PE,
        legs=[
            TradeLeg(action="BUY", instrument=_instrument(atm.strike_price, "PE"),
                     strike=atm.strike_price, option_type="PE", ltp=atm.pe_ltp),
        ],
        direction_bias="Bearish",
        confidence=_confidence_from_score(score),
        score=score,
        entry_timing="Enter now — strong bearish alignment" if _all_bearish(tech) else "Wait for all signals to align bearish",
        technicals_to_check=checks,
        expected_outcome=f"Profitable below breakeven at {int(atm.strike_price - atm.pe_ltp)}",
        max_profit=f"Up to {_fmt_currency(atm.strike_price * lot)} (theoretical)",
        max_loss=_fmt_currency(cost),
        stop_loss=f"Exit if Supertrend flips bullish or spot breaks above {int(tech.bb_upper)} (BB upper)",
        position_size=f"1 lot ({lot} qty) = {_fmt_currency(cost)} cost",
        reasoning=reasons,
    )


# ---------------------------------------------------------------------------
# Evaluator → strategy name mapping (for override lookup)
# ---------------------------------------------------------------------------

_EVALUATOR_STRATEGY_MAP = {
    _eval_short_straddle: "Short Straddle",
    _eval_short_strangle: "Short Strangle",
    _eval_long_straddle: "Long Straddle",
    _eval_long_strangle: "Long Strangle",
    _eval_bull_put_spread: "Bull Put Spread",
    _eval_bear_call_spread: "Bear Call Spread",
    _eval_bull_call_spread: "Bull Call Spread",
    _eval_bear_put_spread: "Bear Put Spread",
    _eval_iron_condor: "Iron Condor",
    _eval_long_ce: "Long Call (CE)",
    _eval_long_pe: "Long Put (PE)",
}

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

_ALL_EVALUATORS = [
    _eval_short_straddle,
    _eval_short_strangle,
    _eval_long_straddle,
    _eval_long_strangle,
    _eval_bull_put_spread,
    _eval_bear_call_spread,
    _eval_bull_call_spread,
    _eval_bear_put_spread,
    _eval_iron_condor,
    _eval_long_ce,
    _eval_long_pe,
]

# ---------------------------------------------------------------------------
# Credit/debit classification for context adjustments
# ---------------------------------------------------------------------------

_CREDIT_STRATEGIES = {
    "Short Straddle", "Short Strangle", "Bull Put Spread",
    "Bear Call Spread", "Iron Condor",
}

_NEUTRAL_CREDIT_STRATEGIES = {"Short Straddle", "Short Strangle", "Iron Condor"}

_DIRECTIONAL_CREDIT_MAP = {
    "Bull Put Spread": "Bullish",
    "Bear Call Spread": "Bearish",
}


# ---------------------------------------------------------------------------
# Context-aware post-processor
# ---------------------------------------------------------------------------

def _apply_context_adjustments(
    suggestions: list[TradeSuggestion],
    context: "MarketContext",
    cfg: dict,
    observation: "ObservationSnapshot | None" = None,
) -> list[TradeSuggestion]:
    """Apply multi-level context adjustments to suggestion scores.

    Runs AFTER all 11 evaluators complete. Adjusts scores based on vol regime,
    multi-day trend, prior day candle, session context, observation bias, weekly
    trend, and vol regime stability. Appends [CTX]-prefixed reasons to each
    suggestion's reasoning list.

    Pure function: returns new list of suggestions with updated scores.
    """
    from core.context_models import MarketContext  # noqa: F811

    adjusted: list[TradeSuggestion] = []
    for s in suggestions:
        strategy_name = s.strategy.value if hasattr(s.strategy, "value") else str(s.strategy)
        is_credit = strategy_name in _CREDIT_STRATEGIES
        is_neutral_credit = strategy_name in _NEUTRAL_CREDIT_STRATEGIES
        is_directional_credit = strategy_name in _DIRECTIONAL_CREDIT_MAP
        dir_credit_bias = _DIRECTIONAL_CREDIT_MAP.get(strategy_name)
        original_score = s.score
        delta = 0.0
        ctx_reasons: list[str] = []
        bias = s.direction_bias

        # --- Vol regime ---
        vol = context.vol
        if is_credit:
            if vol.regime == "sell_premium":
                bonus = cfg.get("vol_sell_premium_bonus", 10)
                delta += bonus
                ctx_reasons.append(f"[CTX] Vol regime sell_premium: +{bonus}")
            elif vol.regime == "stand_down":
                penalty = cfg.get("vol_stand_down_penalty", -10)
                delta += penalty
                ctx_reasons.append(f"[CTX] Vol regime stand_down: {penalty}")
            elif vol.regime == "buy_premium":
                penalty = cfg.get("vol_buy_premium_penalty", -5)
                delta += penalty
                ctx_reasons.append(f"[CTX] Vol regime buy_premium: {penalty}")

            # RV trend
            if vol.rv_trend == "expanding":
                penalty = cfg.get("rv_expanding_penalty", -5)
                delta += penalty
                ctx_reasons.append(f"[CTX] RV expanding: {penalty}")
            elif vol.rv_trend == "contracting":
                bonus = cfg.get("rv_contracting_bonus", 5)
                delta += bonus
                ctx_reasons.append(f"[CTX] RV contracting: +{bonus}")
        else:
            # Debit strategies
            if vol.regime == "buy_premium":
                bonus = cfg.get("vol_sell_premium_bonus", 10)  # reuse same magnitude
                delta += bonus
                ctx_reasons.append(f"[CTX] Vol regime buy_premium (debit): +{bonus}")
            elif vol.regime == "sell_premium":
                penalty = cfg.get("vol_stand_down_penalty", -10)  # symmetric penalty
                delta += penalty
                ctx_reasons.append(f"[CTX] Vol regime sell_premium (debit): {penalty}")

            if vol.rv_trend == "expanding":
                bonus = cfg.get("rv_contracting_bonus", 5)  # expanding helps debit
                delta += bonus
                ctx_reasons.append(f"[CTX] RV expanding (debit): +{bonus}")

        # --- Multi-day trend alignment ---
        trend = context.multi_day_trend
        align_bonus = cfg.get("trend_alignment_bonus", 5)
        conflict_penalty = cfg.get("trend_conflict_penalty", -5)

        if is_neutral_credit:
            # Neutral credit strategies benefit from neutral trend
            if trend == "neutral":
                delta += align_bonus
                ctx_reasons.append(f"[CTX] Neutral trend suits neutral credit: +{align_bonus}")
            elif trend in ("bullish", "bearish"):
                delta += conflict_penalty
                ctx_reasons.append(f"[CTX] Trending market hurts neutral credit: {conflict_penalty}")
        elif is_credit:
            # Directional credit: BPS is bullish, BCS is bearish
            if (bias == "Bullish" and trend == "bullish") or (bias == "Bearish" and trend == "bearish"):
                delta += align_bonus
                ctx_reasons.append(f"[CTX] Trend {trend} aligns with {bias} credit: +{align_bonus}")
            elif (bias == "Bullish" and trend == "bearish") or (bias == "Bearish" and trend == "bullish"):
                delta += conflict_penalty
                ctx_reasons.append(f"[CTX] Trend {trend} conflicts with {bias} credit: {conflict_penalty}")
        else:
            # Debit: trend alignment
            if (bias == "Bullish" and trend == "bullish") or (bias == "Bearish" and trend == "bearish"):
                delta += align_bonus
                ctx_reasons.append(f"[CTX] Trend {trend} aligns with {bias} debit: +{align_bonus}")
            elif (bias == "Bullish" and trend == "bearish") or (bias == "Bearish" and trend == "bullish"):
                delta += conflict_penalty
                ctx_reasons.append(f"[CTX] Trend {trend} conflicts with {bias} debit: {conflict_penalty}")

        # --- Prior day context ---
        if context.prior_day is not None:
            # Doji bonus for neutral credit
            if is_neutral_credit and context.prior_day.candle_type == "doji":
                bonus = cfg.get("prior_day_doji_bonus", 5)
                delta += bonus
                ctx_reasons.append(f"[CTX] Prior day doji (indecision): +{bonus}")

            # Wide prior day range penalty
            wide_range_pct = cfg.get("prior_day_wide_range_pct", 2.0)
            if context.prior_day.range_pct > wide_range_pct:
                penalty = cfg.get("prior_day_wide_range_penalty", -5)
                delta += penalty
                ctx_reasons.append(
                    f"[CTX] Prior day range {context.prior_day.range_pct:.1f}% > {wide_range_pct}%: {penalty}"
                )

        # --- Session context ---
        session = context.session
        if is_neutral_credit and session.session_trend == "range_bound":
            bonus = cfg.get("session_range_bound_bonus", 5)
            delta += bonus
            ctx_reasons.append(f"[CTX] Session range-bound confirms neutral: +{bonus}")

        session_wide_pct = cfg.get("session_wide_range_pct", 1.5)
        if session.session_range_pct > session_wide_pct:
            penalty = cfg.get("session_wide_range_penalty", -5)
            delta += penalty
            ctx_reasons.append(
                f"[CTX] Session range {session.session_range_pct:.1f}% > {session_wide_pct}%: {penalty}"
            )

        # --- Session EMA alignment ---
        ema_bonus = cfg.get("session_ema_alignment_bonus", 5)
        ema_neutral_pen = cfg.get("session_ema_neutral_penalty", -3)
        if session.ema_alignment == "bullish":
            if is_directional_credit and dir_credit_bias == "Bullish":
                delta += ema_bonus
                ctx_reasons.append(f"[CTX] Session EMA bullish confirms BPS: +{ema_bonus}")
            elif is_neutral_credit:
                delta += ema_neutral_pen
                ctx_reasons.append(f"[CTX] Session EMA bullish hurts neutral credit: {ema_neutral_pen}")
        elif session.ema_alignment == "bearish":
            if is_directional_credit and dir_credit_bias == "Bearish":
                delta += ema_bonus
                ctx_reasons.append(f"[CTX] Session EMA bearish confirms BCS: +{ema_bonus}")
            elif is_neutral_credit:
                delta += ema_neutral_pen
                ctx_reasons.append(f"[CTX] Session EMA bearish hurts neutral credit: {ema_neutral_pen}")

        # --- Session RSI trajectory ---
        rsi_bonus = cfg.get("session_rsi_momentum_bonus", 3)
        if session.rsi_trajectory == "rising" and is_directional_credit and dir_credit_bias == "Bullish":
            delta += rsi_bonus
            ctx_reasons.append(f"[CTX] Session RSI rising confirms BPS: +{rsi_bonus}")
        elif session.rsi_trajectory == "falling" and is_directional_credit and dir_credit_bias == "Bearish":
            delta += rsi_bonus
            ctx_reasons.append(f"[CTX] Session RSI falling confirms BCS: +{rsi_bonus}")

        # --- Session BB position reversal ---
        bb_bonus = cfg.get("session_bb_reversal_bonus", 3)
        if session.bb_position == "lower" and is_directional_credit and dir_credit_bias == "Bullish":
            delta += bb_bonus
            ctx_reasons.append(f"[CTX] Session BB lower — mean reversion helps BPS: +{bb_bonus}")
        elif session.bb_position == "upper" and is_directional_credit and dir_credit_bias == "Bearish":
            delta += bb_bonus
            ctx_reasons.append(f"[CTX] Session BB upper — mean reversion helps BCS: +{bb_bonus}")

        # --- Session VWAP confirmation ---
        vwap_bonus = cfg.get("session_vwap_confirmation_bonus", 3)
        vwap_thresh = cfg.get("session_vwap_threshold_pct", 0.3)
        if is_directional_credit and dir_credit_bias == "Bullish" and session.current_vs_vwap_pct > vwap_thresh:
            delta += vwap_bonus
            ctx_reasons.append(f"[CTX] Above VWAP +{session.current_vs_vwap_pct:.1f}% confirms BPS: +{vwap_bonus}")
        elif is_directional_credit and dir_credit_bias == "Bearish" and session.current_vs_vwap_pct < -vwap_thresh:
            delta += vwap_bonus
            ctx_reasons.append(f"[CTX] Below VWAP {session.current_vs_vwap_pct:.1f}% confirms BCS: +{vwap_bonus}")

        # --- Observation bias ---
        if observation is not None and hasattr(observation, "bias"):
            obs_bias_bonus = cfg.get("observation_bias_bonus", 5)
            obs_neutral_bonus = cfg.get("observation_neutral_bonus", 3)
            obs_bias = observation.bias
            if is_directional_credit:
                if (dir_credit_bias == "Bullish" and obs_bias == "bullish") or \
                   (dir_credit_bias == "Bearish" and obs_bias == "bearish"):
                    delta += obs_bias_bonus
                    ctx_reasons.append(f"[CTX] Observation bias {obs_bias} confirms {dir_credit_bias} credit: +{obs_bias_bonus}")
            if is_neutral_credit and obs_bias == "neutral":
                delta += obs_neutral_bonus
                ctx_reasons.append(f"[CTX] Observation neutral confirms range: +{obs_neutral_bonus}")

        # --- Weekly trend ---
        if context.current_week is not None:
            weekly_bonus = cfg.get("weekly_trend_bonus", 3)
            wt = context.current_week.weekly_trend
            if is_directional_credit:
                if (dir_credit_bias == "Bullish" and wt == "bullish") or \
                   (dir_credit_bias == "Bearish" and wt == "bearish"):
                    delta += weekly_bonus
                    ctx_reasons.append(f"[CTX] Weekly trend {wt} confirms {dir_credit_bias} credit: +{weekly_bonus}")
            if is_neutral_credit and wt == "neutral":
                delta += weekly_bonus
                ctx_reasons.append(f"[CTX] Weekly neutral confirms range: +{weekly_bonus}")

        # --- Vol regime stability/instability ---
        stability_bonus = cfg.get("regime_stability_bonus", 3)
        stability_min_days = cfg.get("regime_stability_min_days", 3)
        instability_pen = cfg.get("regime_instability_penalty", -3)
        instability_thresh = cfg.get("regime_instability_threshold", 4)
        if is_credit and vol.regime == "sell_premium" and vol.regime_duration_days >= stability_min_days:
            delta += stability_bonus
            ctx_reasons.append(f"[CTX] Sell regime stable {vol.regime_duration_days}d: +{stability_bonus}")
        if vol.regime_changes_30d >= instability_thresh:
            delta += instability_pen
            ctx_reasons.append(f"[CTX] Regime unstable ({vol.regime_changes_30d} changes/30d): {instability_pen}")

        # --- Consecutive prior day pattern ---
        consec_bonus = cfg.get("consecutive_day_bonus", 3)
        consec_thresh = cfg.get("consecutive_day_threshold", 3)
        if is_directional_credit and len(context.prior_days) >= consec_thresh:
            recent = context.prior_days[:consec_thresh]
            if all(d.close_vs_open == "up" for d in recent) and dir_credit_bias == "Bearish":
                delta += consec_bonus
                ctx_reasons.append(f"[CTX] {consec_thresh} consecutive up days — overextension helps BCS: +{consec_bonus}")
            elif all(d.close_vs_open == "down" for d in recent) and dir_credit_bias == "Bullish":
                delta += consec_bonus
                ctx_reasons.append(f"[CTX] {consec_thresh} consecutive down days — overextension helps BPS: +{consec_bonus}")

        # --- Apply adjustment ---
        new_score = max(0.0, original_score + delta)
        new_confidence = _confidence_from_score(new_score)
        new_reasoning = list(s.reasoning) + ctx_reasons

        if delta != 0:
            logger.info(
                "Context: %s %.1f->%.1f (%+.0f)",
                strategy_name, original_score, new_score, delta,
            )

        adjusted.append(s.model_copy(update={
            "score": new_score,
            "confidence": new_confidence,
            "reasoning": new_reasoning,
        }))

    return adjusted


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_trade_suggestions(
    analytics: OptionsAnalytics,
    technicals: TechnicalIndicators,
    chain: OptionChainData,
    context: "MarketContext | None" = None,
    observation: "ObservationSnapshot | None" = None,
) -> list[TradeSuggestion]:
    """Evaluate all 11 strategies, return the best 4 with diversity."""
    min_score = _min_score()
    suggestions: list[TradeSuggestion] = []

    # Load active parameter overrides from the improvement system
    try:
        from core.database import SentimentDatabase
        _db = SentimentDatabase()
        _all_overrides = _db.get_active_overrides()
    except Exception:
        _all_overrides = {}
        logger.debug("Could not load parameter overrides, using defaults", exc_info=True)

    for evaluator in _ALL_EVALUATORS:
        try:
            strategy_name = _EVALUATOR_STRATEGY_MAP[evaluator]
            result = evaluator(
                analytics, technicals, chain,
                overrides=_all_overrides.get(strategy_name, {}),
            )
            if result is not None and result.score >= min_score:
                suggestions.append(result)
        except Exception:
            logger.debug("Strategy evaluator %s failed", evaluator.__name__, exc_info=True)

    # Sort by score descending
    suggestions.sort(key=lambda s: s.score, reverse=True)

    # Apply context adjustments + re-sort (before dedup so scores affect selection)
    if context is not None:
        ctx_cfg = _trade_cfg().get("context_adjustments", {})
        if ctx_cfg.get("enabled", True):
            suggestions = _apply_context_adjustments(suggestions, context, ctx_cfg, observation=observation)
            suggestions.sort(key=lambda s: s.score, reverse=True)

    # Deduplicate by direction_bias: max 2 per bias for variety
    bias_counts: Counter[str] = Counter()
    selected: list[TradeSuggestion] = []
    for s in suggestions:
        if bias_counts[s.direction_bias] < 2:
            selected.append(s)
            bias_counts[s.direction_bias] += 1
        if len(selected) >= 4:
            break

    return selected
