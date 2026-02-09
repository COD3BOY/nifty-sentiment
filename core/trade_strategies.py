"""Evaluate option strategies and return the best 4 trade suggestions."""

from __future__ import annotations

import logging
from collections import Counter

from core.config import load_config
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


def _bb_width_pct(tech: TechnicalIndicators) -> float:
    if tech.bb_middle > 0:
        return ((tech.bb_upper - tech.bb_lower) / tech.bb_middle) * 100
    return 0.0


def _ema_bullish(tech: TechnicalIndicators) -> bool:
    return tech.ema_9 > tech.ema_21 > tech.ema_50


def _ema_bearish(tech: TechnicalIndicators) -> bool:
    return tech.ema_9 < tech.ema_21 < tech.ema_50


def _all_bullish(tech: TechnicalIndicators) -> bool:
    return (
        _ema_bullish(tech)
        and tech.supertrend_direction == 1
        and tech.spot > tech.vwap
    )


def _all_bearish(tech: TechnicalIndicators) -> bool:
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
) -> TradeSuggestion | None:
    atm = _get_atm_strike_data(chain, analytics.atm_strike)
    if not atm or atm.ce_ltp <= 0 or atm.pe_ltp <= 0:
        return None

    score = 0.0
    reasons: list[str] = []
    checks: list[str] = []

    # Low IV favors short straddle
    if analytics.atm_iv < 15:
        score += 25
        reasons.append(f"ATM IV at {analytics.atm_iv:.1f}% — low, favorable for selling")
    elif analytics.atm_iv < 20:
        score += 15
        reasons.append(f"ATM IV at {analytics.atm_iv:.1f}% — moderate")
    else:
        score -= 10

    # Neutral RSI
    if 40 <= tech.rsi <= 60:
        score += 20
        reasons.append(f"RSI {tech.rsi:.1f} in neutral zone — no strong directional bias")
    elif 30 <= tech.rsi <= 70:
        score += 5
    else:
        score -= 15

    # Spot near max pain
    if analytics.max_pain > 0:
        dist_pct = abs(tech.spot - analytics.max_pain) / analytics.max_pain * 100
        if dist_pct < 0.3:
            score += 20
            reasons.append(f"Spot within 0.3% of Max Pain ({analytics.max_pain:.0f})")
        elif dist_pct < 0.8:
            score += 10
        else:
            score -= 10

    # PCR near neutral
    if 0.8 <= analytics.pcr <= 1.2:
        score += 15
        reasons.append(f"PCR {analytics.pcr:.2f} — balanced options flow")
    else:
        score -= 5

    # Tight BB = range-bound
    bw = _bb_width_pct(tech)
    if bw < 1.0:
        score += 10
        reasons.append(f"Bollinger Bands tight ({bw:.1f}%) — range-bound market")
    checks.append("Ensure no major event/news expected")
    checks.append(f"RSI stays in 40-60 range (currently {tech.rsi:.1f})")
    checks.append(f"Spot stays near {analytics.max_pain:.0f} max pain")

    if score < 0:
        return None

    lot = _lot_size()
    premium_collected = (atm.ce_ltp + atm.pe_ltp) * lot
    return TradeSuggestion(
        strategy=StrategyName.SHORT_STRADDLE,
        legs=[
            TradeLeg(action="SELL", instrument=_instrument(atm.strike_price, "CE"),
                     strike=atm.strike_price, option_type="CE", ltp=atm.ce_ltp),
            TradeLeg(action="SELL", instrument=_instrument(atm.strike_price, "PE"),
                     strike=atm.strike_price, option_type="PE", ltp=atm.pe_ltp),
        ],
        direction_bias="Neutral",
        confidence="High" if score >= 60 else "Medium" if score >= 35 else "Low",
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
) -> TradeSuggestion | None:
    offset = _otm_offset()
    otm_ce = _get_strike_by_offset(chain, analytics.atm_strike, offset, "CE")
    otm_pe = _get_strike_by_offset(chain, analytics.atm_strike, offset, "PE")
    if not otm_ce or not otm_pe or otm_ce.ce_ltp <= 0 or otm_pe.pe_ltp <= 0:
        return None

    score = 0.0
    reasons: list[str] = []
    checks: list[str] = []

    # Low-moderate IV
    if analytics.atm_iv < 18:
        score += 20
        reasons.append(f"ATM IV {analytics.atm_iv:.1f}% — favorable for selling")
    elif analytics.atm_iv < 22:
        score += 10

    # Range-bound: strong support + resistance
    if analytics.support_oi > 0 and analytics.resistance_oi > 0:
        score += 15
        reasons.append(f"Strong support at {analytics.support_strike:.0f} and resistance at {analytics.resistance_strike:.0f}")

    # Neutral RSI
    if 35 <= tech.rsi <= 65:
        score += 15
        reasons.append(f"RSI {tech.rsi:.1f} — no extreme momentum")

    # Spot between support and resistance
    if analytics.support_strike < tech.spot < analytics.resistance_strike:
        score += 15
        reasons.append("Spot within support-resistance band")

    bw = _bb_width_pct(tech)
    if bw < 1.2:
        score += 10
        reasons.append(f"BB width {bw:.1f}% — compressed, range-bound")

    checks.append(f"Support holds at {analytics.support_strike:.0f}")
    checks.append(f"Resistance holds at {analytics.resistance_strike:.0f}")
    checks.append("No breakout signals (Supertrend stable)")

    if score < 0:
        return None

    lot = _lot_size()
    premium = (otm_ce.ce_ltp + otm_pe.pe_ltp) * lot
    return TradeSuggestion(
        strategy=StrategyName.SHORT_STRANGLE,
        legs=[
            TradeLeg(action="SELL", instrument=_instrument(otm_ce.strike_price, "CE"),
                     strike=otm_ce.strike_price, option_type="CE", ltp=otm_ce.ce_ltp),
            TradeLeg(action="SELL", instrument=_instrument(otm_pe.strike_price, "PE"),
                     strike=otm_pe.strike_price, option_type="PE", ltp=otm_pe.pe_ltp),
        ],
        direction_bias="Neutral",
        confidence="High" if score >= 55 else "Medium" if score >= 30 else "Low",
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
) -> TradeSuggestion | None:
    atm = _get_atm_strike_data(chain, analytics.atm_strike)
    if not atm or atm.ce_ltp <= 0 or atm.pe_ltp <= 0:
        return None

    score = 0.0
    reasons: list[str] = []
    checks: list[str] = []

    # High IV skew or BB squeeze → breakout expected
    bw = _bb_width_pct(tech)
    if bw < 0.8:
        score += 25
        reasons.append(f"Bollinger Band squeeze ({bw:.1f}%) — breakout expected")
    elif bw < 1.0:
        score += 15

    if abs(analytics.iv_skew) > 3:
        score += 20
        reasons.append(f"IV skew {analytics.iv_skew:+.1f} — significant imbalance, move expected")

    # High IV makes long straddle cheaper in relative terms only if big move expected
    if analytics.atm_iv > 18:
        score += 10
        reasons.append(f"ATM IV {analytics.atm_iv:.1f}% — elevated, implies expected movement")
    else:
        score -= 10

    # RSI near extremes or neutral (could break either way from squeeze)
    if 45 <= tech.rsi <= 55:
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
        confidence="High" if score >= 50 else "Medium" if score >= 25 else "Low",
        score=score,
        entry_timing="Enter now — squeeze detected, breakout imminent" if bw < 0.8 else "Enter before major event/news",
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
) -> TradeSuggestion | None:
    offset = _otm_offset()
    otm_ce = _get_strike_by_offset(chain, analytics.atm_strike, offset, "CE")
    otm_pe = _get_strike_by_offset(chain, analytics.atm_strike, offset, "PE")
    if not otm_ce or not otm_pe or otm_ce.ce_ltp <= 0 or otm_pe.pe_ltp <= 0:
        return None

    score = 0.0
    reasons: list[str] = []
    checks: list[str] = []

    bw = _bb_width_pct(tech)
    if bw < 1.0:
        score += 20
        reasons.append(f"BB width {bw:.1f}% — breakout setup")

    if analytics.atm_iv < 16:
        score += 20
        reasons.append(f"IV {analytics.atm_iv:.1f}% — cheap premiums, good entry for longs")
    elif analytics.atm_iv < 20:
        score += 10

    if abs(analytics.iv_skew) > 2:
        score += 10
        reasons.append(f"IV skew {analytics.iv_skew:+.1f} — directional pressure building")

    checks.append("Volume surge confirmation needed for breakout")
    checks.append(f"BB width currently {bw:.1f}%")

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
        confidence="High" if score >= 45 else "Medium" if score >= 25 else "Low",
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
) -> TradeSuggestion | None:
    # Sell PE at support, buy PE further OTM
    width = _spread_width()
    sell_pe = _get_strike_by_offset(chain, analytics.atm_strike, 1, "PE")  # 1 step below ATM
    buy_pe = _get_strike_by_offset(chain, analytics.atm_strike, 1 + width, "PE")
    if not sell_pe or not buy_pe or sell_pe.pe_ltp <= 0 or buy_pe.pe_ltp <= 0:
        return None
    if sell_pe.pe_ltp <= buy_pe.pe_ltp:
        return None  # no credit

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

    # PCR > 1.2 → put writing = bullish
    if analytics.pcr > 1.2:
        score += 15
        reasons.append(f"PCR {analytics.pcr:.2f} — heavy put writing (bullish)")
    elif analytics.pcr > 1.0:
        score += 5

    # RSI not overbought
    if tech.rsi < 70:
        score += 5
    else:
        score -= 10
        checks.append(f"RSI {tech.rsi:.1f} — overbought, wait for pullback")

    checks.append(f"Supertrend direction stays bullish")
    checks.append(f"Support at {analytics.support_strike:.0f} holds")
    checks.append(f"RSI stays below 70 (currently {tech.rsi:.1f})")

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
        confidence="High" if score >= 50 else "Medium" if score >= 30 else "Low",
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
) -> TradeSuggestion | None:
    width = _spread_width()
    sell_ce = _get_strike_by_offset(chain, analytics.atm_strike, 1, "CE")
    buy_ce = _get_strike_by_offset(chain, analytics.atm_strike, 1 + width, "CE")
    if not sell_ce or not buy_ce or sell_ce.ce_ltp <= 0 or buy_ce.ce_ltp <= 0:
        return None
    if sell_ce.ce_ltp <= buy_ce.ce_ltp:
        return None

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

    if analytics.pcr < 0.7:
        score += 15
        reasons.append(f"PCR {analytics.pcr:.2f} — low put writing (bearish)")
    elif analytics.pcr < 1.0:
        score += 5

    if tech.rsi > 30:
        score += 5
    else:
        score -= 10
        checks.append(f"RSI {tech.rsi:.1f} — oversold, wait for bounce")

    checks.append("Supertrend direction stays bearish")
    checks.append(f"Resistance at {analytics.resistance_strike:.0f} holds")
    checks.append(f"RSI stays above 30 (currently {tech.rsi:.1f})")

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
        confidence="High" if score >= 50 else "Medium" if score >= 30 else "Low",
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
) -> TradeSuggestion | None:
    width = _spread_width()
    buy_ce = _get_atm_strike_data(chain, analytics.atm_strike)
    sell_ce = _get_strike_by_offset(chain, analytics.atm_strike, width, "CE")
    if not buy_ce or not sell_ce or buy_ce.ce_ltp <= 0 or sell_ce.ce_ltp <= 0:
        return None
    if buy_ce.ce_ltp <= sell_ce.ce_ltp:
        return None

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

    # Momentum confirmation
    if 50 < tech.rsi < 70:
        score += 15
        reasons.append(f"RSI {tech.rsi:.1f} — bullish momentum, not overbought")
    elif tech.rsi >= 70:
        score -= 10

    checks.append("EMA bullish alignment intact")
    checks.append(f"RSI below 70 (currently {tech.rsi:.1f})")
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
        confidence="High" if score >= 50 else "Medium" if score >= 30 else "Low",
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
) -> TradeSuggestion | None:
    width = _spread_width()
    buy_pe = _get_atm_strike_data(chain, analytics.atm_strike)
    sell_pe = _get_strike_by_offset(chain, analytics.atm_strike, width, "PE")
    if not buy_pe or not sell_pe or buy_pe.pe_ltp <= 0 or sell_pe.pe_ltp <= 0:
        return None
    if buy_pe.pe_ltp <= sell_pe.pe_ltp:
        return None

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

    if 30 < tech.rsi < 50:
        score += 15
        reasons.append(f"RSI {tech.rsi:.1f} — bearish momentum, not oversold")
    elif tech.rsi <= 30:
        score -= 10

    checks.append("EMA bearish alignment intact")
    checks.append(f"RSI above 30 (currently {tech.rsi:.1f})")
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
        confidence="High" if score >= 50 else "Medium" if score >= 30 else "Low",
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

    score = 0.0
    reasons: list[str] = []
    checks: list[str] = []

    # Very neutral conditions
    if 40 <= tech.rsi <= 60:
        score += 20
        reasons.append(f"RSI {tech.rsi:.1f} — perfectly neutral")
    elif 35 <= tech.rsi <= 65:
        score += 10

    bw = _bb_width_pct(tech)
    if bw < 1.0:
        score += 15
        reasons.append(f"BB width {bw:.1f}% — tight range")

    if analytics.atm_iv < 18:
        score += 15
        reasons.append(f"IV {analytics.atm_iv:.1f}% — moderate, good for selling")

    if 0.85 <= analytics.pcr <= 1.15:
        score += 10
        reasons.append(f"PCR {analytics.pcr:.2f} — balanced")

    if analytics.support_strike < tech.spot < analytics.resistance_strike:
        score += 10
        reasons.append("Spot between support and resistance")

    checks.append(f"Range {int(sell_pe.strike_price)}-{int(sell_ce.strike_price)} holds")
    checks.append("No breakout triggers (news/events)")
    checks.append(f"PCR stays near {analytics.pcr:.2f}")

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
        confidence="High" if score >= 55 else "Medium" if score >= 30 else "Low",
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
) -> TradeSuggestion | None:
    atm = _get_atm_strike_data(chain, analytics.atm_strike)
    if not atm or atm.ce_ltp <= 0:
        return None

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

    if tech.rsi < 70:
        score += 10
        reasons.append(f"RSI {tech.rsi:.1f} — room to run higher")
    else:
        score -= 15

    if analytics.pcr > 1.2:
        score += 10
        reasons.append(f"PCR {analytics.pcr:.2f} — bullish options flow")

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
        confidence="High" if score >= 50 else "Medium" if score >= 30 else "Low",
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
) -> TradeSuggestion | None:
    atm = _get_atm_strike_data(chain, analytics.atm_strike)
    if not atm or atm.pe_ltp <= 0:
        return None

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

    if tech.rsi > 30:
        score += 10
        reasons.append(f"RSI {tech.rsi:.1f} — room to fall further")
    else:
        score -= 15

    if analytics.pcr < 0.7:
        score += 10
        reasons.append(f"PCR {analytics.pcr:.2f} — bearish options flow")

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
        confidence="High" if score >= 50 else "Medium" if score >= 30 else "Low",
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


def generate_trade_suggestions(
    analytics: OptionsAnalytics,
    technicals: TechnicalIndicators,
    chain: OptionChainData,
) -> list[TradeSuggestion]:
    """Evaluate all 11 strategies, return the best 4 with diversity."""
    min_score = _min_score()
    suggestions: list[TradeSuggestion] = []

    for evaluator in _ALL_EVALUATORS:
        try:
            result = evaluator(analytics, technicals, chain)
            if result is not None and result.score >= min_score:
                suggestions.append(result)
        except Exception:
            logger.debug("Strategy evaluator %s failed", evaluator.__name__, exc_info=True)

    # Sort by score descending
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
