"""Market context snapshot captured at trade entry and exit."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from core.options_models import OptionChainData, OptionsAnalytics, TechnicalIndicators
from core.paper_trading_models import _now_ist


class LegContext(BaseModel):
    """Per-leg IV/OI snapshot from the option chain."""

    strike: float
    option_type: str  # "CE" or "PE"
    ltp: float
    iv: float = 0.0
    oi: float = 0.0
    volume: int = 0


class MarketContextSnapshot(BaseModel):
    """Full market state at a point in time (entry or exit)."""

    timestamp: datetime = Field(default_factory=_now_ist)

    # TechnicalIndicators (flat copy)
    spot: float = 0.0
    vwap: float = 0.0
    ema_9: float = 0.0
    ema_21: float = 0.0
    ema_50: float = 0.0
    rsi: float = 0.0
    supertrend: float = 0.0
    supertrend_direction: int = 0
    bb_upper: float = 0.0
    bb_middle: float = 0.0
    bb_lower: float = 0.0
    spot_change_pct: float = 0.0

    # OptionsAnalytics (flat copy)
    pcr: float = 0.0
    max_pain: float = 0.0
    atm_iv: float = 0.0
    iv_skew: float = 0.0
    support_strike: float = 0.0
    support_oi: float = 0.0
    resistance_strike: float = 0.0
    resistance_oi: float = 0.0

    # Per-leg IV/OI from chain
    leg_contexts: list[LegContext] = Field(default_factory=list)

    # Strategy decision context
    strategy_score: float = 0.0
    strategy_reasoning: list[str] = Field(default_factory=list)

    @classmethod
    def from_snapshot_data(
        cls,
        technicals: TechnicalIndicators | None,
        analytics: OptionsAnalytics | None,
        leg_contexts: list[LegContext] | None = None,
        score: float = 0.0,
        reasoning: list[str] | None = None,
    ) -> MarketContextSnapshot:
        """Build a snapshot from existing indicator/analytics objects."""
        kwargs: dict = {}

        if technicals:
            kwargs.update(
                spot=technicals.spot,
                vwap=technicals.vwap,
                ema_9=technicals.ema_9,
                ema_21=technicals.ema_21,
                ema_50=technicals.ema_50,
                rsi=technicals.rsi,
                supertrend=technicals.supertrend,
                supertrend_direction=technicals.supertrend_direction,
                bb_upper=technicals.bb_upper,
                bb_middle=technicals.bb_middle,
                bb_lower=technicals.bb_lower,
                spot_change_pct=technicals.spot_change_pct,
            )

        if analytics:
            kwargs.update(
                pcr=analytics.pcr,
                max_pain=analytics.max_pain,
                atm_iv=analytics.atm_iv,
                iv_skew=analytics.iv_skew,
                support_strike=analytics.support_strike,
                support_oi=analytics.support_oi,
                resistance_strike=analytics.resistance_strike,
                resistance_oi=analytics.resistance_oi,
            )

        if leg_contexts:
            kwargs["leg_contexts"] = leg_contexts

        kwargs["strategy_score"] = score
        kwargs["strategy_reasoning"] = reasoning or []

        return cls(**kwargs)


def build_leg_contexts(
    chain: OptionChainData,
    legs: list[dict],
) -> list[LegContext]:
    """Look up IV/OI for each leg from the option chain.

    `legs` should be a list of dicts with at least 'strike' and 'option_type' keys.
    """
    # Build lookup: (strike, option_type) -> StrikeData
    strike_lookup: dict[float, object] = {}
    for sd in chain.strikes:
        strike_lookup[sd.strike_price] = sd

    contexts: list[LegContext] = []
    for leg in legs:
        strike = leg["strike"]
        opt_type = leg["option_type"]
        sd = strike_lookup.get(strike)

        if sd:
            if opt_type == "CE":
                contexts.append(LegContext(
                    strike=strike, option_type=opt_type,
                    ltp=sd.ce_ltp, iv=sd.ce_iv, oi=sd.ce_oi, volume=sd.ce_volume,
                ))
            else:
                contexts.append(LegContext(
                    strike=strike, option_type=opt_type,
                    ltp=sd.pe_ltp, iv=sd.pe_iv, oi=sd.pe_oi, volume=sd.pe_volume,
                ))
        else:
            contexts.append(LegContext(
                strike=strike, option_type=opt_type, ltp=leg.get("ltp", 0.0),
            ))

    return contexts
