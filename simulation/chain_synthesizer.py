"""Synthetic option chain generator.

Given a spot price + scenario parameters, builds a complete
``OptionChainData`` with BS-consistent LTPs, parametric IV smile,
exponential OI distribution, and realistic bid-ask spreads.

Reuses real BS pricing (``core.iv_calculator.bs_price``) and delta
computation (``core.greeks.bs_delta``) — only the inputs are synthetic.
"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta, timezone

import numpy as np

from core.greeks import bs_delta
from core.iv_calculator import bs_price
from core.options_models import OptionChainData, StrikeData

_IST = timezone(timedelta(hours=5, minutes=30))
_RISK_FREE_RATE = 0.065  # 6.5% annual


def _iv_smile(
    strike: float,
    spot: float,
    atm_iv: float,
    slope: float,
    curvature: float,
) -> float:
    """Parametric IV smile.

    IV(K) = ATM_IV * (1 + slope * ln(K/S) + curvature * ln(K/S)^2)

    slope < 0 gives put skew (OTM puts have higher IV).
    curvature > 0 gives a smile on both wings.
    """
    if spot <= 0:
        return atm_iv
    moneyness = math.log(strike / spot)
    iv = atm_iv * (1 + slope * moneyness + curvature * moneyness ** 2)
    return max(iv, 1.0)  # floor IV at 1%


def _oi_profile(
    strike: float,
    atm_strike: float,
    base_oi: float,
    decay_rate: float,
    rng: np.random.Generator,
) -> float:
    """Exponential OI decay from ATM with round-number clustering."""
    distance = abs(strike - atm_strike)
    oi = base_oi * math.exp(-decay_rate * distance)

    # Boost OI at 100-multiples by 1.5x, 500-multiples by 2x
    if strike % 500 == 0:
        oi *= 2.0
    elif strike % 100 == 0:
        oi *= 1.5

    # Add noise ±20%
    oi *= rng.uniform(0.8, 1.2)
    return max(oi, 100.0)


def _compute_dynamic_atm_iv(
    base_atm_iv: float,
    current_spot: float,
    open_price: float,
    current_vix: float,
    iv_reactivity: float = 0.5,
) -> float:
    """ATM IV reacts to spot moves and VIX level.

    - Spot move component: |spot_return| * reactivity * 100 → additive IV bump
    - VIX scaling: scale IV proportional to VIX / base_vix
    - Asymmetric: down moves pump IV 1.5x more than up moves (put skew reality)

    Parameters
    ----------
    base_atm_iv : float
        Scenario's base ATM IV (percentage, e.g. 15.0).
    current_spot : float
        Current spot price.
    open_price : float
        Day's opening price (anchor for spot return).
    current_vix : float
        Current VIX level.
    iv_reactivity : float
        Sensitivity (0 = static/old behavior, 1 = very reactive).

    Returns
    -------
    Adjusted ATM IV in percentage.
    """
    if iv_reactivity <= 0 or open_price <= 0:
        return base_atm_iv

    # Spot return from open
    spot_return = (current_spot - open_price) / open_price

    # Asymmetric: down moves pump IV 1.5x more than up moves
    if spot_return < 0:
        spot_iv_bump = abs(spot_return) * iv_reactivity * 150.0  # e.g. -2% → +1.5 IV pts at reactivity=0.5
    else:
        spot_iv_bump = spot_return * iv_reactivity * 100.0 * 0.3  # up moves: mild IV compression effect
        spot_iv_bump = -spot_iv_bump  # up moves slightly reduce IV

    # VIX scaling: if VIX rises above its implied base, scale IV up proportionally
    # Estimate base VIX from base_atm_iv (roughly VIX ≈ ATM IV for near-term)
    base_vix = base_atm_iv
    if base_vix > 0 and current_vix > 0:
        vix_ratio = current_vix / base_vix
        vix_scaling = base_atm_iv * (vix_ratio - 1.0) * iv_reactivity * 0.5
    else:
        vix_scaling = 0.0

    adjusted_iv = base_atm_iv + spot_iv_bump + vix_scaling
    return max(adjusted_iv, 1.0)  # floor at 1%


def _directional_spread_multiplier(
    strike: float,
    spot: float,
    open_price: float,
    option_type: str,
) -> float:
    """Widen spreads for options moving into-the-money.

    When spot drops, PE spreads widen (dealers hedge); CE spreads tighten.
    When spot rises, CE spreads widen; PE spreads tighten.

    Returns a multiplier >= 1.0 applied to the base spread.
    """
    if open_price <= 0:
        return 1.0

    spot_return = (spot - open_price) / open_price

    if abs(spot_return) < 0.002:
        return 1.0  # negligible move

    # For PEs: widen when spot drops (PE goes ITM)
    # For CEs: widen when spot rises (CE goes ITM)
    if option_type == "PE":
        direction_factor = max(-spot_return, 0)  # positive when spot drops
    else:
        direction_factor = max(spot_return, 0)  # positive when spot rises

    # Scale by how close the strike is to current spot (ATM gets widest)
    moneyness = abs(strike - spot) / spot
    proximity = max(1.0 - moneyness * 10, 0.0)  # 1.0 at ATM, 0 at 10%+ OTM

    # Multiplier: 1.0 (no move) up to ~2.5 (big move, ATM strike)
    multiplier = 1.0 + direction_factor * proximity * 50.0
    return min(multiplier, 3.0)  # cap at 3x


def build_chain(
    spot: float,
    sim_datetime: datetime,
    scenario_chain_config,
    seed: int,
    expiry_date: date | None = None,
) -> OptionChainData:
    """Build a complete synthetic option chain.

    Parameters
    ----------
    spot : float
        Current underlying price.
    sim_datetime : datetime
        Simulation timestamp (IST-aware).
    scenario_chain_config : ChainConfig
        Chain parameters from the scenario.
    seed : int
        RNG seed for OI/volume noise.
    expiry_date : date, optional
        If None, derived from ``sim_datetime + dte``.

    Returns
    -------
    OptionChainData with all strikes populated.
    """
    rng = np.random.default_rng(seed + 2000)
    cfg = scenario_chain_config

    # Resolve expiry
    if expiry_date is None:
        expiry_date = sim_datetime.date() + timedelta(days=cfg.expiry_dte)
    expiry_str = expiry_date.strftime("%d-%b-%Y").upper()

    # Time to expiry in years
    days_to_expiry = (expiry_date - sim_datetime.date()).days
    T = max(days_to_expiry / 365.0, 1 / 365.0)  # floor at 1 day

    # ATM strike (rounded to nearest strike_step)
    atm_strike = round(spot / cfg.strike_step) * cfg.strike_step

    # Generate strikes
    strikes = []
    total_ce_oi = 0.0
    total_pe_oi = 0.0

    for i in range(-cfg.num_strikes_each_side, cfg.num_strikes_each_side + 1):
        strike_price = atm_strike + i * cfg.strike_step

        if strike_price <= 0:
            continue

        # IV from smile model (in percentage, e.g. 15.0)
        ce_iv_pct = _iv_smile(
            strike_price, spot, cfg.atm_iv, cfg.iv_skew_slope, cfg.iv_skew_curvature,
        )
        pe_iv_pct = _iv_smile(
            strike_price, spot, cfg.atm_iv,
            cfg.iv_skew_slope * 1.2,  # puts slightly steeper skew
            cfg.iv_skew_curvature,
        )

        # Convert to decimal for BS functions
        ce_iv_dec = ce_iv_pct / 100.0
        pe_iv_dec = pe_iv_pct / 100.0

        # BS-consistent LTPs
        ce_ltp = bs_price(spot, strike_price, T, _RISK_FREE_RATE, ce_iv_dec, "CE")
        pe_ltp = bs_price(spot, strike_price, T, _RISK_FREE_RATE, pe_iv_dec, "PE")

        # Deltas
        ce_delta = bs_delta(spot, strike_price, T, _RISK_FREE_RATE, ce_iv_dec, "CE") or 0.0
        pe_delta = bs_delta(spot, strike_price, T, _RISK_FREE_RATE, pe_iv_dec, "PE") or 0.0

        # OI (higher for PEs on lower strikes, CEs on higher strikes)
        ce_oi = _oi_profile(strike_price, atm_strike, cfg.oi_base, cfg.oi_decay_rate, rng)
        pe_oi = _oi_profile(strike_price, atm_strike, cfg.oi_base * 1.1, cfg.oi_decay_rate, rng)

        # Volume: fraction of OI
        ce_volume = int(ce_oi * rng.uniform(0.05, 0.2))
        pe_volume = int(pe_oi * rng.uniform(0.05, 0.2))

        # Change in OI (small random)
        ce_change_in_oi = ce_oi * rng.uniform(-0.05, 0.05)
        pe_change_in_oi = pe_oi * rng.uniform(-0.05, 0.05)

        # Bid-ask spread
        spread_mult = cfg.stress_spread_multiplier
        ce_spread = max(ce_ltp * cfg.bid_ask_spread_pct * spread_mult, 0.05)
        pe_spread = max(pe_ltp * cfg.bid_ask_spread_pct * spread_mult, 0.05)

        ce_bid = max(ce_ltp - ce_spread / 2, 0.05)
        ce_ask = ce_ltp + ce_spread / 2
        pe_bid = max(pe_ltp - pe_spread / 2, 0.05)
        pe_ask = pe_ltp + pe_spread / 2

        strikes.append(StrikeData(
            strike_price=strike_price,
            ce_oi=round(ce_oi),
            ce_change_in_oi=round(ce_change_in_oi),
            ce_volume=ce_volume,
            ce_iv=round(ce_iv_pct, 2),
            ce_ltp=round(ce_ltp, 2),
            ce_bid=round(ce_bid, 2),
            ce_ask=round(ce_ask, 2),
            ce_delta=round(ce_delta, 4),
            pe_oi=round(pe_oi),
            pe_change_in_oi=round(pe_change_in_oi),
            pe_volume=pe_volume,
            pe_iv=round(pe_iv_pct, 2),
            pe_ltp=round(pe_ltp, 2),
            pe_bid=round(pe_bid, 2),
            pe_ask=round(pe_ask, 2),
            pe_delta=round(pe_delta, 4),
        ))

        total_ce_oi += ce_oi
        total_pe_oi += pe_oi

    return OptionChainData(
        symbol="NIFTY",
        underlying_value=round(spot, 2),
        expiry=expiry_str,
        strikes=strikes,
        total_ce_oi=round(total_ce_oi),
        total_pe_oi=round(total_pe_oi),
        timestamp=sim_datetime,
    )


def update_chain(
    prev_chain: OptionChainData,
    new_spot: float,
    sim_datetime: datetime,
    scenario_chain_config,
    seed: int,
    expiry_date: date | None = None,
    open_price: float = 0.0,
    current_vix: float = 0.0,
    iv_bump: float = 0.0,
    extra_spread_mult: float = 1.0,
) -> OptionChainData:
    """Incrementally update an existing chain as spot moves.

    Recomputes LTPs and deltas from the new spot, preserves OI structure
    (with small random walk), and adds incremental volume.

    Parameters
    ----------
    prev_chain : OptionChainData
        Previous chain to update.
    new_spot : float
        New underlying price.
    sim_datetime : datetime
        Current simulation time.
    scenario_chain_config : ChainConfig
        Chain params (for IV smile, spread).
    seed : int
        RNG seed (changes per tick for variety).
    expiry_date : date, optional
        Resolved expiry date.
    open_price : float
        Day's opening price for dynamic IV computation.
    current_vix : float
        Current VIX level for dynamic IV scaling.
    iv_bump : float
        Additive IV bump from adversarial engine (percentage points).
    extra_spread_mult : float
        Extra spread multiplier from adversarial engine.
    """
    rng = np.random.default_rng(seed + 3000)
    cfg = scenario_chain_config

    if expiry_date is None:
        expiry_date = sim_datetime.date() + timedelta(days=cfg.expiry_dte)

    days_to_expiry = (expiry_date - sim_datetime.date()).days
    # Adjust T for intraday time decay
    market_close = sim_datetime.replace(hour=15, minute=30, second=0)
    minutes_remaining = max((market_close - sim_datetime).total_seconds() / 60, 1)
    T = max(
        (days_to_expiry - 1 + minutes_remaining / _MINUTES_PER_DAY) / 365.0,
        1 / (365.0 * _MINUTES_PER_DAY),
    )

    # Dynamic ATM IV: reacts to spot moves and VIX
    effective_open = open_price if open_price > 0 else new_spot
    effective_vix = current_vix if current_vix > 0 else cfg.atm_iv
    dynamic_atm_iv = _compute_dynamic_atm_iv(
        cfg.atm_iv, new_spot, effective_open, effective_vix,
        cfg.iv_reactivity,
    )
    # Apply adversarial IV bump on top
    dynamic_atm_iv = max(dynamic_atm_iv + iv_bump, 1.0)

    # Combined spread multiplier: base * adversarial
    combined_spread_mult = cfg.stress_spread_multiplier * max(extra_spread_mult, 1.0)

    new_strikes = []
    total_ce_oi = 0.0
    total_pe_oi = 0.0

    for sd in prev_chain.strikes:
        # Recompute IVs from smile model with dynamic ATM IV
        ce_iv_pct = _iv_smile(
            sd.strike_price, new_spot, dynamic_atm_iv,
            cfg.iv_skew_slope, cfg.iv_skew_curvature,
        )
        pe_iv_pct = _iv_smile(
            sd.strike_price, new_spot, dynamic_atm_iv,
            cfg.iv_skew_slope * 1.2, cfg.iv_skew_curvature,
        )
        ce_iv_dec = ce_iv_pct / 100.0
        pe_iv_dec = pe_iv_pct / 100.0

        # BS-consistent LTPs
        ce_ltp = bs_price(new_spot, sd.strike_price, T, _RISK_FREE_RATE, ce_iv_dec, "CE")
        pe_ltp = bs_price(new_spot, sd.strike_price, T, _RISK_FREE_RATE, pe_iv_dec, "PE")

        # Deltas
        ce_delta = bs_delta(new_spot, sd.strike_price, T, _RISK_FREE_RATE, ce_iv_dec, "CE") or 0.0
        pe_delta = bs_delta(new_spot, sd.strike_price, T, _RISK_FREE_RATE, pe_iv_dec, "PE") or 0.0

        # OI random walk (small ±2% change per tick)
        ce_oi = sd.ce_oi * (1 + rng.uniform(-0.02, 0.02))
        pe_oi = sd.pe_oi * (1 + rng.uniform(-0.02, 0.02))

        # Incremental volume
        ce_volume = sd.ce_volume + int(rng.uniform(10, 100))
        pe_volume = sd.pe_volume + int(rng.uniform(10, 100))

        # Change in OI
        ce_change_in_oi = ce_oi - sd.ce_oi
        pe_change_in_oi = pe_oi - sd.pe_oi

        # Directional spread widening per strike
        ce_dir_mult = _directional_spread_multiplier(
            sd.strike_price, new_spot, effective_open, "CE",
        )
        pe_dir_mult = _directional_spread_multiplier(
            sd.strike_price, new_spot, effective_open, "PE",
        )

        # Bid-ask: base * stress * directional * adversarial
        ce_spread = max(ce_ltp * cfg.bid_ask_spread_pct * combined_spread_mult * ce_dir_mult, 0.05)
        pe_spread = max(pe_ltp * cfg.bid_ask_spread_pct * combined_spread_mult * pe_dir_mult, 0.05)

        new_strikes.append(StrikeData(
            strike_price=sd.strike_price,
            ce_oi=round(ce_oi),
            ce_change_in_oi=round(ce_change_in_oi),
            ce_volume=ce_volume,
            ce_iv=round(ce_iv_pct, 2),
            ce_ltp=round(ce_ltp, 2),
            ce_bid=round(max(ce_ltp - ce_spread / 2, 0.05), 2),
            ce_ask=round(ce_ltp + ce_spread / 2, 2),
            ce_delta=round(ce_delta, 4),
            pe_oi=round(pe_oi),
            pe_change_in_oi=round(pe_change_in_oi),
            pe_volume=pe_volume,
            pe_iv=round(pe_iv_pct, 2),
            pe_ltp=round(pe_ltp, 2),
            pe_bid=round(max(pe_ltp - pe_spread / 2, 0.05), 2),
            pe_ask=round(pe_ltp + pe_spread / 2, 2),
            pe_delta=round(pe_delta, 4),
        ))

        total_ce_oi += ce_oi
        total_pe_oi += pe_oi

    return OptionChainData(
        symbol="NIFTY",
        underlying_value=round(new_spot, 2),
        expiry=prev_chain.expiry,
        strikes=new_strikes,
        total_ce_oi=round(total_ce_oi),
        total_pe_oi=round(total_pe_oi),
        timestamp=sim_datetime,
    )


_MINUTES_PER_DAY = 375
