"""Shared test fixtures for the NIFTY options trading system."""

from datetime import date, datetime, timedelta

import pytest

from core.options_models import (
    OptionChainData,
    OptionsAnalytics,
    StrikeData,
    TechnicalIndicators,
    TradeLeg,
    TradeSuggestion,
    StrategyName,
)


@pytest.fixture
def sample_strikes() -> list[StrikeData]:
    """20 strikes around ATM=23000 with realistic data."""
    strikes = []
    base = 22500
    for i in range(20):
        sp = base + i * 50
        # CE premiums decrease away from ATM, PE increases
        dist = abs(sp - 23000)
        ce_ltp = max(5.0, 200.0 - dist * 0.4)
        pe_ltp = max(5.0, 200.0 - (1000 - dist) * 0.2) if sp <= 23000 else max(5.0, dist * 0.3)
        strikes.append(
            StrikeData(
                strike_price=sp,
                ce_oi=50000 + i * 1000,
                ce_volume=2000 + i * 100,
                ce_ltp=round(ce_ltp, 2),
                ce_bid=round(ce_ltp - 1.0, 2),
                ce_ask=round(ce_ltp + 1.0, 2),
                ce_iv=14.5 + i * 0.2,
                pe_oi=40000 + (20 - i) * 1000,
                pe_volume=1500 + (20 - i) * 100,
                pe_ltp=round(pe_ltp, 2),
                pe_bid=round(pe_ltp - 1.0, 2),
                pe_ask=round(pe_ltp + 1.0, 2),
                pe_iv=14.0 + (20 - i) * 0.2,
            )
        )
    return strikes


@pytest.fixture
def sample_chain(sample_strikes) -> OptionChainData:
    """Option chain with 20 strikes, underlying=23000."""
    return OptionChainData(
        symbol="NIFTY",
        underlying_value=23000.0,
        expiry="27-Feb-2026",
        strikes=sample_strikes,
        total_ce_oi=sum(s.ce_oi for s in sample_strikes),
        total_pe_oi=sum(s.pe_oi for s in sample_strikes),
        timestamp=datetime.utcnow(),
    )


@pytest.fixture
def sample_technicals() -> TechnicalIndicators:
    """Neutral-trending technicals."""
    return TechnicalIndicators(
        spot=23000.0,
        spot_change=50.0,
        spot_change_pct=0.22,
        vwap=22980.0,
        ema_9=23010.0,
        ema_20=22990.0,
        ema_21=22985.0,
        ema_50=22950.0,
        rsi=52.0,
        supertrend=22900.0,
        supertrend_direction=1,
        bb_upper=23200.0,
        bb_middle=23000.0,
        bb_lower=22800.0,
        data_staleness_minutes=1.5,
    )


@pytest.fixture
def sample_analytics() -> OptionsAnalytics:
    return OptionsAnalytics(
        pcr=1.1,
        pcr_label="Neutral",
        max_pain=23000.0,
        support_strike=22800.0,
        support_oi=80000.0,
        resistance_strike=23200.0,
        resistance_oi=75000.0,
        atm_strike=23000.0,
        atm_iv=15.0,
        iv_skew=0.5,
        iv_percentile=55.0,
    )


@pytest.fixture
def sample_vol_snapshot():
    from core.vol_distribution import VolSnapshot
    return VolSnapshot(
        rv_5=12.0,
        rv_10=13.0,
        rv_20=14.0,
        vov_20=3.5,
        vix=16.0,
        vrp=2.0,
        p_rv=0.45,
        p_vov=0.50,
        p_vrp=0.55,
        em=350.0,
        date="2026-02-10",
    )


@pytest.fixture
def next_thursday() -> date:
    """Next Thursday from today (common NIFTY expiry)."""
    today = date.today()
    days_ahead = 3 - today.weekday()  # Thursday = 3
    if days_ahead <= 0:
        days_ahead += 7
    return today + timedelta(days=days_ahead)
