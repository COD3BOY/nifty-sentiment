"""Price path generator — GBM + Merton jumps + mean-reversion.

Generates minute-level NIFTY spot + VIX paths from scenario parameters.
All randomness flows from a single ``numpy.random.Generator`` seeded
deterministically for reproducibility.

Output DataFrames use the same column conventions as
``IntradayCandleFetcher``: ``Open, High, Low, Close, Volume`` with
a ``DatetimeIndex`` in IST.
"""

from __future__ import annotations

import math
from datetime import date, datetime, time as dt_time, timedelta, timezone

import numpy as np
import pandas as pd

from simulation.scenario_models import (
    GapConfig,
    PhaseConfig,
    PricePathConfig,
    Scenario,
    VIXConfig,
)

_IST = timezone(timedelta(hours=5, minutes=30))
_MARKET_OPEN = dt_time(9, 15)
_MINUTES_PER_DAY = 375  # 9:15 to 15:30
_TRADING_DAYS_PER_YEAR = 252


def _make_timestamps(sim_date: date, minutes: int = _MINUTES_PER_DAY) -> pd.DatetimeIndex:
    """Create 1-minute bar timestamps from market open."""
    start = datetime.combine(sim_date, _MARKET_OPEN, tzinfo=_IST)
    return pd.DatetimeIndex(
        [start + timedelta(minutes=i) for i in range(minutes)],
        name="datetime",
    )


def _u_shape_volume(
    n_bars: int,
    total_volume: float,
    u_strength: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate U-shaped intraday volume profile with noise.

    High volume at open/close, low mid-day.  ``u_strength`` controls
    the depth of the U (higher = more pronounced).
    """
    x = np.linspace(0, 1, n_bars)
    # Parabola: high at 0 and 1, low at 0.5
    profile = 1.0 + u_strength * (2 * x - 1) ** 2
    # Add noise (±30%)
    noise = rng.uniform(0.7, 1.3, size=n_bars)
    raw = profile * noise
    # Normalize to total volume
    return raw / raw.sum() * total_volume


def _generate_minute_closes(
    phases: list[PhaseConfig],
    open_price: float,
    rng: np.random.Generator,
    n_minutes: int = _MINUTES_PER_DAY,
) -> np.ndarray:
    """Generate minute-level close prices using phase-aware stochastic models.

    Each phase can have different drift, vol, jump, and mean-reversion
    parameters.  Phases are stitched seamlessly — the close of one phase
    becomes the open of the next.
    """
    closes = np.empty(n_minutes)
    price = open_price
    dt = 1.0 / (_TRADING_DAYS_PER_YEAR * _MINUTES_PER_DAY)  # time step in years

    for minute in range(n_minutes):
        # Find active phase
        phase = phases[0]  # default
        for p in phases:
            if p.start_minute <= minute < p.end_minute:
                phase = p
                break

        # GBM component
        sigma = phase.volatility
        mu = phase.drift
        z = rng.standard_normal()
        log_return = (mu - 0.5 * sigma ** 2) * dt + sigma * math.sqrt(dt) * z

        # Mean-reversion (OU) component
        if phase.mean_reversion_kappa > 0:
            target = phase.mean_reversion_level if phase.mean_reversion_level > 0 else open_price
            ou_pull = -phase.mean_reversion_kappa * (price - target) / price * dt
            log_return += ou_pull

        # Merton jump component
        if phase.jump_intensity > 0:
            # Probability of a jump in this minute
            jump_prob = phase.jump_intensity * dt * _TRADING_DAYS_PER_YEAR
            if rng.random() < jump_prob:
                jump = rng.normal(phase.jump_mean, phase.jump_std)
                log_return += jump

        price *= math.exp(log_return)
        closes[minute] = price

    return closes


def _closes_to_ohlcv(
    closes: np.ndarray,
    open_price: float,
    volumes: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert close prices to OHLC bars with realistic wicks.

    Generates Open/High/Low from closes.  The first bar's open is
    ``open_price``; subsequent bars open at the previous close (±tiny noise).
    """
    n = len(closes)
    opens = np.empty(n)
    highs = np.empty(n)
    lows = np.empty(n)

    opens[0] = open_price
    for i in range(1, n):
        # Small gap between bars (±0.01%)
        opens[i] = closes[i - 1] * (1 + rng.normal(0, 0.0001))

    for i in range(n):
        bar_range = abs(closes[i] - opens[i])
        # Wicks extend 0.2–1.0× the body range beyond O/C extremes
        wick_up = bar_range * rng.uniform(0.2, 1.0)
        wick_down = bar_range * rng.uniform(0.2, 1.0)
        highs[i] = max(opens[i], closes[i]) + wick_up
        lows[i] = min(opens[i], closes[i]) - wick_down

    return opens, highs, lows, closes


def generate_price_path(
    scenario: Scenario,
    sim_date: date,
    seed: int,
    prev_close: float | None = None,
) -> pd.DataFrame:
    """Generate a full day's OHLCV candles from a scenario.

    Parameters
    ----------
    scenario : Scenario
        Scenario definition with price path, gap, and volume config.
    sim_date : date
        Date for the simulated trading day.
    seed : int
        RNG seed for reproducibility.
    prev_close : float, optional
        Previous day's close. If None, derived from open_price and gap.

    Returns
    -------
    DataFrame with columns ``Open, High, Low, Close, Volume`` and
    a ``DatetimeIndex`` in IST at 1-minute resolution.
    """
    rng = np.random.default_rng(seed)
    cfg = scenario.price_path
    gap = scenario.gap

    # Resolve opening gap
    open_price = cfg.open_price
    if prev_close is not None and gap.gap_pct != 0:
        open_price = prev_close * (1 + gap.gap_pct)

    # Generate minute closes
    closes = _generate_minute_closes(cfg.phases, open_price, rng)

    # Apply gap-fill dynamics if configured
    if gap.gap_pct != 0 and gap.gap_fill_probability > 0:
        actual_prev_close = prev_close if prev_close else open_price / (1 + gap.gap_pct)
        if rng.random() < gap.gap_fill_probability:
            _apply_gap_fill(closes, open_price, actual_prev_close, gap, rng)

    # Generate volume
    volumes = _u_shape_volume(
        _MINUTES_PER_DAY, cfg.volume_base, cfg.volume_u_shape_strength, rng,
    )

    # Build OHLC
    opens, highs, lows, closes = _closes_to_ohlcv(closes, open_price, volumes, rng)

    # Build DataFrame
    timestamps = _make_timestamps(sim_date)
    df = pd.DataFrame({
        "Open": opens,
        "High": highs,
        "Low": lows,
        "Close": closes,
        "Volume": volumes.astype(int),
    }, index=timestamps)

    return df


def _apply_gap_fill(
    closes: np.ndarray,
    open_price: float,
    prev_close: float,
    gap: GapConfig,
    rng: np.random.Generator,
) -> None:
    """In-place modify closes to partially/fully fill the opening gap."""
    gap_size = open_price - prev_close
    fill_minutes = min(gap.gap_fill_speed, len(closes))

    if fill_minutes <= 0 or abs(gap_size) < 0.01:
        return

    # Linearly pull price toward prev_close over fill_minutes
    for i in range(fill_minutes):
        progress = (i + 1) / fill_minutes
        fill_amount = gap_size * progress * 0.8  # fill ~80% of gap
        closes[i] -= fill_amount * rng.uniform(0.5, 1.0)


def generate_vix_path(
    scenario: Scenario,
    spot_returns: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Generate correlated VIX path for the trading day.

    Uses OU process with negative correlation to spot returns and
    optional scheduled spikes.

    Parameters
    ----------
    scenario : Scenario
        Scenario with VIX configuration.
    spot_returns : array
        Minute-level log returns of the spot price (length N).
    seed : int
        RNG seed (use seed+1000 to avoid correlation with spot RNG).

    Returns
    -------
    1D array of VIX values (same length as spot_returns).
    """
    rng = np.random.default_rng(seed + 1000)
    cfg = scenario.vix
    n = len(spot_returns)
    dt = 1.0 / (_TRADING_DAYS_PER_YEAR * _MINUTES_PER_DAY)

    vix = np.empty(n)
    vix[0] = cfg.open_level

    for i in range(1, n):
        # OU mean-reversion
        ou_pull = cfg.kappa * (cfg.mean_level - vix[i - 1]) * dt

        # Correlated noise: VIX moves opposite to spot
        z_independent = rng.standard_normal()
        rho = cfg.spot_correlation
        z_correlated = rho * spot_returns[i] / (cfg.volatility * math.sqrt(dt) + 1e-10) \
            + math.sqrt(1 - rho ** 2) * z_independent

        diffusion = cfg.volatility * math.sqrt(dt) * z_correlated

        vix[i] = vix[i - 1] + ou_pull + vix[i - 1] * diffusion

        # Floor VIX at 8 (historically never goes below ~9)
        vix[i] = max(vix[i], 8.0)

    # Apply scheduled spikes
    for minute, target in cfg.spikes.items():
        minute = int(minute)
        if 0 <= minute < n:
            # Spike over 5 minutes
            for j in range(min(5, n - minute)):
                blend = (j + 1) / 5.0
                vix[minute + j] = vix[minute + j] * (1 - blend) + target * blend

    return vix


def generate_warmup_days(
    scenario: Scenario,
    sim_date: date,
    seed: int,
    num_days: int = 4,
) -> pd.DataFrame:
    """Generate prior "normal" days for indicator warmup.

    These days use low-volatility, no-jump parameters so indicators
    (EMA-50, RSI) have enough history to produce meaningful values.

    Parameters
    ----------
    scenario : Scenario
        Used for the open_price baseline only.
    sim_date : date
        The main simulation date — warmup days are before this.
    seed : int
        RNG seed for reproducibility.
    num_days : int
        Number of warmup days (default 4).

    Returns
    -------
    DataFrame with OHLCV columns concatenated across all warmup days.
    """
    rng = np.random.default_rng(seed + 5000)

    # Quiet warmup parameters
    warmup_phase = PhaseConfig(
        volatility=0.12,
        drift=0.05,
        jump_intensity=0.0,
        mean_reversion_kappa=0.0,
    )

    frames = []
    price = scenario.price_path.open_price * rng.uniform(0.97, 1.00)

    for day_offset in range(num_days, 0, -1):
        warmup_date = sim_date - timedelta(days=day_offset)
        # Skip weekends
        while warmup_date.weekday() >= 5:
            warmup_date -= timedelta(days=1)

        day_seed = seed + 5000 + day_offset
        day_rng = np.random.default_rng(day_seed)

        closes = _generate_minute_closes([warmup_phase], price, day_rng)
        volumes = _u_shape_volume(
            _MINUTES_PER_DAY, scenario.price_path.volume_base * 0.8,
            scenario.price_path.volume_u_shape_strength, day_rng,
        )
        opens, highs, lows, closes = _closes_to_ohlcv(closes, price, volumes, day_rng)

        timestamps = _make_timestamps(warmup_date)
        df = pd.DataFrame({
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "Volume": volumes.astype(int),
        }, index=timestamps)

        frames.append(df)
        price = closes[-1]  # next day opens near previous close

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames)


def compute_log_returns(closes: np.ndarray) -> np.ndarray:
    """Compute log returns from close prices.  First return is 0."""
    returns = np.zeros(len(closes), dtype=np.float64)
    returns[1:] = np.log(closes[1:] / closes[:-1])
    return returns
