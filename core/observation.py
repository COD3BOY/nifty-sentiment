"""Observation Period Engine — 9:15–10:00 AM market context.

Computes opening range, gap analysis, initial trend, volume profile, and VWAP
context from the existing 5-day 5-min candle DataFrame. No extra API calls.

All functions are pure: DataFrame in, Pydantic model out.
"""

from __future__ import annotations

import logging
from datetime import time as dt_time

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from core.config import load_config
from core.indicators import compute_vwap

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class OpeningGap(BaseModel):
    """Gap between previous close and today's open."""
    prev_close: float = 0.0
    today_open: float = 0.0
    gap_points: float = 0.0
    gap_pct: float = 0.0
    direction: str = "flat"  # "gap_up" | "gap_down" | "flat"
    gap_fill_pct: float = 0.0  # how much of the gap has been filled (0-100+)
    is_gap_filled: bool = False


class OpeningRange(BaseModel):
    """High/low of the observation window (9:15–10:00)."""
    high: float = 0.0
    low: float = 0.0
    range_points: float = 0.0
    range_pct: float = 0.0
    midpoint: float = 0.0
    is_complete: bool = False
    breakout_direction: str = ""  # "above" | "below" | "" (post-10:00 only)
    breakout_distance_pct: float = 0.0


class InitialTrend(BaseModel):
    """Directional move during the observation window."""
    direction: str = "sideways"  # "up" | "down" | "sideways" | "reversal_up" | "reversal_down"
    move_points: float = 0.0
    move_pct: float = 0.0
    max_move_up_pct: float = 0.0
    max_move_down_pct: float = 0.0
    strength: str = "weak"  # "weak" (<0.3%) | "moderate" (0.3-0.7%) | "strong" (>0.7%)
    reversal_detected: bool = False


class ObservationVolumeProfile(BaseModel):
    """Volume analysis during the observation window vs prior days."""
    total_volume: float = 0.0
    avg_bar_volume: float = 0.0
    typical_opening_volume: float = 0.0  # average of prior days' same-window volume
    relative_volume: float = 1.0  # ratio to typical
    classification: str = "normal"  # "high" (>1.3x) | "normal" | "low" (<0.7x)


class VWAPContext(BaseModel):
    """Price relationship to VWAP during the observation window."""
    bars_above_vwap: int = 0
    bars_below_vwap: int = 0
    total_bars: int = 0
    pct_above: float = 50.0
    relationship: str = "crossing"  # "consistently_above" | "consistently_below" | "crossing"
    current_distance_pct: float = 0.0


class ObservationSnapshot(BaseModel):
    """Top-level container for all observation period data."""
    date: str = ""
    is_complete: bool = False
    bars_collected: int = 0
    gap: OpeningGap = Field(default_factory=OpeningGap)
    opening_range: OpeningRange = Field(default_factory=OpeningRange)
    initial_trend: InitialTrend = Field(default_factory=InitialTrend)
    volume: ObservationVolumeProfile = Field(default_factory=ObservationVolumeProfile)
    vwap_context: VWAPContext = Field(default_factory=VWAPContext)
    bias: str = "neutral"  # "bullish" | "bearish" | "neutral"
    bias_reasons: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MARKET_OPEN = dt_time(9, 15)
_DEFAULT_OBS_END = dt_time(10, 0)


# ---------------------------------------------------------------------------
# Internal helper: split DataFrame by day
# ---------------------------------------------------------------------------

def _split_today_and_prior(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split into (today_full, prior_days, today_dates).

    Returns (today_df, prior_df, today_df) where today_df is all bars for the
    latest trading day, and prior_df is everything before that day.
    """
    dates = df.index.date
    unique_dates = sorted(set(dates))
    if not unique_dates:
        empty = df.iloc[0:0]
        return empty, empty, empty

    today = unique_dates[-1]
    today_mask = dates == today
    today_df = df[today_mask]
    prior_df = df[~today_mask]
    return today_df, prior_df, today_df


def _get_observation_window(
    today_df: pd.DataFrame,
    obs_end: dt_time,
) -> pd.DataFrame:
    """Filter today's bars to the observation window (market open to obs_end)."""
    if today_df.empty:
        return today_df

    times = today_df.index.time
    mask = (times >= _MARKET_OPEN) & (times < obs_end)
    return today_df[mask]


def _get_prior_days_observation(
    prior_df: pd.DataFrame,
    obs_end: dt_time,
) -> pd.DataFrame:
    """Get observation-window bars from prior days (for volume comparison)."""
    if prior_df.empty:
        return prior_df

    times = prior_df.index.time
    mask = (times >= _MARKET_OPEN) & (times < obs_end)
    return prior_df[mask]


# ---------------------------------------------------------------------------
# Computation functions
# ---------------------------------------------------------------------------

def _compute_opening_gap(
    today_df: pd.DataFrame,
    prior_df: pd.DataFrame,
    gap_threshold_pct: float = 0.2,
) -> OpeningGap:
    """Compute gap between previous day's close and today's open."""
    if today_df.empty:
        return OpeningGap()

    today_open = float(today_df["Open"].iloc[0])

    # Previous day's last close
    if prior_df.empty:
        return OpeningGap(today_open=today_open)

    dates = prior_df.index.date
    prev_day = max(set(dates))
    prev_day_df = prior_df[dates == prev_day]
    if prev_day_df.empty:
        return OpeningGap(today_open=today_open)

    prev_close = float(prev_day_df["Close"].iloc[-1])
    if prev_close <= 0:
        return OpeningGap(today_open=today_open, prev_close=prev_close)

    gap_points = today_open - prev_close
    gap_pct = (gap_points / prev_close) * 100.0

    if gap_pct > gap_threshold_pct:
        direction = "gap_up"
    elif gap_pct < -gap_threshold_pct:
        direction = "gap_down"
    else:
        direction = "flat"

    # Gap fill: how much of the gap has the current price retraced?
    current_close = float(today_df["Close"].iloc[-1])
    if abs(gap_points) > 0:
        if direction == "gap_up":
            # For gap up: fill = how much price has come back down toward prev_close
            fill_pct = ((today_open - current_close) / gap_points) * 100.0
        elif direction == "gap_down":
            # For gap down: fill = how much price has come back up toward prev_close
            fill_pct = ((current_close - today_open) / abs(gap_points)) * 100.0
        else:
            fill_pct = 0.0
        fill_pct = max(0.0, fill_pct)
    else:
        fill_pct = 0.0

    return OpeningGap(
        prev_close=round(prev_close, 2),
        today_open=round(today_open, 2),
        gap_points=round(gap_points, 2),
        gap_pct=round(gap_pct, 2),
        direction=direction,
        gap_fill_pct=round(fill_pct, 1),
        is_gap_filled=fill_pct >= 100.0,
    )


def _compute_opening_range(
    obs_window_df: pd.DataFrame,
    is_complete: bool,
    current_price: float = 0.0,
) -> OpeningRange:
    """Compute high/low of the observation window."""
    if obs_window_df.empty:
        return OpeningRange()

    high = float(obs_window_df["High"].max())
    low = float(obs_window_df["Low"].min())
    range_points = high - low
    midpoint = (high + low) / 2.0
    range_pct = (range_points / midpoint * 100.0) if midpoint > 0 else 0.0

    breakout_direction = ""
    breakout_distance_pct = 0.0
    if is_complete and current_price > 0:
        if current_price > high:
            breakout_direction = "above"
            breakout_distance_pct = ((current_price - high) / high) * 100.0
        elif current_price < low:
            breakout_direction = "below"
            breakout_distance_pct = ((low - current_price) / low) * 100.0

    return OpeningRange(
        high=round(high, 2),
        low=round(low, 2),
        range_points=round(range_points, 2),
        range_pct=round(range_pct, 2),
        midpoint=round(midpoint, 2),
        is_complete=is_complete,
        breakout_direction=breakout_direction,
        breakout_distance_pct=round(breakout_distance_pct, 2),
    )


def _compute_initial_trend(
    obs_window_df: pd.DataFrame,
    moderate_pct: float = 0.3,
    strong_pct: float = 0.7,
) -> InitialTrend:
    """Compute initial trend direction and strength."""
    if len(obs_window_df) < 2:
        return InitialTrend()

    first_open = float(obs_window_df["Open"].iloc[0])
    current_close = float(obs_window_df["Close"].iloc[-1])

    if first_open <= 0:
        return InitialTrend()

    move_points = current_close - first_open
    move_pct = (move_points / first_open) * 100.0

    # Max moves during the window
    highs = obs_window_df["High"]
    lows = obs_window_df["Low"]
    max_high = float(highs.max())
    min_low = float(lows.min())
    max_move_up_pct = ((max_high - first_open) / first_open) * 100.0
    max_move_down_pct = ((first_open - min_low) / first_open) * 100.0

    # Reversal detection: moved significantly in one direction then reversed
    abs_move = abs(move_pct)
    reversal_detected = False
    if abs_move < moderate_pct:
        # Current move is small, but if max move was significant, it's a reversal
        if max_move_up_pct > strong_pct and move_pct < 0:
            reversal_detected = True
            direction = "reversal_down"
        elif max_move_down_pct > strong_pct and move_pct > 0:
            reversal_detected = True
            direction = "reversal_up"
        else:
            direction = "sideways"
    elif move_pct > 0:
        direction = "up"
        if max_move_down_pct > moderate_pct and max_move_down_pct > abs_move * 0.5:
            reversal_detected = True
            direction = "reversal_up"
    else:
        direction = "down"
        if max_move_up_pct > moderate_pct and max_move_up_pct > abs_move * 0.5:
            reversal_detected = True
            direction = "reversal_down"

    # Strength classification
    if abs_move >= strong_pct:
        strength = "strong"
    elif abs_move >= moderate_pct:
        strength = "moderate"
    else:
        strength = "weak"

    return InitialTrend(
        direction=direction,
        move_points=round(move_points, 2),
        move_pct=round(move_pct, 2),
        max_move_up_pct=round(max_move_up_pct, 2),
        max_move_down_pct=round(max_move_down_pct, 2),
        strength=strength,
        reversal_detected=reversal_detected,
    )


def _compute_volume_profile(
    obs_window_df: pd.DataFrame,
    prior_obs_df: pd.DataFrame,
    high_ratio: float = 1.3,
    low_ratio: float = 0.7,
) -> ObservationVolumeProfile:
    """Compare today's observation window volume to prior days' average."""
    if obs_window_df.empty:
        return ObservationVolumeProfile()

    total_volume = float(obs_window_df["Volume"].sum())
    n_bars = len(obs_window_df)
    avg_bar_volume = total_volume / n_bars if n_bars > 0 else 0.0

    # Compute typical opening volume from prior days
    typical_opening_volume = 0.0
    if not prior_obs_df.empty:
        prior_dates = sorted(set(prior_obs_df.index.date))
        daily_volumes = []
        for d in prior_dates:
            day_mask = prior_obs_df.index.date == d
            day_vol = float(prior_obs_df[day_mask]["Volume"].sum())
            if day_vol > 0:
                daily_volumes.append(day_vol)
        if daily_volumes:
            typical_opening_volume = sum(daily_volumes) / len(daily_volumes)

    # Relative volume
    if typical_opening_volume > 0:
        relative_volume = total_volume / typical_opening_volume
    else:
        relative_volume = 1.0

    # Classification
    if relative_volume > high_ratio:
        classification = "high"
    elif relative_volume < low_ratio:
        classification = "low"
    else:
        classification = "normal"

    return ObservationVolumeProfile(
        total_volume=round(total_volume, 0),
        avg_bar_volume=round(avg_bar_volume, 0),
        typical_opening_volume=round(typical_opening_volume, 0),
        relative_volume=round(relative_volume, 2),
        classification=classification,
    )


def _compute_vwap_context(
    obs_window_df: pd.DataFrame,
    consistency_pct: float = 70.0,
) -> VWAPContext:
    """Count bars above/below VWAP during the observation window."""
    if obs_window_df.empty:
        return VWAPContext()

    vwap = compute_vwap(obs_window_df)
    closes = obs_window_df["Close"]
    total_bars = len(obs_window_df)

    above = int((closes > vwap).sum())
    below = int((closes < vwap).sum())
    pct_above = (above / total_bars * 100.0) if total_bars > 0 else 50.0

    if pct_above >= consistency_pct:
        relationship = "consistently_above"
    elif pct_above <= (100.0 - consistency_pct):
        relationship = "consistently_below"
    else:
        relationship = "crossing"

    # Current distance from VWAP
    current_close = float(closes.iloc[-1])
    current_vwap = float(vwap.iloc[-1])
    if current_vwap > 0:
        distance_pct = ((current_close - current_vwap) / current_vwap) * 100.0
    else:
        distance_pct = 0.0

    return VWAPContext(
        bars_above_vwap=above,
        bars_below_vwap=below,
        total_bars=total_bars,
        pct_above=round(pct_above, 1),
        relationship=relationship,
        current_distance_pct=round(distance_pct, 2),
    )


def _compute_bias(
    gap: OpeningGap,
    opening_range: OpeningRange,
    trend: InitialTrend,
    volume: ObservationVolumeProfile,
    vwap: VWAPContext,
) -> tuple[str, list[str]]:
    """Score-based aggregation into bullish/bearish/neutral with reasons."""
    score = 0
    reasons: list[str] = []

    # Gap contribution
    if gap.direction == "gap_up" and not gap.is_gap_filled:
        score += 1
        reasons.append(f"Gap up {gap.gap_pct:+.2f}% holding")
    elif gap.direction == "gap_down" and not gap.is_gap_filled:
        score -= 1
        reasons.append(f"Gap down {gap.gap_pct:+.2f}% holding")
    elif gap.is_gap_filled:
        reasons.append(f"Gap ({gap.gap_pct:+.2f}%) filled — mean-reversion tendency")

    # Trend contribution
    if trend.direction in ("up", "reversal_up"):
        score += 1
        reasons.append(f"Initial trend {trend.direction}: {trend.move_pct:+.2f}% ({trend.strength})")
    elif trend.direction in ("down", "reversal_down"):
        score -= 1
        reasons.append(f"Initial trend {trend.direction}: {trend.move_pct:+.2f}% ({trend.strength})")
    else:
        reasons.append(f"Sideways trend: {trend.move_pct:+.2f}% ({trend.strength})")

    # Stronger weight for strong trends
    if trend.strength == "strong":
        if trend.direction in ("up", "reversal_up"):
            score += 1
        elif trend.direction in ("down", "reversal_down"):
            score -= 1

    # Volume contribution
    if volume.classification == "high":
        # High volume confirms direction
        if score > 0:
            score += 1
            reasons.append(f"High volume ({volume.relative_volume:.1f}x typical) confirms bullish")
        elif score < 0:
            score -= 1
            reasons.append(f"High volume ({volume.relative_volume:.1f}x typical) confirms bearish")
        else:
            reasons.append(f"High volume ({volume.relative_volume:.1f}x typical) — watching direction")
    elif volume.classification == "low":
        reasons.append(f"Low volume ({volume.relative_volume:.1f}x typical) — weak conviction")

    # VWAP contribution
    if vwap.relationship == "consistently_above":
        score += 1
        reasons.append(f"VWAP: consistently above ({vwap.pct_above:.0f}% of bars)")
    elif vwap.relationship == "consistently_below":
        score -= 1
        reasons.append(f"VWAP: consistently below ({vwap.pct_above:.0f}% of bars)")
    else:
        reasons.append(f"VWAP: crossing ({vwap.pct_above:.0f}% above)")

    # Classify
    if score >= 2:
        bias = "bullish"
    elif score <= -2:
        bias = "bearish"
    else:
        bias = "neutral"

    return bias, reasons


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_observation_snapshot(
    df: pd.DataFrame,
    observation_end_time: str = "10:00",
) -> ObservationSnapshot:
    """Compute full observation snapshot from the candle DataFrame.

    Parameters
    ----------
    df : 5-day 5-min candle DataFrame with DatetimeIndex and OHLCV columns.
    observation_end_time : HH:MM string for end of observation window.

    Returns
    -------
    ObservationSnapshot with all sub-models populated.
    """
    if df is None or df.empty:
        return ObservationSnapshot()

    # Load observation config
    cfg = load_config().get("paper_trading", {}).get("observation", {})
    gap_threshold = cfg.get("gap_threshold_pct", 0.2)
    moderate_pct = cfg.get("trend_moderate_pct", 0.3)
    strong_pct = cfg.get("trend_strong_pct", 0.7)
    high_ratio = cfg.get("volume_high_ratio", 1.3)
    low_ratio = cfg.get("volume_low_ratio", 0.7)
    consistency_pct = cfg.get("vwap_consistency_pct", 70.0)

    # Parse observation end time
    try:
        h, m = (int(x) for x in observation_end_time.split(":"))
        obs_end = dt_time(h, m)
    except (ValueError, TypeError):
        obs_end = _DEFAULT_OBS_END

    # Split data
    today_df, prior_df, _ = _split_today_and_prior(df)
    if today_df.empty:
        return ObservationSnapshot()

    obs_window_df = _get_observation_window(today_df, obs_end)
    prior_obs_df = _get_prior_days_observation(prior_df, obs_end)

    n_bars = len(obs_window_df)
    today_str = str(today_df.index.date[0])

    # Check if observation is complete (current time past observation end)
    last_bar_time = today_df.index[-1].time() if not today_df.empty else _MARKET_OPEN
    is_complete = last_bar_time >= obs_end

    # Current price (latest close in today's full data)
    current_price = float(today_df["Close"].iloc[-1])

    # Compute all sub-models
    gap = _compute_opening_gap(today_df, prior_df, gap_threshold)
    opening_range = _compute_opening_range(obs_window_df, is_complete, current_price)
    trend = _compute_initial_trend(obs_window_df, moderate_pct, strong_pct)
    volume = _compute_volume_profile(obs_window_df, prior_obs_df, high_ratio, low_ratio)
    vwap_ctx = _compute_vwap_context(obs_window_df, consistency_pct)
    bias, bias_reasons = _compute_bias(gap, opening_range, trend, volume, vwap_ctx)

    snapshot = ObservationSnapshot(
        date=today_str,
        is_complete=is_complete,
        bars_collected=n_bars,
        gap=gap,
        opening_range=opening_range,
        initial_trend=trend,
        volume=volume,
        vwap_context=vwap_ctx,
        bias=bias,
        bias_reasons=bias_reasons,
    )

    logger.info(
        "Observation: %d bars | gap=%s %.2f%% | trend=%s %s | vol=%s %.1fx | VWAP=%s | bias=%s",
        n_bars, gap.direction, gap.gap_pct, trend.direction, trend.strength,
        volume.classification, volume.relative_volume, vwap_ctx.relationship, bias,
    )

    return snapshot
