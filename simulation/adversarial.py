"""Adversarial engine — targeted perturbations to stress-test algorithms.

Post-processes base price paths to target algorithm weaknesses.
Reads public state (open positions, SL levels) — does NOT access
algorithm internals.

Usage::

    perturber = AdversarialPerturber(mode="sl_hunt")
    modified_closes = perturber.perturb(
        closes, tick_idx, state, chain_config,
    )
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

from core.paper_trading_models import PaperTradingState
from simulation.scenario_models import ChainConfig

logger = logging.getLogger(__name__)


class PerturbMode(str, Enum):
    """Available adversarial perturbation strategies."""
    SL_HUNT = "sl_hunt"
    FALSE_BREAKOUT = "false_breakout"
    WHIPSAW = "whipsaw"
    LIQUIDITY_VACUUM = "liquidity_vacuum"
    DELTA_SPIKE = "delta_spike"
    IV_EXPANSION = "iv_expansion"
    SPREAD_BLOW = "spread_blow"


@dataclass
class PerturbResult:
    """Result of an adversarial perturbation."""
    modified_close: float
    spread_multiplier: float  # applied to chain bid-ask
    description: str
    iv_bump: float = 0.0  # additive IV percentage points to apply to chain


class AdversarialPerturber:
    """Applies adversarial perturbations to price paths.

    Parameters
    ----------
    mode : PerturbMode or str
        Perturbation strategy.
    intensity : float
        Strength of perturbation (0.0 = none, 1.0 = full strength).
    seed : int
        RNG seed.
    """

    def __init__(
        self,
        mode: PerturbMode | str = PerturbMode.SL_HUNT,
        intensity: float = 1.0,
        seed: int = 42,
    ) -> None:
        self.mode = PerturbMode(mode) if isinstance(mode, str) else mode
        self.intensity = max(0.0, min(1.0, intensity))
        self.rng = np.random.default_rng(seed + 9000)
        self._oscillation_phase = 0.0

    def perturb(
        self,
        base_close: float,
        tick_idx: int,
        state: PaperTradingState,
        chain_config: ChainConfig | None = None,
    ) -> PerturbResult:
        """Apply adversarial perturbation to the next tick's close.

        Parameters
        ----------
        base_close : float
            Unperturbed close price for this tick.
        tick_idx : int
            Current tick index (0-based).
        state : PaperTradingState
            Current algorithm state (read-only).
        chain_config : ChainConfig, optional
            Chain config for spread adjustments.

        Returns
        -------
        PerturbResult with modified close and spread multiplier.
        """
        if self.intensity <= 0:
            return PerturbResult(base_close, 1.0, "no perturbation")

        dispatch = {
            PerturbMode.SL_HUNT: self._sl_hunt,
            PerturbMode.FALSE_BREAKOUT: self._false_breakout,
            PerturbMode.WHIPSAW: self._whipsaw,
            PerturbMode.LIQUIDITY_VACUUM: self._liquidity_vacuum,
            PerturbMode.DELTA_SPIKE: self._delta_spike,
            PerturbMode.IV_EXPANSION: self._iv_expansion,
            PerturbMode.SPREAD_BLOW: self._spread_blow,
        }
        fn = dispatch.get(self.mode, self._noop)
        return fn(base_close, tick_idx, state, chain_config)

    # ------------------------------------------------------------------
    # Perturbation strategies
    # ------------------------------------------------------------------

    def _sl_hunt(
        self, base_close, tick_idx, state, chain_config,
    ) -> PerturbResult:
        """Oscillate spot near positions' stop-loss trigger prices.

        Identifies positions' SL levels and nudges the price to cluster
        around them, triggering SLs on high-vol ticks.
        """
        if not state.open_positions:
            return PerturbResult(base_close, 1.0, "no positions to hunt")

        # Find the nearest SL level
        sl_prices = []
        for pos in state.open_positions:
            # Estimate SL trigger price from net_premium and SL amount
            # For credit spreads: SL triggers when loss exceeds SL amount
            # Approximate: we push price toward strikes
            for leg in pos.legs:
                if leg.action == "SELL":
                    # Short positions are vulnerable to moves against them
                    if leg.option_type == "CE":
                        # Short call — vulnerable to upside
                        sl_prices.append(leg.strike + 100)
                    else:
                        # Short put — vulnerable to downside
                        sl_prices.append(leg.strike - 100)

        if not sl_prices:
            return PerturbResult(base_close, 1.0, "no SL levels found")

        # Find closest SL to current price
        closest_sl = min(sl_prices, key=lambda x: abs(x - base_close))

        # Oscillate around the SL level
        self._oscillation_phase += 0.3
        oscillation = math.sin(self._oscillation_phase) * 30 * self.intensity
        pull_toward_sl = (closest_sl - base_close) * 0.15 * self.intensity

        modified = base_close + pull_toward_sl + oscillation
        return PerturbResult(
            modified,
            1.5,  # slightly wider spreads during SL hunting
            f"SL hunt: targeting {closest_sl:.0f}, oscillation={oscillation:.1f}",
        )

    def _false_breakout(
        self, base_close, tick_idx, state, chain_config,
    ) -> PerturbResult:
        """Push past a level then reverse within 3 bars.

        Simulates false breakouts that trigger directional entries
        then immediately reverse.
        """
        cycle_pos = tick_idx % 20  # 20-bar cycle

        if cycle_pos < 7:
            # Breakout phase: strong directional move
            direction = 1 if self.rng.random() > 0.5 else -1
            breakout_magnitude = 40 * self.intensity * (cycle_pos / 7)
            modified = base_close + direction * breakout_magnitude
            desc = f"false breakout: push {direction:+d} phase ({cycle_pos}/7)"
        elif cycle_pos < 10:
            # Reversal phase: snap back
            reversal_magnitude = -60 * self.intensity * ((cycle_pos - 7) / 3)
            modified = base_close + reversal_magnitude
            desc = f"false breakout: reversal phase ({cycle_pos - 7}/3)"
        else:
            modified = base_close
            desc = "false breakout: recovery"

        return PerturbResult(modified, 1.0, desc)

    def _whipsaw(
        self, base_close, tick_idx, state, chain_config,
    ) -> PerturbResult:
        """Rapid direction changes exceeding 1 ATR each.

        Creates choppy conditions that trigger both bullish and bearish
        signals in rapid succession.
        """
        # High-frequency oscillation with increasing amplitude
        freq = 0.5  # changes direction every ~2 bars
        amplitude = 50 * self.intensity * (1 + 0.5 * math.sin(tick_idx * 0.05))

        whipsaw = amplitude * math.sin(freq * tick_idx)
        # Add random spikes
        if self.rng.random() < 0.1:
            whipsaw += self.rng.choice([-1, 1]) * 80 * self.intensity

        modified = base_close + whipsaw
        return PerturbResult(
            modified,
            1.2,
            f"whipsaw: displacement={whipsaw:+.1f}",
        )

    def _liquidity_vacuum(
        self, base_close, tick_idx, state, chain_config,
    ) -> PerturbResult:
        """Widen bid-ask spreads dramatically during critical moments.

        Price itself moves modestly but execution would be terrible.
        """
        # Determine if this is a "critical moment"
        has_positions = len(state.open_positions) > 0
        near_sl = False
        for pos in state.open_positions:
            unrealized = pos.total_unrealized_pnl
            if unrealized < -pos.stop_loss_amount * 0.7:
                near_sl = True
                break

        if has_positions and near_sl:
            # Vacuum: huge spreads when algo needs to exit
            spread_mult = 3.0 + 2.0 * self.intensity
            price_jitter = self.rng.normal(0, 15) * self.intensity
            return PerturbResult(
                base_close + price_jitter,
                spread_mult,
                f"liquidity vacuum: spread {spread_mult:.1f}x (near SL)",
            )
        elif has_positions:
            spread_mult = 1.5 + 0.5 * self.intensity
            return PerturbResult(
                base_close,
                spread_mult,
                f"liquidity vacuum: spread {spread_mult:.1f}x",
            )
        else:
            return PerturbResult(base_close, 1.0, "liquidity vacuum: no positions")

    def _delta_spike(
        self, base_close, tick_idx, state, chain_config,
    ) -> PerturbResult:
        """Fast spot movement to push short deltas past exit thresholds.

        A sudden +/- 2% move that resolves in ~10 ticks, designed to
        push delta-based exits.
        """
        spike_start = 150  # minute 150 (~11:45 AM)
        spike_duration = 10

        if spike_start <= tick_idx < spike_start + spike_duration:
            progress = (tick_idx - spike_start) / spike_duration
            # Rapid exponential spike
            spike_magnitude = base_close * 0.02 * self.intensity * progress
            modified = base_close + spike_magnitude
            return PerturbResult(
                modified,
                2.0,
                f"delta spike: +{spike_magnitude:.0f} pts ({progress:.0%} complete)",
            )
        elif spike_start + spike_duration <= tick_idx < spike_start + spike_duration + 5:
            # Brief hold at new level
            hold = base_close * 0.02 * self.intensity
            return PerturbResult(
                base_close + hold * 0.9,
                1.5,
                "delta spike: holding at peak",
            )
        else:
            return PerturbResult(base_close, 1.0, "delta spike: inactive")

    def _iv_expansion(
        self, base_close, tick_idx, state, chain_config,
    ) -> PerturbResult:
        """Progressively ramp IV when positions are open.

        Simulates reflexive IV expansion during a sell-off — the more
        positions are open and the longer they've been open, the more
        IV expands.  This is the "vega risk" that kills credit sellers.
        """
        if not state.open_positions:
            # Gentle background IV creep even without positions
            iv_bump = 1.0 * self.intensity * min(tick_idx / 375, 1.0)
            return PerturbResult(
                base_close, 1.0,
                f"iv_expansion: background creep +{iv_bump:.1f}%",
                iv_bump=iv_bump,
            )

        # Ramp IV proportional to number of positions and time held
        num_pos = len(state.open_positions)
        # Base IV bump: 3-8% depending on intensity and position count
        iv_bump = (3.0 + 5.0 * self.intensity) * min(num_pos, 3) / 3.0

        # Accelerate if positions are losing
        total_unrealized = sum(
            p.total_unrealized_pnl for p in state.open_positions
        )
        if total_unrealized < 0:
            # Losing positions → IV expands faster (reflexivity)
            loss_factor = min(abs(total_unrealized) / 20_000, 2.0)
            iv_bump *= (1.0 + loss_factor)

        # Progressive ramp: IV bump grows over the day
        time_factor = min(tick_idx / 200, 1.5)
        iv_bump *= time_factor

        return PerturbResult(
            base_close, 1.0,
            f"iv_expansion: +{iv_bump:.1f}% ({num_pos} pos, unrealized={total_unrealized:.0f})",
            iv_bump=iv_bump,
        )

    def _spread_blow(
        self, base_close, tick_idx, state, chain_config,
    ) -> PerturbResult:
        """Blow out bid-ask spreads when algo has losing positions.

        Simulates liquidity vacuum at the worst time — when the algo
        needs to exit, execution cost skyrockets.
        """
        if not state.open_positions:
            return PerturbResult(base_close, 1.0, "spread_blow: no positions")

        total_unrealized = sum(
            p.total_unrealized_pnl for p in state.open_positions
        )

        if total_unrealized < 0:
            # Losing → massive spread widening (3-5x)
            loss_severity = min(abs(total_unrealized) / 15_000, 1.0)
            spread_mult = 3.0 + 2.0 * self.intensity * loss_severity
            return PerturbResult(
                base_close, spread_mult,
                f"spread_blow: {spread_mult:.1f}x (losing {total_unrealized:.0f})",
            )
        else:
            # Winning but still elevated (1.5-2x)
            spread_mult = 1.5 + 0.5 * self.intensity
            return PerturbResult(
                base_close, spread_mult,
                f"spread_blow: {spread_mult:.1f}x (winning)",
            )

    def _noop(self, base_close, tick_idx, state, chain_config) -> PerturbResult:
        return PerturbResult(base_close, 1.0, "noop")
