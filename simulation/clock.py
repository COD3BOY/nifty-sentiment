"""VirtualClock — time compression for market simulation.

Patches ``_now_ist()`` and ``is_market_open()`` across all import sites so
algorithms believe they are running during real market hours.

Usage::

    clock = VirtualClock(date(2026, 2, 10), tick_interval=60, speed=6.0)
    with clock.patch():
        while not clock.is_day_complete():
            clock.tick()
            # ... run algo loop ...
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from datetime import date, datetime, time as dt_time, timedelta, timezone
from typing import Generator
from unittest.mock import patch

_IST = timezone(timedelta(hours=5, minutes=30))

# Market open/close times
_MARKET_OPEN = dt_time(9, 15)
_MARKET_CLOSE = dt_time(15, 30)

# All modules that import _now_ist by name — must be patched individually.
_NOW_IST_TARGETS = [
    "core.paper_trading_models._now_ist",
    "core.paper_trading_engine._now_ist",
    "core.options_utils._now_ist",
    "core.trade_context._now_ist",
    "core.trade_audit._now_ist",
    "core.improvement_models._now_ist",
    "core.criticizer_models._now_ist",
    "analyzers.daily_review._now_ist",
]

# All modules that import is_market_open by name.
# Note: core.options_utils does a local import inside is_observation_period(),
# so we patch the source (core.market_hours) rather than the import site.
_MARKET_OPEN_TARGETS = [
    "core.market_hours.is_market_open",  # source definition (covers local imports)
    "core.paper_trading_engine.is_market_open",
    "algorithms.jarvis.is_market_open",
    "algorithms.optimus.is_market_open",
    "algorithms.atlas.is_market_open",
]


class VirtualClock:
    """Simulated clock that advances through a trading day.

    Parameters
    ----------
    sim_date : date
        The simulated trading date.
    tick_interval : int
        Seconds to advance per ``tick()`` call (default 60 = 1 minute).
    speed : float
        Wall-clock compression factor. At 6.0×, a 375-minute day takes
        ~62.5 minutes.  Only affects ``sleep_for_tick()`` — the clock
        always advances by exactly ``tick_interval`` per call.
    """

    def __init__(
        self,
        sim_date: date,
        tick_interval: int = 60,
        speed: float = 6.0,
    ) -> None:
        self.sim_date = sim_date
        self.tick_interval = tick_interval
        self.speed = speed

        # Start at market open
        self._current = datetime.combine(
            sim_date, _MARKET_OPEN, tzinfo=_IST,
        )
        self._market_close = datetime.combine(
            sim_date, _MARKET_CLOSE, tzinfo=_IST,
        )
        self._tick_count = 0

    # ------------------------------------------------------------------
    # Time accessors
    # ------------------------------------------------------------------

    @property
    def now(self) -> datetime:
        """Current virtual time (IST-aware)."""
        return self._current

    @property
    def tick_count(self) -> int:
        return self._tick_count

    @property
    def elapsed_minutes(self) -> float:
        """Minutes elapsed since market open."""
        delta = self._current - datetime.combine(
            self.sim_date, _MARKET_OPEN, tzinfo=_IST,
        )
        return delta.total_seconds() / 60.0

    # ------------------------------------------------------------------
    # Time control
    # ------------------------------------------------------------------

    def tick(self) -> datetime:
        """Advance by ``tick_interval`` seconds and return new time."""
        self._current += timedelta(seconds=self.tick_interval)
        self._tick_count += 1
        return self._current

    def set_time(self, t: dt_time) -> None:
        """Jump to a specific time on the simulation date."""
        self._current = datetime.combine(self.sim_date, t, tzinfo=_IST)

    def is_day_complete(self) -> bool:
        """True when virtual time is at or past market close."""
        return self._current >= self._market_close

    def sleep_for_tick(self) -> None:
        """Sleep for the wall-clock duration of one tick (respects speed)."""
        wall_seconds = self.tick_interval / self.speed
        time.sleep(wall_seconds)

    def start_day(self, sim_date: date) -> None:
        """Reset clock to market open of a new date."""
        self.sim_date = sim_date
        self._current = datetime.combine(
            sim_date, _MARKET_OPEN, tzinfo=_IST,
        )
        self._market_close = datetime.combine(
            sim_date, _MARKET_CLOSE, tzinfo=_IST,
        )
        self._tick_count = 0

    # ------------------------------------------------------------------
    # Monkey-patching
    # ------------------------------------------------------------------

    def _make_now_ist(self):
        """Return a closure that returns the current virtual time."""
        def _virtual_now_ist() -> datetime:
            return self._current
        return _virtual_now_ist

    def _make_is_market_open(self):
        """Return a closure for virtual is_market_open."""
        def _virtual_is_market_open() -> bool:
            t = self._current.time()
            return _MARKET_OPEN <= t <= _MARKET_CLOSE
        return _virtual_is_market_open

    @contextmanager
    def patch(self) -> Generator[VirtualClock, None, None]:
        """Context manager that patches all import sites.

        Usage::

            with clock.patch():
                # All _now_ist() calls return virtual time
                # All is_market_open() calls use virtual time
                ...
        """
        now_fn = self._make_now_ist()
        market_fn = self._make_is_market_open()

        patches = []
        for target in _NOW_IST_TARGETS:
            try:
                patches.append(patch(target, now_fn))
            except (ModuleNotFoundError, AttributeError):
                # Module not imported yet — skip silently
                pass

        for target in _MARKET_OPEN_TARGETS:
            try:
                patches.append(patch(target, market_fn))
            except (ModuleNotFoundError, AttributeError):
                pass

        # Enter all patches
        for p in patches:
            p.start()

        try:
            yield self
        finally:
            for p in patches:
                p.stop()

    def patch_start(self) -> list:
        """Non-context-manager version: start patches, return list for later stop."""
        now_fn = self._make_now_ist()
        market_fn = self._make_is_market_open()

        active_patches = []
        for target in _NOW_IST_TARGETS:
            try:
                p = patch(target, now_fn)
                p.start()
                active_patches.append(p)
            except (ModuleNotFoundError, AttributeError):
                pass

        for target in _MARKET_OPEN_TARGETS:
            try:
                p = patch(target, market_fn)
                p.start()
                active_patches.append(p)
            except (ModuleNotFoundError, AttributeError):
                pass

        return active_patches

    @staticmethod
    def patch_stop(active_patches: list) -> None:
        """Stop patches started by patch_start()."""
        for p in active_patches:
            p.stop()
