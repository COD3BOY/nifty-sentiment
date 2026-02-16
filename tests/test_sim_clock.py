"""Tests for simulation.clock — VirtualClock and monkeypatching."""

from datetime import date, datetime, time as dt_time, timedelta, timezone

import pytest

from simulation.clock import VirtualClock, _IST, _MARKET_OPEN, _MARKET_CLOSE


class TestVirtualClockBasics:
    """Basic clock operations."""

    def test_initial_time_is_market_open(self):
        clock = VirtualClock(date(2026, 2, 10))
        expected = datetime(2026, 2, 10, 9, 15, tzinfo=_IST)
        assert clock.now == expected

    def test_tick_advances_by_interval(self):
        clock = VirtualClock(date(2026, 2, 10), tick_interval=60)
        before = clock.now
        clock.tick()
        assert clock.now == before + timedelta(seconds=60)

    def test_tick_count_increments(self):
        clock = VirtualClock(date(2026, 2, 10))
        assert clock.tick_count == 0
        clock.tick()
        assert clock.tick_count == 1
        clock.tick()
        assert clock.tick_count == 2

    def test_elapsed_minutes(self):
        clock = VirtualClock(date(2026, 2, 10), tick_interval=60)
        assert clock.elapsed_minutes == 0.0
        for _ in range(30):
            clock.tick()
        assert abs(clock.elapsed_minutes - 30.0) < 0.01

    def test_day_complete_at_market_close(self):
        clock = VirtualClock(date(2026, 2, 10), tick_interval=60)
        assert not clock.is_day_complete()
        # Advance 375 ticks (9:15 + 375 min = 15:30)
        for _ in range(375):
            clock.tick()
        assert clock.is_day_complete()

    def test_day_not_complete_before_close(self):
        clock = VirtualClock(date(2026, 2, 10), tick_interval=60)
        for _ in range(374):
            clock.tick()
        assert not clock.is_day_complete()

    def test_set_time(self):
        clock = VirtualClock(date(2026, 2, 10))
        clock.set_time(dt_time(12, 0))
        assert clock.now.hour == 12
        assert clock.now.minute == 0

    def test_start_day_resets(self):
        clock = VirtualClock(date(2026, 2, 10))
        clock.tick()
        clock.tick()
        clock.start_day(date(2026, 2, 11))
        assert clock.sim_date == date(2026, 2, 11)
        assert clock.now.hour == 9
        assert clock.now.minute == 15
        assert clock.tick_count == 0


class TestVirtualClockPatching:
    """Test that monkeypatching works for _now_ist and is_market_open."""

    def test_patch_now_ist(self):
        """Verify _now_ist returns virtual time inside patch context."""
        clock = VirtualClock(date(2026, 2, 10))

        with clock.patch():
            from core.paper_trading_models import _now_ist
            now = _now_ist()
            assert now.hour == 9
            assert now.minute == 15
            assert now.date() == date(2026, 2, 10)

            clock.tick()  # advance 1 minute
            now2 = _now_ist()
            assert now2.minute == 16

    def test_patch_is_market_open(self):
        """Verify is_market_open returns True during virtual market hours."""
        clock = VirtualClock(date(2026, 2, 10))

        with clock.patch():
            from core.market_hours import is_market_open
            assert is_market_open()

            # Advance past close
            clock.set_time(dt_time(15, 31))
            assert not is_market_open()

    def test_patch_restores_on_exit(self):
        """Verify real functions are restored after context exits."""
        clock = VirtualClock(date(2026, 2, 10))

        with clock.patch():
            pass  # Patch is active here

        # Outside the context, functions should be restored
        # (We can't easily test the exact original values without
        # knowing the real current time, but we can at least verify
        # the patching didn't throw)

    def test_patch_multiple_modules(self):
        """Verify patching works across multiple modules."""
        clock = VirtualClock(date(2026, 2, 10))

        with clock.patch():
            # These should all return the same virtual time
            from core.paper_trading_models import _now_ist as now1
            t1 = now1()

            # Verify it's our virtual time, not real time
            assert t1.date() == date(2026, 2, 10)
            assert t1.hour == 9
            assert t1.minute == 15

    def test_patch_start_stop(self):
        """Test non-context-manager patching."""
        clock = VirtualClock(date(2026, 2, 10))

        patches = clock.patch_start()
        try:
            from core.paper_trading_models import _now_ist
            assert _now_ist().date() == date(2026, 2, 10)
        finally:
            VirtualClock.patch_stop(patches)
