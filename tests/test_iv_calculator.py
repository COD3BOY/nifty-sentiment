"""Comprehensive tests for core.iv_calculator module.

Tests cover:
  - bs_price: boundary conditions, put-call parity, intrinsic value edge cases
  - bs_vega: ATM peak, boundary returns
  - implied_volatility: round-trip recovery, edge cases, convergence
  - compute_iv_for_chain: empty input, NaN handling, ATM windowing
"""

import math
from datetime import date, timedelta

import pytest

from core.iv_calculator import (
    bs_price,
    bs_vega,
    implied_volatility,
    compute_iv_for_chain,
)
from core.options_models import StrikeData


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Standard parameters used throughout tests
S = 23000.0  # NIFTY spot
K_ATM = 23000.0
K_ITM_CALL = 22000.0  # deep ITM call (S >> K)
K_OTM_CALL = 25000.0  # deep OTM call (K >> S)
R = 0.065  # risk-free rate
SIGMA = 0.15  # 15% annual vol
T = 10 / 365.0  # ~10 calendar days to expiry

# Expiry date approximately 10 days from now (ensures T > 0 in chain tests)
EXPIRY = date.today() + timedelta(days=10)


# ===================================================================
# 1. bs_price tests
# ===================================================================


class TestBsPrice:
    """Tests for the Black-Scholes pricing function."""

    def test_atm_call_and_put_close(self):
        """ATM call and put should be close in value (put-call parity gap is small)."""
        call = bs_price(S, K_ATM, T, R, SIGMA, "CE")
        put = bs_price(S, K_ATM, T, R, SIGMA, "PE")
        # At ATM, the difference is approximately S*(1 - exp(-rT)) which is small
        assert call > 0
        assert put > 0
        # For short-dated ATM, call and put are very similar
        assert abs(call - put) < 0.02 * S  # within 2% of spot

    def test_put_call_parity(self):
        """C - P = S - K*exp(-rT) should hold for any strike."""
        for K in [22000.0, 23000.0, 24000.0]:
            call = bs_price(S, K, T, R, SIGMA, "CE")
            put = bs_price(S, K, T, R, SIGMA, "PE")
            expected_diff = S - K * math.exp(-R * T)
            assert abs((call - put) - expected_diff) < 0.01, (
                f"Put-call parity violated at K={K}"
            )

    def test_deep_itm_call_approx_intrinsic(self):
        """Deep ITM call ~ S - K*exp(-rT)."""
        call = bs_price(S, K_ITM_CALL, T, R, SIGMA, "CE")
        forward_intrinsic = S - K_ITM_CALL * math.exp(-R * T)
        # Should be very close to forward intrinsic value
        assert abs(call - forward_intrinsic) < 10.0  # within 10 pts on NIFTY scale

    def test_deep_otm_call_near_zero(self):
        """Deep OTM call should be very close to 0."""
        call = bs_price(S, K_OTM_CALL, T, R, SIGMA, "CE")
        assert call < 1.0  # effectively zero for a 2000-pt OTM option

    def test_deep_itm_put_approx_intrinsic(self):
        """Deep ITM put (K >> S) ~ K*exp(-rT) - S."""
        K_itm_put = 25000.0  # deep ITM put
        put = bs_price(S, K_itm_put, T, R, SIGMA, "PE")
        forward_intrinsic = K_itm_put * math.exp(-R * T) - S
        assert abs(put - forward_intrinsic) < 10.0

    def test_deep_otm_put_near_zero(self):
        """Deep OTM put (S >> K) should be very close to 0."""
        K_otm_put = 20000.0
        put = bs_price(S, K_otm_put, T, R, SIGMA, "PE")
        assert put < 1.0

    def test_T_zero_returns_intrinsic_call(self):
        """T=0 should return intrinsic value for a call."""
        # ITM call
        price = bs_price(S, K_ITM_CALL, 0.0, R, SIGMA, "CE")
        assert price == pytest.approx(S - K_ITM_CALL, abs=0.01)
        # OTM call
        price = bs_price(S, K_OTM_CALL, 0.0, R, SIGMA, "CE")
        assert price == 0.0

    def test_T_zero_returns_intrinsic_put(self):
        """T=0 should return intrinsic value for a put."""
        K_itm_put = 25000.0
        price = bs_price(S, K_itm_put, 0.0, R, SIGMA, "PE")
        assert price == pytest.approx(K_itm_put - S, abs=0.01)
        # OTM put
        K_otm_put = 20000.0
        price = bs_price(S, K_otm_put, 0.0, R, SIGMA, "PE")
        assert price == 0.0

    def test_negative_T_returns_intrinsic(self):
        """Negative T (expired) should also return intrinsic."""
        price = bs_price(S, K_ITM_CALL, -0.01, R, SIGMA, "CE")
        assert price == pytest.approx(S - K_ITM_CALL, abs=0.01)

    def test_sigma_zero_returns_intrinsic_call(self):
        """sigma=0 should return intrinsic value for a call."""
        price = bs_price(S, K_ITM_CALL, T, R, 0.0, "CE")
        assert price == pytest.approx(max(S - K_ITM_CALL, 0.0), abs=0.01)

    def test_sigma_zero_returns_intrinsic_put(self):
        """sigma=0 should return intrinsic value for a put."""
        K_itm_put = 25000.0
        price = bs_price(S, K_itm_put, T, R, 0.0, "PE")
        assert price == pytest.approx(max(K_itm_put - S, 0.0), abs=0.01)

    def test_sigma_zero_otm_call_is_zero(self):
        """sigma=0 for OTM call returns 0."""
        price = bs_price(S, K_OTM_CALL, T, R, 0.0, "CE")
        assert price == 0.0

    def test_S_zero_returns_zero(self):
        """S=0 should return 0.0."""
        assert bs_price(0.0, K_ATM, T, R, SIGMA, "CE") == 0.0
        assert bs_price(0.0, K_ATM, T, R, SIGMA, "PE") == 0.0

    def test_S_negative_returns_zero(self):
        """S<0 should return 0.0."""
        assert bs_price(-100.0, K_ATM, T, R, SIGMA, "CE") == 0.0

    def test_K_zero_returns_zero(self):
        """K=0 should return 0.0."""
        assert bs_price(S, 0.0, T, R, SIGMA, "CE") == 0.0
        assert bs_price(S, 0.0, T, R, SIGMA, "PE") == 0.0

    def test_K_negative_returns_zero(self):
        """K<0 should return 0.0."""
        assert bs_price(S, -100.0, T, R, SIGMA, "CE") == 0.0

    def test_call_price_positive(self):
        """Any valid call should have non-negative price."""
        price = bs_price(S, K_ATM, T, R, SIGMA, "CE")
        assert price >= 0.0

    def test_put_price_positive(self):
        """Any valid put should have non-negative price."""
        price = bs_price(S, K_ATM, T, R, SIGMA, "PE")
        assert price >= 0.0

    def test_higher_vol_higher_price(self):
        """Higher sigma should produce a higher option price (for both CE/PE)."""
        low_vol = bs_price(S, K_ATM, T, R, 0.10, "CE")
        high_vol = bs_price(S, K_ATM, T, R, 0.30, "CE")
        assert high_vol > low_vol

        low_vol_put = bs_price(S, K_ATM, T, R, 0.10, "PE")
        high_vol_put = bs_price(S, K_ATM, T, R, 0.30, "PE")
        assert high_vol_put > low_vol_put

    def test_longer_time_higher_price(self):
        """Longer T should produce a higher option price (holding other inputs constant)."""
        short_t = bs_price(S, K_ATM, 5 / 365, R, SIGMA, "CE")
        long_t = bs_price(S, K_ATM, 30 / 365, R, SIGMA, "CE")
        assert long_t > short_t


# ===================================================================
# 2. bs_vega tests
# ===================================================================


class TestBsVega:
    """Tests for the Black-Scholes vega function."""

    def test_atm_has_highest_vega(self):
        """ATM vega should be higher than OTM/ITM vega."""
        vega_atm = bs_vega(S, K_ATM, T, R, SIGMA)
        vega_otm = bs_vega(S, K_OTM_CALL, T, R, SIGMA)
        vega_itm = bs_vega(S, K_ITM_CALL, T, R, SIGMA)
        assert vega_atm > vega_otm
        assert vega_atm > vega_itm

    def test_vega_positive(self):
        """Vega should always be non-negative for valid inputs."""
        assert bs_vega(S, K_ATM, T, R, SIGMA) > 0

    def test_T_zero_returns_zero(self):
        """T=0 should return vega=0."""
        assert bs_vega(S, K_ATM, 0.0, R, SIGMA) == 0.0

    def test_T_negative_returns_zero(self):
        """T<0 should return vega=0."""
        assert bs_vega(S, K_ATM, -0.01, R, SIGMA) == 0.0

    def test_S_zero_returns_zero(self):
        """S<=0 should return 0."""
        assert bs_vega(0.0, K_ATM, T, R, SIGMA) == 0.0

    def test_S_negative_returns_zero(self):
        """S<0 should return 0."""
        assert bs_vega(-100.0, K_ATM, T, R, SIGMA) == 0.0

    def test_K_zero_returns_zero(self):
        """K<=0 should return 0."""
        assert bs_vega(S, 0.0, T, R, SIGMA) == 0.0

    def test_K_negative_returns_zero(self):
        """K<0 should return 0."""
        assert bs_vega(S, -100.0, T, R, SIGMA) == 0.0

    def test_sigma_zero_returns_zero(self):
        """sigma=0 should return 0."""
        assert bs_vega(S, K_ATM, T, R, 0.0) == 0.0

    def test_sigma_negative_returns_zero(self):
        """sigma<0 should return 0."""
        assert bs_vega(S, K_ATM, T, R, -0.1) == 0.0

    def test_vega_increases_with_time(self):
        """Longer T produces higher vega (ATM)."""
        short_vega = bs_vega(S, K_ATM, 5 / 365, R, SIGMA)
        long_vega = bs_vega(S, K_ATM, 60 / 365, R, SIGMA)
        assert long_vega > short_vega

    def test_vega_symmetric_around_atm(self):
        """Vega should be roughly symmetric around ATM for equal distance strikes."""
        K_above = K_ATM + 500
        K_below = K_ATM - 500
        vega_above = bs_vega(S, K_above, T, R, SIGMA)
        vega_below = bs_vega(S, K_below, T, R, SIGMA)
        # Lognormal skew means this is not perfectly symmetric; use 25% tolerance
        assert abs(vega_above - vega_below) / max(vega_above, vega_below) < 0.25


# ===================================================================
# 3. implied_volatility tests
# ===================================================================


class TestImpliedVolatility:
    """Tests for the Newton-Raphson + bisection IV solver."""

    def test_atm_call_round_trip(self):
        """Price an ATM call at 15% vol, then recover IV ~ 15%."""
        target_sigma = 0.15
        price = bs_price(S, K_ATM, T, R, target_sigma, "CE")
        iv = implied_volatility(price, S, K_ATM, T, R, "CE")
        assert iv is not None
        assert iv == pytest.approx(target_sigma * 100, abs=0.1)  # within 0.1% IV

    def test_atm_put_round_trip(self):
        """Price an ATM put at 15% vol, then recover IV ~ 15%."""
        target_sigma = 0.15
        price = bs_price(S, K_ATM, T, R, target_sigma, "PE")
        iv = implied_volatility(price, S, K_ATM, T, R, "PE")
        assert iv is not None
        assert iv == pytest.approx(target_sigma * 100, abs=0.1)

    def test_round_trip_various_strikes(self):
        """Round-trip IV recovery across multiple strikes and vols."""
        for K in [22000, 22500, 23000, 23500, 24000]:
            for sigma in [0.10, 0.15, 0.25, 0.40]:
                for opt in ["CE", "PE"]:
                    price = bs_price(S, K, T, R, sigma, opt)
                    if price < 0.01:
                        continue  # skip near-zero prices
                    # Skip cases where BS price is below intrinsic due to
                    # discounting — the solver rejects these as arbitrage.
                    intrinsic = max(S - K, 0.0) if opt == "CE" else max(K - S, 0.0)
                    if price < intrinsic:
                        continue
                    iv = implied_volatility(price, S, K, T, R, opt)
                    assert iv is not None, (
                        f"IV solve failed: K={K}, sigma={sigma}, opt={opt}, price={price:.2f}"
                    )
                    assert iv == pytest.approx(sigma * 100, abs=0.5), (
                        f"IV mismatch: K={K}, sigma={sigma}, opt={opt}: got {iv:.2f}"
                    )

    def test_market_price_zero_returns_none(self):
        """market_price=0 should return None."""
        assert implied_volatility(0.0, S, K_ATM, T, R, "CE") is None

    def test_market_price_negative_returns_none(self):
        """market_price<0 should return None."""
        assert implied_volatility(-10.0, S, K_ATM, T, R, "CE") is None

    def test_T_zero_returns_none(self):
        """T=0 should return None."""
        assert implied_volatility(100.0, S, K_ATM, 0.0, R, "CE") is None

    def test_T_negative_returns_none(self):
        """T<0 should return None."""
        assert implied_volatility(100.0, S, K_ATM, -0.01, R, "CE") is None

    def test_S_zero_returns_none(self):
        """S=0 should return None."""
        assert implied_volatility(100.0, 0.0, K_ATM, T, R, "CE") is None

    def test_S_negative_returns_none(self):
        """S<0 should return None."""
        assert implied_volatility(100.0, -1.0, K_ATM, T, R, "CE") is None

    def test_K_zero_returns_none(self):
        """K=0 should return None."""
        assert implied_volatility(100.0, S, 0.0, T, R, "CE") is None

    def test_K_negative_returns_none(self):
        """K<0 should return None."""
        assert implied_volatility(100.0, S, -1.0, T, R, "CE") is None

    def test_price_below_intrinsic_returns_none(self):
        """Market price below intrinsic should return None (arbitrage)."""
        # ITM call: intrinsic = S - K = 23000 - 22000 = 1000
        intrinsic = S - K_ITM_CALL  # 1000.0
        price_below = intrinsic - 50.0  # 950, below intrinsic
        assert implied_volatility(price_below, S, K_ITM_CALL, T, R, "CE") is None

    def test_price_below_intrinsic_put_returns_none(self):
        """Market price below put intrinsic should return None."""
        K_itm_put = 25000.0
        intrinsic = K_itm_put - S  # 2000.0
        price_below = intrinsic - 50.0
        assert implied_volatility(price_below, S, K_itm_put, T, R, "PE") is None

    def test_deep_itm_call_convergence(self):
        """Deep ITM call should still converge."""
        target_sigma = 0.20
        price = bs_price(S, K_ITM_CALL, T, R, target_sigma, "CE")
        iv = implied_volatility(price, S, K_ITM_CALL, T, R, "CE")
        assert iv is not None
        assert iv == pytest.approx(target_sigma * 100, abs=1.0)

    def test_deep_otm_call_convergence(self):
        """Deep OTM call — may use bisection fallback.

        A moderately OTM option with enough premium to have a solvable IV.
        """
        # Use a strike only 500 pts OTM with higher vol to ensure non-trivial price
        K_mod_otm = 23500.0
        target_sigma = 0.25
        price = bs_price(S, K_mod_otm, T, R, target_sigma, "CE")
        iv = implied_volatility(price, S, K_mod_otm, T, R, "CE")
        assert iv is not None
        assert iv == pytest.approx(target_sigma * 100, abs=1.0)

    def test_deep_otm_put_convergence(self):
        """Deep OTM put convergence (may hit bisection)."""
        K_mod_otm = 22500.0
        target_sigma = 0.25
        price = bs_price(S, K_mod_otm, T, R, target_sigma, "PE")
        iv = implied_volatility(price, S, K_mod_otm, T, R, "PE")
        assert iv is not None
        assert iv == pytest.approx(target_sigma * 100, abs=1.0)

    def test_high_iv_round_trip(self):
        """High IV (80%) should round-trip correctly."""
        target_sigma = 0.80
        price = bs_price(S, K_ATM, T, R, target_sigma, "CE")
        iv = implied_volatility(price, S, K_ATM, T, R, "CE")
        assert iv is not None
        assert iv == pytest.approx(target_sigma * 100, abs=1.0)

    def test_low_iv_round_trip(self):
        """Low IV (5%) should round-trip correctly."""
        target_sigma = 0.05
        price = bs_price(S, K_ATM, T, R, target_sigma, "CE")
        iv = implied_volatility(price, S, K_ATM, T, R, "CE")
        assert iv is not None
        assert iv == pytest.approx(target_sigma * 100, abs=0.5)

    def test_returns_percentage_not_decimal(self):
        """IV should be returned as percentage (e.g. 15.0 not 0.15)."""
        target_sigma = 0.15
        price = bs_price(S, K_ATM, T, R, target_sigma, "CE")
        iv = implied_volatility(price, S, K_ATM, T, R, "CE")
        assert iv is not None
        assert iv > 1.0  # Must be percentage, not decimal
        assert iv < 100.0  # Reasonable upper bound for 15% vol

    def test_very_small_price_near_zero(self):
        """Very small but positive price should either solve or return None, not crash."""
        result = implied_volatility(0.01, S, K_OTM_CALL, T, R, "CE")
        # Result can be a valid IV or None, but must not raise
        assert result is None or result > 0

    def test_very_large_price(self):
        """Very large price (exceeds BS at max vol) may return None or high IV."""
        # BS at sigma=5.0 (500%) is the solver max; price beyond that can't be solved
        max_price = bs_price(S, K_ATM, T, R, 5.0, "CE")
        result = implied_volatility(max_price * 1.5, S, K_ATM, T, R, "CE")
        # Solver may return None since price exceeds max-vol bound
        assert result is None or result > 0

    def test_expiry_day_short_T(self):
        """Very short T (expiry day) should still work for ATM options."""
        short_T = 0.5 / 365.0  # half a day
        target_sigma = 0.15
        price = bs_price(S, K_ATM, short_T, R, target_sigma, "CE")
        if price > 0.01:
            iv = implied_volatility(price, S, K_ATM, short_T, R, "CE")
            assert iv is not None
            assert iv == pytest.approx(target_sigma * 100, abs=2.0)


# ===================================================================
# 4. compute_iv_for_chain tests
# ===================================================================


class TestComputeIvForChain:
    """Tests for batch IV computation on an option chain."""

    @staticmethod
    def _make_strike(strike_price: float, ce_ltp: float = 100.0, pe_ltp: float = 100.0) -> StrikeData:
        """Helper to create a minimal StrikeData."""
        return StrikeData(
            strike_price=strike_price,
            ce_ltp=ce_ltp,
            pe_ltp=pe_ltp,
        )

    def test_empty_strikes_returns_empty(self):
        """Empty strikes list should return empty list."""
        result = compute_iv_for_chain([], 23000.0, EXPIRY)
        assert result == []

    def test_underlying_zero_returns_unchanged(self):
        """underlying=0 should return strikes unchanged."""
        strikes = [self._make_strike(23000)]
        result = compute_iv_for_chain(strikes, 0.0, EXPIRY)
        assert len(result) == 1
        assert result[0].ce_iv == 0.0  # unchanged default

    def test_underlying_negative_returns_unchanged(self):
        """underlying<0 should return strikes unchanged."""
        strikes = [self._make_strike(23000)]
        result = compute_iv_for_chain(strikes, -100.0, EXPIRY)
        assert len(result) == 1
        assert result[0].ce_iv == 0.0

    def test_returns_correct_count(self):
        """Output list length should match input."""
        strikes = [self._make_strike(22500 + i * 50) for i in range(20)]
        result = compute_iv_for_chain(strikes, 23000.0, EXPIRY)
        assert len(result) == len(strikes)

    def test_atm_strike_gets_iv(self):
        """ATM strike should have non-zero IV computed."""
        # Build a chain around 23000 with realistic premiums
        underlying = 23000.0
        strikes = []
        for sp in range(22500, 23600, 50):
            # Use BS to generate realistic premiums at 15% vol
            T_days = (EXPIRY - date.today()).days
            T_years = max(T_days / 365.0, 0.5 / 365.0)
            ce_price = bs_price(underlying, sp, T_years, R, 0.15, "CE")
            pe_price = bs_price(underlying, sp, T_years, R, 0.15, "PE")
            strikes.append(self._make_strike(sp, max(ce_price, 0.05), max(pe_price, 0.05)))

        result = compute_iv_for_chain(strikes, underlying, EXPIRY)

        # Find ATM strike in result
        atm = min(result, key=lambda s: abs(s.strike_price - underlying))
        assert atm.ce_iv > 0, "ATM CE IV should be computed"
        assert atm.pe_iv > 0, "ATM PE IV should be computed"
        # Should be close to the 15% we used to generate prices
        assert atm.ce_iv == pytest.approx(15.0, abs=2.0)
        assert atm.pe_iv == pytest.approx(15.0, abs=2.0)

    def test_far_otm_strikes_outside_atm_range_unchanged(self):
        """Strikes outside +-atm_range should not have IV computed."""
        underlying = 23000.0
        # Create 40 strikes with 50pt spacing: 22000 to 24000
        strikes = [self._make_strike(22000 + i * 50, ce_ltp=50.0, pe_ltp=50.0) for i in range(40)]

        # Use a small atm_range=3 so only ~7 strikes get IV
        result = compute_iv_for_chain(strikes, underlying, EXPIRY, atm_range=3)

        # ATM index is the closest to 23000 => strike index 20 (22000 + 20*50 = 23000)
        atm_idx = min(range(len(strikes)), key=lambda i: abs(strikes[i].strike_price - underlying))

        # Strikes well outside range should be unchanged (ce_iv/pe_iv = 0.0)
        far_strike = result[0]  # 22000, well outside atm_range=3
        assert far_strike.ce_iv == 0.0
        assert far_strike.pe_iv == 0.0

        far_strike_hi = result[-1]  # 23950, also outside range
        assert far_strike_hi.ce_iv == 0.0
        assert far_strike_hi.pe_iv == 0.0

    def test_nan_premium_handling(self):
        """NaN LTP should result in IV=0 (not crash)."""
        strikes = [
            StrikeData(
                strike_price=23000.0,
                ce_ltp=float("nan"),
                pe_ltp=float("nan"),
            )
        ]
        result = compute_iv_for_chain(strikes, 23000.0, EXPIRY)
        assert len(result) == 1
        assert result[0].ce_iv == 0.0
        assert result[0].pe_iv == 0.0

    def test_zero_premium_gets_iv_zero(self):
        """ce_ltp=0 or pe_ltp=0 should leave IV as 0."""
        strikes = [self._make_strike(23000.0, ce_ltp=0.0, pe_ltp=0.0)]
        result = compute_iv_for_chain(strikes, 23000.0, EXPIRY)
        assert result[0].ce_iv == 0.0
        assert result[0].pe_iv == 0.0

    def test_negative_premium_gets_iv_zero(self):
        """Negative LTP should leave IV as 0 (ce_ltp > 0 check fails)."""
        strikes = [self._make_strike(23000.0, ce_ltp=-5.0, pe_ltp=-5.0)]
        result = compute_iv_for_chain(strikes, 23000.0, EXPIRY)
        assert result[0].ce_iv == 0.0
        assert result[0].pe_iv == 0.0

    def test_original_strikes_not_mutated(self):
        """Input list should not be modified in place."""
        strikes = [self._make_strike(23000.0, ce_ltp=150.0, pe_ltp=150.0)]
        original_ce_iv = strikes[0].ce_iv
        original_pe_iv = strikes[0].pe_iv
        compute_iv_for_chain(strikes, 23000.0, EXPIRY)
        assert strikes[0].ce_iv == original_ce_iv
        assert strikes[0].pe_iv == original_pe_iv

    def test_iv_within_floor_cap(self):
        """Computed IVs should be between the floor (0.5%) and cap (200%)."""
        underlying = 23000.0
        strikes = []
        for sp in range(22500, 23600, 100):
            T_days = (EXPIRY - date.today()).days
            T_years = max(T_days / 365.0, 0.5 / 365.0)
            ce_price = bs_price(underlying, sp, T_years, R, 0.15, "CE")
            pe_price = bs_price(underlying, sp, T_years, R, 0.15, "PE")
            strikes.append(self._make_strike(sp, max(ce_price, 0.05), max(pe_price, 0.05)))

        result = compute_iv_for_chain(strikes, underlying, EXPIRY)
        for s in result:
            if s.ce_iv > 0:
                assert 0.5 <= s.ce_iv <= 200.0, f"CE IV {s.ce_iv} out of bounds at K={s.strike_price}"
            if s.pe_iv > 0:
                assert 0.5 <= s.pe_iv <= 200.0, f"PE IV {s.pe_iv} out of bounds at K={s.strike_price}"

    def test_single_strike_chain(self):
        """Single-strike chain should work without errors."""
        underlying = 23000.0
        T_days = (EXPIRY - date.today()).days
        T_years = max(T_days / 365.0, 0.5 / 365.0)
        ce_price = bs_price(underlying, 23000, T_years, R, 0.15, "CE")
        pe_price = bs_price(underlying, 23000, T_years, R, 0.15, "PE")
        strikes = [self._make_strike(23000, ce_price, pe_price)]

        result = compute_iv_for_chain(strikes, underlying, EXPIRY)
        assert len(result) == 1
        assert result[0].ce_iv > 0
        assert result[0].pe_iv > 0

    def test_custom_risk_free_rate(self):
        """Passing a different risk_free_rate should affect computed IV."""
        underlying = 23000.0
        T_days = (EXPIRY - date.today()).days
        T_years = max(T_days / 365.0, 0.5 / 365.0)

        # Generate price at one rate, solve at a different rate
        price = bs_price(underlying, 23000, T_years, 0.065, 0.15, "CE")
        strikes = [self._make_strike(23000, ce_ltp=price, pe_ltp=50.0)]

        result_high_r = compute_iv_for_chain(strikes, underlying, EXPIRY, risk_free_rate=0.10)
        result_low_r = compute_iv_for_chain(strikes, underlying, EXPIRY, risk_free_rate=0.02)

        # The IV should differ when using different rates to invert the same price
        if result_high_r[0].ce_iv > 0 and result_low_r[0].ce_iv > 0:
            assert result_high_r[0].ce_iv != pytest.approx(result_low_r[0].ce_iv, abs=0.01)

    def test_iv_rounded_to_two_decimals(self):
        """Computed IV should be rounded to 2 decimal places."""
        underlying = 23000.0
        T_days = (EXPIRY - date.today()).days
        T_years = max(T_days / 365.0, 0.5 / 365.0)
        ce_price = bs_price(underlying, 23000, T_years, R, 0.15, "CE")
        pe_price = bs_price(underlying, 23000, T_years, R, 0.15, "PE")
        strikes = [self._make_strike(23000, ce_price, pe_price)]

        result = compute_iv_for_chain(strikes, underlying, EXPIRY)
        if result[0].ce_iv > 0:
            # Check it has at most 2 decimal places
            assert result[0].ce_iv == round(result[0].ce_iv, 2)
        if result[0].pe_iv > 0:
            assert result[0].pe_iv == round(result[0].pe_iv, 2)
