"""Comprehensive tests for core.greeks — BS delta, chain deltas, and POP."""

import math

import pytest

from core.greeks import bs_delta, compute_chain_deltas, compute_pop
from core.options_models import OptionChainData, StrikeData


# ---------------------------------------------------------------------------
# Constants shared across tests
# ---------------------------------------------------------------------------
ATM_SPOT = 23000.0
ATM_STRIKE = 23000.0
RISK_FREE = 0.065
TYPICAL_IV = 0.15  # 15% annualized
TYPICAL_T = 7 / 365  # 7 days to expiry


# ===================================================================
# bs_delta tests
# ===================================================================
class TestBsDelta:
    """Unit tests for the bs_delta function."""

    # --- ATM behaviour --------------------------------------------------

    def test_atm_call_delta_near_half(self):
        """ATM call delta should be approximately 0.5."""
        d = bs_delta(ATM_SPOT, ATM_STRIKE, TYPICAL_T, RISK_FREE, TYPICAL_IV, "CE")
        assert d is not None
        assert d == pytest.approx(0.5, abs=0.05)

    def test_atm_put_delta_near_neg_half(self):
        """ATM put delta should be approximately -0.5."""
        d = bs_delta(ATM_SPOT, ATM_STRIKE, TYPICAL_T, RISK_FREE, TYPICAL_IV, "PE")
        assert d is not None
        assert d == pytest.approx(-0.5, abs=0.05)

    # --- Deep ITM / OTM ------------------------------------------------

    def test_deep_itm_call_delta_near_one(self):
        """Deep ITM call (S >> K) should have delta near 1.0."""
        d = bs_delta(25000.0, 20000.0, TYPICAL_T, RISK_FREE, TYPICAL_IV, "CE")
        assert d is not None
        assert d > 0.99

    def test_deep_otm_call_delta_near_zero(self):
        """Deep OTM call (S << K) should have delta near 0.0."""
        d = bs_delta(20000.0, 25000.0, TYPICAL_T, RISK_FREE, TYPICAL_IV, "CE")
        assert d is not None
        assert d < 0.01

    def test_deep_itm_put_delta_near_neg_one(self):
        """Deep ITM put (S << K) should have delta near -1.0."""
        d = bs_delta(20000.0, 25000.0, TYPICAL_T, RISK_FREE, TYPICAL_IV, "PE")
        assert d is not None
        assert d < -0.99

    def test_deep_otm_put_delta_near_zero(self):
        """Deep OTM put (S >> K) should have delta near 0.0."""
        d = bs_delta(25000.0, 20000.0, TYPICAL_T, RISK_FREE, TYPICAL_IV, "PE")
        assert d is not None
        assert d > -0.01

    # --- Put-call parity ------------------------------------------------

    def test_put_call_parity(self):
        """call_delta - put_delta should equal 1.0 for same inputs."""
        call_d = bs_delta(ATM_SPOT, ATM_STRIKE, TYPICAL_T, RISK_FREE, TYPICAL_IV, "CE")
        put_d = bs_delta(ATM_SPOT, ATM_STRIKE, TYPICAL_T, RISK_FREE, TYPICAL_IV, "PE")
        assert call_d is not None and put_d is not None
        assert (call_d - put_d) == pytest.approx(1.0, abs=1e-10)

    def test_put_call_parity_otm(self):
        """Put-call parity holds for OTM strikes too."""
        K = 23500.0
        call_d = bs_delta(ATM_SPOT, K, TYPICAL_T, RISK_FREE, TYPICAL_IV, "CE")
        put_d = bs_delta(ATM_SPOT, K, TYPICAL_T, RISK_FREE, TYPICAL_IV, "PE")
        assert call_d is not None and put_d is not None
        assert (call_d - put_d) == pytest.approx(1.0, abs=1e-10)

    # --- T=0 / sigma=0 intrinsic delta ----------------------------------

    def test_t_zero_call_itm(self):
        """At expiry, ITM call should return intrinsic delta = 1.0."""
        d = bs_delta(23100.0, 23000.0, 0.0, RISK_FREE, TYPICAL_IV, "CE")
        assert d == 1.0

    def test_t_zero_call_otm(self):
        """At expiry, OTM call should return intrinsic delta = 0.0."""
        d = bs_delta(22900.0, 23000.0, 0.0, RISK_FREE, TYPICAL_IV, "CE")
        assert d == 0.0

    def test_t_zero_call_atm(self):
        """At expiry, ATM call (S == K) should return 0.0 (not ITM)."""
        d = bs_delta(23000.0, 23000.0, 0.0, RISK_FREE, TYPICAL_IV, "CE")
        assert d == 0.0

    def test_t_zero_put_itm(self):
        """At expiry, ITM put should return intrinsic delta = -1.0."""
        d = bs_delta(22900.0, 23000.0, 0.0, RISK_FREE, TYPICAL_IV, "PE")
        assert d == -1.0

    def test_t_zero_put_otm(self):
        """At expiry, OTM put should return intrinsic delta = 0.0."""
        d = bs_delta(23100.0, 23000.0, 0.0, RISK_FREE, TYPICAL_IV, "PE")
        assert d == 0.0

    def test_t_zero_put_atm(self):
        """At expiry, ATM put (S == K) should return 0.0 (not ITM)."""
        d = bs_delta(23000.0, 23000.0, 0.0, RISK_FREE, TYPICAL_IV, "PE")
        assert d == 0.0

    def test_sigma_zero_call_itm(self):
        """With zero vol, ITM call returns intrinsic delta 1.0."""
        d = bs_delta(23100.0, 23000.0, TYPICAL_T, RISK_FREE, 0.0, "CE")
        assert d == 1.0

    def test_sigma_zero_call_otm(self):
        """With zero vol, OTM call returns intrinsic delta 0.0."""
        d = bs_delta(22900.0, 23000.0, TYPICAL_T, RISK_FREE, 0.0, "CE")
        assert d == 0.0

    def test_sigma_zero_put_itm(self):
        """With zero vol, ITM put returns intrinsic delta -1.0."""
        d = bs_delta(22900.0, 23000.0, TYPICAL_T, RISK_FREE, 0.0, "PE")
        assert d == -1.0

    def test_sigma_zero_put_otm(self):
        """With zero vol, OTM put returns intrinsic delta 0.0."""
        d = bs_delta(23100.0, 23000.0, TYPICAL_T, RISK_FREE, 0.0, "PE")
        assert d == 0.0

    def test_negative_t_uses_intrinsic(self):
        """Negative T should also fall through to intrinsic delta."""
        d = bs_delta(23100.0, 23000.0, -0.01, RISK_FREE, TYPICAL_IV, "CE")
        assert d == 1.0

    def test_negative_sigma_uses_intrinsic(self):
        """Negative sigma should also fall through to intrinsic delta."""
        d = bs_delta(23100.0, 23000.0, TYPICAL_T, RISK_FREE, -0.15, "CE")
        assert d == 1.0

    # --- Invalid inputs returning None ----------------------------------

    def test_s_zero_returns_none(self):
        """S=0 is invalid; should return None."""
        assert bs_delta(0.0, ATM_STRIKE, TYPICAL_T, RISK_FREE, TYPICAL_IV, "CE") is None

    def test_s_negative_returns_none(self):
        """Negative S is invalid; should return None."""
        assert bs_delta(-100.0, ATM_STRIKE, TYPICAL_T, RISK_FREE, TYPICAL_IV, "CE") is None

    def test_k_zero_returns_none(self):
        """K=0 is invalid; should return None."""
        assert bs_delta(ATM_SPOT, 0.0, TYPICAL_T, RISK_FREE, TYPICAL_IV, "CE") is None

    def test_k_negative_returns_none(self):
        """Negative K is invalid; should return None."""
        assert bs_delta(ATM_SPOT, -100.0, TYPICAL_T, RISK_FREE, TYPICAL_IV, "PE") is None

    def test_both_s_and_k_zero_returns_none(self):
        """Both S=0 and K=0 should return None."""
        assert bs_delta(0.0, 0.0, TYPICAL_T, RISK_FREE, TYPICAL_IV, "CE") is None

    # --- Delta monotonicity / sanity -----------------------------------

    def test_call_delta_increases_with_spot(self):
        """Call delta should increase as spot moves above strike."""
        d_otm = bs_delta(22800.0, ATM_STRIKE, TYPICAL_T, RISK_FREE, TYPICAL_IV, "CE")
        d_atm = bs_delta(ATM_SPOT, ATM_STRIKE, TYPICAL_T, RISK_FREE, TYPICAL_IV, "CE")
        d_itm = bs_delta(23200.0, ATM_STRIKE, TYPICAL_T, RISK_FREE, TYPICAL_IV, "CE")
        assert d_otm < d_atm < d_itm

    def test_put_delta_decreases_with_spot(self):
        """Put delta should decrease (more negative) as spot moves below strike."""
        d_otm = bs_delta(23200.0, ATM_STRIKE, TYPICAL_T, RISK_FREE, TYPICAL_IV, "PE")
        d_atm = bs_delta(ATM_SPOT, ATM_STRIKE, TYPICAL_T, RISK_FREE, TYPICAL_IV, "PE")
        d_itm = bs_delta(22800.0, ATM_STRIKE, TYPICAL_T, RISK_FREE, TYPICAL_IV, "PE")
        # ITM put has most negative delta
        assert d_itm < d_atm < d_otm

    def test_call_delta_bounded_zero_one(self):
        """Call delta should always be in [0, 1]."""
        for S in [20000.0, 22000.0, 23000.0, 24000.0, 26000.0]:
            d = bs_delta(S, ATM_STRIKE, TYPICAL_T, RISK_FREE, TYPICAL_IV, "CE")
            assert d is not None
            assert 0.0 <= d <= 1.0

    def test_put_delta_bounded_neg_one_zero(self):
        """Put delta should always be in [-1, 0]."""
        for S in [20000.0, 22000.0, 23000.0, 24000.0, 26000.0]:
            d = bs_delta(S, ATM_STRIKE, TYPICAL_T, RISK_FREE, TYPICAL_IV, "PE")
            assert d is not None
            assert -1.0 <= d <= 0.0

    def test_higher_vol_atm_delta_stays_near_half(self):
        """ATM call delta should remain near 0.5 regardless of volatility level."""
        d_low = bs_delta(ATM_SPOT, ATM_STRIKE, TYPICAL_T, RISK_FREE, 0.10, "CE")
        d_high = bs_delta(ATM_SPOT, ATM_STRIKE, TYPICAL_T, RISK_FREE, 0.50, "CE")
        assert d_low is not None and d_high is not None
        # Both should be close to 0.5
        assert d_low == pytest.approx(0.5, abs=0.1)
        assert d_high == pytest.approx(0.5, abs=0.1)

    def test_higher_vol_increases_otm_call_delta(self):
        """Higher IV should increase delta for an OTM call (more chance of finishing ITM)."""
        K_otm = 23500.0  # well OTM
        d_low = bs_delta(ATM_SPOT, K_otm, 30 / 365, RISK_FREE, 0.10, "CE")
        d_high = bs_delta(ATM_SPOT, K_otm, 30 / 365, RISK_FREE, 0.40, "CE")
        assert d_low is not None and d_high is not None
        assert d_high > d_low

    def test_longer_dte_call_delta_slightly_above_half(self):
        """Longer DTE slightly raises ATM call delta due to drift term."""
        d_short = bs_delta(ATM_SPOT, ATM_STRIKE, 1 / 365, RISK_FREE, TYPICAL_IV, "CE")
        d_long = bs_delta(ATM_SPOT, ATM_STRIKE, 30 / 365, RISK_FREE, TYPICAL_IV, "CE")
        assert d_short is not None and d_long is not None
        # With positive risk-free rate, longer DTE pushes call delta slightly higher
        assert d_long > d_short


# ===================================================================
# compute_chain_deltas tests
# ===================================================================
class TestComputeChainDeltas:
    """Tests for compute_chain_deltas using OptionChainData."""

    def _make_chain(
        self, underlying: float = 23000.0, strikes: list[StrikeData] | None = None
    ) -> OptionChainData:
        if strikes is None:
            strikes = [
                StrikeData(
                    strike_price=22900.0,
                    ce_iv=16.0,
                    pe_iv=15.5,
                    ce_ltp=150.0,
                    pe_ltp=50.0,
                ),
                StrikeData(
                    strike_price=23000.0,
                    ce_iv=15.0,
                    pe_iv=15.0,
                    ce_ltp=100.0,
                    pe_ltp=100.0,
                ),
                StrikeData(
                    strike_price=23100.0,
                    ce_iv=14.5,
                    pe_iv=16.0,
                    ce_ltp=50.0,
                    pe_ltp=150.0,
                ),
            ]
        return OptionChainData(
            symbol="NIFTY",
            underlying_value=underlying,
            expiry="27-Feb-2026",
            strikes=strikes,
        )

    def test_deltas_populated_for_nonzero_iv(self):
        """Strikes with ce_iv > 0 and pe_iv > 0 should get deltas."""
        chain = self._make_chain()
        result = compute_chain_deltas(chain, T=TYPICAL_T)
        for strike in result:
            assert strike.ce_delta != 0.0, f"ce_delta not set for {strike.strike_price}"
            assert strike.pe_delta != 0.0, f"pe_delta not set for {strike.strike_price}"

    def test_atm_strike_deltas_near_half(self):
        """ATM strike should have ce_delta near 0.5 and pe_delta near -0.5."""
        chain = self._make_chain()
        result = compute_chain_deltas(chain, T=TYPICAL_T)
        atm = result[1]  # 23000 strike
        assert atm.ce_delta == pytest.approx(0.5, abs=0.07)
        assert atm.pe_delta == pytest.approx(-0.5, abs=0.07)

    def test_itm_call_delta_above_atm(self):
        """ITM call (22900 < 23000 spot) should have higher delta than ATM."""
        chain = self._make_chain()
        result = compute_chain_deltas(chain, T=TYPICAL_T)
        itm_call = result[0]  # 22900 strike
        atm_call = result[1]  # 23000 strike
        assert itm_call.ce_delta > atm_call.ce_delta

    def test_deltas_rounded_to_four_decimals(self):
        """All deltas should be rounded to 4 decimal places."""
        chain = self._make_chain()
        result = compute_chain_deltas(chain, T=TYPICAL_T)
        for strike in result:
            if strike.ce_delta != 0.0:
                # Check that rounding to 4 decimals doesn't change the value
                assert strike.ce_delta == round(strike.ce_delta, 4)
            if strike.pe_delta != 0.0:
                assert strike.pe_delta == round(strike.pe_delta, 4)

    def test_zero_iv_strike_not_updated(self):
        """Strike with ce_iv=0 and pe_iv=0 should keep default delta=0."""
        chain = self._make_chain(
            strikes=[
                StrikeData(strike_price=23000.0, ce_iv=0.0, pe_iv=0.0),
            ]
        )
        result = compute_chain_deltas(chain, T=TYPICAL_T)
        assert result[0].ce_delta == 0.0
        assert result[0].pe_delta == 0.0

    def test_mixed_iv_partial_update(self):
        """Strike with only ce_iv>0 should only populate ce_delta."""
        chain = self._make_chain(
            strikes=[
                StrikeData(strike_price=23000.0, ce_iv=15.0, pe_iv=0.0),
            ]
        )
        result = compute_chain_deltas(chain, T=TYPICAL_T)
        assert result[0].ce_delta != 0.0
        assert result[0].pe_delta == 0.0

    def test_s_zero_returns_unmodified(self):
        """S <= 0 should return original strikes unmodified."""
        chain = self._make_chain(underlying=0.0)
        result = compute_chain_deltas(chain, T=TYPICAL_T)
        # Should be the original strikes with default deltas
        for strike in result:
            assert strike.ce_delta == 0.0
            assert strike.pe_delta == 0.0

    def test_s_negative_returns_unmodified(self):
        """Negative underlying should return original strikes unmodified."""
        chain = self._make_chain(underlying=-100.0)
        result = compute_chain_deltas(chain, T=TYPICAL_T)
        for strike in result:
            assert strike.ce_delta == 0.0
            assert strike.pe_delta == 0.0

    def test_t_zero_returns_unmodified(self):
        """T <= 0 should return the original strikes unmodified."""
        chain = self._make_chain()
        result = compute_chain_deltas(chain, T=0.0)
        for strike in result:
            assert strike.ce_delta == 0.0
            assert strike.pe_delta == 0.0

    def test_t_negative_returns_unmodified(self):
        """Negative T should return the original strikes unmodified."""
        chain = self._make_chain()
        result = compute_chain_deltas(chain, T=-1.0)
        for strike in result:
            assert strike.ce_delta == 0.0
            assert strike.pe_delta == 0.0

    def test_result_is_new_list(self):
        """compute_chain_deltas should return a new list, not mutate input."""
        chain = self._make_chain()
        result = compute_chain_deltas(chain, T=TYPICAL_T)
        assert result is not chain.strikes
        # Original chain strikes should still have default deltas
        for orig in chain.strikes:
            assert orig.ce_delta == 0.0
            assert orig.pe_delta == 0.0

    def test_preserves_strike_count(self):
        """Output should contain the same number of strikes as input."""
        chain = self._make_chain()
        result = compute_chain_deltas(chain, T=TYPICAL_T)
        assert len(result) == len(chain.strikes)

    def test_custom_risk_free_rate(self):
        """Different risk-free rate should produce different deltas."""
        chain = self._make_chain()
        result_low = compute_chain_deltas(chain, T=TYPICAL_T, risk_free_rate=0.01)
        result_high = compute_chain_deltas(chain, T=TYPICAL_T, risk_free_rate=0.10)
        # Higher risk-free rate should push call delta slightly higher
        atm_low = result_low[1].ce_delta
        atm_high = result_high[1].ce_delta
        assert atm_high > atm_low

    def test_uses_conftest_chain(self, sample_chain):
        """Works with the shared conftest fixture (20 strikes around 23000)."""
        result = compute_chain_deltas(sample_chain, T=TYPICAL_T)
        assert len(result) == len(sample_chain.strikes)
        # All strikes have non-zero IV in conftest, so all should get deltas
        for strike in result:
            assert strike.ce_delta != 0.0
            assert strike.pe_delta != 0.0

    def test_put_call_parity_holds_for_chain(self):
        """For each strike, ce_delta - pe_delta should be approximately 1.0."""
        chain = self._make_chain()
        # Use same IV for CE and PE to test parity precisely
        for s in chain.strikes:
            s_copy = s.model_copy(update={"pe_iv": s.ce_iv})
            chain.strikes[chain.strikes.index(s)] = s_copy
        result = compute_chain_deltas(chain, T=TYPICAL_T)
        for strike in result:
            if strike.ce_delta != 0.0 and strike.pe_delta != 0.0:
                diff = strike.ce_delta - strike.pe_delta
                assert diff == pytest.approx(1.0, abs=0.001)


# ===================================================================
# compute_pop tests
# ===================================================================
class TestComputePop:
    """Tests for the compute_pop probability-of-profit function."""

    # --- Credit strategy ------------------------------------------------

    def test_credit_pop_basic(self):
        """Credit POP = (1 - |short_delta|) * 100."""
        pop = compute_pop("credit", short_strike_delta=0.3)
        assert pop == pytest.approx(70.0, abs=0.1)

    def test_credit_pop_negative_delta(self):
        """Negative short delta (put spread) should use abs value."""
        pop = compute_pop("credit", short_strike_delta=-0.25)
        assert pop == pytest.approx(75.0, abs=0.1)

    def test_credit_pop_zero_delta(self):
        """Short delta = 0 means POP = 100%."""
        pop = compute_pop("credit", short_strike_delta=0.0)
        assert pop == pytest.approx(100.0, abs=0.1)

    def test_credit_pop_delta_one(self):
        """Short delta = 1.0 means POP = 0%."""
        pop = compute_pop("credit", short_strike_delta=1.0)
        assert pop == pytest.approx(0.0, abs=0.1)

    def test_credit_pop_high_delta(self):
        """High |delta| = low POP."""
        pop = compute_pop("credit", short_strike_delta=0.8)
        assert pop == pytest.approx(20.0, abs=0.1)

    def test_credit_pop_ignores_other_params(self):
        """Credit POP should only depend on short_strike_delta."""
        pop1 = compute_pop("credit", short_strike_delta=0.3)
        pop2 = compute_pop(
            "credit",
            short_strike_delta=0.3,
            breakeven=23100.0,
            spot=23000.0,
            atm_iv=15.0,
            T=0.02,
        )
        assert pop1 == pop2

    # --- Debit strategy -------------------------------------------------

    def test_debit_pop_basic(self):
        """Debit POP should return a value between 0 and 100."""
        pop = compute_pop(
            "debit",
            breakeven=23200.0,
            spot=23000.0,
            atm_iv=15.0,
            T=TYPICAL_T,
        )
        assert 0.0 <= pop <= 100.0

    def test_debit_pop_breakeven_above_spot(self):
        """Long call: breakeven > spot. Should give reasonable POP."""
        pop = compute_pop(
            "debit",
            breakeven=23100.0,
            spot=23000.0,
            atm_iv=15.0,
            T=TYPICAL_T,
        )
        assert 0.0 < pop < 100.0

    def test_debit_pop_breakeven_below_spot(self):
        """Long put: breakeven < spot. Should give reasonable POP."""
        pop = compute_pop(
            "debit",
            breakeven=22900.0,
            spot=23000.0,
            atm_iv=15.0,
            T=TYPICAL_T,
        )
        assert 0.0 < pop < 100.0

    def test_debit_pop_closer_breakeven_higher_pop(self):
        """Breakeven closer to spot should yield higher POP."""
        pop_close = compute_pop(
            "debit",
            breakeven=23050.0,
            spot=23000.0,
            atm_iv=15.0,
            T=TYPICAL_T,
        )
        pop_far = compute_pop(
            "debit",
            breakeven=23300.0,
            spot=23000.0,
            atm_iv=15.0,
            T=TYPICAL_T,
        )
        assert pop_close > pop_far

    def test_debit_pop_higher_iv_increases_pop_for_far_breakeven(self):
        """Higher IV should increase POP when breakeven is far from spot."""
        pop_low_iv = compute_pop(
            "debit",
            breakeven=23500.0,
            spot=23000.0,
            atm_iv=10.0,
            T=30 / 365,
        )
        pop_high_iv = compute_pop(
            "debit",
            breakeven=23500.0,
            spot=23000.0,
            atm_iv=30.0,
            T=30 / 365,
        )
        assert pop_high_iv > pop_low_iv

    def test_debit_pop_longer_dte_increases_pop(self):
        """Longer time to expiry should increase POP for debit strategies."""
        pop_short = compute_pop(
            "debit",
            breakeven=23200.0,
            spot=23000.0,
            atm_iv=15.0,
            T=3 / 365,
        )
        pop_long = compute_pop(
            "debit",
            breakeven=23200.0,
            spot=23000.0,
            atm_iv=15.0,
            T=30 / 365,
        )
        assert pop_long > pop_short

    # --- Edge cases / zero params ----------------------------------------

    def test_debit_t_zero_returns_zero(self):
        """Debit strategy with T=0 should return 0.0."""
        pop = compute_pop(
            "debit",
            breakeven=23100.0,
            spot=23000.0,
            atm_iv=15.0,
            T=0.0,
        )
        assert pop == 0.0

    def test_debit_t_negative_returns_zero(self):
        """Debit strategy with T<0 should return 0.0."""
        pop = compute_pop(
            "debit",
            breakeven=23100.0,
            spot=23000.0,
            atm_iv=15.0,
            T=-0.01,
        )
        assert pop == 0.0

    def test_debit_all_zero_returns_zero(self):
        """All-zero params for debit should return 0.0."""
        pop = compute_pop("debit")
        assert pop == 0.0

    def test_debit_spot_zero_returns_zero(self):
        """Debit with spot=0 should return 0.0."""
        pop = compute_pop(
            "debit",
            breakeven=23100.0,
            spot=0.0,
            atm_iv=15.0,
            T=TYPICAL_T,
        )
        assert pop == 0.0

    def test_debit_breakeven_zero_returns_zero(self):
        """Debit with breakeven=0 should return 0.0."""
        pop = compute_pop(
            "debit",
            breakeven=0.0,
            spot=23000.0,
            atm_iv=15.0,
            T=TYPICAL_T,
        )
        assert pop == 0.0

    def test_debit_iv_zero_returns_zero(self):
        """Debit with atm_iv=0 should return 0.0."""
        pop = compute_pop(
            "debit",
            breakeven=23100.0,
            spot=23000.0,
            atm_iv=0.0,
            T=TYPICAL_T,
        )
        assert pop == 0.0

    # --- Unknown strategy type ------------------------------------------

    def test_unknown_strategy_type_returns_zero(self):
        """Unknown strategy_type should fall through to the debit branch.
        With all default params, returns 0.0."""
        pop = compute_pop("unknown_type")
        assert pop == 0.0

    # --- Return type / rounding -----------------------------------------

    def test_credit_pop_returns_float(self):
        """POP should always be a float."""
        pop = compute_pop("credit", short_strike_delta=0.3)
        assert isinstance(pop, float)

    def test_debit_pop_returns_float(self):
        """POP should always be a float."""
        pop = compute_pop(
            "debit",
            breakeven=23100.0,
            spot=23000.0,
            atm_iv=15.0,
            T=TYPICAL_T,
        )
        assert isinstance(pop, float)

    def test_credit_pop_rounded_to_one_decimal(self):
        """Credit POP should be rounded to 1 decimal place."""
        pop = compute_pop("credit", short_strike_delta=0.333)
        # (1 - 0.333) * 100 = 66.7 after round(..., 1)
        assert pop == pytest.approx(66.7, abs=0.01)

    def test_debit_pop_rounded_to_one_decimal(self):
        """Debit POP should be rounded to 1 decimal place."""
        pop = compute_pop(
            "debit",
            breakeven=23100.0,
            spot=23000.0,
            atm_iv=15.0,
            T=TYPICAL_T,
        )
        # Verify it's rounded to 1 decimal
        assert pop == round(pop, 1)
