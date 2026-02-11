"""Tests for core.validation — every validator with valid, invalid, and boundary inputs."""

import math

import numpy as np
import pandas as pd
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
from core.validation import (
    ValidationResult,
    validate_option_chain,
    validate_strike_data,
    validate_iv_value,
    validate_vol_snapshot,
    validate_technicals,
    validate_candle_dataframe,
    validate_trade_suggestion,
    validate_paper_trading_state,
    validate_iv_for_storage,
    validate_yfinance_df,
    validate_config,
)


# ── ValidationResult ──────────────────────────────────────────────────────

class TestValidationResult:
    def test_valid_is_truthy(self):
        assert ValidationResult(valid=True)

    def test_invalid_is_falsy(self):
        assert not ValidationResult(valid=False, errors=["bad"])

    def test_merge_both_valid(self):
        a = ValidationResult(valid=True, warnings=["w1"])
        b = ValidationResult(valid=True, warnings=["w2"])
        merged = a.merge(b)
        assert merged.valid
        assert merged.warnings == ["w1", "w2"]

    def test_merge_one_invalid(self):
        a = ValidationResult(valid=True)
        b = ValidationResult(valid=False, errors=["e1"])
        merged = a.merge(b)
        assert not merged.valid
        assert merged.errors == ["e1"]


# ── validate_option_chain ─────────────────────────────────────────────────

class TestValidateOptionChain:
    def test_valid_chain(self, sample_chain):
        result = validate_option_chain(sample_chain)
        assert result.valid

    def test_none_chain(self):
        result = validate_option_chain(None)
        assert not result.valid
        assert "None" in result.errors[0]

    def test_zero_underlying(self, sample_chain):
        chain = sample_chain.model_copy(update={"underlying_value": 0.0})
        result = validate_option_chain(chain)
        assert not result.valid

    def test_negative_underlying(self, sample_chain):
        chain = sample_chain.model_copy(update={"underlying_value": -100.0})
        result = validate_option_chain(chain)
        assert not result.valid

    def test_too_few_strikes(self):
        strikes = [StrikeData(strike_price=23000 + i * 50) for i in range(5)]
        chain = OptionChainData(underlying_value=23000, strikes=strikes)
        result = validate_option_chain(chain)
        assert not result.valid
        assert "strikes" in result.errors[0].lower() or "10" in result.errors[0]

    def test_no_oi_strikes(self):
        strikes = [StrikeData(strike_price=23000 + i * 50) for i in range(15)]
        chain = OptionChainData(underlying_value=23000, strikes=strikes)
        result = validate_option_chain(chain)
        assert not result.valid

    def test_negative_premium(self):
        strikes = [
            StrikeData(strike_price=23000 + i * 50, ce_oi=10000, ce_ltp=-5.0)
            for i in range(15)
        ]
        chain = OptionChainData(underlying_value=23000, strikes=strikes)
        result = validate_option_chain(chain)
        assert not result.valid


# ── validate_strike_data ──────────────────────────────────────────────────

class TestValidateStrikeData:
    def test_valid_strike(self):
        s = StrikeData(strike_price=23000, ce_ltp=100, pe_ltp=90, ce_iv=15.0, pe_iv=14.0)
        assert validate_strike_data(s).valid

    def test_zero_strike(self):
        s = StrikeData(strike_price=0)
        assert not validate_strike_data(s).valid

    def test_negative_premium(self):
        s = StrikeData(strike_price=23000, ce_ltp=-1.0)
        assert not validate_strike_data(s).valid

    def test_iv_out_of_range_warns(self):
        s = StrikeData(strike_price=23000, ce_iv=250.0)
        result = validate_strike_data(s)
        assert result.valid  # warning, not error
        assert len(result.warnings) > 0

    def test_bid_ask_inversion_warns(self):
        s = StrikeData(strike_price=23000, ce_bid=110, ce_ask=100)
        result = validate_strike_data(s)
        assert result.valid  # warning
        assert any("bid" in w.lower() for w in result.warnings)


# ── validate_iv_value ─────────────────────────────────────────────────────

class TestValidateIvValue:
    def test_valid_iv(self):
        assert validate_iv_value(15.0).valid

    def test_none_iv(self):
        result = validate_iv_value(None)
        assert result.valid  # None is a valid "could not compute" signal
        assert len(result.warnings) > 0

    def test_nan_iv(self):
        assert not validate_iv_value(float("nan")).valid

    def test_inf_iv(self):
        assert not validate_iv_value(float("inf")).valid

    def test_below_floor(self):
        assert not validate_iv_value(0.1).valid

    def test_above_cap(self):
        assert not validate_iv_value(250.0).valid

    def test_at_floor(self):
        assert validate_iv_value(0.5).valid

    def test_at_cap(self):
        assert validate_iv_value(200.0).valid


# ── validate_vol_snapshot ─────────────────────────────────────────────────

class TestValidateVolSnapshot:
    def test_valid_snapshot(self, sample_vol_snapshot):
        assert validate_vol_snapshot(sample_vol_snapshot).valid

    def test_none_snapshot(self):
        assert not validate_vol_snapshot(None).valid

    def test_percentile_out_of_range(self, sample_vol_snapshot):
        sample_vol_snapshot.p_rv = 1.5
        assert not validate_vol_snapshot(sample_vol_snapshot).valid

    def test_negative_percentile(self, sample_vol_snapshot):
        sample_vol_snapshot.p_vov = -0.1
        assert not validate_vol_snapshot(sample_vol_snapshot).valid

    def test_nan_vix(self, sample_vol_snapshot):
        sample_vol_snapshot.vix = float("nan")
        assert not validate_vol_snapshot(sample_vol_snapshot).valid

    def test_negative_vix(self, sample_vol_snapshot):
        sample_vol_snapshot.vix = -5.0
        assert not validate_vol_snapshot(sample_vol_snapshot).valid


# ── validate_technicals ───────────────────────────────────────────────────

class TestValidateTechnicals:
    def test_valid_technicals(self, sample_technicals):
        assert validate_technicals(sample_technicals).valid

    def test_none_technicals(self):
        assert not validate_technicals(None).valid

    def test_zero_spot(self):
        t = TechnicalIndicators(spot=0.0)
        assert not validate_technicals(t).valid

    def test_rsi_over_100(self):
        t = TechnicalIndicators(spot=23000, rsi=105.0)
        assert not validate_technicals(t).valid

    def test_stale_data_warns(self):
        t = TechnicalIndicators(spot=23000, rsi=50, data_staleness_minutes=15)
        result = validate_technicals(t)
        assert result.valid
        assert any("stale" in w.lower() for w in result.warnings)


# ── validate_candle_dataframe ─────────────────────────────────────────────

class TestValidateCandleDataframe:
    def test_valid_df(self):
        df = pd.DataFrame({
            "Open": [100, 101, 102],
            "High": [103, 104, 105],
            "Low": [99, 100, 101],
            "Close": [102, 103, 104],
            "Volume": [1000, 1100, 1200],
        })
        assert validate_candle_dataframe(df).valid

    def test_none_df(self):
        assert not validate_candle_dataframe(None).valid

    def test_empty_df(self):
        assert not validate_candle_dataframe(pd.DataFrame()).valid

    def test_missing_columns(self):
        df = pd.DataFrame({"Open": [100], "Close": [101]})
        assert not validate_candle_dataframe(df).valid

    def test_all_nan_close(self):
        df = pd.DataFrame({
            "Open": [100], "High": [103], "Low": [99],
            "Close": [float("nan")], "Volume": [1000],
        })
        assert not validate_candle_dataframe(df).valid

    def test_few_rows_warns(self):
        df = pd.DataFrame({
            "Open": [100, 101], "High": [103, 104], "Low": [99, 100],
            "Close": [102, 103], "Volume": [1000, 1100],
        })
        result = validate_candle_dataframe(df)
        assert result.valid
        assert any("rows" in w for w in result.warnings)


# ── validate_trade_suggestion ─────────────────────────────────────────────

class TestValidateTradeSuggestion:
    def _make_suggestion(self, **overrides):
        defaults = dict(
            strategy=StrategyName.IRON_CONDOR,
            legs=[
                TradeLeg(action="SELL", instrument="NIFTY 23000 CE", strike=23000, option_type="CE", ltp=100),
                TradeLeg(action="BUY", instrument="NIFTY 23100 CE", strike=23100, option_type="CE", ltp=50),
            ],
            direction_bias="Neutral",
            confidence="Medium",
            score=65.0,
            entry_timing="Enter now",
            technicals_to_check=["RSI neutral"],
            expected_outcome="Profit if range-bound",
            max_profit="₹5000",
            max_loss="₹3000",
            stop_loss="2x credit",
            position_size="1 lot",
            reasoning=["Good setup"],
            net_credit_debit=50.0,
        )
        defaults.update(overrides)
        return TradeSuggestion(**defaults)

    def test_valid_suggestion(self):
        assert validate_trade_suggestion(self._make_suggestion()).valid

    def test_no_legs(self):
        assert not validate_trade_suggestion(self._make_suggestion(legs=[])).valid

    def test_zero_ltp_leg(self):
        s = self._make_suggestion(
            legs=[TradeLeg(action="SELL", instrument="X", strike=23000, option_type="CE", ltp=0)]
        )
        assert not validate_trade_suggestion(s).valid

    def test_credit_strategy_with_debit(self):
        s = self._make_suggestion(
            strategy=StrategyName.SHORT_STRADDLE,
            net_credit_debit=-100.0,
        )
        assert not validate_trade_suggestion(s).valid


# ── validate_iv_for_storage ───────────────────────────────────────────────

class TestValidateIvForStorage:
    def test_valid(self):
        assert validate_iv_for_storage(15.0).valid

    def test_nan(self):
        assert not validate_iv_for_storage(float("nan")).valid

    def test_inf(self):
        assert not validate_iv_for_storage(float("inf")).valid

    def test_zero(self):
        assert not validate_iv_for_storage(0.0).valid

    def test_negative(self):
        assert not validate_iv_for_storage(-5.0).valid

    def test_below_floor(self):
        assert not validate_iv_for_storage(0.3).valid

    def test_above_cap(self):
        assert not validate_iv_for_storage(250.0).valid

    def test_at_boundaries(self):
        assert validate_iv_for_storage(0.5).valid
        assert validate_iv_for_storage(200.0).valid


# ── validate_yfinance_df ──────────────────────────────────────────────────

class TestValidateYfinanceDf:
    def test_valid(self):
        df = pd.DataFrame({"Close": [100, 101], "Volume": [1000, 1100]})
        assert validate_yfinance_df(df).valid

    def test_none(self):
        assert not validate_yfinance_df(None).valid

    def test_empty(self):
        assert not validate_yfinance_df(pd.DataFrame()).valid

    def test_multiindex_warns(self):
        arrays = [["Close", "Close"], ["^NSEI", "^NSEI"]]
        tuples = list(zip(*arrays))
        idx = pd.MultiIndex.from_tuples(tuples)
        df = pd.DataFrame([[100, 100]], columns=idx)
        result = validate_yfinance_df(df)
        assert result.valid
        assert any("MultiIndex" in w for w in result.warnings)

    def test_missing_expected_cols(self):
        df = pd.DataFrame({"Close": [100]})
        result = validate_yfinance_df(df, expected_cols=["Close", "Volume"])
        assert not result.valid


# ── validate_config ───────────────────────────────────────────────────────

class TestValidateConfig:
    def test_valid_config(self):
        cfg = {
            "sources": {},
            "engine": {},
            "options_desk": {},
            "paper_trading": {"lot_size": 65, "initial_capital": 5000000},
        }
        assert validate_config(cfg).valid

    def test_missing_section(self):
        cfg = {"sources": {}, "engine": {}}
        result = validate_config(cfg)
        assert not result.valid

    def test_invalid_lot_size(self):
        cfg = {
            "sources": {},
            "engine": {},
            "options_desk": {},
            "paper_trading": {"lot_size": -1, "initial_capital": 5000000},
        }
        assert not validate_config(cfg).valid

    def test_invalid_capital(self):
        cfg = {
            "sources": {},
            "engine": {},
            "options_desk": {},
            "paper_trading": {"lot_size": 65, "initial_capital": 0},
        }
        assert not validate_config(cfg).valid

    def test_daily_loss_out_of_range_warns(self):
        cfg = {
            "sources": {},
            "engine": {},
            "options_desk": {},
            "paper_trading": {
                "lot_size": 65,
                "initial_capital": 5000000,
                "daily_loss_limit_pct": 15.0,
            },
        }
        result = validate_config(cfg)
        assert result.valid
        assert any("daily_loss" in w for w in result.warnings)
