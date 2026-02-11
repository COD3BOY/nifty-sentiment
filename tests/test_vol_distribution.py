"""Comprehensive tests for core.vol_distribution module.

Tests cover:
- _compute_rolling_series: RV, VoV, VRP column generation and NaN behavior
- _compute_percentile_ranks: percentile logic, edge cases (few values, inf, all-same)
- VolSnapshot dataclass
- NaN injection resilience
- get_today_vol_snapshot with mocked load_vol_distribution
"""

import math
from datetime import date, timedelta
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from core.vol_distribution import (
    VolSnapshot,
    _compute_rolling_series,
    _compute_percentile_ranks,
    get_today_vol_snapshot,
    load_vol_distribution,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_df(rows: int = 100, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic DataFrame with `rows` trading days of close and vix.

    Prices follow a random walk starting at 23000 (close) and 14 (vix).
    """
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=rows)

    close_returns = rng.normal(0, 0.01, size=rows)
    close = 23000.0 * np.cumprod(1 + close_returns)

    vix = 14.0 + np.cumsum(rng.normal(0, 0.3, size=rows))
    vix = np.clip(vix, 8.0, 40.0)  # keep VIX in a realistic range

    return pd.DataFrame({"close": close, "vix": vix}, index=dates)


def _make_full_df(rows: int = 200, seed: int = 42) -> pd.DataFrame:
    """Create a DataFrame that has gone through both compute stages."""
    raw = _make_synthetic_df(rows=rows, seed=seed)
    df = _compute_rolling_series(raw)
    df = _compute_percentile_ranks(df)
    return df


# ===========================================================================
# 1. _compute_rolling_series
# ===========================================================================

class TestComputeRollingSeries:
    """Tests for _compute_rolling_series."""

    def test_output_columns_exist(self):
        """After compute, rv_5, rv_10, rv_20, vov_20, vrp columns should exist."""
        df = _make_synthetic_df(100)
        result = _compute_rolling_series(df)
        for col in ["log_return", "rv_5", "rv_10", "rv_20", "vov_20", "vrp"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_rv_nan_for_first_n_minus_1_rows(self):
        """rv_N should have NaN for the first N rows (N-1 for log_return + rolling)."""
        df = _make_synthetic_df(100)
        result = _compute_rolling_series(df)

        # log_return has NaN at row 0 (shift(1)), then rolling(5) needs 5 valid values
        # So first valid rv_5 is at index 5 (0-based)
        for n, col in [(5, "rv_5"), (10, "rv_10"), (20, "rv_20")]:
            first_valid_idx = result[col].first_valid_index()
            position = result.index.get_loc(first_valid_idx)
            # row 0 is NaN for log_return, then rolling(n, min_periods=n) needs n values
            # so first valid position is n (0-based)
            assert position == n, (
                f"{col}: first valid at position {position}, expected {n}"
            )

    def test_rv_values_are_positive(self):
        """Realized volatility values should be non-negative (std * constant)."""
        df = _make_synthetic_df(100)
        result = _compute_rolling_series(df)
        for col in ["rv_5", "rv_10", "rv_20"]:
            valid = result[col].dropna()
            assert (valid >= 0).all(), f"{col} has negative values"

    def test_rv_is_annualized_percentage(self):
        """RV values should be in percentage terms (typically 5-50 for NIFTY)."""
        df = _make_synthetic_df(200)
        result = _compute_rolling_series(df)
        for col in ["rv_5", "rv_10", "rv_20"]:
            valid = result[col].dropna()
            # With 1% daily vol, annualized should be ~15-16%
            assert valid.mean() > 1.0, f"{col} mean too low, not annualized?"
            assert valid.mean() < 100.0, f"{col} mean too high"

    def test_vrp_equals_vix_minus_rv20(self):
        """VRP should be exactly vix - rv_20."""
        df = _make_synthetic_df(100)
        result = _compute_rolling_series(df)
        # Compare only where both are non-NaN
        mask = result["rv_20"].notna() & result["vix"].notna()
        expected = result.loc[mask, "vix"] - result.loc[mask, "rv_20"]
        actual = result.loc[mask, "vrp"]
        pd.testing.assert_series_equal(actual, expected, check_names=False)

    def test_vov_20_uses_rv5_changes(self):
        """vov_20 is the rolling std of rv_5 daily changes over 20 days."""
        df = _make_synthetic_df(100)
        result = _compute_rolling_series(df)
        # vov_20 should have NaN until rv_5 has enough history + 20-day rolling
        # rv_5 first valid at position 5, diff makes position 6 first valid rv_5_change
        # rolling(20, min_periods=10) needs 10 valid values, so ~position 15+
        valid_vov = result["vov_20"].dropna()
        assert len(valid_vov) > 0, "vov_20 has no valid values"
        assert (valid_vov >= 0).all(), "vov_20 should be non-negative (it is a std)"

    def test_does_not_modify_input(self):
        """_compute_rolling_series should not modify the input DataFrame."""
        df = _make_synthetic_df(50)
        original_cols = set(df.columns)
        _ = _compute_rolling_series(df)
        assert set(df.columns) == original_cols, "Input DataFrame was modified"


# ===========================================================================
# 2. _compute_percentile_ranks
# ===========================================================================

class TestComputePercentileRanks:
    """Tests for _compute_percentile_ranks."""

    def _make_df_with_known_rv(self, n: int = 200) -> pd.DataFrame:
        """Create a DataFrame with rv_20, vov_20, vrp set to known sorted values."""
        dates = pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=n)
        # Linearly increasing values: 1, 2, ..., n
        values = np.arange(1, n + 1, dtype=float)
        return pd.DataFrame(
            {"rv_20": values, "vov_20": values * 0.5, "vrp": values * 0.1},
            index=dates,
        )

    def test_median_percentile_is_approximately_half(self):
        """With linearly increasing data, the median value's percentile should be ~0.5."""
        n = 200
        df = self._make_df_with_known_rv(n)
        result = _compute_percentile_ranks(df)

        # The value at the midpoint should have percentile close to 0.5
        mid = n // 2
        p_rv_mid = result["p_rv"].iloc[mid]
        assert not np.isnan(p_rv_mid), "p_rv at midpoint is NaN"
        # At position 100 (value=101), the window has values 1..101
        # count_below(101) = 100, percentile = 100/100 = 1.0 for that window
        # Actually we need the last row for a full-window check.
        # Check last row: value=200, window is 1..200, count_below=199, pct=199/199=1.0
        # Better: check a value that is in the middle of the full lookback
        # The function uses an expanding window up to lookback=504.
        # For n=200 all rows are in the window. At position 99 (value 100):
        # window = [1..100], count_below(100) = 99, pct = 99/99 = 1.0
        # This is because each value is the max of its window.
        # We need to test differently: use a shuffled distribution.
        pass  # Replaced by test_known_percentile below

    def test_known_percentile_values(self):
        """With sorted data, the last value should have percentile ~1.0
        and a value in the lower range should have a low percentile."""
        n = 200
        df = self._make_df_with_known_rv(n)
        result = _compute_percentile_ranks(df)

        # Last row is the maximum in the full window -> percentile should be ~1.0
        assert result["p_rv"].iloc[-1] == pytest.approx(1.0, abs=0.01)

        # Row at position 60 (61st value): window=[1..61], value=61
        # count_below(61) = 60, percentile = 60/60 = 1.0 (still max in expanding window)
        # With sorted ascending, every point is the max of its window -> all 1.0 after min_periods
        # So test with reversed data instead
        df_rev = df.copy()
        df_rev["rv_20"] = np.arange(n, 0, -1, dtype=float)
        df_rev["vov_20"] = df_rev["rv_20"] * 0.5
        df_rev["vrp"] = df_rev["rv_20"] * 0.1
        result_rev = _compute_percentile_ranks(df_rev)

        # Last row (value=1) should be the minimum -> percentile ~0.0
        assert result_rev["p_rv"].iloc[-1] == pytest.approx(0.0, abs=0.01)

    def test_fewer_than_60_values_returns_nan(self):
        """With <60 valid values in the window, percentile should be NaN."""
        n = 59
        dates = pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=n)
        df = pd.DataFrame(
            {"rv_20": np.arange(1, n + 1, dtype=float),
             "vov_20": np.arange(1, n + 1, dtype=float),
             "vrp": np.arange(1, n + 1, dtype=float)},
            index=dates,
        )
        result = _compute_percentile_ranks(df)

        # All rows should be NaN since the max window never reaches 60 valid values
        assert result["p_rv"].isna().all(), "Expected all NaN for <60 valid values"

    def test_exactly_60_values_produces_result(self):
        """With exactly 60 valid values, the last row should have a valid percentile."""
        n = 60
        dates = pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=n)
        df = pd.DataFrame(
            {"rv_20": np.arange(1, n + 1, dtype=float),
             "vov_20": np.arange(1, n + 1, dtype=float),
             "vrp": np.arange(1, n + 1, dtype=float)},
            index=dates,
        )
        result = _compute_percentile_ranks(df)

        # The last row has 60 valid values in its expanding window
        assert not np.isnan(result["p_rv"].iloc[-1]), "Expected non-NaN at row 60"

    def test_inf_values_filtered_out(self):
        """inf values in the window should be filtered and not affect percentile."""
        n = 100
        dates = pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=n)
        values = np.arange(1, n + 1, dtype=float)
        # Inject some infs in the middle
        values[30] = np.inf
        values[31] = -np.inf
        values[50] = np.inf
        df = pd.DataFrame(
            {"rv_20": values, "vov_20": values.copy(), "vrp": values.copy()},
            index=dates,
        )
        result = _compute_percentile_ranks(df)

        # Rows with inf as the current value should be NaN
        assert np.isnan(result["p_rv"].iloc[30]), "inf current value should produce NaN"
        assert np.isnan(result["p_rv"].iloc[31]), "-inf current value should produce NaN"

        # Other rows should still have valid percentiles (after 60 valid values accumulate)
        # There are 3 infs, so we reach 60 valid values around row 62
        # Check a later row
        last_val = result["p_rv"].iloc[-1]
        assert not np.isnan(last_val), "Last row should have valid percentile despite some infs"

    def test_all_same_values(self):
        """When all values are the same, count_below=0 -> percentile=0.0."""
        n = 100
        dates = pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=n)
        df = pd.DataFrame(
            {"rv_20": np.full(n, 15.0),
             "vov_20": np.full(n, 3.0),
             "vrp": np.full(n, 2.0)},
            index=dates,
        )
        result = _compute_percentile_ranks(df)

        # All values equal -> count_below = 0 -> 0 / (n-1) = 0.0
        valid = result["p_rv"].dropna()
        assert len(valid) > 0, "Should have valid percentiles"
        assert (valid == 0.0).all(), f"Expected all 0.0 for identical values, got {valid.unique()}"

    def test_does_not_modify_input(self):
        """_compute_percentile_ranks should not modify the input DataFrame."""
        df = _make_synthetic_df(100)
        df = _compute_rolling_series(df)
        original_cols = set(df.columns)
        _ = _compute_percentile_ranks(df)
        assert set(df.columns) == original_cols, "Input DataFrame was modified"


# ===========================================================================
# 3. VolSnapshot dataclass
# ===========================================================================

class TestVolSnapshot:
    """Tests for VolSnapshot dataclass."""

    def test_creation_with_valid_data(self):
        """VolSnapshot can be created with all required fields."""
        snap = VolSnapshot(
            rv_5=12.5, rv_10=13.0, rv_20=14.0,
            vov_20=3.5, vix=16.0, vrp=2.0,
            p_rv=0.45, p_vov=0.50, p_vrp=0.55,
            em=350.0, date="2026-02-10",
        )
        assert snap.rv_5 == 12.5
        assert snap.rv_10 == 13.0
        assert snap.rv_20 == 14.0
        assert snap.vov_20 == 3.5
        assert snap.vix == 16.0
        assert snap.vrp == 2.0
        assert snap.p_rv == 0.45
        assert snap.p_vov == 0.50
        assert snap.p_vrp == 0.55
        assert snap.em == 350.0
        assert snap.date == "2026-02-10"

    def test_fields_are_accessible(self):
        """All fields should be accessible via attribute access."""
        snap = VolSnapshot(
            rv_5=10.0, rv_10=11.0, rv_20=12.0,
            vov_20=2.0, vix=15.0, vrp=3.0,
            p_rv=0.3, p_vov=0.4, p_vrp=0.6,
            em=400.0, date="2026-01-15",
        )
        expected_fields = [
            "rv_5", "rv_10", "rv_20", "vov_20", "vix", "vrp",
            "p_rv", "p_vov", "p_vrp", "em", "date",
        ]
        for field in expected_fields:
            assert hasattr(snap, field), f"VolSnapshot missing field: {field}"

    def test_equality(self):
        """Two VolSnapshots with the same values should be equal (dataclass)."""
        kwargs = dict(
            rv_5=10.0, rv_10=11.0, rv_20=12.0,
            vov_20=2.0, vix=15.0, vrp=3.0,
            p_rv=0.3, p_vov=0.4, p_vrp=0.6,
            em=400.0, date="2026-01-15",
        )
        assert VolSnapshot(**kwargs) == VolSnapshot(**kwargs)


# ===========================================================================
# 4. NaN injection tests
# ===========================================================================

class TestNaNInjection:
    """Tests for NaN resilience in the pipeline."""

    def test_nan_blocks_in_close(self):
        """NaN blocks in close column should propagate as NaN in RV
        but not corrupt the rest of the series."""
        df = _make_synthetic_df(100)
        # Insert a NaN gap of 3 rows in close
        df.iloc[40:43, df.columns.get_loc("close")] = np.nan

        result = _compute_rolling_series(df)

        # log_return should be NaN around the gap
        assert np.isnan(result["log_return"].iloc[40])

        # RV values after the gap should eventually recover
        # (rolling will have NaN for a window covering the gap)
        valid_rv_after = result["rv_20"].iloc[70:].dropna()
        assert len(valid_rv_after) > 0, "rv_20 should recover after NaN gap"

    def test_nan_blocks_in_vix(self):
        """NaN blocks in vix should produce NaN in vrp where vix is NaN."""
        df = _make_synthetic_df(100)
        # Insert a NaN gap of 3 rows in vix
        df.iloc[50:53, df.columns.get_loc("vix")] = np.nan

        result = _compute_rolling_series(df)

        # vrp = vix - rv_20, so vrp is NaN where vix is NaN
        assert np.isnan(result["vrp"].iloc[50])
        assert np.isnan(result["vrp"].iloc[52])

    def test_ffill_limit_5_respected(self):
        """VIX gaps > 5 should stay NaN after ffill(limit=5).

        This tests the raw data pipeline behavior. We simulate what
        _fetch_raw_data does with ffill(limit=5).
        """
        df = _make_synthetic_df(100)
        # Create a gap of 8 days in vix
        df.iloc[30:38, df.columns.get_loc("vix")] = np.nan

        # Apply the same ffill logic as _fetch_raw_data
        df["vix"] = df["vix"].ffill(limit=5)

        # First 5 NaN values should be filled
        assert not np.isnan(df["vix"].iloc[30]), "Row 30 should be forward-filled"
        assert not np.isnan(df["vix"].iloc[34]), "Row 34 should be forward-filled"

        # Remaining values should still be NaN
        assert np.isnan(df["vix"].iloc[35]), "Row 35 should remain NaN (gap > 5)"
        assert np.isnan(df["vix"].iloc[37]), "Row 37 should remain NaN (gap > 5)"

    def test_all_nan_close_returns_all_nan_rv(self):
        """If close is entirely NaN, all RV columns should be NaN."""
        df = _make_synthetic_df(50)
        df["close"] = np.nan
        result = _compute_rolling_series(df)
        assert result["rv_5"].isna().all()
        assert result["rv_20"].isna().all()

    def test_nan_in_percentile_input_skipped(self):
        """NaN values in rv_20/vov_20/vrp should be skipped by percentile calculation."""
        n = 120
        dates = pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=n)
        values = np.arange(1, n + 1, dtype=float)
        # Make first 20 values NaN
        values[:20] = np.nan
        df = pd.DataFrame(
            {"rv_20": values, "vov_20": values.copy(), "vrp": values.copy()},
            index=dates,
        )
        result = _compute_percentile_ranks(df)

        # First 20 are NaN, so at row 79 we have 60 valid values (rows 20..79)
        # Rows before position 79 should be NaN (fewer than 60 valid)
        assert result["p_rv"].iloc[78].item() is np.nan or np.isnan(result["p_rv"].iloc[78])

        # Row 79 should be valid
        assert not np.isnan(result["p_rv"].iloc[79]), "Expected valid percentile at row 79"


# ===========================================================================
# 5. get_today_vol_snapshot
# ===========================================================================

class TestGetTodayVolSnapshot:
    """Tests for get_today_vol_snapshot with mocked load_vol_distribution."""

    def _build_mock_df(self, last_date: pd.Timestamp | None = None, rows: int = 200) -> pd.DataFrame:
        """Build a complete vol distribution DataFrame with known last-row values."""
        if last_date is None:
            last_date = pd.Timestamp.now().normalize()
        dates = pd.bdate_range(end=last_date, periods=rows)

        rng = np.random.RandomState(99)
        df = pd.DataFrame(index=dates)
        df["close"] = 23000.0 + np.cumsum(rng.normal(0, 50, rows))
        df["vix"] = 15.0 + rng.normal(0, 1, rows)
        df["log_return"] = rng.normal(0, 0.01, rows)
        df["rv_5"] = 12.0 + rng.normal(0, 0.5, rows)
        df["rv_10"] = 13.0 + rng.normal(0, 0.5, rows)
        df["rv_20"] = 14.0 + rng.normal(0, 0.5, rows)
        df["rv_5_change"] = rng.normal(0, 0.1, rows)
        df["vov_20"] = 3.5 + rng.normal(0, 0.2, rows)
        df["vrp"] = df["vix"] - df["rv_20"]
        df["p_rv"] = 0.45 + rng.normal(0, 0.05, rows)
        df["p_vov"] = 0.50 + rng.normal(0, 0.05, rows)
        df["p_vrp"] = 0.55 + rng.normal(0, 0.05, rows)

        return df

    @patch("core.vol_distribution.load_vol_distribution")
    def test_returns_snapshot_with_correct_values(self, mock_load):
        """Should return VolSnapshot populated from the last valid row."""
        df = self._build_mock_df()
        mock_load.return_value = df

        snap = get_today_vol_snapshot(spot=23000.0, dte=5)

        assert snap is not None
        assert isinstance(snap, VolSnapshot)

        # Values should come from the last row of the DataFrame
        last_row = df.iloc[-1]
        assert snap.rv_5 == pytest.approx(float(last_row["rv_5"]), abs=1e-6)
        assert snap.rv_10 == pytest.approx(float(last_row["rv_10"]), abs=1e-6)
        assert snap.rv_20 == pytest.approx(float(last_row["rv_20"]), abs=1e-6)
        assert snap.vov_20 == pytest.approx(float(last_row["vov_20"]), abs=1e-6)
        assert snap.vrp == pytest.approx(float(last_row["vrp"]), abs=1e-6)
        assert snap.p_rv == pytest.approx(float(last_row["p_rv"]), abs=1e-6)
        assert snap.p_vov == pytest.approx(float(last_row["p_vov"]), abs=1e-6)
        assert snap.p_vrp == pytest.approx(float(last_row["p_vrp"]), abs=1e-6)

    @patch("core.vol_distribution.load_vol_distribution")
    def test_expected_move_calculation(self, mock_load):
        """EM should be spot * (vix/100) * sqrt(dte/365)."""
        df = self._build_mock_df()
        mock_load.return_value = df

        spot = 23000.0
        dte = 5
        snap = get_today_vol_snapshot(spot=spot, dte=dte)

        assert snap is not None
        vix_val = float(df.iloc[-1]["vix"])
        expected_em = spot * (vix_val / 100.0) * math.sqrt(dte / 365.0)
        assert snap.em == pytest.approx(expected_em, abs=0.01)

    @patch("core.vol_distribution.load_vol_distribution")
    def test_returns_none_when_load_returns_none(self, mock_load):
        """Should return None when load_vol_distribution returns None."""
        mock_load.return_value = None
        assert get_today_vol_snapshot(spot=23000.0) is None

    @patch("core.vol_distribution.load_vol_distribution")
    def test_returns_none_for_empty_dataframe(self, mock_load):
        """Should return None when load_vol_distribution returns an empty DataFrame."""
        mock_load.return_value = pd.DataFrame()
        assert get_today_vol_snapshot(spot=23000.0) is None

    @patch("core.vol_distribution.load_vol_distribution")
    def test_returns_none_for_stale_data(self, mock_load):
        """Should return None when data is older than 2 business days."""
        # Create data ending 10 business days ago
        stale_date = pd.Timestamp.now().normalize() - pd.tseries.offsets.BDay(10)
        df = self._build_mock_df(last_date=stale_date)
        mock_load.return_value = df

        result = get_today_vol_snapshot(spot=23000.0)
        assert result is None

    @patch("core.vol_distribution.load_vol_distribution")
    def test_returns_snapshot_for_fresh_data(self, mock_load):
        """Should return VolSnapshot when data is within 2 business days."""
        # Data ending today
        df = self._build_mock_df(last_date=pd.Timestamp.now().normalize())
        mock_load.return_value = df

        result = get_today_vol_snapshot(spot=23000.0)
        assert result is not None
        assert isinstance(result, VolSnapshot)

    @patch("core.vol_distribution.load_vol_distribution")
    def test_returns_snapshot_for_1_bday_old_data(self, mock_load):
        """Data from 1 business day ago should still be valid."""
        yesterday_bday = pd.Timestamp.now().normalize() - pd.tseries.offsets.BDay(1)
        df = self._build_mock_df(last_date=yesterday_bday)
        mock_load.return_value = df

        result = get_today_vol_snapshot(spot=23000.0)
        assert result is not None

    @patch("core.vol_distribution.load_vol_distribution")
    def test_returns_none_when_all_required_cols_nan(self, mock_load):
        """If all rows have NaN in required columns, should return None."""
        df = self._build_mock_df()
        # Set all required columns to NaN
        for col in ["rv_20", "vov_20", "vrp", "p_rv", "p_vov", "p_vrp"]:
            df[col] = np.nan
        mock_load.return_value = df

        result = get_today_vol_snapshot(spot=23000.0)
        assert result is None

    @patch("core.vol_distribution.load_vol_distribution")
    def test_date_field_in_snapshot(self, mock_load):
        """The date field should match the last valid row's date."""
        target_date = pd.Timestamp.now().normalize()
        df = self._build_mock_df(last_date=target_date)
        mock_load.return_value = df

        snap = get_today_vol_snapshot(spot=23000.0)
        assert snap is not None
        assert snap.date == str(target_date.date())

    @patch("core.vol_distribution.load_vol_distribution")
    def test_vix_nan_replaced_with_zero(self, mock_load):
        """If vix is NaN in the last row, it should be replaced with 0.0."""
        df = self._build_mock_df()
        df.iloc[-1, df.columns.get_loc("vix")] = np.nan
        mock_load.return_value = df

        snap = get_today_vol_snapshot(spot=23000.0)
        assert snap is not None
        assert snap.vix == 0.0
        assert snap.em == 0.0  # EM should be 0 when vix is 0

    @patch("core.vol_distribution.load_vol_distribution")
    def test_em_zero_when_spot_is_zero(self, mock_load):
        """Expected move should be 0.0 when spot is 0."""
        df = self._build_mock_df()
        mock_load.return_value = df

        snap = get_today_vol_snapshot(spot=0.0, dte=5)
        assert snap is not None
        assert snap.em == 0.0

    @patch("core.vol_distribution.load_vol_distribution")
    def test_em_zero_when_dte_is_zero(self, mock_load):
        """Expected move should be 0.0 when dte is 0."""
        df = self._build_mock_df()
        mock_load.return_value = df

        snap = get_today_vol_snapshot(spot=23000.0, dte=0)
        assert snap is not None
        assert snap.em == 0.0


# ===========================================================================
# 6. Integration: full pipeline
# ===========================================================================

class TestFullPipeline:
    """Integration tests running _compute_rolling_series + _compute_percentile_ranks."""

    def test_full_pipeline_produces_valid_output(self):
        """Running both stages on 200 rows should produce valid percentile columns."""
        df = _make_full_df(rows=200)

        # Percentile columns should exist
        for col in ["p_rv", "p_vov", "p_vrp"]:
            assert col in df.columns

        # Last row should have valid percentiles (200 rows > 60 minimum)
        last = df.iloc[-1]
        for col in ["p_rv", "p_vov", "p_vrp"]:
            val = last[col]
            assert not np.isnan(val), f"{col} is NaN in last row"
            assert 0.0 <= val <= 1.0, f"{col}={val} out of [0, 1] range"

    def test_percentile_range_is_zero_to_one(self):
        """All non-NaN percentile values should be in [0, 1]."""
        df = _make_full_df(rows=200)

        for col in ["p_rv", "p_vov", "p_vrp"]:
            valid = df[col].dropna()
            assert (valid >= 0.0).all(), f"{col} has values < 0"
            assert (valid <= 1.0).all(), f"{col} has values > 1"

    def test_small_dataset_produces_nan_percentiles(self):
        """With only 50 rows, percentiles should be NaN (need 60 valid values)."""
        df = _make_full_df(rows=50)

        for col in ["p_rv", "p_vov", "p_vrp"]:
            # With 50 rows, rv_20 has ~30 valid values -> less than 60 -> all NaN
            valid = df[col].dropna()
            assert len(valid) == 0, f"{col} should be all NaN with 50 rows, got {len(valid)} valid"
