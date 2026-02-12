"""Tests for core.paper_trading_engine — persistence, state safety."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from core.paper_trading_engine import save_state, load_state, force_save_state, _state_file
from core.paper_trading_models import PaperTradingState, PaperPosition, PositionLeg, StrategyType, TradeRecord


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Redirect state files to a temp directory."""
    with patch("core.paper_trading_engine._DATA_DIR", tmp_path):
        yield tmp_path


class TestSaveLoadState:
    def test_roundtrip(self, tmp_data_dir):
        state = PaperTradingState(initial_capital=5000000)
        save_state(state, algo_name="test_algo")
        loaded = load_state(algo_name="test_algo")
        assert loaded is not None
        assert loaded.initial_capital == 5000000

    def test_load_nonexistent(self, tmp_data_dir):
        result = load_state(algo_name="nonexistent")
        assert result is None

    def test_backup_created(self, tmp_data_dir):
        state1 = PaperTradingState(initial_capital=1000000)
        save_state(state1, algo_name="test_bak")
        # Save again with a position to trigger backup (trade count changes)
        state2 = _make_state_with_position()
        save_state(state2, algo_name="test_bak")

        bak = tmp_data_dir / "paper_trading_state_test_bak.bak"
        assert bak.exists()
        bak_state = PaperTradingState.model_validate_json(bak.read_text())
        assert bak_state.initial_capital == 1000000

    def test_atomic_write_no_tmp_left(self, tmp_data_dir):
        state = PaperTradingState(initial_capital=5000000)
        save_state(state, algo_name="test_atomic")
        tmp = tmp_data_dir / "paper_trading_state_test_atomic.tmp"
        assert not tmp.exists()

    def test_state_preserves_positions(self, tmp_data_dir):
        from core.paper_trading_models import PaperPosition, PositionLeg, StrategyType
        state = PaperTradingState(
            initial_capital=5000000,
            open_positions=[
                PaperPosition(
                    strategy="Short Straddle",
                    strategy_type=StrategyType.CREDIT,
                    direction_bias="Neutral",
                    confidence="Medium",
                    score=60.0,
                    legs=[
                        PositionLeg(
                            action="SELL", instrument="CE", strike=23000,
                            option_type="CE", lots=1, lot_size=65,
                            entry_ltp=100.0, current_ltp=80.0,
                        ),
                    ],
                ),
            ],
        )
        save_state(state, algo_name="test_pos")
        loaded = load_state(algo_name="test_pos")
        assert loaded is not None
        assert len(loaded.open_positions) == 1
        assert loaded.open_positions[0].legs[0].entry_ltp == 100.0


def _make_state_with_position() -> PaperTradingState:
    """Helper: create a state with one open position."""
    return PaperTradingState(
        initial_capital=5000000,
        open_positions=[
            PaperPosition(
                strategy="Short Straddle",
                strategy_type=StrategyType.CREDIT,
                direction_bias="Neutral",
                confidence="Medium",
                score=60.0,
                legs=[
                    PositionLeg(
                        action="SELL", instrument="CE", strike=23000,
                        option_type="CE", lots=1, lot_size=65,
                        entry_ltp=100.0, current_ltp=80.0,
                    ),
                ],
            ),
        ],
    )


def _make_state_with_trade_log() -> PaperTradingState:
    """Helper: create a state with one closed trade in the log."""
    from datetime import datetime, timezone
    return PaperTradingState(
        initial_capital=5000000,
        trade_log=[
            TradeRecord(
                id="test-001",
                strategy="Short Straddle",
                strategy_type="credit",
                direction_bias="Neutral",
                confidence="Medium",
                score=60.0,
                legs_summary=[{"action": "SELL", "strike": 23000}],
                lots=1,
                entry_time=datetime(2025, 1, 1, tzinfo=timezone.utc),
                exit_time=datetime(2025, 1, 1, tzinfo=timezone.utc),
                exit_reason="closed_profit_target",
                realized_pnl=5000.0,
                net_premium=100.0,
                stop_loss_amount=200.0,
                profit_target_amount=400.0,
            ),
        ],
        total_realized_pnl=5000.0,
    )


class TestStateSafety:
    """Tests for state persistence safety features."""

    def test_load_falls_back_to_backup(self, tmp_data_dir):
        """Corrupt primary file → load_state should recover from .bak."""
        state = _make_state_with_position()
        save_state(state, algo_name="test_fallback")

        # Save again with different data to create a .bak with the original
        state2 = _make_state_with_trade_log()
        save_state(state2, algo_name="test_fallback")

        # Corrupt primary file
        primary = tmp_data_dir / "paper_trading_state_test_fallback.json"
        primary.write_text("{invalid json!!!}")

        loaded = load_state(algo_name="test_fallback")
        assert loaded is not None
        # Should have loaded from .bak (which has the first state with position)
        assert len(loaded.open_positions) == 1

    def test_save_blocks_empty_overwrite(self, tmp_data_dir):
        """save_state() must refuse to overwrite data with empty state."""
        state_with_data = _make_state_with_position()
        save_state(state_with_data, algo_name="test_guard")

        # Try to save an empty state — should be blocked
        empty_state = PaperTradingState(initial_capital=5000000)
        save_state(empty_state, algo_name="test_guard")

        # Verify original data is preserved
        loaded = load_state(algo_name="test_guard")
        assert loaded is not None
        assert len(loaded.open_positions) == 1, "Empty state overwrote data!"

    def test_backup_only_rotates_on_change(self, tmp_data_dir):
        """Saving the same state twice should NOT rotate backups."""
        state = _make_state_with_position()
        save_state(state, algo_name="test_norotate")

        # Save again with same trade/position count to create .bak
        # (first save has no .bak to rotate)
        save_state(state, algo_name="test_norotate")

        bak = tmp_data_dir / "paper_trading_state_test_norotate.bak"
        # .bak should NOT exist because trade/position count didn't change
        assert not bak.exists(), "Backup rotated despite no change in trade/position count"

    def test_force_save_bypasses_guard(self, tmp_data_dir):
        """force_save_state() should overwrite even when existing has data."""
        state_with_data = _make_state_with_trade_log()
        save_state(state_with_data, algo_name="test_force")

        # Force-save an empty state
        empty_state = PaperTradingState(initial_capital=5000000)
        force_save_state(empty_state, algo_name="test_force")

        # Verify empty state was written
        loaded = load_state(algo_name="test_force")
        assert loaded is not None
        assert len(loaded.trade_log) == 0, "force_save_state didn't overwrite"

        # Verify old data still exists in .bak
        bak = tmp_data_dir / "paper_trading_state_test_force.bak"
        assert bak.exists()
        bak_state = PaperTradingState.model_validate_json(bak.read_text())
        assert len(bak_state.trade_log) == 1
