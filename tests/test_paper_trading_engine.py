"""Tests for core.paper_trading_engine — persistence, state safety."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from core.paper_trading_engine import save_state, load_state, _state_file
from core.paper_trading_models import PaperTradingState


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
        # Save again to trigger backup
        state2 = PaperTradingState(initial_capital=2000000)
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
