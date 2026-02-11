"""Tests for core.paper_trading_models — PnL calculations, state properties, migration."""

import json
from datetime import datetime, timezone, timedelta

import pytest

from core.paper_trading_models import (
    PaperPosition,
    PaperTradingState,
    PositionLeg,
    PositionStatus,
    StrategyType,
    TradeRecord,
    _now_ist,
)


# ── PositionLeg PnL ──────────────────────────────────────────────────────

class TestPositionLegPnl:
    def test_buy_leg_profit(self):
        leg = PositionLeg(
            action="BUY", instrument="NIFTY 23000 CE", strike=23000,
            option_type="CE", lots=1, lot_size=65,
            entry_ltp=100.0, current_ltp=120.0,
        )
        assert leg.unrealized_pnl == (120.0 - 100.0) * 1 * 65

    def test_buy_leg_loss(self):
        leg = PositionLeg(
            action="BUY", instrument="NIFTY 23000 CE", strike=23000,
            option_type="CE", lots=2, lot_size=65,
            entry_ltp=100.0, current_ltp=80.0,
        )
        assert leg.unrealized_pnl == (80.0 - 100.0) * 2 * 65

    def test_sell_leg_profit(self):
        leg = PositionLeg(
            action="SELL", instrument="NIFTY 23000 PE", strike=23000,
            option_type="PE", lots=1, lot_size=65,
            entry_ltp=100.0, current_ltp=70.0,
        )
        assert leg.unrealized_pnl == (100.0 - 70.0) * 1 * 65

    def test_sell_leg_loss(self):
        leg = PositionLeg(
            action="SELL", instrument="NIFTY 23000 PE", strike=23000,
            option_type="PE", lots=1, lot_size=65,
            entry_ltp=100.0, current_ltp=130.0,
        )
        assert leg.unrealized_pnl == (100.0 - 130.0) * 1 * 65

    def test_zero_movement(self):
        leg = PositionLeg(
            action="BUY", instrument="X", strike=23000,
            option_type="CE", lots=1, lot_size=65,
            entry_ltp=100.0, current_ltp=100.0,
        )
        assert leg.unrealized_pnl == 0.0


# ── PaperPosition ────────────────────────────────────────────────────────

class TestPaperPosition:
    def test_total_unrealized_pnl_multi_leg(self):
        pos = PaperPosition(
            strategy="Short Straddle",
            strategy_type=StrategyType.CREDIT,
            direction_bias="Neutral",
            confidence="Medium",
            score=60.0,
            legs=[
                PositionLeg(
                    action="SELL", instrument="CE", strike=23000, option_type="CE",
                    lots=1, lot_size=65, entry_ltp=100.0, current_ltp=80.0,
                ),
                PositionLeg(
                    action="SELL", instrument="PE", strike=23000, option_type="PE",
                    lots=1, lot_size=65, entry_ltp=90.0, current_ltp=70.0,
                ),
            ],
        )
        # Sell CE profit: (100-80)*65 = 1300
        # Sell PE profit: (90-70)*65 = 1300
        assert pos.total_unrealized_pnl == 2600.0


# ── PaperTradingState ────────────────────────────────────────────────────

class TestPaperTradingState:
    def test_net_realized_pnl(self):
        state = PaperTradingState(
            total_realized_pnl=10000.0,
            total_execution_costs=500.0,
        )
        assert state.net_realized_pnl == 9500.0

    def test_capital_remaining_no_positions(self):
        state = PaperTradingState(
            initial_capital=5000000,
            total_realized_pnl=10000.0,
            total_execution_costs=500.0,
        )
        # 5000000 + 10000 - 500 - 0 (no margin)
        assert state.capital_remaining == 5009500.0

    def test_capital_remaining_with_margin(self):
        state = PaperTradingState(
            initial_capital=5000000,
            open_positions=[
                PaperPosition(
                    strategy="X", strategy_type=StrategyType.CREDIT,
                    direction_bias="N", confidence="L", score=50,
                    legs=[], margin_required=150000.0,
                ),
            ],
        )
        assert state.capital_remaining == 5000000 - 150000

    def test_unrealized_pnl(self):
        state = PaperTradingState(
            open_positions=[
                PaperPosition(
                    strategy="X", strategy_type=StrategyType.CREDIT,
                    direction_bias="N", confidence="L", score=50,
                    legs=[
                        PositionLeg(
                            action="SELL", instrument="CE", strike=23000, option_type="CE",
                            lots=1, lot_size=65, entry_ltp=100.0, current_ltp=80.0,
                        ),
                    ],
                ),
            ],
        )
        assert state.unrealized_pnl == 1300.0

    def test_win_rate_empty(self):
        state = PaperTradingState()
        assert state.win_rate == 0.0

    def test_win_rate_all_wins(self):
        state = PaperTradingState(
            trade_log=[
                TradeRecord(
                    id="1", strategy="X", strategy_type=StrategyType.CREDIT,
                    direction_bias="N", confidence="L", score=50,
                    legs_summary=[], entry_time=_now_ist(), exit_time=_now_ist(),
                    exit_reason=PositionStatus.CLOSED_PROFIT_TARGET,
                    realized_pnl=1000.0, net_premium=100, stop_loss_amount=200,
                    profit_target_amount=100,
                ),
                TradeRecord(
                    id="2", strategy="X", strategy_type=StrategyType.CREDIT,
                    direction_bias="N", confidence="L", score=50,
                    legs_summary=[], entry_time=_now_ist(), exit_time=_now_ist(),
                    exit_reason=PositionStatus.CLOSED_PROFIT_TARGET,
                    realized_pnl=500.0, net_premium=100, stop_loss_amount=200,
                    profit_target_amount=100,
                ),
            ],
        )
        assert state.win_rate == 100.0

    def test_win_rate_mixed(self):
        def _trade(pnl):
            return TradeRecord(
                id=str(pnl), strategy="X", strategy_type=StrategyType.CREDIT,
                direction_bias="N", confidence="L", score=50,
                legs_summary=[], entry_time=_now_ist(), exit_time=_now_ist(),
                exit_reason=PositionStatus.CLOSED_PROFIT_TARGET,
                realized_pnl=pnl, net_premium=100, stop_loss_amount=200,
                profit_target_amount=100,
            )
        state = PaperTradingState(
            trade_log=[_trade(1000), _trade(-500), _trade(200), _trade(-100)],
        )
        assert state.win_rate == 50.0

    def test_sharpe_ratio_empty(self):
        state = PaperTradingState()
        assert state.sharpe_ratio == 0.0

    def test_sharpe_ratio_single_trade(self):
        state = PaperTradingState(
            trade_log=[
                TradeRecord(
                    id="1", strategy="X", strategy_type=StrategyType.CREDIT,
                    direction_bias="N", confidence="L", score=50,
                    legs_summary=[], entry_time=_now_ist(), exit_time=_now_ist(),
                    exit_reason=PositionStatus.CLOSED_PROFIT_TARGET,
                    realized_pnl=1000.0, net_pnl=1000.0,
                    net_premium=100, stop_loss_amount=200, profit_target_amount=100,
                ),
            ],
        )
        assert state.sharpe_ratio == 0.0  # need >=2 trades

    def test_migration_single_position(self):
        """Old format with current_position should migrate to open_positions."""
        old_data = {
            "initial_capital": 5000000,
            "current_position": {
                "strategy": "Short Straddle",
                "strategy_type": "credit",
                "direction_bias": "Neutral",
                "confidence": "Medium",
                "score": 60,
                "legs": [],
            },
            "total_realized_pnl": 0,
            "total_execution_costs": 0,
        }
        state = PaperTradingState.model_validate(old_data)
        assert len(state.open_positions) == 1

    def test_serialization_roundtrip(self):
        state = PaperTradingState(initial_capital=5000000)
        json_str = state.model_dump_json()
        restored = PaperTradingState.model_validate_json(json_str)
        assert restored.initial_capital == 5000000
        assert restored.net_realized_pnl == 0.0
