"""V2 Jarvis M algorithm — Jarvis + full MarketContext wiring.

Reuses all Jarvis helper functions (no code duplication). Adds:
  - Context scoring via V1's generate_trade_suggestions(context=, observation=)
  - IV expansion exit for credit positions
  - Context-driven exits (neutral trend, directional reversal, vol regime shift)
  - Vol stand-down min score bump
  - Re-entry tracking (block after SL/IV/context exit; cooldown after PT)
  - Score gates (min_score_to_trade, debit_min_score_to_trade)
  - Confidence loosening for credit with 3+ positive CTX signals
  - Max trades per day gate
  - Entry cutoff time
"""

from __future__ import annotations

import logging
import time as _time
from collections import Counter
from datetime import time

from typing import TYPE_CHECKING

from algorithms import register_algorithm
from algorithms.base import TradingAlgorithm
from algorithms.jarvis import (
    _CREDIT_STRATEGIES,
    _DEBIT_STRATEGIES,
    _SPREAD_STRATEGIES,
    _NAKED_STRATEGIES,
    _get_vix_from_analytics,
    _check_credit_entry_filters,
    _check_debit_entry_filters,
    _validate_credit_trade,
    _validate_debit_trade,
    _check_global_gates,
    _compute_jarvis_lots,
    _apply_realistic_fills,
    _enrich_suggestion,
)
from core.config import load_config
from core.market_hours import is_market_open
from core.options_models import (
    OptionChainData,
    OptionsAnalytics,
    StrategyName,
    TechnicalIndicators,
    TradeSuggestion,
)
from core.paper_trading_engine import (
    check_exit_conditions,
    classify_strategy,
    close_position,
    open_position,
    update_position_ltp,
)
from core.paper_trading_models import (
    PaperPosition,
    PaperTradingState,
    PositionStatus,
    StrategyType,
    TradeRecord,
    _now_ist,
)
from core.options_utils import (
    compute_spread_width as _compute_spread_width,
    is_observation_period as _is_observation_period,
)

if TYPE_CHECKING:
    from core.context_models import MarketContext
    from core.observation import ObservationSnapshot

logger = logging.getLogger(__name__)


@register_algorithm
class JarvisMAlgorithm(TradingAlgorithm):
    """Jarvis + MarketContext: context scoring, context exits, vol stand-down."""

    name = "jarvis_m"
    display_name = "V2 Jarvis M"
    description = "Jarvis + MarketContext: context scoring, context exits, vol stand-down"

    # ---------------------------------------------------------------
    # generate_suggestions — Jarvis filters + context/observation forwarding
    # ---------------------------------------------------------------

    def generate_suggestions(
        self,
        chain: OptionChainData,
        technicals: TechnicalIndicators,
        analytics: OptionsAnalytics,
        observation: ObservationSnapshot | None = None,
        context: MarketContext | None = None,
    ) -> list[TradeSuggestion]:
        if not is_market_open():
            return []

        # V1 suggestions WITH context + observation for scoring
        from core.trade_strategies import generate_trade_suggestions
        raw_suggestions = generate_trade_suggestions(
            analytics, technicals, chain,
            context=context, observation=observation,
        )

        cfg = self.config
        accepted: list[TradeSuggestion] = []
        rejected: list[TradeSuggestion] = []

        for suggestion in raw_suggestions:
            is_credit = suggestion.strategy in _CREDIT_STRATEGIES
            rejection: str | None = None

            # Entry logic filters
            if is_credit:
                rejection = _check_credit_entry_filters(analytics, technicals, cfg)
            else:
                rejection = _check_debit_entry_filters(analytics, technicals, cfg)

            # Strategy-specific validation
            if rejection is None:
                if is_credit:
                    rejection = _validate_credit_trade(suggestion, chain, analytics, cfg)
                else:
                    rejection = _validate_debit_trade(suggestion, chain, analytics, technicals, cfg)

            # Realistic execution fills
            if rejection is None:
                new_legs, fill_reject = _apply_realistic_fills(suggestion, chain, cfg)
                if fill_reject:
                    rejection = fill_reject
                else:
                    suggestion = suggestion.model_copy(update={"legs": new_legs})

                    # Re-check RR after realistic fills
                    if is_credit:
                        sell_prem = sum(l.ltp for l in new_legs if l.action == "SELL")
                        buy_prem = sum(l.ltp for l in new_legs if l.action == "BUY")
                        net_credit = sell_prem - buy_prem
                        if net_credit <= 0:
                            rejection = "no net credit after realistic fills"
                        elif suggestion.strategy in _SPREAD_STRATEGIES:
                            width = _compute_spread_width(new_legs)
                            if width > 0:
                                rr = (width - net_credit) / net_credit
                                max_rr = cfg.get("credit_max_risk_reward", 2.5)
                                if rr > max_rr:
                                    rejection = f"RR {rr:.1f}:1 > {max_rr}:1 after realistic fills"

            if rejection:
                rejected.append(suggestion.model_copy(update={
                    "rejection_reason": rejection,
                    "score": 0.0,
                }))
                logger.info("JM REJECTED: %s — %s", suggestion.strategy.value, rejection)
            else:
                enriched = _enrich_suggestion(suggestion, chain, analytics, cfg)
                accepted.append(enriched)
                logger.info(
                    "JM ACCEPTED: %s | RR=%.2f | POP=%.1f%% | liq=%.0f | score=%.1f",
                    enriched.strategy.value, enriched.risk_reward_ratio,
                    enriched.pop, enriched.liquidity_score, enriched.score,
                )

        return sorted(accepted, key=lambda s: s.score, reverse=True) + rejected

    # ---------------------------------------------------------------
    # evaluate_and_manage — Jarvis logic + context features
    # ---------------------------------------------------------------

    def evaluate_and_manage(
        self,
        state: PaperTradingState,
        suggestions: list[TradeSuggestion] | None,
        chain: OptionChainData | None,
        technicals: TechnicalIndicators | None = None,
        analytics: OptionsAnalytics | None = None,
        lot_size: int | None = None,
        refresh_ts: float = 0.0,
        observation: ObservationSnapshot | None = None,
        context: MarketContext | None = None,
    ) -> PaperTradingState:
        if not is_market_open():
            return state

        if lot_size is None:
            lot_size = self.config.get("lot_size", 65)
        cfg = self.config

        notes: list[str] = []

        # Context summary
        if context is not None:
            vol = context.vol
            notes.append(f"Vol regime: {vol.regime} | RV trend: {vol.rv_trend}")
            notes.append(f"Context bias: {context.context_bias} | Trend: {context.multi_day_trend}")
            if context.prior_day is not None:
                notes.append(
                    f"Prior day: {context.prior_day.candle_type} | range {context.prior_day.range_pct:.1f}%"
                )
            notes.append(
                f"Session: {context.session.session_trend} | range {context.session.session_range_pct:.1f}%"
            )

        # Observation context
        if observation:
            notes.append(f"Observation: {observation.bias} bias ({observation.bars_collected} bars)")
            if observation.gap.direction != "flat":
                notes.append(f"Gap: {observation.gap.direction} {observation.gap.gap_pct:+.2f}%")
            notes.append(f"Trend: {observation.initial_trend.direction} ({observation.initial_trend.strength})")

        vix = _get_vix_from_analytics(analytics)
        if vix > 0:
            notes.append(f"VIX (ATM IV proxy): {vix:.1f}")
            vix_threshold = cfg.get("vix_spread_only_threshold", 28.0)
            if vix > vix_threshold:
                notes.append(f"Naked strategies blocked: VIX {vix:.1f} > {vix_threshold}")

        # =====================================================================
        # Phase 1: Manage existing positions
        # =====================================================================
        still_open: list[PaperPosition] = []
        new_records: list[TradeRecord] = []
        added_realized = 0.0
        added_costs = 0.0
        new_pending_critiques: list[str] = []

        for position in state.open_positions:
            if position.status != PositionStatus.OPEN:
                continue

            if chain:
                position = update_position_ltp(position, chain)

            pnl = position.total_unrealized_pnl
            position = position.model_copy(update={
                "peak_pnl": max(position.peak_pnl, pnl),
                "trough_pnl": min(position.trough_pnl, pnl),
            })

            # V2 credit SL: exit at 2.2x credit for spreads
            exit_reason = None
            strategy_type = classify_strategy(position.strategy)

            if strategy_type == StrategyType.CREDIT:
                sl_mult = cfg.get("credit_sl_multiplier", 2.2)
                if position.net_premium > 0 and pnl <= -(abs(position.net_premium) * sl_mult):
                    exit_reason = PositionStatus.CLOSED_STOP_LOSS
            elif strategy_type == StrategyType.DEBIT:
                sl_pct = cfg.get("debit_sl_premium_loss_pct", 50.0)
                if position.net_premium < 0 and pnl <= -(abs(position.net_premium) * sl_pct / 100):
                    exit_reason = PositionStatus.CLOSED_STOP_LOSS

                max_sessions = cfg.get("debit_max_sessions", 3)
                if position.sessions_held >= max_sessions:
                    exit_reason = PositionStatus.CLOSED_TIME_LIMIT

            # Standard exit conditions (PT, EOD)
            if exit_reason is None:
                exit_reason = check_exit_conditions(position)

            # IV expansion exit for credit positions
            if (
                exit_reason is None
                and analytics
                and position.entry_atm_iv > 0
                and position.strategy_type == StrategyType.CREDIT.value
                and analytics.atm_iv > 0
            ):
                iv_threshold = cfg.get("iv_expansion_exit_pts", 3.0)
                iv_expansion = analytics.atm_iv - position.entry_atm_iv
                if iv_expansion >= iv_threshold:
                    exit_reason = PositionStatus.CLOSED_IV_EXPANSION
                    logger.info(
                        "JM IV expansion exit: %s | IV %.1f → %.1f (+%.1f pts > %.1f)",
                        position.strategy, position.entry_atm_iv,
                        analytics.atm_iv, iv_expansion, iv_threshold,
                    )

            # Context-driven exits
            if exit_reason is None and context is not None and cfg.get("context_exit_enabled", True):
                ctx_min_hold = cfg.get("context_exit_min_hold_minutes", 15)
                hold_minutes = (_now_ist() - position.entry_time).total_seconds() / 60
                if hold_minutes >= ctx_min_hold:
                    # Neutral credit exit on strong trend emergence
                    if (
                        position.strategy in ("Short Straddle", "Short Strangle", "Iron Condor")
                        and context.session.session_trend in ("trending_up", "trending_down")
                        and context.session.session_range_pct > cfg.get("context_exit_range_pct", 1.2)
                    ):
                        exit_reason = PositionStatus.CLOSED_CONTEXT_EXIT
                        logger.info(
                            "JM context exit (neutral trend): %s | session %s range %.1f%%",
                            position.strategy, context.session.session_trend,
                            context.session.session_range_pct,
                        )

                    # Directional credit exit on trend reversal
                    if exit_reason is None and position.strategy == "Bull Put Spread":
                        if (
                            context.multi_day_trend == "bearish"
                            and context.session.ema_alignment == "bearish"
                        ):
                            exit_reason = PositionStatus.CLOSED_CONTEXT_EXIT
                            logger.info("JM context exit (BPS reversal): trend bearish + EMA bearish")
                    if exit_reason is None and position.strategy == "Bear Call Spread":
                        if (
                            context.multi_day_trend == "bullish"
                            and context.session.ema_alignment == "bullish"
                        ):
                            exit_reason = PositionStatus.CLOSED_CONTEXT_EXIT
                            logger.info("JM context exit (BCS reversal): trend bullish + EMA bullish")

                    # Vol regime shift exit
                    if (
                        exit_reason is None
                        and position.entry_vol_regime == "sell_premium"
                        and position.strategy_type == StrategyType.CREDIT.value
                        and context.vol.regime == "stand_down"
                    ):
                        exit_reason = PositionStatus.CLOSED_CONTEXT_EXIT
                        logger.info(
                            "JM context exit (regime shift): %s entered sell_premium, now stand_down",
                            position.strategy,
                        )

            if exit_reason:
                _closed_pos, record = close_position(
                    position, exit_reason,
                    technicals=technicals, analytics=analytics, chain=chain,
                )
                logger.info(
                    "JM CLOSED: %s | reason=%s | pnl=%.2f",
                    record.strategy,
                    exit_reason.value if hasattr(exit_reason, 'value') else exit_reason,
                    record.net_pnl,
                )
                new_records.append(record)
                added_realized += record.realized_pnl
                added_costs += record.execution_cost
                if record.entry_context:
                    new_pending_critiques.append(record.id)
            else:
                still_open.append(position)

        # Portfolio defense — consecutive losses
        consecutive_losses = state.consecutive_losses
        trading_halted = state.trading_halted

        for record in new_records:
            if record.net_pnl < 0:
                consecutive_losses += 1
            else:
                consecutive_losses = 0

        halt_after = cfg.get("consecutive_loss_halt", 3)
        if consecutive_losses >= halt_after:
            trading_halted = True
            logger.warning("JM HALTED: %d consecutive losses >= %d", consecutive_losses, halt_after)

        # Track peak capital
        current_capital = (
            state.initial_capital + state.total_realized_pnl + added_realized
            - (state.total_execution_costs + added_costs)
        )
        peak_capital = max(state.peak_capital, current_capital)

        # Build intermediate state
        state = state.model_copy(update={
            "open_positions": still_open,
            "trade_log": state.trade_log + new_records,
            "total_realized_pnl": state.total_realized_pnl + added_realized,
            "total_execution_costs": state.total_execution_costs + added_costs,
            "pending_critiques": state.pending_critiques + new_pending_critiques,
            "consecutive_losses": consecutive_losses,
            "trading_halted": trading_halted,
            "peak_capital": peak_capital,
        })

        # Initialize session start capital
        if state.session_start_capital == 0.0:
            state = state.model_copy(update={
                "session_start_capital": state.initial_capital + state.net_realized_pnl,
            })

        # Daily loss > 2% -> full shutdown
        daily_loss_pct = cfg.get("daily_loss_shutdown_pct", 2.0)
        if state.session_start_capital > 0:
            session_pnl = (state.initial_capital + state.net_realized_pnl) - state.session_start_capital
            if session_pnl <= -(state.session_start_capital * daily_loss_pct / 100):
                logger.warning("JM SHUTDOWN: daily loss %.0f > %.1f%%", abs(session_pnl), daily_loss_pct)
                notes.append(f"Blocked: Daily loss shutdown (loss {abs(session_pnl):.0f} > {daily_loss_pct}%)")
                return state.model_copy(update={"trade_status_notes": notes})

        # =====================================================================
        # Phase 2: Open new positions
        # =====================================================================

        # Observation period guard
        entry_start = cfg.get("entry_start_time",
                              load_config().get("paper_trading", {}).get("entry_start_time", "10:00"))
        if _is_observation_period(entry_start):
            notes.append(f"Observation period active — collecting data before {entry_start}")
            return state.model_copy(update={"trade_status_notes": notes})

        # Entry end time — no new trades after this time
        entry_end = cfg.get("entry_end_time", "14:45")
        eh, em = (int(x) for x in entry_end.split(":"))
        now = _now_ist()
        if now.time() >= time(eh, em):
            notes.append(f"Entry cutoff: {now.strftime('%H:%M')} >= {entry_end}")
            return state.model_copy(update={"trade_status_notes": notes})

        if not state.is_auto_trading:
            notes.append("Auto-trading is OFF")
            return state.model_copy(update={"trade_status_notes": notes})
        if not suggestions:
            notes.append("No trade suggestions available")
            return state.model_copy(update={"trade_status_notes": notes})
        if not chain:
            notes.append("No option chain data available")
            return state.model_copy(update={"trade_status_notes": notes})

        if state.trading_halted:
            logger.info("JM: Trading halted — skipping new positions")
            notes.append(f"Blocked: Trading halted ({state.consecutive_losses} consecutive losses)")
            return state.model_copy(update={"trade_status_notes": notes})

        # Data staleness guard
        if technicals and technicals.data_staleness_minutes > 20:
            notes.append(f"Blocked: Data stale ({technicals.data_staleness_minutes:.0f} min > 20 min limit)")
            return state.model_copy(update={"trade_status_notes": notes})

        # Max trades per day gate
        max_trades_per_day = cfg.get("max_trades_per_day", 4)
        today = _now_ist().date()
        today_trade_count = sum(1 for t in state.trade_log if t.entry_time.date() == today)
        today_trade_count += len([p for p in state.open_positions if p.status == PositionStatus.OPEN])
        if today_trade_count >= max_trades_per_day:
            notes.append(f"Max trades/day ({max_trades_per_day}) reached")
            return state.model_copy(update={"trade_status_notes": notes})

        # Score gates
        min_score = cfg.get("min_score_to_trade", 50)
        debit_min_score = cfg.get("debit_min_score_to_trade", 70)

        # Vol stand-down: raise min score when regime is stand_down
        if context is not None and context.vol.regime == "stand_down":
            standdown_bump = cfg.get("standdown_min_score_bump", 10)
            min_score += standdown_bump
            debit_min_score += standdown_bump
            notes.append(f"Stand-down bump: min_score +{standdown_bump} (credit={min_score}, debit={debit_min_score})")

        # Debit unlock in buy_premium regime
        if context is not None and context.vol.regime == "buy_premium":
            debit_unlock_threshold = cfg.get("debit_context_unlock_threshold", 55)
            if debit_min_score > debit_unlock_threshold:
                trend_matches = context.multi_day_trend in ("bullish", "bearish")
                session_matches = context.session.ema_alignment in ("bullish", "bearish")
                if trend_matches and session_matches:
                    old_debit = debit_min_score
                    debit_min_score = debit_unlock_threshold
                    notes.append(f"Debit unlock: {old_debit} -> {debit_unlock_threshold} (buy_premium + trend + EMA)")

        # Cooldown
        cooldown = cfg.get("trade_cooldown_seconds", 60)
        now_ts = refresh_ts if refresh_ts > 0 else _time.time()
        if state.last_trade_opened_ts > 0 and (now_ts - state.last_trade_opened_ts) < cooldown:
            notes.append(f"Trade cooldown active ({cooldown}s between trades)")
            return state.model_copy(update={"trade_status_notes": notes})

        # Re-entry tracking: block same strategy after SL/IV/context exit; allow after PT with cooldown
        held_strategies = {p.strategy for p in state.open_positions if p.status == PositionStatus.OPEN}
        reentry_cooldown = cfg.get("reentry_after_pt_cooldown_minutes", 30)
        for record in state.trade_log:
            if record.entry_time.date() == today:
                if record.exit_reason in (
                    PositionStatus.CLOSED_STOP_LOSS.value,
                    PositionStatus.CLOSED_IV_EXPANSION.value,
                    PositionStatus.CLOSED_CONTEXT_EXIT.value,
                ):
                    held_strategies.add(record.strategy)
                elif record.exit_reason == PositionStatus.CLOSED_PROFIT_TARGET.value:
                    if record.exit_time:
                        minutes_since = (_now_ist() - record.exit_time).total_seconds() / 60
                        if minutes_since < reentry_cooldown:
                            held_strategies.add(record.strategy)
                else:
                    held_strategies.add(record.strategy)

        opened_trade = False
        for suggestion in suggestions:
            if suggestion.rejection_reason:
                continue
            if suggestion.score <= 0:
                continue

            # Global gates
            gate_reason = _check_global_gates(state, suggestion, chain, analytics, cfg)
            if gate_reason:
                logger.info("JM GATE BLOCKED: %s — %s", suggestion.strategy.value, gate_reason)
                notes.append(f"Blocked {suggestion.strategy.value}: {gate_reason}")
                continue

            # Re-entry check
            if suggestion.strategy.value in held_strategies:
                notes.append(f"Blocked {suggestion.strategy.value}: already held or re-entry blocked")
                continue

            if state.capital_remaining <= 0:
                notes.append("Blocked: no capital remaining")
                break

            # Score gate
            is_debit = classify_strategy(suggestion.strategy) == StrategyType.DEBIT
            effective_min = debit_min_score if is_debit else min_score
            if suggestion.score < effective_min:
                logger.debug("JM: Skipping %s — score %.1f < min %d", suggestion.strategy.value, suggestion.score, effective_min)
                continue

            # Confidence gate — High for credit, Medium+ for debit
            # Context-confirmed credit accepts Medium (3+ positive CTX, 0 negative)
            min_confidence = {"High"} if not is_debit else {"High", "Medium"}
            if not is_debit:
                ctx_reasons = [r for r in suggestion.reasoning if r.startswith("[CTX]")]
                positive_ctx = sum(1 for r in ctx_reasons if "+" in r)
                negative_ctx = sum(1 for r in ctx_reasons if any(w in r for w in ["-", "penalty", "conflict", "hurts"]))
                context_confirmed = positive_ctx >= 3 and negative_ctx == 0
                if context_confirmed:
                    min_confidence = {"High", "Medium"}
            if suggestion.confidence not in min_confidence:
                logger.info(
                    "JM: Skipping %s — confidence %s (require %s)",
                    suggestion.strategy.value, suggestion.confidence,
                    "High" if not is_debit else "Medium+",
                )
                continue

            # Position sizing
            account_value = state.initial_capital + state.net_realized_pnl
            jarvis_lots, size_reject = _compute_jarvis_lots(
                suggestion, account_value, cfg, state.consecutive_losses,
            )
            if size_reject:
                logger.info("JM SIZING REJECTED: %s — %s", suggestion.strategy.value, size_reject)
                notes.append(f"Blocked {suggestion.strategy.value}: sizing — {size_reject}")
                continue

            # Open position
            position = open_position(
                suggestion, lot_size, state.capital_remaining,
                expiry=chain.expiry,
                technicals=technicals, analytics=analytics, chain=chain,
            )
            if jarvis_lots > 0 and jarvis_lots != position.lots:
                position = position.model_copy(update={"lots": jarvis_lots})

            # Set entry metadata
            updates: dict = {
                "entry_date": _now_ist().strftime("%Y-%m-%d"),
            }
            if context is not None:
                updates["entry_vol_regime"] = context.vol.regime
            position = position.model_copy(update=updates)

            logger.info(
                "JM OPENED: %s | lots=%d | margin=%.0f | premium=%.2f | score=%.1f",
                position.strategy, position.lots, position.margin_required,
                position.net_premium, position.score,
            )

            state = state.model_copy(update={
                "open_positions": state.open_positions + [position],
                "last_trade_opened_ts": refresh_ts if refresh_ts > 0 else _time.time(),
            })
            notes.append(f"Opened: {position.strategy} ({position.lots} lots)")
            opened_trade = True
            break  # max 1 trade per cycle

        if not opened_trade and not any("Opened:" in n for n in notes):
            notes.append("All suggestions blocked by gates/sizing")

        return state.model_copy(update={"trade_status_notes": notes})
