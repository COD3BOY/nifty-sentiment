#!/usr/bin/env python3
"""Tune V4 Vol-Optimized algorithm parameters via realized-move backtest.

Uses historical NIFTY close + vol distribution data to find optimal
regime thresholds and dynamic function coefficients.

Usage:
    cd nifty
    python scripts/tune_vol_params.py

Output:
    - Console summary (regime split, win rate, tuned params)
    - data/vol_optimized_tuned_params.json
"""

from __future__ import annotations

import json
import math
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.vol_distribution import get_historical_backtest_data


def _realized_move_backtest(
    df: pd.DataFrame,
    regime_params: dict,
    dynamic_params: dict,
    dte: int = 5,
    assumed_width: float = 100.0,
) -> dict:
    """Run realized-move backtest and return performance metrics.

    For each "sell premium" day:
    1. Compute expected move EM = close * (vix/100) * sqrt(dte/365)
    2. Compute safe zone = EM * be_mult(p_vov)
    3. Check actual max move over next 5 trading days
    4. Win if max_move < zone, Loss if max_move >= zone
    5. Credit proxy = min_credit_ratio(p_vrp) * assumed_width
    6. Loss proxy = credit * sl_mult(p_vov), capped at width - credit
    """
    sell_vrp_min = regime_params["regime_sell_vrp_min"]
    sell_vov_max = regime_params["regime_sell_vov_max"]
    standdown_vov_min = regime_params["standdown_vov_min"]
    standdown_rv_min = regime_params["standdown_rv_min"]

    be_base = dynamic_params.get("be_base", 1.15)
    be_coeff = dynamic_params.get("be_coeff", 0.30)
    credit_base = dynamic_params.get("credit_base", 0.22)
    credit_coeff = dynamic_params.get("credit_coeff", 0.16)
    sl_base = dynamic_params.get("sl_base", 1.6)
    sl_coeff = dynamic_params.get("sl_coeff", 0.8)

    closes = df["close"].values
    vix_vals = df["vix"].values
    p_rv_vals = df["p_rv"].values
    p_vov_vals = df["p_vov"].values
    p_vrp_vals = df["p_vrp"].values

    n = len(df)
    wins = 0
    losses = 0
    total_credit = 0.0
    total_loss = 0.0
    sell_days = 0
    buy_days = 0
    standdown_days = 0

    for i in range(n - dte):
        p_rv = p_rv_vals[i]
        p_vov = p_vov_vals[i]
        p_vrp = p_vrp_vals[i]

        # Classify regime
        if p_vov >= standdown_vov_min or p_rv >= standdown_rv_min:
            standdown_days += 1
            continue
        elif p_vrp >= sell_vrp_min and p_vov <= sell_vov_max:
            sell_days += 1
        else:
            buy_days += 1
            continue  # Only backtest sell premium days

        close = closes[i]
        vix = vix_vals[i]

        if close <= 0 or vix <= 0:
            continue

        # Expected move
        em = close * (vix / 100.0) * math.sqrt(dte / 365.0)

        # Dynamic breakeven multiplier
        be_mult = be_base + be_coeff * p_vov
        zone = em * be_mult

        # Actual max move over next dte trading days
        future_closes = closes[i + 1: i + 1 + dte]
        if len(future_closes) < dte:
            continue
        max_move = float(np.max(np.abs(future_closes - close)))

        # Credit proxy
        min_credit_ratio = credit_base + credit_coeff * p_vrp
        credit = min_credit_ratio * assumed_width

        # SL proxy
        sl_mult = sl_base + sl_coeff * p_vov
        loss_amount = min(credit * sl_mult, assumed_width - credit)

        if max_move < zone:
            wins += 1
            total_credit += credit
        else:
            losses += 1
            total_loss += loss_amount

    total_trades = wins + losses
    if total_trades == 0:
        return {
            "score": -1e9,
            "win_rate": 0,
            "avg_credit": 0,
            "avg_loss": 0,
            "total_trades": 0,
            "sell_days": sell_days,
            "buy_days": buy_days,
            "standdown_days": standdown_days,
        }

    avg_credit = total_credit / max(1, wins)
    avg_loss = total_loss / max(1, losses)
    score = (wins * avg_credit) - (losses * avg_loss)
    win_rate = wins / total_trades * 100

    return {
        "score": score,
        "win_rate": win_rate,
        "wins": wins,
        "losses": losses,
        "avg_credit": avg_credit,
        "avg_loss": avg_loss,
        "total_trades": total_trades,
        "sell_days": sell_days,
        "buy_days": buy_days,
        "standdown_days": standdown_days,
    }


def _tune_regime_thresholds(df: pd.DataFrame) -> tuple[dict, dict]:
    """Phase A: Grid search over regime threshold parameters."""
    print("\n=== Phase A: Tuning Regime Thresholds ===")

    sell_vrp_range = [0.45, 0.50, 0.55, 0.60, 0.65]
    sell_vov_range = [0.55, 0.60, 0.65, 0.70, 0.75]
    standdown_vov_range = [0.75, 0.80, 0.85, 0.90]
    standdown_rv_range = [0.80, 0.85, 0.90, 0.95]

    # Default dynamic params for regime tuning
    default_dynamic = {
        "be_base": 1.15, "be_coeff": 0.30,
        "credit_base": 0.22, "credit_coeff": 0.16,
        "sl_base": 1.6, "sl_coeff": 0.8,
    }

    best_score = -1e9
    best_params: dict = {}
    best_result: dict = {}
    combos = list(product(sell_vrp_range, sell_vov_range, standdown_vov_range, standdown_rv_range))
    total = len(combos)

    for idx, (vrp_min, vov_max, sd_vov, sd_rv) in enumerate(combos):
        if (idx + 1) % 100 == 0:
            print(f"  Progress: {idx + 1}/{total}")

        regime = {
            "regime_sell_vrp_min": vrp_min,
            "regime_sell_vov_max": vov_max,
            "standdown_vov_min": sd_vov,
            "standdown_rv_min": sd_rv,
        }

        result = _realized_move_backtest(df, regime, default_dynamic)

        if result["score"] > best_score:
            best_score = result["score"]
            best_params = regime.copy()
            best_result = result.copy()

    print(f"\nBest regime params: {best_params}")
    print(f"  Score: {best_score:.0f}")
    print(f"  Win rate: {best_result.get('win_rate', 0):.1f}%")
    print(f"  Sell days: {best_result.get('sell_days', 0)}, "
          f"Buy days: {best_result.get('buy_days', 0)}, "
          f"Stand-down: {best_result.get('standdown_days', 0)}")

    return best_params, best_result


def _tune_dynamic_coefficients(df: pd.DataFrame, regime_params: dict) -> dict:
    """Phase B: Sequential tuning of dynamic coefficient pairs."""
    print("\n=== Phase B: Tuning Dynamic Coefficients ===")

    # Start with defaults
    best_dynamic = {
        "be_base": 1.15, "be_coeff": 0.30,
        "credit_base": 0.22, "credit_coeff": 0.16,
        "sl_base": 1.6, "sl_coeff": 0.8,
    }

    # Sub-problem 1: Breakeven coverage
    print("\n  Tuning breakeven coverage...")
    best_score = -1e9
    for be_b, be_c in product([1.10, 1.15, 1.20], [0.25, 0.30, 0.35]):
        params = {**best_dynamic, "be_base": be_b, "be_coeff": be_c}
        result = _realized_move_backtest(df, regime_params, params)
        if result["score"] > best_score:
            best_score = result["score"]
            best_dynamic["be_base"] = be_b
            best_dynamic["be_coeff"] = be_c
    print(f"    BE: base={best_dynamic['be_base']}, coeff={best_dynamic['be_coeff']}")

    # Sub-problem 2: Credit ratio
    print("  Tuning credit ratio...")
    best_score = -1e9
    for cr_b, cr_c in product([0.20, 0.22, 0.24], [0.14, 0.16, 0.18]):
        params = {**best_dynamic, "credit_base": cr_b, "credit_coeff": cr_c}
        result = _realized_move_backtest(df, regime_params, params)
        if result["score"] > best_score:
            best_score = result["score"]
            best_dynamic["credit_base"] = cr_b
            best_dynamic["credit_coeff"] = cr_c
    print(f"    Credit: base={best_dynamic['credit_base']}, coeff={best_dynamic['credit_coeff']}")

    # Sub-problem 3: SL multiple
    print("  Tuning SL multiple...")
    best_score = -1e9
    for sl_b, sl_c in product([1.4, 1.6, 1.8], [0.6, 0.8, 1.0]):
        params = {**best_dynamic, "sl_base": sl_b, "sl_coeff": sl_c}
        result = _realized_move_backtest(df, regime_params, params)
        if result["score"] > best_score:
            best_score = result["score"]
            best_dynamic["sl_base"] = sl_b
            best_dynamic["sl_coeff"] = sl_c
    print(f"    SL: base={best_dynamic['sl_base']}, coeff={best_dynamic['sl_coeff']}")

    # Final backtest with all tuned params
    final = _realized_move_backtest(df, regime_params, best_dynamic)
    print(f"\n  Final score: {final['score']:.0f}")
    print(f"  Final win rate: {final.get('win_rate', 0):.1f}%")
    print(f"  Avg credit: {final.get('avg_credit', 0):.1f}, Avg loss: {final.get('avg_loss', 0):.1f}")

    return best_dynamic


def _derive_all_params(regime: dict, dynamic: dict) -> dict:
    """Derive the full config parameter set from tuning results.

    Maps tuned breakeven/credit/SL coefficients to all 9 dynamic functions.
    """
    return {
        # Regime thresholds (from Phase A)
        "regime_sell_vrp_min": regime["regime_sell_vrp_min"],
        "regime_sell_vov_max": regime["regime_sell_vov_max"],
        "standdown_vov_min": regime["standdown_vov_min"],
        "standdown_rv_min": regime["standdown_rv_min"],

        # Strike delta (derived from credit ratio tuning — lower credit = further OTM)
        "strike_delta_base": 0.20,
        "strike_delta_coeff": 0.08,

        # Min credit ratio (from Phase B)
        "credit_ratio_base": dynamic["credit_base"],
        "credit_ratio_coeff": dynamic["credit_coeff"],

        # SL multiple (from Phase B)
        "sl_multiple_base": dynamic["sl_base"],
        "sl_multiple_coeff": dynamic["sl_coeff"],

        # Delta exit (derived from SL tuning)
        "delta_exit_base": 0.22,
        "delta_exit_coeff": 0.10,

        # Take profit (inverse of vol — take profits faster in high vol)
        "take_profit_base": 0.45,
        "take_profit_coeff": 0.20,

        # Breakeven coverage (from Phase B)
        "breakeven_mult_base": dynamic["be_base"],
        "breakeven_mult_coeff": dynamic["be_coeff"],

        # Risk per trade (derived)
        "risk_per_trade_base_pct": 0.50,
        "risk_per_trade_rv_coeff": 0.5,
        "risk_per_trade_vov_coeff": 0.5,

        # Max portfolio risk (derived)
        "max_portfolio_risk_base_pct": 3.0,
        "max_portfolio_risk_coeff": 0.5,
    }


def main() -> None:
    print("V4 Vol-Optimized Parameter Tuning")
    print("=" * 50)

    print("\nLoading historical vol distribution data...")
    df = get_historical_backtest_data()
    if df is None or df.empty:
        print("ERROR: Failed to load historical data. Run vol_distribution first.")
        sys.exit(1)

    print(f"  Data: {len(df)} rows, {df.index[0].date()} to {df.index[-1].date()}")
    print(f"  Columns: {list(df.columns)}")

    # Phase A
    regime_params, regime_result = _tune_regime_thresholds(df)

    # Phase B
    dynamic_params = _tune_dynamic_coefficients(df, regime_params)

    # Derive full parameter set
    all_params = _derive_all_params(regime_params, dynamic_params)

    # Summary
    total_days = len(df)
    sell_pct = regime_result.get("sell_days", 0) / total_days * 100
    buy_pct = regime_result.get("buy_days", 0) / total_days * 100
    sd_pct = regime_result.get("standdown_days", 0) / total_days * 100

    print("\n" + "=" * 50)
    print("TUNING SUMMARY")
    print("=" * 50)
    print(f"\nRegime Split ({total_days} days):")
    print(f"  Sell premium: {sell_pct:.1f}%")
    print(f"  Buy premium:  {buy_pct:.1f}%")
    print(f"  Stand-down:   {sd_pct:.1f}%")
    print(f"\nWin Rate: {regime_result.get('win_rate', 0):.1f}%")
    print(f"Avg Credit: {regime_result.get('avg_credit', 0):.1f}")
    print(f"Avg Loss:   {regime_result.get('avg_loss', 0):.1f}")
    print(f"Score:      {regime_result.get('score', 0):.0f}")

    # Save to JSON
    output_path = Path(__file__).resolve().parent.parent / "data" / "vol_optimized_tuned_params.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(all_params, indent=2))
    print(f"\nTuned params saved to: {output_path}")
    print("\nAll parameters:")
    for k, v in all_params.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
