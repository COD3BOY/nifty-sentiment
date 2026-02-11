"""Central data validation for the NIFTY trading system.

Every function returns a ``ValidationResult``.  Callers decide whether to
raise, warn, or silently degrade.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── IV bounds (match iv_calculator.py constants) ─────────────────────────
IV_FLOOR = 0.5
IV_CAP = 200.0


@dataclass
class ValidationResult:
    """Outcome of a validation check."""

    valid: bool
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.valid

    def merge(self, other: ValidationResult) -> ValidationResult:
        return ValidationResult(
            valid=self.valid and other.valid,
            warnings=self.warnings + other.warnings,
            errors=self.errors + other.errors,
        )


# ── Helpers ──────────────────────────────────────────────────────────────

def _is_finite(v: float) -> bool:
    """True if v is a real finite number (not NaN, not ±inf)."""
    if isinstance(v, (int, float)):
        return math.isfinite(v)
    # numpy scalar
    try:
        return bool(np.isfinite(v))
    except (TypeError, ValueError):
        return False


# ── Validators ───────────────────────────────────────────────────────────

def validate_option_chain(chain) -> ValidationResult:
    """Validate an OptionChainData instance before it reaches algorithms.

    Checks:
    - underlying > 0
    - ≥ 10 strikes present
    - ≥ 3 strikes with OI > 0
    - no negative premiums
    """
    errors: list[str] = []
    warnings: list[str] = []

    if chain is None:
        return ValidationResult(valid=False, errors=["chain is None"])

    if not _is_finite(chain.underlying_value) or chain.underlying_value <= 0:
        errors.append(f"underlying_value invalid: {chain.underlying_value}")

    n_strikes = len(chain.strikes)
    if n_strikes < 10:
        errors.append(f"only {n_strikes} strikes (need ≥10)")

    n_oi = sum(
        1
        for s in chain.strikes
        if (s.ce_oi > 0 or s.pe_oi > 0)
    )
    if n_oi < 3:
        errors.append(f"only {n_oi} strikes with OI > 0 (need ≥3)")

    neg_premiums = 0
    for s in chain.strikes:
        if s.ce_ltp < 0 or s.pe_ltp < 0:
            neg_premiums += 1
    if neg_premiums:
        errors.append(f"{neg_premiums} strikes have negative premiums")

    return ValidationResult(valid=not errors, warnings=warnings, errors=errors)


def validate_strike_data(strike) -> ValidationResult:
    """Validate a single StrikeData row."""
    errors: list[str] = []
    warnings: list[str] = []
    sp = strike.strike_price

    if not _is_finite(sp) or sp <= 0:
        errors.append(f"strike_price invalid: {sp}")
        return ValidationResult(valid=False, errors=errors)

    for label, ltp in [("ce_ltp", strike.ce_ltp), ("pe_ltp", strike.pe_ltp)]:
        if ltp < 0:
            errors.append(f"{label} negative: {ltp}")

    for label, iv in [("ce_iv", strike.ce_iv), ("pe_iv", strike.pe_iv)]:
        if iv > 0 and not (IV_FLOOR <= iv <= IV_CAP):
            warnings.append(f"{label}={iv} outside [{IV_FLOOR},{IV_CAP}]")

    if strike.ce_bid > 0 and strike.ce_ask > 0 and strike.ce_bid > strike.ce_ask:
        warnings.append(f"CE bid ({strike.ce_bid}) > ask ({strike.ce_ask})")
    if strike.pe_bid > 0 and strike.pe_ask > 0 and strike.pe_bid > strike.pe_ask:
        warnings.append(f"PE bid ({strike.pe_bid}) > ask ({strike.pe_ask})")

    return ValidationResult(valid=not errors, warnings=warnings, errors=errors)


def validate_iv_value(iv: float | None) -> ValidationResult:
    """Validate a computed IV value.

    - None means "could not compute" (valid signal)
    - 0.0 is now ambiguous and should not be used
    - Must be in [IV_FLOOR, IV_CAP] when present
    """
    if iv is None:
        return ValidationResult(valid=True, warnings=["IV is None (could not compute)"])

    if not _is_finite(iv):
        return ValidationResult(valid=False, errors=[f"IV not finite: {iv}"])

    if iv == 0.0:
        return ValidationResult(valid=True, warnings=["IV is 0.0 — ambiguous"])

    if not (IV_FLOOR <= iv <= IV_CAP):
        return ValidationResult(valid=False, errors=[f"IV {iv} outside [{IV_FLOOR},{IV_CAP}]"])

    return ValidationResult(valid=True)


def validate_vol_snapshot(vol) -> ValidationResult:
    """Validate a VolSnapshot before it drives algorithm parameters."""
    if vol is None:
        return ValidationResult(valid=False, errors=["vol snapshot is None"])

    errors: list[str] = []
    warnings: list[str] = []

    for pctl in ["p_rv", "p_vov", "p_vrp"]:
        v = getattr(vol, pctl, None)
        if v is None or not _is_finite(v):
            errors.append(f"{pctl} not finite: {v}")
        elif not (0 <= v <= 1):
            errors.append(f"{pctl}={v} outside [0,1]")

    for rv_name in ["rv_5", "rv_10", "rv_20"]:
        v = getattr(vol, rv_name, None)
        if v is None or not _is_finite(v):
            errors.append(f"{rv_name} not finite: {v}")
        elif v < 0:
            warnings.append(f"{rv_name}={v} negative")

    if not _is_finite(vol.vix) or vol.vix < 0:
        errors.append(f"vix invalid: {vol.vix}")

    if not _is_finite(vol.vrp):
        errors.append(f"vrp not finite: {vol.vrp}")

    return ValidationResult(valid=not errors, warnings=warnings, errors=errors)


def validate_technicals(tech) -> ValidationResult:
    """Validate TechnicalIndicators before signal generation."""
    if tech is None:
        return ValidationResult(valid=False, errors=["technicals is None"])

    errors: list[str] = []
    warnings: list[str] = []

    if not _is_finite(tech.spot) or tech.spot <= 0:
        errors.append(f"spot invalid: {tech.spot}")

    if _is_finite(tech.rsi) and not (0 <= tech.rsi <= 100):
        errors.append(f"RSI out of range: {tech.rsi}")

    for ema_name in ["ema_9", "ema_20", "ema_21"]:
        v = getattr(tech, ema_name, 0.0)
        if v != 0.0 and (not _is_finite(v) or v <= 0):
            warnings.append(f"{ema_name} invalid: {v}")

    if tech.data_staleness_minutes > 10:
        warnings.append(f"data stale: {tech.data_staleness_minutes:.1f} min")

    return ValidationResult(valid=not errors, warnings=warnings, errors=errors)


def validate_candle_dataframe(df: pd.DataFrame | None) -> ValidationResult:
    """Validate an OHLCV DataFrame before indicator computation."""
    if df is None:
        return ValidationResult(valid=False, errors=["DataFrame is None"])

    if df.empty:
        return ValidationResult(valid=False, errors=["DataFrame is empty"])

    errors: list[str] = []
    warnings: list[str] = []

    required_cols = {"Open", "High", "Low", "Close", "Volume"}
    present = set(df.columns)
    missing = required_cols - present
    if missing:
        errors.append(f"missing columns: {missing}")
        return ValidationResult(valid=False, errors=errors)

    nan_close = df["Close"].isna().sum()
    if nan_close == len(df):
        errors.append("all Close values are NaN")
    elif nan_close > 0:
        warnings.append(f"{nan_close}/{len(df)} Close values are NaN")

    if len(df) < 5:
        warnings.append(f"only {len(df)} rows — some indicators may be NaN")

    return ValidationResult(valid=not errors, warnings=warnings, errors=errors)


def validate_trade_suggestion(sug) -> ValidationResult:
    """Validate a TradeSuggestion before opening a position."""
    errors: list[str] = []
    warnings: list[str] = []

    if not sug.legs:
        errors.append("no legs")
        return ValidationResult(valid=False, errors=errors)

    for i, leg in enumerate(sug.legs):
        if not _is_finite(leg.ltp) or leg.ltp <= 0:
            errors.append(f"leg[{i}] ltp invalid: {leg.ltp}")
        if not _is_finite(leg.strike) or leg.strike <= 0:
            errors.append(f"leg[{i}] strike invalid: {leg.strike}")

    if not _is_finite(sug.score) or sug.score <= 0:
        warnings.append(f"score non-positive: {sug.score}")

    # Credit strategies must have positive net credit
    credit_strategies = {
        "Short Straddle", "Short Strangle",
        "Bull Put Spread", "Bear Call Spread", "Iron Condor",
    }
    if sug.strategy.value in credit_strategies and sug.net_credit_debit < 0:
        errors.append(
            f"credit strategy {sug.strategy.value} has net debit: {sug.net_credit_debit}"
        )

    return ValidationResult(valid=not errors, warnings=warnings, errors=errors)


def validate_paper_trading_state(state) -> ValidationResult:
    """Validate PaperTradingState for corruption."""
    errors: list[str] = []
    warnings: list[str] = []

    if not _is_finite(state.initial_capital) or state.initial_capital <= 0:
        errors.append(f"initial_capital invalid: {state.initial_capital}")

    if not _is_finite(state.net_realized_pnl):
        errors.append(f"net_realized_pnl not finite: {state.net_realized_pnl}")

    for i, pos in enumerate(state.open_positions):
        for leg in pos.legs:
            if not _is_finite(leg.entry_ltp) or leg.entry_ltp <= 0:
                errors.append(f"position[{i}] leg entry_ltp invalid: {leg.entry_ltp}")

    return ValidationResult(valid=not errors, warnings=warnings, errors=errors)


def validate_iv_for_storage(atm_iv: float) -> ValidationResult:
    """Validate an IV value before writing to iv_history database."""
    if not _is_finite(atm_iv):
        return ValidationResult(valid=False, errors=[f"IV not finite: {atm_iv}"])
    if atm_iv <= 0:
        return ValidationResult(valid=False, errors=[f"IV non-positive: {atm_iv}"])
    if not (IV_FLOOR <= atm_iv <= IV_CAP):
        return ValidationResult(
            valid=False,
            errors=[f"IV {atm_iv} outside storage bounds [{IV_FLOOR},{IV_CAP}]"],
        )
    return ValidationResult(valid=True)


def validate_yfinance_df(df: pd.DataFrame | None, expected_cols: list[str] | None = None) -> ValidationResult:
    """Validate a DataFrame returned by yfinance.

    Handles MultiIndex column edge cases and NaN blocks.
    """
    if df is None:
        return ValidationResult(valid=False, errors=["yfinance returned None"])
    if df.empty:
        return ValidationResult(valid=False, errors=["yfinance returned empty DataFrame"])

    warnings: list[str] = []

    if isinstance(df.columns, pd.MultiIndex):
        warnings.append("MultiIndex columns detected — caller should flatten")

    if expected_cols:
        cols = set(df.columns.get_level_values(0) if isinstance(df.columns, pd.MultiIndex) else df.columns)
        missing = set(expected_cols) - cols
        if missing:
            return ValidationResult(valid=False, errors=[f"missing columns: {missing}"])

    if not isinstance(df.columns, pd.MultiIndex):
        all_nan_cols = [c for c in df.columns if df[c].isna().all()]
        if all_nan_cols:
            warnings.append(f"all-NaN columns: {all_nan_cols}")

    return ValidationResult(valid=True, warnings=warnings)


def validate_config(config: dict) -> ValidationResult:
    """Validate top-level config.yaml structure."""
    errors: list[str] = []
    warnings: list[str] = []

    required_sections = ["sources", "engine", "options_desk", "paper_trading"]
    for section in required_sections:
        if section not in config:
            errors.append(f"missing config section: {section}")

    pt = config.get("paper_trading", {})
    lot_size = pt.get("lot_size", 0)
    if not isinstance(lot_size, (int, float)) or lot_size <= 0:
        errors.append(f"paper_trading.lot_size invalid: {lot_size}")

    initial_cap = pt.get("initial_capital", 0)
    if not isinstance(initial_cap, (int, float)) or initial_cap <= 0:
        errors.append(f"paper_trading.initial_capital invalid: {initial_cap}")

    daily_loss = pt.get("daily_loss_limit_pct", 0)
    if isinstance(daily_loss, (int, float)) and not (0.1 <= daily_loss <= 10.0):
        warnings.append(f"daily_loss_limit_pct={daily_loss} outside [0.1, 10.0]")

    return ValidationResult(valid=not errors, warnings=warnings, errors=errors)
