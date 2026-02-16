"""Pydantic models for simulation scenario definitions.

Scenarios are defined in YAML files under ``simulation/scenarios/`` and
loaded via ``load_scenario()``.  Every numeric parameter has a default
so minimal YAML is sufficient for simple scenarios.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)

SCENARIOS_DIR = Path(__file__).parent / "scenarios"


# ---------------------------------------------------------------------------
# Phase configuration — allows different params for segments of the day
# ---------------------------------------------------------------------------

class PhaseConfig(BaseModel):
    """Parameters for a single intraday phase (e.g. morning calm, midday crash)."""

    start_minute: int = 0  # minutes from 9:15 (0 = market open)
    end_minute: int = 375  # 375 = market close (15:30)
    drift: float = 0.0  # annualized drift (μ) — 0 = no trend
    volatility: float = 0.15  # annualized vol (σ) for GBM
    jump_intensity: float = 0.0  # Poisson λ (jumps per day)
    jump_mean: float = -0.02  # mean log-jump size
    jump_std: float = 0.03  # std of log-jump size
    mean_reversion_kappa: float = 0.0  # OU pull strength (0 = pure GBM)
    mean_reversion_level: float = 0.0  # OU pull target (0 = use open price)


class PricePathConfig(BaseModel):
    """Full-day price generation parameters."""

    open_price: float = 23000.0  # NIFTY spot at 9:15
    phases: list[PhaseConfig] = Field(default_factory=lambda: [PhaseConfig()])
    volume_base: float = 5_000_000.0  # total daily volume
    volume_u_shape_strength: float = 2.0  # higher = more pronounced U-shape


# ---------------------------------------------------------------------------
# Gap configuration
# ---------------------------------------------------------------------------

class GapConfig(BaseModel):
    """Opening gap from previous close."""

    gap_pct: float = 0.0  # e.g. -0.03 for -3% gap down
    prev_close: float = 0.0  # if 0, derived from open_price / (1 + gap_pct)
    gap_fill_probability: float = 0.5  # probability the gap fills during the day
    gap_fill_speed: int = 60  # minutes to fill the gap (if filling)

    @model_validator(mode="after")
    def _derive_prev_close(self):
        if self.prev_close == 0.0 and self.gap_pct != 0.0:
            # open = prev_close * (1 + gap_pct)
            # so prev_close = open / (1 + gap_pct)  — but open is on PricePathConfig
            # We'll resolve this in the engine; leave as 0 here
            pass
        return self


# ---------------------------------------------------------------------------
# VIX configuration
# ---------------------------------------------------------------------------

class VIXConfig(BaseModel):
    """VIX (India VIX) co-generation parameters."""

    open_level: float = 14.0  # VIX at market open
    mean_level: float = 14.0  # OU mean-reversion target
    volatility: float = 0.3  # annualized vol of VIX itself
    kappa: float = 5.0  # OU mean-reversion speed
    spot_correlation: float = -0.7  # correlation with NIFTY spot returns
    # Scheduled VIX events (minute_from_open → target_level)
    spikes: dict[int, float] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Volatility regime (for V4 Atlas VolSnapshot)
# ---------------------------------------------------------------------------

class VolRegimeConfig(BaseModel):
    """Pre-configured volatility regime for the scenario."""

    rv_5: float = 12.0  # 5-day realized vol (%)
    rv_10: float = 13.0
    rv_20: float = 14.0
    vov_20: float = 3.0  # vol-of-vol
    vrp: float = 2.0  # VIX - RV_20
    p_rv: float = 0.5  # percentile [0,1]
    p_vov: float = 0.5
    p_vrp: float = 0.5
    regime_duration_days: int = 5   # how long current regime has persisted
    regime_changes_30d: int = 2     # regime transitions in last 30 days
    rv_trend: str = "stable"        # "expanding" | "contracting" | "stable"


# ---------------------------------------------------------------------------
# Option chain configuration
# ---------------------------------------------------------------------------

class ChainConfig(BaseModel):
    """Parameters for synthetic option chain generation."""

    expiry_dte: int = 5  # days to expiry
    atm_iv: float = 15.0  # ATM implied vol (%)
    iv_skew_slope: float = -0.08  # put skew (negative = puts more expensive)
    iv_skew_curvature: float = 0.02  # smile curvature
    oi_base: float = 500_000.0  # ATM OI
    oi_decay_rate: float = 0.03  # exponential decay from ATM
    bid_ask_spread_pct: float = 0.01  # base spread as fraction of premium
    stress_spread_multiplier: float = 1.0  # multiplied during stress phases
    strike_step: float = 50.0  # distance between strikes
    num_strikes_each_side: int = 20  # strikes above/below ATM
    iv_percentile: float = 50.0  # for OptionsAnalytics
    iv_reactivity: float = 0.5  # how aggressively IV reacts to spot moves (0=static, 1=very reactive)


# ---------------------------------------------------------------------------
# Adversarial configuration
# ---------------------------------------------------------------------------

class AdversarialConfig(BaseModel):
    """Configuration for adversarial perturbation engine."""

    mode: str = ""  # empty = disabled; see PerturbMode enum in adversarial.py
    intensity: float = 0.8  # 0.0 = no effect, 1.0 = full strength


# ---------------------------------------------------------------------------
# Top-level scenario
# ---------------------------------------------------------------------------

class DayConfig(BaseModel):
    """Per-day config for multi-day scenarios.

    Each field is optional — only specified fields override the base
    scenario's defaults.
    """

    price_path: PricePathConfig = Field(default_factory=PricePathConfig)
    gap: GapConfig = Field(default_factory=GapConfig)
    vix: VIXConfig = Field(default_factory=VIXConfig)
    vol_regime: VolRegimeConfig = Field(default_factory=VolRegimeConfig)
    chain: ChainConfig = Field(default_factory=ChainConfig)
    adversarial: AdversarialConfig = Field(default_factory=AdversarialConfig)


class Scenario(BaseModel):
    """Complete scenario definition."""

    name: str = "unnamed"
    description: str = ""
    category: str = "normal"  # normal, gaps, crisis, volatility, patterns, adversarial
    price_path: PricePathConfig = Field(default_factory=PricePathConfig)
    gap: GapConfig = Field(default_factory=GapConfig)
    vix: VIXConfig = Field(default_factory=VIXConfig)
    vol_regime: VolRegimeConfig = Field(default_factory=VolRegimeConfig)
    chain: ChainConfig = Field(default_factory=ChainConfig)
    adversarial: AdversarialConfig = Field(default_factory=AdversarialConfig)
    num_warmup_days: int = 4  # prior "normal" days for indicator warmup
    tags: list[str] = Field(default_factory=list)

    # Multi-day scenarios: list of per-day configs
    # If non-empty, len(day_configs) is the number of days to simulate.
    # Each DayConfig provides that day's specific parameters.
    day_configs: list[DayConfig] = Field(default_factory=list)

    @property
    def num_days(self) -> int:
        """Number of trading days to simulate."""
        return max(1, len(self.day_configs))

    @property
    def is_multi_day(self) -> bool:
        return len(self.day_configs) > 1

    def get_day_scenario(self, day_idx: int) -> "Scenario":
        """Return a single-day Scenario for the given day index.

        For multi-day scenarios, returns a copy with the per-day config
        applied.  For single-day scenarios, returns self.
        """
        if not self.day_configs or day_idx >= len(self.day_configs):
            return self

        dc = self.day_configs[day_idx]
        return Scenario(
            name=f"{self.name}_day{day_idx}",
            description=self.description,
            category=self.category,
            price_path=dc.price_path,
            gap=dc.gap,
            vix=dc.vix,
            vol_regime=dc.vol_regime,
            chain=dc.chain,
            adversarial=dc.adversarial,
            num_warmup_days=self.num_warmup_days if day_idx == 0 else 0,
            tags=self.tags,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_multi_day(data: dict) -> None:
    """Convert YAML ``days`` list to ``day_configs`` for the Scenario model.

    Multi-day YAML scenarios use a ``days`` list where each element has
    per-day config (price_path, gap, vix, etc.).  This converts it to
    the ``day_configs`` field that the Pydantic model expects.
    """
    if "days" in data and isinstance(data["days"], list):
        data["day_configs"] = data.pop("days")


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_scenario(path: str | Path) -> Scenario:
    """Load a scenario from a YAML file.

    Parameters
    ----------
    path : str or Path
        Either a full path, or a relative name like ``"crisis/flash_crash"``
        which resolves to ``simulation/scenarios/crisis.yaml`` key ``flash_crash``.
    """
    path = str(path)

    # Handle "category/name" shorthand
    if "/" in path and not Path(path).suffix:
        category, name = path.split("/", 1)
        yaml_path = SCENARIOS_DIR / f"{category}.yaml"
        if not yaml_path.exists():
            raise FileNotFoundError(f"Scenario file not found: {yaml_path}")

        with open(yaml_path) as f:
            raw = yaml.safe_load(f)

        if name not in raw:
            available = list(raw.keys())
            raise KeyError(
                f"Scenario '{name}' not found in {yaml_path}. "
                f"Available: {available}"
            )

        data = raw[name]
        data.setdefault("name", name)
        data.setdefault("category", category)
        _normalize_multi_day(data)
        return Scenario.model_validate(data)

    # Full path to YAML
    yaml_path = Path(path)
    if not yaml_path.exists():
        # Try scenarios dir
        yaml_path = SCENARIOS_DIR / path
    if not yaml_path.exists():
        raise FileNotFoundError(f"Scenario file not found: {path}")

    with open(yaml_path) as f:
        raw = yaml.safe_load(f)

    # If the file has a single scenario (flat dict with 'name')
    if "name" in raw:
        _normalize_multi_day(raw)
        return Scenario.model_validate(raw)

    # If the file has multiple scenarios, return all as a dict
    raise ValueError(
        f"YAML file {yaml_path} has multiple scenarios. "
        f"Use 'category/name' format to select one."
    )


def list_scenarios() -> list[dict[str, str]]:
    """List all available scenarios from the YAML library.

    Returns list of dicts with keys: name, category, description, path.
    """
    results = []
    if not SCENARIOS_DIR.exists():
        return results

    for yaml_file in sorted(SCENARIOS_DIR.glob("*.yaml")):
        category = yaml_file.stem
        try:
            with open(yaml_file) as f:
                raw = yaml.safe_load(f)
            if not isinstance(raw, dict):
                continue
            for key, val in raw.items():
                if isinstance(val, dict):
                    results.append({
                        "name": f"{category}/{key}",
                        "category": category,
                        "description": val.get("description", ""),
                        "path": str(yaml_file),
                    })
        except Exception as e:
            logger.warning("Failed to parse %s: %s", yaml_file, e)

    return results
