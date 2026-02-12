"""Pydantic schema validation for config.yaml.

Called at startup to catch misconfigurations before any trading logic runs.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


class ObservationConfig(BaseModel):
    """Validation for observation period parameters."""
    gap_threshold_pct: float = Field(ge=0.0, le=5.0, default=0.2)
    trend_moderate_pct: float = Field(ge=0.0, le=5.0, default=0.3)
    trend_strong_pct: float = Field(ge=0.0, le=10.0, default=0.7)
    volume_high_ratio: float = Field(gt=0.0, le=10.0, default=1.3)
    volume_low_ratio: float = Field(gt=0.0, le=2.0, default=0.7)
    vwap_consistency_pct: float = Field(ge=50.0, le=100.0, default=70.0)
    historical_days: int = Field(ge=1, le=30, default=5)

    @model_validator(mode="after")
    def validate_trend_ordering(self) -> "ObservationConfig":
        if self.trend_moderate_pct >= self.trend_strong_pct:
            raise ValueError(
                f"trend_moderate_pct ({self.trend_moderate_pct}) must be < trend_strong_pct ({self.trend_strong_pct})"
            )
        return self

    @model_validator(mode="after")
    def validate_volume_ordering(self) -> "ObservationConfig":
        if self.volume_low_ratio >= self.volume_high_ratio:
            raise ValueError(
                f"volume_low_ratio ({self.volume_low_ratio}) must be < volume_high_ratio ({self.volume_high_ratio})"
            )
        return self

    class Config:
        extra = "allow"


class PaperTradingConfig(BaseModel):
    initial_capital: float = Field(gt=0)
    lot_size: int = Field(gt=0)
    auto_execute: bool = True
    min_score_to_trade: float = Field(ge=0, le=100, default=50)
    trade_cooldown_seconds: int = Field(ge=0, default=60)
    max_lots: int = Field(gt=0, default=4)
    capital_utilization: float = Field(gt=0, le=1.0, default=0.70)
    daily_loss_limit_pct: float = Field(ge=0.1, le=10.0, default=3.0)
    max_open_positions: int = Field(gt=0, default=3)
    max_staleness_minutes: float = Field(gt=0, default=20)
    eod_close_time: str = "15:20"
    entry_start_time: str = "10:00"
    entry_cutoff_time: str = "15:10"
    observation: ObservationConfig = Field(default_factory=ObservationConfig)

    @field_validator("eod_close_time", "entry_start_time", "entry_cutoff_time")
    @classmethod
    def validate_time_str(cls, v: str) -> str:
        parts = v.split(":")
        if len(parts) != 2:
            raise ValueError(f"invalid time format: {v!r} (expected HH:MM)")
        h, m = int(parts[0]), int(parts[1])
        if not (0 <= h <= 23 and 0 <= m <= 59):
            raise ValueError(f"time out of range: {v!r}")
        return v

    class Config:
        extra = "allow"  # allow unknown keys


class AlgorithmConfig(BaseModel):
    """Validates per-algorithm config. Allows extra keys for algorithm-specific params."""
    enabled: bool = True
    display_name: str = ""
    initial_capital: float = Field(gt=0, default=5_000_000)
    lot_size: int = Field(gt=0, default=65)

    class Config:
        extra = "allow"


class SelfImprovementConfig(BaseModel):
    """Validation for self-improvement / daily review safety rails."""
    enabled: bool = True
    min_sample_size: int = Field(ge=5, le=100, default=15)
    max_step_pct: float = Field(gt=0.0, le=0.5, default=0.15)
    max_drift_pct: float = Field(gt=0.0, le=1.0, default=0.40)
    cooling_period_trades: int = Field(ge=1, le=100, default=10)
    reversion_trigger_losses: int = Field(ge=2, le=20, default=5)
    max_changes_per_session: int = Field(ge=1, le=10, default=3)
    rolling_window_days: int = Field(ge=5, le=90, default=20)

    @model_validator(mode="after")
    def validate_drift_exceeds_step(self) -> "SelfImprovementConfig":
        if self.max_drift_pct < self.max_step_pct:
            raise ValueError(
                f"max_drift_pct ({self.max_drift_pct}) must be >= max_step_pct ({self.max_step_pct})"
            )
        return self

    class Config:
        extra = "allow"


class OptionsDeskConfig(BaseModel):
    symbol: str = "NIFTY"
    lot_size: int = Field(gt=0, default=65)

    class Config:
        extra = "allow"


class EngineConfig(BaseModel):
    default_timeout: int = Field(gt=0, default=30)

    class Config:
        extra = "allow"


class DatabaseConfig(BaseModel):
    path: str = "data/nifty_sentiment.db"


class NiftyConfig(BaseModel):
    """Top-level config schema."""
    sources: dict[str, Any] = Field(default_factory=dict)
    engine: EngineConfig = Field(default_factory=EngineConfig)
    options_desk: dict[str, Any] = Field(default_factory=dict)
    paper_trading: PaperTradingConfig
    algorithms: dict[str, Any] = Field(default_factory=dict)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    self_improvement: SelfImprovementConfig = Field(default_factory=SelfImprovementConfig)

    class Config:
        extra = "allow"

    @model_validator(mode="after")
    def validate_algorithm_lot_sizes(self) -> "NiftyConfig":
        """Warn if any algorithm lot_size differs from paper_trading.lot_size."""
        pt_lot = self.paper_trading.lot_size
        for algo_name, algo_cfg in self.algorithms.items():
            if isinstance(algo_cfg, dict):
                algo_lot = algo_cfg.get("lot_size", pt_lot)
                if algo_lot != pt_lot:
                    logger.warning(
                        "algorithms.%s.lot_size=%d differs from paper_trading.lot_size=%d",
                        algo_name, algo_lot, pt_lot,
                    )
        return self


def validate_config_dict(raw: dict[str, Any]) -> NiftyConfig:
    """Validate a raw config dict. Raises pydantic.ValidationError on failure."""
    return NiftyConfig.model_validate(raw)
