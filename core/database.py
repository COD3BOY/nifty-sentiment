"""SQLAlchemy + SQLite storage for sentiment data."""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


class _SafeEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types and other non-standard types."""

    def default(self, obj):
        try:
            import numpy as np
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
        except ImportError:
            pass
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


def _dumps(obj: Any) -> str:
    return json.dumps(obj, cls=_SafeEncoder)

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine, func
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from core.config import load_config
from core.models import AggregatedSentiment, SentimentScore

Base = declarative_base()


class AggregatedSentimentRow(Base):
    __tablename__ = "aggregated_sentiment"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    overall_score = Column(Float, nullable=False)
    level = Column(String(30), nullable=False)
    confidence = Column(Float, nullable=False)
    sources_used = Column(Integer, default=0)
    sources_failed = Column(Integer, default=0)
    bullish_factors = Column(Text, default="[]")
    bearish_factors = Column(Text, default="[]")


class SourceScoreRow(Base):
    __tablename__ = "source_scores"

    id = Column(Integer, primary_key=True, autoincrement=True)
    aggregated_id = Column(Integer, nullable=False, index=True)
    source_name = Column(String(50), nullable=False)
    score = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    explanation = Column(Text, default="")
    raw_data = Column(Text, default="{}")
    bullish_factors = Column(Text, default="[]")
    bearish_factors = Column(Text, default="[]")


class MarketActualRow(Base):
    __tablename__ = "market_actuals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(DateTime, nullable=False, unique=True, index=True)
    nifty_open = Column(Float)
    nifty_close = Column(Float)
    nifty_change_pct = Column(Float)
    direction = Column(String(10))  # 'up', 'down', 'flat'


class TradeCritiqueRow(Base):
    __tablename__ = "trade_critiques"

    id = Column(Integer, primary_key=True, autoincrement=True)
    trade_id = Column(String(20), nullable=False, unique=True, index=True)
    strategy = Column(String(50), nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)
    overall_grade = Column(String(20), nullable=False)
    summary = Column(Text, default="")
    pnl_assessment = Column(Text, default="{}")
    entry_signal_analysis = Column(Text, default="{}")
    strategy_fitness = Column(Text, default="{}")
    parameter_recommendations = Column(Text, default="[]")
    patterns_observed = Column(Text, default="[]")
    risk_management_notes = Column(Text, default="")
    full_trade_data = Column(Text, default="{}")


class ParameterAdjustmentRow(Base):
    __tablename__ = "parameter_adjustments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    trade_id = Column(String(20), nullable=False, index=True)
    strategy_name = Column(String(50), nullable=False, index=True)
    parameter_name = Column(String(50), nullable=False, index=True)
    current_value = Column(Float, nullable=False)
    recommended_value = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    reasoning = Column(Text, default="")
    condition = Column(Text, default="")
    applied = Column(Integer, default=0)  # 0=pending, 1=applied, -1=rejected


class SentimentDatabase:
    def __init__(self):
        config = load_config()
        db_path = Path(__file__).resolve().parent.parent / config["database"]["path"]
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)

    def save_aggregated_sentiment(self, sentiment: AggregatedSentiment) -> int:
        with self.SessionLocal() as session:
            row = AggregatedSentimentRow(
                timestamp=sentiment.timestamp,
                overall_score=sentiment.overall_score,
                level=sentiment.level.value,
                confidence=sentiment.confidence,
                sources_used=sentiment.sources_used,
                sources_failed=sentiment.sources_failed,
                bullish_factors=_dumps(sentiment.bullish_factors),
                bearish_factors=_dumps(sentiment.bearish_factors),
            )
            session.add(row)
            session.flush()
            agg_id = row.id

            for ss in sentiment.source_scores:
                self._save_source_score(session, agg_id, ss)

            session.commit()
            return agg_id

    def _save_source_score(self, session: Session, aggregated_id: int, ss: SentimentScore) -> None:
        row = SourceScoreRow(
            aggregated_id=aggregated_id,
            source_name=ss.source_name,
            score=ss.score,
            confidence=ss.confidence,
            timestamp=ss.timestamp,
            explanation=ss.explanation,
            raw_data=_dumps(ss.raw_data),
            bullish_factors=_dumps(ss.bullish_factors),
            bearish_factors=_dumps(ss.bearish_factors),
        )
        session.add(row)

    def get_historical_sentiment(self, days: int = 30) -> list[dict[str, Any]]:
        cutoff = datetime.utcnow() - timedelta(days=days)
        with self.SessionLocal() as session:
            rows = (
                session.query(AggregatedSentimentRow)
                .filter(AggregatedSentimentRow.timestamp >= cutoff)
                .order_by(AggregatedSentimentRow.timestamp.asc())
                .all()
            )
            return [
                {
                    "id": r.id,
                    "timestamp": r.timestamp,
                    "overall_score": r.overall_score,
                    "level": r.level,
                    "confidence": r.confidence,
                    "sources_used": r.sources_used,
                    "bullish_factors": json.loads(r.bullish_factors),
                    "bearish_factors": json.loads(r.bearish_factors),
                }
                for r in rows
            ]

    def get_source_scores_for(self, aggregated_id: int) -> list[dict[str, Any]]:
        with self.SessionLocal() as session:
            rows = (
                session.query(SourceScoreRow)
                .filter(SourceScoreRow.aggregated_id == aggregated_id)
                .all()
            )
            return [
                {
                    "source_name": r.source_name,
                    "score": r.score,
                    "confidence": r.confidence,
                    "explanation": r.explanation,
                    "bullish_factors": json.loads(r.bullish_factors),
                    "bearish_factors": json.loads(r.bearish_factors),
                }
                for r in rows
            ]

    def save_market_actual(self, date: datetime, open_price: float, close_price: float) -> None:
        change_pct = ((close_price - open_price) / open_price) * 100 if open_price else 0
        direction = "up" if change_pct > 0.05 else ("down" if change_pct < -0.05 else "flat")
        with self.SessionLocal() as session:
            existing = session.query(MarketActualRow).filter(MarketActualRow.date == date).first()
            if existing:
                existing.nifty_open = open_price
                existing.nifty_close = close_price
                existing.nifty_change_pct = change_pct
                existing.direction = direction
            else:
                session.add(MarketActualRow(
                    date=date,
                    nifty_open=open_price,
                    nifty_close=close_price,
                    nifty_change_pct=change_pct,
                    direction=direction,
                ))
            session.commit()

    # ------------------------------------------------------------------
    # Trade critique persistence
    # ------------------------------------------------------------------

    def save_critique(self, critique, trade_record_dict: dict | None = None) -> None:
        """Save a TradeCritique and its parameter recommendations.

        Recommendations are stored as informational only (applied=0).
        Use the EOD report to review aggregated recommendations.
        """
        from core.criticizer_models import TradeCritique

        if not isinstance(critique, TradeCritique):
            raise TypeError("Expected TradeCritique instance")

        with self.SessionLocal() as session:
            row = TradeCritiqueRow(
                trade_id=critique.trade_id,
                strategy=(trade_record_dict or {}).get("strategy", ""),
                timestamp=critique.timestamp,
                overall_grade=critique.overall_grade,
                summary=critique.summary,
                pnl_assessment=_dumps(critique.pnl_assessment),
                entry_signal_analysis=_dumps(critique.entry_signal_analysis),
                strategy_fitness=_dumps(critique.strategy_fitness),
                parameter_recommendations=_dumps(
                    [r.model_dump(mode="json") for r in critique.parameter_recommendations]
                ),
                patterns_observed=_dumps(critique.patterns_observed),
                risk_management_notes=critique.risk_management_notes,
                full_trade_data=_dumps(trade_record_dict or {}),
            )
            session.merge(row)

            # Denormalized parameter adjustments — stored as informational only
            for rec in critique.parameter_recommendations:
                adj = ParameterAdjustmentRow(
                    trade_id=critique.trade_id,
                    strategy_name=rec.strategy_name,
                    parameter_name=rec.parameter_name,
                    current_value=rec.current_value,
                    recommended_value=rec.recommended_value,
                    confidence=rec.confidence,
                    reasoning=rec.reasoning,
                    condition=rec.condition,
                    applied=0,
                )
                session.add(adj)

            session.commit()

    def get_critiques_for_strategy(self, strategy: str, limit: int = 10) -> list[dict[str, Any]]:
        """Return recent critiques for a strategy."""
        with self.SessionLocal() as session:
            rows = (
                session.query(TradeCritiqueRow)
                .filter(TradeCritiqueRow.strategy == strategy)
                .order_by(TradeCritiqueRow.timestamp.desc())
                .limit(limit)
                .all()
            )
            return [
                {
                    "trade_id": r.trade_id,
                    "strategy": r.strategy,
                    "timestamp": r.timestamp,
                    "overall_grade": r.overall_grade,
                    "summary": r.summary,
                    "pnl_assessment": json.loads(r.pnl_assessment),
                    "entry_signal_analysis": json.loads(r.entry_signal_analysis),
                    "strategy_fitness": json.loads(r.strategy_fitness),
                    "parameter_recommendations": json.loads(r.parameter_recommendations),
                    "patterns_observed": json.loads(r.patterns_observed),
                    "risk_management_notes": r.risk_management_notes,
                }
                for r in rows
            ]

    def get_all_critiques(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return all recent critiques."""
        with self.SessionLocal() as session:
            rows = (
                session.query(TradeCritiqueRow)
                .order_by(TradeCritiqueRow.timestamp.desc())
                .limit(limit)
                .all()
            )
            return [
                {
                    "trade_id": r.trade_id,
                    "strategy": r.strategy,
                    "timestamp": r.timestamp,
                    "overall_grade": r.overall_grade,
                    "summary": r.summary,
                    "pnl_assessment": json.loads(r.pnl_assessment),
                    "entry_signal_analysis": json.loads(r.entry_signal_analysis),
                    "strategy_fitness": json.loads(r.strategy_fitness),
                    "parameter_recommendations": json.loads(r.parameter_recommendations),
                    "patterns_observed": json.loads(r.patterns_observed),
                    "risk_management_notes": r.risk_management_notes,
                }
                for r in rows
            ]

    def get_pending_adjustments(self, strategy: str | None = None) -> list[dict[str, Any]]:
        """Return pending parameter adjustments, optionally filtered by strategy."""
        with self.SessionLocal() as session:
            query = session.query(ParameterAdjustmentRow).filter(
                ParameterAdjustmentRow.applied == 0,
            )
            if strategy:
                query = query.filter(ParameterAdjustmentRow.strategy_name == strategy)
            rows = query.order_by(ParameterAdjustmentRow.id.desc()).all()
            return [
                {
                    "id": r.id,
                    "trade_id": r.trade_id,
                    "strategy_name": r.strategy_name,
                    "parameter_name": r.parameter_name,
                    "current_value": r.current_value,
                    "recommended_value": r.recommended_value,
                    "confidence": r.confidence,
                    "reasoning": r.reasoning,
                    "condition": r.condition,
                }
                for r in rows
            ]

    def get_active_overrides(self) -> dict[str, dict[str, float]]:
        """Return the latest applied override per (strategy, parameter).

        Returns ``{strategy_name: {parameter_name: recommended_value}}``.
        Only rows with ``applied=1`` are considered; for each
        (strategy, param) pair the row with the highest ``id`` wins.
        """
        with self.SessionLocal() as session:
            # Subquery: max id per (strategy, param) among applied rows
            sub = (
                session.query(
                    ParameterAdjustmentRow.strategy_name,
                    ParameterAdjustmentRow.parameter_name,
                    func.max(ParameterAdjustmentRow.id).label("max_id"),
                )
                .filter(ParameterAdjustmentRow.applied == 1)
                .group_by(
                    ParameterAdjustmentRow.strategy_name,
                    ParameterAdjustmentRow.parameter_name,
                )
                .subquery()
            )
            rows = (
                session.query(ParameterAdjustmentRow)
                .join(sub, ParameterAdjustmentRow.id == sub.c.max_id)
                .all()
            )
            result: dict[str, dict[str, float]] = {}
            for r in rows:
                result.setdefault(r.strategy_name, {})[r.parameter_name] = r.recommended_value
            return result

    def clear_critiques(self) -> None:
        """Delete all trade critiques and parameter adjustments."""
        with self.SessionLocal() as session:
            session.query(ParameterAdjustmentRow).delete()
            session.query(TradeCritiqueRow).delete()
            session.commit()

    def compute_accuracy(self, days: int = 30) -> dict[str, Any]:
        """Compare predicted direction vs actual NIFTY close direction."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        with self.SessionLocal() as session:
            sentiments = (
                session.query(AggregatedSentimentRow)
                .filter(AggregatedSentimentRow.timestamp >= cutoff)
                .order_by(AggregatedSentimentRow.timestamp.asc())
                .all()
            )
            actuals = (
                session.query(MarketActualRow)
                .filter(MarketActualRow.date >= cutoff)
                .order_by(MarketActualRow.date.asc())
                .all()
            )

        actual_by_date = {a.date.date(): a.direction for a in actuals}
        correct = 0
        total = 0
        for s in sentiments:
            s_date = s.timestamp.date()
            if s_date in actual_by_date:
                predicted = "up" if s.overall_score > 0 else ("down" if s.overall_score < 0 else "flat")
                actual = actual_by_date[s_date]
                if predicted == actual:
                    correct += 1
                total += 1

        return {
            "correct": correct,
            "total": total,
            "accuracy": (correct / total * 100) if total > 0 else 0.0,
            "days": days,
        }
