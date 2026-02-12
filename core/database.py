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


class IVHistoryRow(Base):
    __tablename__ = "iv_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    atm_iv = Column(Float, nullable=False)
    vix = Column(Float, nullable=True)


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


class ImprovementLedgerRow(Base):
    """Tracks the lifecycle of a parameter change from proposal to confirmation/reversion."""

    __tablename__ = "improvement_ledger"

    id = Column(String(20), primary_key=True)  # UUID prefix
    date = Column(String(10), nullable=False, index=True)  # trading date
    algorithm = Column(String(30), nullable=False, index=True)
    strategy_name = Column(String(50), nullable=False, index=True)
    parameter_name = Column(String(50), nullable=False, index=True)
    default_value = Column(Float, nullable=False)
    old_value = Column(Float, nullable=False)
    new_value = Column(Float, nullable=False)
    change_pct = Column(Float, nullable=False)
    evidence_trade_count = Column(Integer, nullable=False)
    evidence_json = Column(Text, default="{}")
    confidence = Column(Float, default=0.0)
    status = Column(String(20), nullable=False, default="proposed")
    pre_metrics_json = Column(Text, default="{}")
    post_metrics_json = Column(Text, default="{}")
    created_at = Column(DateTime, nullable=False)
    applied_at = Column(DateTime, nullable=True)
    reverted_at = Column(DateTime, nullable=True)
    trades_since_applied = Column(Integer, default=0)
    consecutive_losses = Column(Integer, default=0)


class ReviewSessionRow(Base):
    """Stores daily review session summaries."""

    __tablename__ = "review_sessions"

    id = Column(String(20), primary_key=True)
    date = Column(String(10), nullable=False, index=True)
    algorithm = Column(String(30), nullable=False, index=True)
    trades_reviewed = Column(Integer, default=0)
    summary_json = Column(Text, default="{}")
    created_at = Column(DateTime, nullable=False)


class SentimentDatabase:
    def __init__(self):
        config = load_config()
        db_path = Path(__file__).resolve().parent.parent / config["database"]["path"]
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(
            f"sqlite:///{db_path}",
            echo=False,
            connect_args={"timeout": 30},
        )

        # Enable WAL mode and busy timeout for better concurrency
        from sqlalchemy import event

        @event.listens_for(self.engine, "connect")
        def _set_sqlite_pragma(dbapi_conn, connection_record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA busy_timeout=5000")
            cursor.close()

        try:
            Base.metadata.create_all(self.engine)
        except Exception:
            # Race condition: another worker already created the tables
            pass
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

    # ------------------------------------------------------------------
    # Improvement ledger persistence
    # ------------------------------------------------------------------

    def save_ledger_entry(self, entry: dict[str, Any]) -> str:
        """Save a new improvement ledger entry. Returns the entry ID."""
        with self.SessionLocal() as session:
            row = ImprovementLedgerRow(
                id=entry["id"],
                date=entry["date"],
                algorithm=entry["algorithm"],
                strategy_name=entry["strategy_name"],
                parameter_name=entry["parameter_name"],
                default_value=entry["default_value"],
                old_value=entry["old_value"],
                new_value=entry["new_value"],
                change_pct=entry["change_pct"],
                evidence_trade_count=entry["evidence_trade_count"],
                evidence_json=_dumps(entry.get("evidence", {})),
                confidence=entry.get("confidence", 0.0),
                status=entry.get("status", "proposed"),
                pre_metrics_json=_dumps(entry.get("pre_metrics", {})),
                post_metrics_json=_dumps(entry.get("post_metrics", {})),
                created_at=entry.get("created_at", datetime.utcnow()),
            )
            session.add(row)
            session.commit()
            return entry["id"]

    def update_ledger_status(
        self, entry_id: str, status: str, **kwargs: Any
    ) -> bool:
        """Update the status (and optional fields) of a ledger entry."""
        with self.SessionLocal() as session:
            row = session.query(ImprovementLedgerRow).filter(
                ImprovementLedgerRow.id == entry_id
            ).first()
            if not row:
                return False
            row.status = status
            if "applied_at" in kwargs:
                row.applied_at = kwargs["applied_at"]
            if "reverted_at" in kwargs:
                row.reverted_at = kwargs["reverted_at"]
            if "post_metrics" in kwargs:
                row.post_metrics_json = _dumps(kwargs["post_metrics"])
            if "trades_since_applied" in kwargs:
                row.trades_since_applied = kwargs["trades_since_applied"]
            if "consecutive_losses" in kwargs:
                row.consecutive_losses = kwargs["consecutive_losses"]
            session.commit()
            return True

    def get_ledger_entries(
        self,
        algorithm: str | None = None,
        status: str | None = None,
        days: int = 90,
    ) -> list[dict[str, Any]]:
        """Return ledger entries filtered by algorithm, status, and age."""
        with self.SessionLocal() as session:
            query = session.query(ImprovementLedgerRow)
            if algorithm:
                query = query.filter(ImprovementLedgerRow.algorithm == algorithm)
            if status:
                query = query.filter(ImprovementLedgerRow.status == status)
            rows = query.order_by(ImprovementLedgerRow.created_at.desc()).all()
            return [
                {
                    "id": r.id,
                    "date": r.date,
                    "algorithm": r.algorithm,
                    "strategy_name": r.strategy_name,
                    "parameter_name": r.parameter_name,
                    "default_value": r.default_value,
                    "old_value": r.old_value,
                    "new_value": r.new_value,
                    "change_pct": r.change_pct,
                    "evidence_trade_count": r.evidence_trade_count,
                    "evidence": json.loads(r.evidence_json),
                    "confidence": r.confidence,
                    "status": r.status,
                    "pre_metrics": json.loads(r.pre_metrics_json),
                    "post_metrics": json.loads(r.post_metrics_json),
                    "created_at": r.created_at,
                    "applied_at": r.applied_at,
                    "reverted_at": r.reverted_at,
                    "trades_since_applied": r.trades_since_applied,
                    "consecutive_losses": r.consecutive_losses,
                }
                for r in rows
            ]

    def get_active_ledger_for_param(
        self, strategy_name: str, parameter_name: str
    ) -> dict[str, Any] | None:
        """Return the currently applied ledger entry for a specific parameter, if any."""
        with self.SessionLocal() as session:
            row = (
                session.query(ImprovementLedgerRow)
                .filter(
                    ImprovementLedgerRow.strategy_name == strategy_name,
                    ImprovementLedgerRow.parameter_name == parameter_name,
                    ImprovementLedgerRow.status.in_(["applied", "monitoring"]),
                )
                .order_by(ImprovementLedgerRow.created_at.desc())
                .first()
            )
            if not row:
                return None
            return {
                "id": row.id,
                "date": row.date,
                "algorithm": row.algorithm,
                "strategy_name": row.strategy_name,
                "parameter_name": row.parameter_name,
                "default_value": row.default_value,
                "old_value": row.old_value,
                "new_value": row.new_value,
                "change_pct": row.change_pct,
                "status": row.status,
                "created_at": row.created_at,
                "applied_at": row.applied_at,
                "trades_since_applied": row.trades_since_applied,
                "consecutive_losses": row.consecutive_losses,
            }

    # ------------------------------------------------------------------
    # Review session persistence
    # ------------------------------------------------------------------

    def save_review_session(self, session_data: dict[str, Any]) -> str:
        """Save a review session summary. Returns the session ID."""
        with self.SessionLocal() as session:
            row = ReviewSessionRow(
                id=session_data["id"],
                date=session_data["date"],
                algorithm=session_data["algorithm"],
                trades_reviewed=session_data.get("trades_reviewed", 0),
                summary_json=_dumps(session_data.get("summary", {})),
                created_at=session_data.get("created_at", datetime.utcnow()),
            )
            session.add(row)
            session.commit()
            return session_data["id"]

    def get_review_sessions(
        self, algorithm: str | None = None, limit: int = 30
    ) -> list[dict[str, Any]]:
        """Return recent review sessions."""
        with self.SessionLocal() as session:
            query = session.query(ReviewSessionRow)
            if algorithm:
                query = query.filter(ReviewSessionRow.algorithm == algorithm)
            rows = (
                query.order_by(ReviewSessionRow.created_at.desc())
                .limit(limit)
                .all()
            )
            return [
                {
                    "id": r.id,
                    "date": r.date,
                    "algorithm": r.algorithm,
                    "trades_reviewed": r.trades_reviewed,
                    "summary": json.loads(r.summary_json),
                    "created_at": r.created_at,
                }
                for r in rows
            ]
