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

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine
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
