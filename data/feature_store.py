"""
ARGUS V5 — Universal Feature Store & Audit Database
Thread-safe, leak-free storage layer backed by SQLAlchemy.
Seamlessly swappable between local SQLite and production PostgreSQL (Railway / Render / Fly.io).
"""

import json
import os
import threading
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Tuple
import pandas as pd
from sqlalchemy import create_engine, select, desc, func, and_
from sqlalchemy.orm import sessionmaker, Session

from config import BASE_DIR
from data.db import (
    Base,
    PredictionRecord,
    OutcomeRecord,
    SentimentRecord,
    SeenArticleRecord,
    DriftEventRecord,
    DEFAULT_SQLITE_PATH,
    DATABASE_URL,
    engine as default_engine,
    SessionLocal as default_sessionmaker,
)


class FeatureStore:
    """
    Universal Feature Store for ARGUS V5:
    - Point-in-time features and auditable predictions
    - Strict T+5 outcome matching and realization
    - Point-in-time FinBERT sentiment audit logs
    - Concept drift event telemetry
    - Deduplicated news caching
    """

    def __init__(self, db_path: Optional[str] = None):
        self._lock = threading.Lock()
        
        # If custom db_path passed (e.g. in test suites), spin up an isolated SQLite engine
        if db_path and db_path != DEFAULT_SQLITE_PATH:
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            self.db_path = db_path
            self.engine = create_engine(
                f"sqlite:///{db_path}",
                connect_args={"check_same_thread": False},
                pool_pre_ping=True,
            )
            self.Session = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
            Base.metadata.create_all(bind=self.engine)
        else:
            self.db_path = DEFAULT_SQLITE_PATH
            self.engine = default_engine
            self.Session = default_sessionmaker
            Base.metadata.create_all(bind=self.engine)

    def _get_session(self) -> Session:
        return self.Session()

    # ── Prediction & Outcome Methods ──────────────────────────────────────────

    def save_prediction(
        self,
        pred_id: str,
        timestamp: str,
        ticker: str,
        features_asof: str,
        features: Dict[str, Any],
        prob_online: float,
        action_online: str,
        base_price: float,
        prob_dl: Optional[float] = None,
        action_dl: Optional[str] = None,
        daily_asof: Optional[str] = None,
    ):
        """Log a new prediction made at time T with auditable features_asof and daily_asof."""
        with self._lock, self._get_session() as session:
            record = PredictionRecord(
                id=pred_id,
                timestamp=str(timestamp),
                ticker=ticker.upper(),
                features_asof=str(features_asof),
                daily_asof=str(daily_asof) if daily_asof else None,
                features_json=json.dumps(features),
                prob_online=float(prob_online),
                action_online=str(action_online),
                prob_dl=float(prob_dl) if prob_dl is not None else None,
                action_dl=str(action_dl) if action_dl is not None else None,
                base_price=float(base_price),
                status="pending",
            )
            session.merge(record)
            session.commit()

    def resolve_prediction(
        self,
        pred_id: str,
        realized_timestamp: str,
        realized_price: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Match prediction at T with realized price at T+5min.
        Computes true binary label: 1 if Price(T+5) > Price(T) else 0.
        Updates outcomes table and prediction status.
        """
        with self._lock, self._get_session() as session:
            pred = session.query(PredictionRecord).filter(PredictionRecord.id == pred_id).first()
            if not pred:
                return None

            base_price = float(pred.base_price)
            realized_price = float(realized_price)
            price_change = realized_price - base_price
            price_change_pct = (price_change / base_price) * 100.0 if base_price > 0 else 0.0

            # Ground truth binary label: 1 for up, 0 for flat/down
            realized_label = 1 if realized_price > base_price else 0

            # Did the model correctly anticipate direction?
            pred_up_online = 1 if pred.prob_online >= 0.5 else 0
            correct_online = 1 if pred_up_online == realized_label else 0

            correct_dl = None
            if pred.prob_dl is not None:
                pred_up_dl = 1 if pred.prob_dl >= 0.5 else 0
                correct_dl = 1 if pred_up_dl == realized_label else 0

            # Trading PnL based on executed action
            pnl_pct = 0.0
            if pred.action_online == "BUY":
                pnl_pct = price_change_pct
            elif pred.action_online == "SELL":
                pnl_pct = -price_change_pct

            outcome = OutcomeRecord(
                prediction_id=pred_id,
                realized_timestamp=str(realized_timestamp),
                realized_price=realized_price,
                price_change_pct=round(price_change_pct, 4),
                realized_label=realized_label,
                correct_online=correct_online,
                correct_dl=correct_dl,
                pnl_pct=round(pnl_pct, 4),
            )
            session.merge(outcome)
            pred.status = "resolved"
            session.commit()

            features = json.loads(pred.features_json)
            return {
                "prediction_id": pred_id,
                "ticker": pred.ticker,
                "features_asof": pred.features_asof,
                "daily_asof": pred.daily_asof,
                "features": features,
                "base_price": base_price,
                "realized_price": realized_price,
                "price_change_pct": price_change_pct,
                "realized_label": realized_label,
                "prob_online": pred.prob_online,
                "action_online": pred.action_online,
                "correct_online": bool(correct_online),
                "prob_dl": pred.prob_dl,
                "action_dl": pred.action_dl,
                "correct_dl": bool(correct_dl) if correct_dl is not None else None,
                "pnl_pct": pnl_pct,
            }

    def get_pending_predictions(self, ticker: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetch all unresolved predictions awaiting realization at T+15min."""
        with self._lock, self._get_session() as session:
            query = session.query(PredictionRecord).filter(PredictionRecord.status == "pending")
            if ticker:
                query = query.filter(PredictionRecord.ticker == ticker.upper())
            rows = query.order_by(PredictionRecord.timestamp.asc()).all()

            results = []
            for r in rows:
                results.append({
                    "id": r.id,
                    "timestamp": r.timestamp,
                    "ticker": r.ticker,
                    "features_asof": r.features_asof,
                    "daily_asof": r.daily_asof,
                    "features": json.loads(r.features_json),
                    "prob_online": r.prob_online,
                    "action_online": r.action_online,
                    "base_price": r.base_price,
                    "prob_dl": r.prob_dl,
                    "action_dl": r.action_dl,
                })
            return results

    # ── Point-in-Time News Sentiment ──────────────────────────────────────────

    def save_sentiment(
        self,
        timestamp: str,
        ticker: Optional[str],
        sector: Optional[str],
        sentiment_score: float,
        sentiment_delta: float = 0.0,
        article_count: int = 1,
    ):
        """Save point-in-time sentiment score for a ticker or sector."""
        with self._lock, self._get_session() as session:
            record = SentimentRecord(
                timestamp=str(timestamp),
                ticker=ticker.upper() if ticker else None,
                sector=sector.upper() if sector else None,
                sentiment_score=float(sentiment_score),
                sentiment_delta=float(sentiment_delta),
                article_count=int(article_count),
            )
            session.add(record)
            session.commit()

    def get_latest_sentiment(
        self,
        ticker: Optional[str] = None,
        sector: Optional[str] = None,
        asof_time: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Point-in-time query: Returns the most recent sentiment timestamped <= asof_time.
        STRICT TIME LEAKAGE GUARD: Will NEVER return data timestamped after asof_time.
        """
        if asof_time is None:
            asof_time = datetime.now(timezone.utc).isoformat()

        with self._lock, self._get_session() as session:
            row = None
            if ticker:
                row = (
                    session.query(SentimentRecord)
                    .filter(
                        SentimentRecord.ticker == ticker.upper(),
                        SentimentRecord.timestamp <= str(asof_time),
                    )
                    .order_by(desc(SentimentRecord.timestamp))
                    .first()
                )

            if not row and sector:
                row = (
                    session.query(SentimentRecord)
                    .filter(
                        SentimentRecord.sector == sector.upper(),
                        SentimentRecord.timestamp <= str(asof_time),
                    )
                    .order_by(desc(SentimentRecord.timestamp))
                    .first()
                )

            if not row:
                return {
                    "sentiment_score": 0.0,
                    "sentiment_delta": 0.0,
                    "sentiment_asof": asof_time,
                    "article_count": 0,
                    "is_fallback": True,
                }

            return {
                "sentiment_score": float(row.sentiment_score),
                "sentiment_delta": float(row.sentiment_delta),
                "sentiment_asof": row.timestamp,
                "article_count": int(row.article_count),
                "is_fallback": False,
            }

    # ── Seen Articles Deduplication ───────────────────────────────────────────

    def is_article_seen(self, article_hash: str) -> bool:
        """Check if an article hash has already been scored."""
        with self._lock, self._get_session() as session:
            return (
                session.query(SeenArticleRecord)
                .filter(SeenArticleRecord.article_hash == article_hash)
                .first()
                is not None
            )

    def mark_article_seen(
        self,
        article_hash: str,
        title: str,
        source: str,
        published_at: Optional[str],
        scraped_at: str,
    ):
        """Mark article as processed in seen_articles table."""
        with self._lock, self._get_session() as session:
            record = SeenArticleRecord(
                article_hash=article_hash,
                title=title,
                source=source,
                published_at=str(published_at) if published_at else None,
                scraped_at=str(scraped_at),
            )
            session.merge(record)
            session.commit()

    # ── Concept Drift Logging ─────────────────────────────────────────────────

    def log_drift_event(
        self,
        timestamp: str,
        metric_name: str,
        value_before: Optional[float],
        value_after: Optional[float],
        message: str,
    ):
        """Record an ADWIN concept drift detection event."""
        with self._lock, self._get_session() as session:
            record = DriftEventRecord(
                timestamp=str(timestamp),
                metric_name=metric_name,
                value_before=value_before,
                value_after=value_after,
                message=message,
            )
            session.add(record)
            session.commit()

    def get_drift_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve recent drift events."""
        with self._lock, self._get_session() as session:
            rows = (
                session.query(DriftEventRecord)
                .order_by(desc(DriftEventRecord.timestamp))
                .limit(limit)
                .all()
            )
            return [
                {
                    "id": r.id,
                    "timestamp": r.timestamp,
                    "metric_name": r.metric_name,
                    "value_before": r.value_before,
                    "value_after": r.value_after,
                    "message": r.message,
                }
                for r in rows
            ]

    # ── Training / Evaluation Export ──────────────────────────────────────────

    def get_resolved_dataset(
        self,
        ticker: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Export completed (feature_vector, realized_label) pairs as a DataFrame
        for deep learning batch retraining or walk-forward validation.
        """
        with self._lock, self._get_session() as session:
            query = (
                session.query(PredictionRecord, OutcomeRecord)
                .join(OutcomeRecord, PredictionRecord.id == OutcomeRecord.prediction_id)
            )
            if ticker:
                query = query.filter(PredictionRecord.ticker == ticker.upper())
            query = query.order_by(PredictionRecord.timestamp.asc())
            if limit:
                query = query.limit(limit)

            pairs = query.all()

        if not pairs:
            return pd.DataFrame()

        records = []
        for p, o in pairs:
            rec = {
                "id": p.id,
                "timestamp": p.timestamp,
                "ticker": p.ticker,
                "features_asof": p.features_asof,
                "features_json": p.features_json,
                "prob_online": p.prob_online,
                "action_online": p.action_online,
                "prob_dl": p.prob_dl,
                "action_dl": p.action_dl,
                "base_price": p.base_price,
                "realized_timestamp": o.realized_timestamp,
                "realized_price": o.realized_price,
                "price_change_pct": o.price_change_pct,
                "realized_label": o.realized_label,
                "correct_online": o.correct_online,
                "correct_dl": o.correct_dl,
                "pnl_pct": o.pnl_pct,
            }
            records.append(rec)

        df = pd.DataFrame(records)
        features_list = df["features_json"].apply(json.loads).tolist()
        feat_df = pd.DataFrame(features_list)
        feat_df = feat_df.add_prefix("feat_")
        merged = pd.concat([df.drop(columns=["features_json"]), feat_df], axis=1)
        return merged
