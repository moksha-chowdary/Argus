"""
ARGUS V5 — Universal Database & ORM Layer
Supports SQLite for local development and PostgreSQL for persistent cloud hosting (Railway / Render / Fly.io).
Automatically translates 'postgres://' to 'postgresql://' for SQLAlchemy compatibility.
"""

import os
from datetime import datetime, timezone
from typing import Generator, Dict, Any
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    Text,
    ForeignKey,
    Index,
    DateTime,
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, Session
from config import BASE_DIR

DEFAULT_SQLITE_PATH = os.path.join(BASE_DIR, "data", "argus_v5.db")

def get_database_url() -> str:
    """
    Resolves the database URL:
    - If DATABASE_URL env var exists, normalizes 'postgres://' -> 'postgresql://'
    - Otherwise falls back to local SQLite WAL file
    """
    raw_url = os.getenv("DATABASE_URL")
    if raw_url:
        if raw_url.startswith("postgres://"):
            return raw_url.replace("postgres://", "postgresql://", 1)
        return raw_url
    os.makedirs(os.path.dirname(DEFAULT_SQLITE_PATH), exist_ok=True)
    return f"sqlite:///{DEFAULT_SQLITE_PATH}"

DATABASE_URL = get_database_url()

# Engine configuration
is_sqlite = DATABASE_URL.startswith("sqlite")
connect_args = {"check_same_thread": False} if is_sqlite else {}

engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    pool_pre_ping=True,
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ── SQLAlchemy Database Models ────────────────────────────────────────────────

class PredictionRecord(Base):
    __tablename__ = "predictions"

    id = Column(String(64), primary_key=True, index=True)
    timestamp = Column(String(64), nullable=False, index=True)
    ticker = Column(String(32), nullable=False, index=True)
    features_asof = Column(String(64), nullable=False)
    features_json = Column(Text, nullable=False)
    prob_online = Column(Float, nullable=False)
    action_online = Column(String(16), nullable=False)
    prob_dl = Column(Float, nullable=True)
    action_dl = Column(String(16), nullable=True)
    base_price = Column(Float, nullable=False)
    status = Column(String(16), nullable=False, default="pending", index=True)  # 'pending' | 'resolved' | 'expired'

    outcome = relationship("OutcomeRecord", back_populates="prediction", uselist=False, cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_predictions_ticker_time", "ticker", "timestamp"),
    )


class OutcomeRecord(Base):
    __tablename__ = "outcomes"

    prediction_id = Column(String(64), ForeignKey("predictions.id", ondelete="CASCADE"), primary_key=True)
    realized_timestamp = Column(String(64), nullable=False)
    realized_price = Column(Float, nullable=False)
    price_change_pct = Column(Float, nullable=False)
    realized_label = Column(Integer, nullable=False)  # 1 = Up, 0 = Flat/Down
    correct_online = Column(Integer, nullable=False)
    correct_dl = Column(Integer, nullable=True)
    pnl_pct = Column(Float, nullable=True)

    prediction = relationship("PredictionRecord", back_populates="outcome")


class SentimentRecord(Base):
    __tablename__ = "sentiment_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(String(64), nullable=False, index=True)
    ticker = Column(String(32), nullable=True, index=True)
    sector = Column(String(32), nullable=True, index=True)
    sentiment_score = Column(Float, nullable=False)
    sentiment_delta = Column(Float, nullable=False, default=0.0)
    article_count = Column(Integer, nullable=False, default=1)

    __table_args__ = (
        Index("idx_sentiment_ticker_time", "ticker", "timestamp"),
        Index("idx_sentiment_sector_time", "sector", "timestamp"),
    )


class SeenArticleRecord(Base):
    __tablename__ = "seen_articles"

    article_hash = Column(String(64), primary_key=True, index=True)
    title = Column(Text, nullable=False)
    source = Column(String(128), nullable=False)
    published_at = Column(String(64), nullable=True)
    scraped_at = Column(String(64), nullable=False)


class DriftEventRecord(Base):
    __tablename__ = "drift_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(String(64), nullable=False, index=True)
    metric_name = Column(String(64), nullable=False)
    value_before = Column(Float, nullable=True)
    value_after = Column(Float, nullable=True)
    message = Column(Text, nullable=False)


# ── Initialization & Session Utilities ────────────────────────────────────────

def init_db():
    """Initializes all database tables if they do not already exist."""
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency for obtaining a thread-safe database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def check_db_health() -> Dict[str, Any]:
    """Tests DB connectivity and returns dialect and status."""
    try:
        with engine.connect() as conn:
            conn.execute(Base.metadata.tables["predictions"].select().limit(1))
        return {
            "reachable": True,
            "dialect": engine.dialect.name,
            "database_url_type": "postgresql" if not is_sqlite else "sqlite",
        }
    except Exception as e:
        return {
            "reachable": False,
            "dialect": engine.dialect.name,
            "error": str(e),
        }
