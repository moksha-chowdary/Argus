"""
ARGUS V5 — Persistent Feature Store & Audit Database
Stores predictions, realized outcomes, point-in-time sentiment, seen news articles,
and ADWIN concept drift events using SQLite for auditability and leakage prevention.
"""

import json
import sqlite3
import os
import threading
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Tuple
import pandas as pd

from config import BASE_DIR

DEFAULT_DB_PATH = os.path.join(BASE_DIR, "data", "argus_v5.db")


class FeatureStore:
    """
    Central storage for ARGUS V5:
    - Predictions & realized 5-minute outcomes
    - Historical point-in-time sentiment (leakage prevention)
    - Deduplicated news article cache
    - Concept drift alerts
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _init_db(self):
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()

            # 1. Predictions Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                ticker TEXT NOT NULL,
                features_asof TEXT NOT NULL,
                features_json TEXT NOT NULL,
                prob_online REAL NOT NULL,
                action_online TEXT NOT NULL,
                prob_dl REAL,
                action_dl TEXT,
                base_price REAL NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending' -- 'pending' | 'resolved' | 'expired'
            );
            """)

            cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_predictions_ticker_time
            ON predictions(ticker, timestamp);
            """)
            cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_predictions_status
            ON predictions(status);
            """)

            # 2. Outcomes Table (Resolved at T+5min)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS outcomes (
                prediction_id TEXT PRIMARY KEY,
                realized_timestamp TEXT NOT NULL,
                realized_price REAL NOT NULL,
                price_change_pct REAL NOT NULL,
                realized_label INTEGER NOT NULL, -- 1 = Upward (P_{t+5} > P_t), 0 = Downward/Flat
                correct_online INTEGER NOT NULL,
                correct_dl INTEGER,
                pnl_pct REAL,
                FOREIGN KEY(prediction_id) REFERENCES predictions(id) ON DELETE CASCADE
            );
            """)

            # 3. Sentiment History Table (Point-in-time auditability)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS sentiment_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                ticker TEXT,
                sector TEXT,
                sentiment_score REAL NOT NULL,
                sentiment_delta REAL NOT NULL DEFAULT 0.0,
                article_count INTEGER NOT NULL DEFAULT 1
            );
            """)
            cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_sentiment_ticker_time
            ON sentiment_history(ticker, timestamp);
            """)
            cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_sentiment_sector_time
            ON sentiment_history(sector, timestamp);
            """)

            # 4. Seen Articles Table (Deduplication)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS seen_articles (
                article_hash TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                source TEXT NOT NULL,
                published_at TEXT,
                scraped_at TEXT NOT NULL
            );
            """)

            # 5. Concept Drift Events Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS drift_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                value_before REAL,
                value_after REAL,
                message TEXT NOT NULL
            );
            """)

            conn.commit()

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
    ):
        """Log a new prediction made at time T with auditable features_asof."""
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO predictions (
                id, timestamp, ticker, features_asof, features_json,
                prob_online, action_online, prob_dl, action_dl, base_price, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending');
            """, (
                pred_id,
                timestamp,
                ticker.upper(),
                features_asof,
                json.dumps(features),
                float(prob_online),
                action_online,
                float(prob_dl) if prob_dl is not None else None,
                action_dl,
                float(base_price),
            ))
            conn.commit()

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
        Returns outcome details or None if prediction not found.
        """
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM predictions WHERE id = ?;", (pred_id,))
            row = cursor.fetchone()
            if not row:
                return None

            base_price = float(row["base_price"])
            price_change = realized_price - base_price
            price_change_pct = (price_change / base_price) * 100.0 if base_price > 0 else 0.0

            # Ground truth binary label: 1 for up, 0 for flat/down
            realized_label = 1 if realized_price > base_price else 0

            # Did the model correctly anticipate direction?
            # Model predicts up if prob_online >= 0.5
            pred_up_online = 1 if row["prob_online"] >= 0.5 else 0
            correct_online = 1 if pred_up_online == realized_label else 0

            correct_dl = None
            if row["prob_dl"] is not None:
                pred_up_dl = 1 if row["prob_dl"] >= 0.5 else 0
                correct_dl = 1 if pred_up_dl == realized_label else 0

            # Trading PnL based on executed action
            pnl_pct = 0.0
            if row["action_online"] == "BUY":
                pnl_pct = price_change_pct
            elif row["action_online"] == "SELL":
                pnl_pct = -price_change_pct

            cursor.execute("""
            INSERT OR REPLACE INTO outcomes (
                prediction_id, realized_timestamp, realized_price,
                price_change_pct, realized_label, correct_online, correct_dl, pnl_pct
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                pred_id,
                realized_timestamp,
                float(realized_price),
                round(price_change_pct, 4),
                realized_label,
                correct_online,
                correct_dl,
                round(pnl_pct, 4),
            ))

            cursor.execute("UPDATE predictions SET status = 'resolved' WHERE id = ?;", (pred_id,))
            conn.commit()

            features = json.loads(row["features_json"])
            return {
                "prediction_id": pred_id,
                "ticker": row["ticker"],
                "features_asof": row["features_asof"],
                "features": features,
                "base_price": base_price,
                "realized_price": realized_price,
                "price_change_pct": price_change_pct,
                "realized_label": realized_label,
                "prob_online": row["prob_online"],
                "action_online": row["action_online"],
                "correct_online": bool(correct_online),
                "prob_dl": row["prob_dl"],
                "action_dl": row["action_dl"],
                "correct_dl": bool(correct_dl) if correct_dl is not None else None,
                "pnl_pct": pnl_pct,
            }

    def get_pending_predictions(self, ticker: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetch all unresolved predictions awaiting realization at T+5min."""
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            if ticker:
                cursor.execute(
                    "SELECT * FROM predictions WHERE status = 'pending' AND ticker = ? ORDER BY timestamp ASC;",
                    (ticker.upper(),),
                )
            else:
                cursor.execute(
                    "SELECT * FROM predictions WHERE status = 'pending' ORDER BY timestamp ASC;"
                )
            rows = cursor.fetchall()
            results = []
            for r in rows:
                results.append({
                    "id": r["id"],
                    "timestamp": r["timestamp"],
                    "ticker": r["ticker"],
                    "features_asof": r["features_asof"],
                    "features": json.loads(r["features_json"]),
                    "prob_online": r["prob_online"],
                    "action_online": r["action_online"],
                    "base_price": r["base_price"],
                    "prob_dl": r["prob_dl"],
                    "action_dl": r["action_dl"],
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
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO sentiment_history (
                timestamp, ticker, sector, sentiment_score, sentiment_delta, article_count
            ) VALUES (?, ?, ?, ?, ?, ?);
            """, (
                timestamp,
                ticker.upper() if ticker else None,
                sector.upper() if sector else None,
                float(sentiment_score),
                float(sentiment_delta),
                int(article_count),
            ))
            conn.commit()

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

        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            row = None
            if ticker:
                cursor.execute("""
                SELECT * FROM sentiment_history
                WHERE ticker = ? AND timestamp <= ?
                ORDER BY timestamp DESC LIMIT 1;
                """, (ticker.upper(), asof_time))
                row = cursor.fetchone()

            if not row and sector:
                cursor.execute("""
                SELECT * FROM sentiment_history
                WHERE sector = ? AND timestamp <= ?
                ORDER BY timestamp DESC LIMIT 1;
                """, (sector.upper(), asof_time))
                row = cursor.fetchone()

            if not row:
                # Return neutral baseline
                return {
                    "sentiment_score": 0.0,
                    "sentiment_delta": 0.0,
                    "sentiment_asof": asof_time,
                    "article_count": 0,
                    "is_fallback": True,
                }

            return {
                "sentiment_score": float(row["sentiment_score"]),
                "sentiment_delta": float(row["sentiment_delta"]),
                "sentiment_asof": row["timestamp"],
                "article_count": int(row["article_count"]),
                "is_fallback": False,
            }

    # ── Seen Articles Deduplication ───────────────────────────────────────────

    def is_article_seen(self, article_hash: str) -> bool:
        """Check if an article hash has already been scored."""
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM seen_articles WHERE article_hash = ? LIMIT 1;", (article_hash,))
            return cursor.fetchone() is not None

    def mark_article_seen(
        self,
        article_hash: str,
        title: str,
        source: str,
        published_at: Optional[str],
        scraped_at: str,
    ):
        """Mark article as processed in SQLite seen_articles table."""
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT OR IGNORE INTO seen_articles (
                article_hash, title, source, published_at, scraped_at
            ) VALUES (?, ?, ?, ?, ?);
            """, (article_hash, title, source, published_at, scraped_at))
            conn.commit()

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
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO drift_events (
                timestamp, metric_name, value_before, value_after, message
            ) VALUES (?, ?, ?, ?, ?);
            """, (timestamp, metric_name, value_before, value_after, message))
            conn.commit()

    def get_drift_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve recent drift events."""
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT * FROM drift_events ORDER BY timestamp DESC LIMIT ?;
            """, (limit,))
            return [dict(r) for r in cursor.fetchall()]

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
        with self._lock, self._get_connection() as conn:
            query = """
            SELECT
                p.id, p.timestamp, p.ticker, p.features_asof, p.features_json,
                p.prob_online, p.action_online, p.prob_dl, p.action_dl, p.base_price,
                o.realized_timestamp, o.realized_price, o.price_change_pct,
                o.realized_label, o.correct_online, o.correct_dl, o.pnl_pct
            FROM predictions p
            JOIN outcomes o ON p.id = o.prediction_id
            """
            params = []
            if ticker:
                query += " WHERE p.ticker = ? "
                params.append(ticker.upper())

            query += " ORDER BY p.timestamp ASC "
            if limit:
                query += f" LIMIT {int(limit)} "

            df = pd.read_sql_query(query, conn, params=params)

        if df.empty:
            return df

        # Parse features_json into structured columns
        features_list = df["features_json"].apply(json.loads).tolist()
        feat_df = pd.DataFrame(features_list)
        # Prefix feature columns
        feat_df = feat_df.add_prefix("feat_")

        merged = pd.concat([df.drop(columns=["features_json"]), feat_df], axis=1)
        return merged
