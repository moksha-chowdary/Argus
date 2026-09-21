"""
ARGUS V5 — REST API & WebSocket Endpoint Unit Tests
Tests FastAPI lifespan, health checks, prediction history, drift events,
news sentiment, signal endpoints, and WebSocket subscriptions.
"""

import os
import json
import pytest
from fastapi.testclient import TestClient
import pandas as pd
import numpy as np

from api.main import app
from data.db import SessionLocal, PredictionRecord, OutcomeRecord, SentimentRecord, DriftEventRecord


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_api_health_check(client):
    """Test /api/health endpoint structure and database reachability."""
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()

    assert "status" in data
    assert data["status"] in ["healthy", "degraded"]
    assert "uptime_seconds" in data
    assert "database" in data
    assert data["database"]["reachable"] is True
    assert "scheduler" in data
    assert "online_learner" in data
    assert data["online_learner"]["model_type"] == "HoeffdingAdaptiveTreeClassifier"
    assert "telemetry" in data


def test_predictions_history_endpoint(client):
    """Test /api/predictions/history retrieves prediction-outcome pairs."""
    # Seed a sample record
    with SessionLocal() as session:
        pred = PredictionRecord(
            id="test-pred-api-001",
            timestamp="2026-09-21T10:00:00Z",
            ticker="RELIANCE.NS",
            features_asof="2026-09-21T10:00:00Z",
            features_json=json.dumps({"ret_1": 0.001}),
            prob_online=0.62,
            action_online="BUY",
            prob_dl=0.58,
            action_dl="BUY",
            base_price=2500.0,
            status="resolved",
        )
        session.merge(pred)

        outcome = OutcomeRecord(
            prediction_id="test-pred-api-001",
            realized_timestamp="2026-09-21T10:05:00Z",
            realized_price=2510.0,
            price_change_pct=0.40,
            realized_label=1,
            correct_online=1,
            correct_dl=1,
            pnl_pct=0.40,
        )
        session.merge(outcome)
        session.commit()

    resp = client.get("/api/predictions/history?ticker=RELIANCE.NS&limit=10")
    assert resp.status_code == 200
    items = resp.json()
    assert isinstance(items, list)
    assert len(items) >= 1

    matched = next((i for i in items if i["prediction_id"] == "test-pred-api-001"), None)
    assert matched is not None
    assert matched["ticker"] == "RELIANCE.NS"
    assert matched["action_online"] == "BUY"
    assert matched["realized"] is not None
    assert matched["realized"]["label"] == 1
    assert matched["realized"]["correct"] is True


def test_drift_events_endpoint(client):
    """Test /api/drift-events endpoint."""
    with SessionLocal() as session:
        drift = DriftEventRecord(
            timestamp="2026-09-21T11:00:00Z",
            metric_name="online_accuracy",
            value_before=0.55,
            value_after=0.48,
            message="ADWIN detected concept drift in model error rate.",
        )
        session.add(drift)
        session.commit()

    resp = client.get("/api/drift-events?limit=20")
    assert resp.status_code == 200
    events = resp.json()
    assert isinstance(events, list)
    assert len(events) >= 1
    assert "metric_name" in events[0]
    assert "message" in events[0]


def test_news_sentiment_endpoint(client):
    """Test /api/news/sentiment/{ticker} endpoint."""
    with SessionLocal() as session:
        sent = SentimentRecord(
            timestamp="2026-09-21T12:00:00Z",
            ticker="TCS.NS",
            sentiment_score=0.42,
            sentiment_delta=0.15,
            article_count=3,
        )
        session.add(sent)
        session.commit()

    resp = client.get("/api/news/sentiment/TCS.NS")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ticker"] == "TCS.NS"
    assert "latest" in data
    assert "history" in data
    assert isinstance(data["history"], list)


def test_websocket_signal_stream(client):
    """Test real-time WebSocket connection and subscription handshake."""
    with client.websocket_connect("/ws/signals/RELIANCE.NS") as ws:
        msg = ws.receive_json()
        assert msg["event"] == "connected"
        assert msg["ticker"] == "RELIANCE.NS"

        # Send ping, expect pong
        ws.send_text("ping")
        resp_text = ws.receive_text()
        assert resp_text == "pong"


def test_signals_endpoint_with_mock_data(client, monkeypatch):
    """Test /api/signals/{ticker} returns complete signal payload."""
    dates = pd.date_range("2026-09-21 09:15", periods=50, freq="5min")
    mock_df = pd.DataFrame({
        "Open": np.linspace(2500, 2550, 50),
        "High": np.linspace(2505, 2555, 50),
        "Low": np.linspace(2495, 2545, 50),
        "Close": np.linspace(2502, 2552, 50),
        "Volume": np.random.randint(1000, 5000, 50),
    }, index=dates)

    import yfinance as yf
    monkeypatch.setattr(yf, "download", lambda *args, **kwargs: mock_df)

    resp = client.get("/api/signals/RELIANCE.NS")
    assert resp.status_code == 200
    data = resp.json()

    assert data["ticker"] == "RELIANCE.NS"
    assert data["signal"] in ["BUY", "SELL", "WAIT"]
    assert "probability_up" in data
    assert 0.0 <= data["probability_up"] <= 1.0
    assert "confidence_band" in data
    assert data["confidence_band"]["buy_threshold"] == 0.58
    assert data["confidence_band"]["sell_threshold"] == 0.42
    assert "pricing" in data
    assert "technical_indicators" in data
    assert "reasoning" in data
    assert isinstance(data["reasoning"], list)


def test_backtest_endpoint_with_mock_data(client, monkeypatch):
    """Test /api/backtest/{ticker} returns JSON metrics and chart data."""
    dates = pd.date_range("2026-08-01 09:15", periods=200, freq="5min")
    ret = np.random.normal(0.0001, 0.002, 200)
    price = 2500.0 * np.exp(np.cumsum(ret))
    mock_df = pd.DataFrame({
        "Open": price,
        "High": price * 1.002,
        "Low": price * 0.998,
        "Close": price * 1.0005,
        "Volume": np.random.randint(500, 5000, 200),
    }, index=dates)

    import yfinance as yf
    monkeypatch.setattr(yf, "download", lambda *args, **kwargs: mock_df)

    resp = client.get("/api/backtest/RELIANCE.NS?period=5d")
    assert resp.status_code == 200
    data = resp.json()

    assert data["ticker"] == "RELIANCE.NS"
    assert "directional_accuracy" in data
    assert "financial_performance" in data
    assert "charts" in data
    assert "equity_curve" in data["charts"]
    assert len(data["charts"]["equity_curve"]) > 0
    assert "rolling_accuracy" in data["charts"]
    assert "calibration" in data["charts"]
