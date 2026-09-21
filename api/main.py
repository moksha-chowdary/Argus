"""
ARGUS V5 — Production REST API & WebSocket Streaming Service
Designed for persistent deployment on Railway / Render / Fly.io with a decoupled Vercel frontend.

Exposes:
- GET /api/health                   : Health status, DB reachable, scheduler state, last prediction/outcome
- GET /api/signals/{ticker}         : Real-time signal, P(Up), confidence band, reasoning, technicals
- GET /api/predictions/history      : Recent predictions + realized T+5 outcomes
- GET /api/backtest/{ticker}        : Auditable backtest metrics & time-series JSON (for frontend charts)
- GET /api/drift-events             : Concept drift log entries from ADWIN
- GET /api/news/sentiment/{ticker}  : FinBERT point-in-time sentiment score & history
- WS  /ws/signals/{ticker}          : Real-time WebSocket signal streaming
- GET /api/stream/signals           : Server-Sent Events (SSE) fallback stream
"""

import os
import json
import asyncio
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import pandas as pd
import yfinance as yf

from config import BASE_DIR, CAPITAL
from data.db import init_db, check_db_health, SessionLocal, PredictionRecord, OutcomeRecord, SentimentRecord
from data.feature_store import FeatureStore
from intelligence.online_learner import OnlineLearner
from intelligence.dl_retrainer import DeepLearningRetrainer
from intelligence.outcome_tracker import OutcomeTracker
from intelligence.brain import ArgusBrain, TradeSignal
from news.scheduler import start_news_scheduler

# In-memory backtest cache to serve frontend charts instantly
BACKTEST_CACHE: Dict[str, Dict[str, Any]] = {}
ACTIVE_WEBSOCKETS: Dict[str, List[WebSocket]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Persistent process lifespan manager:
    - Runs DB migrations/table setup
    - Holds persistent online learner and neural net in memory
    - Starts the 4x daily FinBERT news scheduler
    - Starts the T+5 outcome resolution loop
    """
    print("\n[ARGUS V5 API] Initializing backend services...")
    init_db()

    feature_store = FeatureStore()
    online_learner = OnlineLearner(feature_store=feature_store)
    dl_retrainer = DeepLearningRetrainer(feature_store=feature_store)
    outcome_tracker = OutcomeTracker(
        online_learner=online_learner,
        feature_store=feature_store,
        buy_threshold=0.58,
        sell_threshold=0.42,
    )
    brain = ArgusBrain(
        feature_store=feature_store,
        online_learner=online_learner,
        dl_retrainer=dl_retrainer,
        outcome_tracker=outcome_tracker,
    )

    # Start persistent background services
    news_scheduler = start_news_scheduler(feature_store=feature_store)
    outcome_tracker.start_background_tracker(interval_sec=30)

    # Attach to app state for endpoint access
    app.state.feature_store = feature_store
    app.state.online_learner = online_learner
    app.state.dl_retrainer = dl_retrainer
    app.state.outcome_tracker = outcome_tracker
    app.state.brain = brain
    app.state.news_scheduler = news_scheduler
    app.state.start_time = datetime.now(timezone.utc)

    print("[ARGUS V5 API] Persistent backend services initialized successfully.\n")
    yield

    print("[ARGUS V5 API] Shutting down background services...")
    if news_scheduler and news_scheduler.running:
        news_scheduler.shutdown(wait=False)
    outcome_tracker.stop_background_tracker()


app = FastAPI(
    title="ARGUS V5 Market Intelligence API",
    description="Probabilistic ML/DL Intraday Directional Forecasting for NSE Equities",
    version="5.0.0",
    lifespan=lifespan,
)

# ── Cross-Origin Resource Sharing (CORS) ──────────────────────────────────────
# Configured for Next.js / Vite frontends hosted on Vercel or localhost
cors_origins_env = os.getenv("CORS_ORIGINS", "*")
allowed_origins = [o.strip() for o in cors_origins_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins if allowed_origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health Check Endpoint ─────────────────────────────────────────────────────

@app.get("/api/health", tags=["System"])
def get_health() -> Dict[str, Any]:
    """
    Comprehensive health check:
    - Database reachability & dialect (PostgreSQL vs SQLite)
    - APScheduler daemon status
    - In-memory online learner statistics
    - Last successful prediction and outcome resolution timestamps
    """
    now = datetime.now(timezone.utc)
    uptime_seconds = (now - app.state.start_time).total_seconds()
    db_health = check_db_health()

    # Query last prediction and outcome timestamps
    last_pred_time = None
    last_outcome_time = None
    try:
        with SessionLocal() as session:
            latest_pred = session.query(PredictionRecord).order_by(PredictionRecord.timestamp.desc()).first()
            if latest_pred:
                last_pred_time = latest_pred.timestamp

            latest_outcome = session.query(OutcomeRecord).order_by(OutcomeRecord.realized_timestamp.desc()).first()
            if latest_outcome:
                last_outcome_time = latest_outcome.realized_timestamp
    except Exception:
        pass

    scheduler_running = bool(
        app.state.news_scheduler and app.state.news_scheduler.running
    )

    return {
        "status": "healthy" if db_health.get("reachable") else "degraded",
        "timestamp": now.isoformat(),
        "uptime_seconds": round(uptime_seconds, 1),
        "database": db_health,
        "scheduler": {
            "running": scheduler_running,
            "jobs": [j.name for j in app.state.news_scheduler.get_jobs()] if scheduler_running else [],
        },
        "online_learner": {
            "model_type": "HoeffdingAdaptiveTreeClassifier",
            "samples_seen": app.state.online_learner.samples_seen,
            "drifts_detected": app.state.online_learner.drift_count,
        },
        "telemetry": {
            "last_prediction_timestamp": last_pred_time,
            "last_outcome_timestamp": last_outcome_time,
        },
    }


# ── Live Market Signal Endpoint ───────────────────────────────────────────────

@app.get("/api/signals/{ticker}", tags=["Signals"])
async def get_live_signal(ticker: str) -> Dict[str, Any]:
    """
    Generates real-time probabilistic trading signal for a given NSE ticker:
    - Calibrated upward probability P(Up) from River online tree
    - Deep learning LSTM probability
    - Conviction band (BUY >= 0.58, SELL <= 0.42, WAIT: 0.42-0.58)
    - Technical indicators (RSI, EMA 20/50, MACD)
    - Auditable reasoning points and point-in-time FinBERT news score
    """
    clean_ticker = ticker.upper()
    if not clean_ticker.endswith(".NS"):
        clean_ticker = f"{clean_ticker}.NS"

    try:
        # Download recent 5-minute intraday bars
        df = yf.download(clean_ticker, period="5d", interval="5m", progress=False, auto_adjust=True)
        if hasattr(df.columns, "get_level_values"):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna()

        if len(df) < 25:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient intraday candle history returned for {clean_ticker} (got {len(df)} bars)",
            )

        # Generate signal via ARGUS Brain
        signal: TradeSignal = app.state.brain.generate_signal(
            ticker=clean_ticker,
            candle_df=df,
        )

        curr_price = float(df["Close"].iloc[-1])
        sentiment = app.state.feature_store.get_latest_sentiment(ticker=clean_ticker)

        response = {
            "ticker": clean_ticker,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "signal": signal.action,
            "conviction": signal.conviction,
            "confidence_score": signal.confidence,
            "probability_up": round(signal.prob_online, 4),
            "probability_dl": round(signal.prob_dl, 4),
            "confidence_band": {
                "buy_threshold": 0.58,
                "sell_threshold": 0.42,
                "wait_band": [0.42, 0.58],
                "in_wait_zone": 0.42 < signal.prob_online < 0.58,
            },
            "pricing": {
                "current_price": round(curr_price, 2),
                "entry_price": round(signal.entry_price, 2),
                "target_price": round(signal.target_price, 2),
                "stop_price": round(signal.stop_price, 2),
                "risk_reward_ratio": signal.rr_ratio,
            },
            "technical_indicators": {
                "rsi": round(signal.rsi, 2) if signal.rsi else None,
                "ema_20": round(signal.ema_20, 2) if signal.ema_20 else None,
                "ema_50": round(signal.ema_50, 2) if signal.ema_50 else None,
                "macd_cross": signal.macd_cross,
            },
            "news_sentiment": {
                "score": sentiment["sentiment_score"],
                "delta": sentiment["sentiment_delta"],
                "asof": sentiment["sentiment_asof"],
                "is_fallback": sentiment["is_fallback"],
            },
            "reasoning": signal.matched_rules + ([signal.final_advice] if signal.final_advice else []),
            "warnings": signal.warnings,
            "drift_status": signal.drift_status,
            "prediction_id": signal.prediction_id,
        }

        # Broadcast update to connected WebSockets
        asyncio.create_task(_broadcast_signal(clean_ticker, response))
        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Signal generation error for {clean_ticker}: {str(e)}")


# ── Historical Predictions & Realized Outcomes ────────────────────────────────

@app.get("/api/predictions/history", tags=["Predictions"])
def get_prediction_history(
    ticker: Optional[str] = None,
    limit: int = Query(50, ge=1, le=500),
) -> List[Dict[str, Any]]:
    """
    Returns recent predictions with matched realized T+5 outcomes and PnL:
    Allows the frontend to audit past predictions and directional accuracy.
    """
    clean_ticker = ticker.upper() if ticker else None
    if clean_ticker and not clean_ticker.endswith(".NS"):
        clean_ticker = f"{clean_ticker}.NS"

    with SessionLocal() as session:
        query = (
            session.query(PredictionRecord)
            .outerjoin(OutcomeRecord, PredictionRecord.id == OutcomeRecord.prediction_id)
        )
        if clean_ticker:
            query = query.filter(PredictionRecord.ticker == clean_ticker)

        records = query.order_by(PredictionRecord.timestamp.desc()).limit(limit).all()

        results = []
        for p in records:
            item = {
                "prediction_id": p.id,
                "timestamp": p.timestamp,
                "ticker": p.ticker,
                "prob_online": p.prob_online,
                "action_online": p.action_online,
                "base_price": p.base_price,
                "status": p.status,
                "realized": None,
            }
            if p.outcome:
                item["realized"] = {
                    "timestamp": p.outcome.realized_timestamp,
                    "price": p.outcome.realized_price,
                    "change_pct": p.outcome.price_change_pct,
                    "label": p.outcome.realized_label,
                    "correct": bool(p.outcome.correct_online),
                    "pnl_pct": p.outcome.pnl_pct,
                }
            results.append(item)

        return results


# ── Auditable Backtest Metrics & Chart Data (JSON) ───────────────────────────

@app.get("/api/backtest/{ticker}", tags=["Backtesting"])
def get_backtest_data(
    ticker: str,
    period: str = "60d",
) -> Dict[str, Any]:
    """
    Returns latest backtest performance metrics and time-series chart data as JSON:
    Designed so Next.js / Vite frontends can render interactive charts (Recharts / Chart.js)
    without server-side image rendering.
    """
    clean_ticker = ticker.upper()
    if not clean_ticker.endswith(".NS"):
        clean_ticker = f"{clean_ticker}.NS"

    cache_key = f"{clean_ticker}_{period}"
    if cache_key in BACKTEST_CACHE:
        return BACKTEST_CACHE[cache_key]

    try:
        from evaluation.backtest import WalkForwardBacktest
        engine = WalkForwardBacktest()

        df = yf.download(clean_ticker, period=period, interval="5m", progress=False, auto_adjust=True)
        if hasattr(df.columns, "get_level_values"):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna()

        if len(df) < 150:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient candle history for backtest on {clean_ticker} (got {len(df)} bars)",
            )

        report, res_df = engine.run_single(df, ticker=clean_ticker, warmup_bars=80, run_dl=False)

        # Downsample time-series to max 250 points for efficient JSON transfer
        step = max(1, len(res_df) // 250)
        downsampled = res_df.iloc[::step]

        equity_curve = [
            {
                "time": str(row["time"]),
                "equity_strategy": round(float(row["eq_online"]), 4),
                "equity_benchmark": round(float(row["eq_bh"]), 4),
            }
            for _, row in downsampled.iterrows()
        ]

        # 50-bar rolling accuracy series
        roll_acc = (res_df["pred_online"] == res_df["y_true"]).rolling(50).mean() * 100.0
        rolling_accuracy = [
            {
                "time": str(res_df["time"].iloc[i]),
                "accuracy": round(float(roll_acc.iloc[i]), 2),
            }
            for i in range(49, len(res_df), step)
            if not pd.isna(roll_acc.iloc[i])
        ]

        # Probability calibration diagram data (10 bins)
        probs = res_df["prob_online"].values
        y_true = res_df["y_true"].values
        calibration = []
        for b_low, b_high in [(i / 10.0, (i + 1) / 10.0) for i in range(10)]:
            mask = (probs >= b_low) & (probs < b_high)
            if mask.sum() > 0:
                calibration.append({
                    "bin_center": round((b_low + b_high) / 2.0, 2),
                    "predicted_prob": round(float(probs[mask].mean()), 3),
                    "empirical_prob": round(float(y_true[mask].mean()), 3),
                    "sample_count": int(mask.sum()),
                })

        payload = {
            "ticker": clean_ticker,
            "period": period,
            "bars_evaluated": report["bars_evaluated"],
            "directional_accuracy": report["directional_accuracy"],
            "financial_performance": report["financial_performance"],
            "classification_metrics": report["classification_metrics"],
            "charts": {
                "equity_curve": equity_curve,
                "rolling_accuracy": rolling_accuracy,
                "calibration": calibration,
            },
        }

        # Cache result
        BACKTEST_CACHE[cache_key] = payload
        return payload

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backtest error on {clean_ticker}: {str(e)}")


# ── Concept Drift Events ──────────────────────────────────────────────────────

@app.get("/api/drift-events", tags=["Monitoring"])
def get_drift_events(limit: int = Query(50, ge=1, le=200)) -> List[Dict[str, Any]]:
    """Returns recent ADWIN concept drift detections."""
    return app.state.feature_store.get_drift_events(limit=limit)


# ── News Sentiment Telemetry ──────────────────────────────────────────────────

@app.get("/api/news/sentiment/{ticker}", tags=["News & Sentiment"])
def get_news_sentiment(
    ticker: str,
    limit: int = Query(20, ge=1, le=100),
) -> Dict[str, Any]:
    """Returns latest FinBERT sentiment and point-in-time historical records."""
    clean_ticker = ticker.upper()
    if not clean_ticker.endswith(".NS"):
        clean_ticker = f"{clean_ticker}.NS"

    latest = app.state.feature_store.get_latest_sentiment(ticker=clean_ticker)

    with SessionLocal() as session:
        history_rows = (
            session.query(SentimentRecord)
            .filter(SentimentRecord.ticker == clean_ticker)
            .order_by(SentimentRecord.timestamp.desc())
            .limit(limit)
            .all()
        )
        history = [
            {
                "timestamp": r.timestamp,
                "score": r.sentiment_score,
                "delta": r.sentiment_delta,
                "article_count": r.article_count,
            }
            for r in history_rows
        ]

    return {
        "ticker": clean_ticker,
        "latest": latest,
        "history": history,
    }


# ── Real-Time WebSocket Streaming ─────────────────────────────────────────────

@app.websocket("/ws/signals/{ticker}")
async def websocket_signals(websocket: WebSocket, ticker: str):
    """
    WebSocket endpoint for real-time signal streaming to browser clients.
    Broadcasts new signal ticks whenever generated.
    """
    clean_ticker = ticker.upper()
    if not clean_ticker.endswith(".NS"):
        clean_ticker = f"{clean_ticker}.NS"

    await websocket.accept()
    if clean_ticker not in ACTIVE_WEBSOCKETS:
        ACTIVE_WEBSOCKETS[clean_ticker] = []
    ACTIVE_WEBSOCKETS[clean_ticker].append(websocket)

    try:
        # Send initial confirmation message
        await websocket.send_json({
            "event": "connected",
            "ticker": clean_ticker,
            "message": f"Subscribed to real-time signals for {clean_ticker}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        while True:
            # Keep connection alive with heartbeat
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        ACTIVE_WEBSOCKETS[clean_ticker].remove(websocket)
    except Exception:
        if websocket in ACTIVE_WEBSOCKETS.get(clean_ticker, []):
            ACTIVE_WEBSOCKETS[clean_ticker].remove(websocket)


async def _broadcast_signal(ticker: str, payload: Dict[str, Any]):
    """Helper to broadcast generated signal to active WebSockets."""
    clients = ACTIVE_WEBSOCKETS.get(ticker, [])
    dead_clients = []
    for ws in clients:
        try:
            await ws.send_json({"event": "signal_update", "data": payload})
        except Exception:
            dead_clients.append(ws)
    for ws in dead_clients:
        if ws in clients:
            clients.remove(ws)


# ── Server-Sent Events (SSE) Streaming Fallback ───────────────────────────────

@app.get("/api/stream/signals", tags=["Signals"])
async def stream_signals(ticker: str = "RELIANCE.NS"):
    """
    Server-Sent Events (SSE) endpoint for environments or firewalls
    where WebSockets are unavailable or restricted.
    """
    clean_ticker = ticker.upper()
    if not clean_ticker.endswith(".NS"):
        clean_ticker = f"{clean_ticker}.NS"

    async def event_generator():
        while True:
            try:
                # Query latest signal
                sig = get_live_signal(clean_ticker)
                yield f"data: {json.dumps(sig)}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
            await asyncio.sleep(5)  # 5-second interval

    return StreamingResponse(event_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    print(f"Starting ARGUS V5 API Server on {host}:{port}...")
    uvicorn.run("api.main:app", host=host, port=port, reload=False)
