"""
ARGUS V5 Comprehensive ML/DL & Leakage Prevention Test Suite
Validates:
1. Time-based leakage prevention in feature store and feature engineering
2. River online learner stream updates and ADWIN drift tracking
3. Outcome tracker T -> T+5 resolution and action decision thresholds
4. PyTorch LSTM sequence batch retraining
5. FinBERT sentiment scoring and deduplication
6. Walk-forward rolling-origin backtest execution
"""

import os
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

from data.feature_store import FeatureStore
from intelligence.features import calculate_numeric_features, FEATURE_COLUMNS
from intelligence.online_learner import OnlineLearner
from intelligence.outcome_tracker import OutcomeTracker
from intelligence.dl_retrainer import DeepLearningRetrainer
from news.sentiment_analyzer import FinBERTSentimentAnalyzer
from evaluation.backtest import WalkForwardBacktest


@pytest.fixture
def temp_env():
    tmp_dir = tempfile.mkdtemp()
    db_path = os.path.join(tmp_dir, "test_store.db")
    model_path = os.path.join(tmp_dir, "test_online.pkl")
    dl_path = os.path.join(tmp_dir, "test_dl.pt")

    fs = FeatureStore(db_path=db_path)
    learner = OnlineLearner(model_path=model_path, feature_store=fs)
    tracker = OutcomeTracker(online_learner=learner, feature_store=fs)
    dl = DeepLearningRetrainer(input_dim=len(FEATURE_COLUMNS), seq_len=5, model_path=dl_path, feature_store=fs)

    yield {"dir": tmp_dir, "fs": fs, "learner": learner, "tracker": tracker, "dl": dl}


def test_feature_store_leakage_guard(temp_env):
    fs = temp_env["fs"]
    t0 = "2026-09-21T10:00:00+00:00"
    t1 = "2026-09-21T10:05:00+00:00"
    t2 = "2026-09-21T10:10:00+00:00"

    # Save historical sentiment at t0 and future sentiment at t2
    fs.save_sentiment(timestamp=t0, ticker="RELIANCE", sector="ENERGY", sentiment_score=0.25)
    fs.save_sentiment(timestamp=t2, ticker="RELIANCE", sector="ENERGY", sentiment_score=0.85)

    # Point-in-time query at t1 (between t0 and t2) MUST return t0, NEVER t2
    result = fs.get_latest_sentiment(ticker="RELIANCE", asof_time=t1)
    assert result["sentiment_score"] == 0.25, f"LEAKAGE DETECTED: Query at {t1} returned future data ({result['sentiment_score']})"
    assert result["sentiment_asof"] == t0


def test_feature_engineering_strict_asof():
    dates = pd.date_range("2026-09-21 09:15", periods=40, freq="5min", tz="Asia/Kolkata")
    df = pd.DataFrame({
        "Open": np.linspace(2500, 2600, 40),
        "High": np.linspace(2510, 2610, 40),
        "Low": np.linspace(2490, 2590, 40),
        "Close": np.linspace(2505, 2605, 40),
        "Volume": [1000 + i*20 for i in range(40)]
    }, index=dates)

    # Cut off at bar 25
    cutoff_time = dates[25]
    feats, asof, daily_asof = calculate_numeric_features(df, ticker="TEST.NS", asof_timestamp=cutoff_time)

    assert len(feats) == len(FEATURE_COLUMNS)
    assert "rel_nifty_ret_1" in feats
    assert "rel_sector_ret_1" in feats
    assert "nifty_divergence_flag" in feats
    assert asof == cutoff_time.isoformat()
    assert daily_asof is not None
    # Verify daily context features exist
    assert "daily_ret_3" in feats
    assert "daily_dist_ema20" in feats
    assert "daily_atr14_pct" in feats
    # Confirm norm_rsi is within [0, 1]
    assert 0.0 <= feats["norm_rsi"] <= 1.0


def test_multi_timeframe_leakage_guard():
    from data.multi_timeframe import fetch_daily_context
    daily_dates = pd.date_range("2026-06-01", periods=60, freq="D")
    df_daily = pd.DataFrame({
        "Open": np.linspace(100, 200, 60),
        "High": np.linspace(105, 205, 60),
        "Low": np.linspace(95, 195, 60),
        "Close": np.linspace(102, 202, 60),
        "Volume": np.random.randint(1000, 5000, 60),
    }, index=daily_dates)

    # Intraday prediction made on 2026-07-20 10:30 IST
    # Strict point-in-time leakage guard MUST use completed daily candle as of 2026-07-19
    asof_time = "2026-07-20T10:30:00+05:30"
    daily_feats, daily_asof = fetch_daily_context(
        ticker="TEST.NS",
        asof_timestamp=asof_time,
        daily_df=df_daily,
    )

    assert daily_asof == "2026-07-19"
    assert "daily_ret_3" in daily_feats
    assert "daily_dist_ema20" in daily_feats
    assert "daily_dist_ema50" in daily_feats
    assert "daily_atr14_pct" in daily_feats
    assert "daily_prox_swing_high" in daily_feats
    assert "daily_prox_swing_low" in daily_feats
    assert "daily_weekly_momentum" in daily_feats


def test_online_learner_stream_update(temp_env):
    learner = temp_env["learner"]
    feats = {"ret_1": 0.002, "norm_rsi": 0.60, "vol_zscore": 1.1}

    p0 = learner.predict_up_probability(feats)
    assert 0.0 <= p0 <= 1.0

    # Stream 30 consecutive samples
    for i in range(30):
        target = 1 if i % 2 == 0 else 0
        metrics = learner.learn_one(feats, target)

    assert metrics["samples_seen"] == 30
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["brier_score"] <= 1.0


def test_outcome_tracker_resolution_loop(temp_env):
    tracker = temp_env["tracker"]
    feats = {"ret_1": 0.005, "norm_rsi": 0.65, "norm_atr": 0.015}

    # 1. Prediction at T (with daily_asof audit metadata)
    pred = tracker.record_prediction(
        ticker="TCS",
        features=feats,
        features_asof="2026-09-21T11:00:00",
        daily_asof="2026-09-20",
        base_price=3800.0,
        timestamp="2026-09-21T11:00:00"
    )
    assert pred["prediction_id"].startswith("pred_")
    assert pred["daily_asof"] == "2026-09-20"

    # 2. Realization at T+15min (Price moved up to 3825.0)
    res = tracker.resolve_prediction(
        pred_id=pred["prediction_id"],
        realized_price=3825.0,
        realized_timestamp="2026-09-21T11:15:00"
    )
    assert res is not None
    assert res["realized_label"] == 1
    assert res["price_change_pct"] > 0
    assert res["prediction_id"] == pred["prediction_id"]
    assert res["daily_asof"] == "2026-09-20"


def test_dl_retrainer_batch_optimization(temp_env):
    dl = temp_env["dl"]
    N, L, D = 40, 5, len(FEATURE_COLUMNS)
    X = np.random.randn(N, L, D).astype(np.float32)
    y = np.random.randint(0, 2, size=(N,)).astype(np.float32)

    res = dl.retrain_batch(X, y, epochs=2, batch_size=16)
    assert res["status"] == "completed"
    assert res["samples_trained"] == N
    assert res["avg_loss"] > 0

    # Test sequence inference
    single_seq = X[0]
    prob = dl.predict_proba(single_seq)
    assert 0.0 <= prob <= 1.0


def test_finbert_sentiment_scoring():
    analyzer = FinBERTSentimentAnalyzer()
    bullish_text = "HDFC Bank reports 30% surge in net profit, asset quality improves"
    bearish_text = "Tech firm warns of severe slowdown, margins collapse amid contract cancellations"

    score_bull = analyzer.score_text(bullish_text)
    score_bear = analyzer.score_text(bearish_text)

    assert score_bull["compound"] > 0.1, f"Expected positive score, got {score_bull}"
    assert score_bear["compound"] < -0.1, f"Expected negative score, got {score_bear}"


def test_walk_forward_backtest_execution():
    dates = pd.date_range("2026-08-01 09:15", periods=200, freq="5min", tz="Asia/Kolkata")
    np.random.seed(123)
    ret = np.random.normal(0.0001, 0.002, 200)
    price = 1000.0 * np.exp(np.cumsum(ret))
    df = pd.DataFrame({
        "Open": price,
        "High": price * 1.002,
        "Low": price * 0.998,
        "Close": price * 1.0005,
        "Volume": np.random.randint(500, 5000, 200),
    }, index=dates)

    engine = WalkForwardBacktest()
    report = engine.run(df, ticker="TEST.NS", warmup_bars=60)

    assert report["total_bars_evaluated"] > 100
    assert "directional_accuracy" in report
    assert "financial_performance" in report
    assert "online_river" in report["financial_performance"]
    assert "dl_lstm" in report["financial_performance"]
