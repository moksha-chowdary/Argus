"""
ARGUS V5 — Quantitative Feature Engineering Engine
Transforms raw intraday OHLCV candles + point-in-time sentiment into standardized,
numeric feature vectors for the continuous online learner and deep learning models.

Enforces strict time-based leakage prevention:
- Slices OHLCV strictly up to the 'asof_timestamp'
- Sentiment is queried strictly with timestamp <= asof_timestamp
"""

import math
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd

from data.feature_store import FeatureStore


def calculate_numeric_features(
    df: pd.DataFrame,
    ticker: str,
    asof_timestamp: Optional[datetime | str] = None,
    feature_store: Optional[FeatureStore] = None,
    sector: Optional[str] = None,
) -> Tuple[Dict[str, float], str]:
    """
    Extracts continuous numeric features from OHLCV data up to asof_timestamp.

    Args:
        df: DataFrame with columns ['Open', 'High', 'Low', 'Close', 'Volume'] and DatetimeIndex.
        ticker: Stock symbol (e.g. 'RELIANCE.NS' or 'RELIANCE')
        asof_timestamp: Cut-off time for features. If None, uses df.index[-1].
        feature_store: FeatureStore instance to query point-in-time sentiment.
        sector: Sector name for sector sentiment fallback.

    Returns:
        (features_dict, features_asof_iso_string)
    """
    if df is None or len(df) < 20:
        raise ValueError(f"Insufficient candle history for feature extraction (got {0 if df is None else len(df)}, need >= 20)")

    clean_df = df.copy()

    # Ensure datetime index
    if not isinstance(clean_df.index, pd.DatetimeIndex):
        clean_df.index = pd.to_datetime(clean_df.index)

    # Sort chronologically
    clean_df = clean_df.sort_index()

    # Enforce strict time-based leakage prevention
    if asof_timestamp is not None:
        if isinstance(asof_timestamp, str):
            asof_dt = pd.to_datetime(asof_timestamp)
        else:
            asof_dt = asof_timestamp

        # Localize or strip timezone to match dataframe index
        if clean_df.index.tz is not None and asof_dt.tzinfo is None:
            asof_dt = asof_dt.replace(tzinfo=clean_df.index.tz)
        elif clean_df.index.tz is None and asof_dt.tzinfo is not None:
            asof_dt = asof_dt.tz_convert(None)

        # Strictly filter data to <= asof_dt
        clean_df = clean_df[clean_df.index <= asof_dt]
        if len(clean_df) < 20:
            raise ValueError(f"Insufficient history up to asof_timestamp {asof_dt}")

    # Canonical asof string
    last_candle_time = clean_df.index[-1]
    if hasattr(last_candle_time, "isoformat"):
        features_asof = last_candle_time.isoformat()
    else:
        features_asof = str(last_candle_time)

    # ── 1. Price Basics & Log Returns ─────────────────────────────────────────
    close = clean_df["Close"].astype(float)
    high = clean_df["High"].astype(float)
    low = clean_df["Low"].astype(float)
    open_p = clean_df["Open"].astype(float)
    volume = clean_df["Volume"].astype(float)

    curr_close = float(close.iloc[-1])
    curr_open = float(open_p.iloc[-1])
    curr_high = float(high.iloc[-1])
    curr_low = float(low.iloc[-1])
    curr_vol = float(volume.iloc[-1])

    # Multi-lag log returns
    # r_k = ln(C_t / C_{t-k})
    def safe_log_return(k: int) -> float:
        if len(close) > k and float(close.iloc[-k-1]) > 0:
            return float(np.log(curr_close / float(close.iloc[-k-1])))
        return 0.0

    ret_1 = safe_log_return(1)
    ret_3 = safe_log_return(3)
    ret_5 = safe_log_return(5)
    ret_15 = safe_log_return(15)

    # ── 2. Volatility & Price Action Dynamics ─────────────────────────────────
    # True Range (TR) & Average True Range (ATR 14)
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = float(tr.rolling(min(14, len(tr))).mean().iloc[-1])
    norm_atr = (atr_14 / curr_close) if curr_close > 0 else 0.0

    # Realized volatility over 10 periods
    rolling_std_10 = float(close.pct_change().rolling(min(10, len(close))).std().iloc[-1])
    if np.isnan(rolling_std_10):
        rolling_std_10 = 0.0

    # Candle geometry
    candle_range = curr_high - curr_low
    eps = 1e-8
    body_pct = (curr_close - curr_open) / (candle_range + eps)
    upper_wick_pct = (curr_high - max(curr_open, curr_close)) / (candle_range + eps)
    lower_wick_pct = (min(curr_open, curr_close) - curr_low) / (candle_range + eps)

    # ── 3. Oscillators: RSI, MACD, Bollinger Bands ─────────────────────────────
    # RSI (14) normalized to [0, 1]
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-9)
    rsi_val = float((100.0 - (100.0 / (1.0 + rs))).iloc[-1])
    norm_rsi = (rsi_val / 100.0) if not np.isnan(rsi_val) else 0.5

    # MACD (12, 26, 9) normalized by close price
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_diff = macd_line - macd_signal

    norm_macd = float(macd_line.iloc[-1] / (curr_close + eps))
    norm_macd_signal = float(macd_signal.iloc[-1] / (curr_close + eps))
    norm_macd_diff = float(macd_diff.iloc[-1] / (curr_close + eps))

    # Bollinger Bands (20, 2)
    bb_mid = close.rolling(min(20, len(close))).mean()
    bb_std = close.rolling(min(20, len(close))).std()
    bb_upper = bb_mid + 2.0 * bb_std
    bb_lower = bb_mid - 2.0 * bb_std

    curr_bb_mid = float(bb_mid.iloc[-1])
    curr_bb_up = float(bb_upper.iloc[-1])
    curr_bb_low = float(bb_lower.iloc[-1])
    bb_width = (curr_bb_up - curr_bb_low) / (curr_bb_mid + eps)
    bb_pct_b = (curr_close - curr_bb_low) / ((curr_bb_up - curr_bb_low) + eps)

    # ── 4. Moving Average Spreads ─────────────────────────────────────────────
    ema20 = float(close.ewm(span=min(20, len(close)), adjust=False).mean().iloc[-1])
    ema50 = float(close.ewm(span=min(50, len(close)), adjust=False).mean().iloc[-1])
    ema200 = float(close.ewm(span=min(200, len(close)), adjust=False).mean().iloc[-1])

    spread_price_ema20 = (curr_close - ema20) / (ema20 + eps)
    spread_ema20_ema50 = (ema20 - ema50) / (ema50 + eps)
    spread_ema50_ema200 = (ema50 - ema200) / (ema200 + eps)

    # ── 5. Volume Statistics ──────────────────────────────────────────────────
    vol_mean_20 = float(volume.rolling(min(20, len(volume))).mean().iloc[-1])
    vol_std_20 = float(volume.rolling(min(20, len(volume))).std().iloc[-1])
    vol_zscore = (curr_vol - vol_mean_20) / (vol_std_20 + eps) if vol_std_20 > 0 else 0.0
    vol_ratio = curr_vol / (vol_mean_20 + eps) if vol_mean_20 > 0 else 1.0

    # ── 6. Time-of-Day Cyclical Encodings (Intraday Seasonality) ──────────────
    # Convert candle timestamp to minute of day (0..1439) and day of week (0..6)
    minute_of_day = 0
    day_of_week = 0
    if hasattr(last_candle_time, "hour"):
        h = last_candle_time.hour
        m = last_candle_time.minute
        # If UTC, convert roughly to Indian Standard Time (UTC+5:30)
        if clean_df.index.tz is not None and "UTC" in str(clean_df.index.tz):
            total_m = (h * 60 + m + 330) % 1440
        else:
            total_m = h * 60 + m
        minute_of_day = total_m
        day_of_week = getattr(last_candle_time, "dayofweek", 0)

    # Cyclical sin / cos
    time_sin = float(math.sin(2.0 * math.pi * minute_of_day / 1440.0))
    time_cos = float(math.cos(2.0 * math.pi * minute_of_day / 1440.0))
    day_sin = float(math.sin(2.0 * math.pi * day_of_week / 7.0))
    day_cos = float(math.cos(2.0 * math.pi * day_of_week / 7.0))

    # Market regime markers in IST (9:15 AM is minute 555, 15:30 is minute 930)
    is_morning_rush = 1.0 if (555 <= minute_of_day <= 615) else 0.0
    is_midday = 1.0 if (675 <= minute_of_day <= 795) else 0.0
    is_power_hour = 1.0 if (840 <= minute_of_day <= 915) else 0.0

    # ── 7. Point-in-Time News Sentiment ───────────────────────────────────────
    sent_score = 0.0
    sent_delta = 0.0
    sent_age_min = 1440.0  # default 24h decay

    if feature_store is not None:
        raw_sent = feature_store.get_latest_sentiment(
            ticker=ticker, sector=sector, asof_time=features_asof
        )
        sent_score = float(raw_sent.get("sentiment_score", 0.0))
        sent_delta = float(raw_sent.get("sentiment_delta", 0.0))

        sent_asof = raw_sent.get("sentiment_asof")
        if sent_asof:
            try:
                t_sent = pd.to_datetime(sent_asof)
                t_feat = pd.to_datetime(features_asof)
                # Compute elapsed minutes
                diff_m = (t_feat - t_sent).total_seconds() / 60.0
                sent_age_min = max(0.0, min(1440.0, diff_m))
            except Exception:
                sent_age_min = 1440.0

    # Exponential decay weight on sentiment based on age (half-life of 2 hours / 120 mins)
    sentiment_decay = math.exp(-0.693 * (sent_age_min / 120.0))
    effective_sentiment = sent_score * sentiment_decay

    # ── Assemble Clean Numeric Vector ─────────────────────────────────────────
    features: Dict[str, float] = {
        "ret_1": round(ret_1, 6),
        "ret_3": round(ret_3, 6),
        "ret_5": round(ret_5, 6),
        "ret_15": round(ret_15, 6),
        "norm_atr": round(norm_atr, 6),
        "rolling_std_10": round(rolling_std_10, 6),
        "body_pct": round(body_pct, 4),
        "upper_wick_pct": round(upper_wick_pct, 4),
        "lower_wick_pct": round(lower_wick_pct, 4),
        "norm_rsi": round(norm_rsi, 4),
        "norm_macd": round(norm_macd, 6),
        "norm_macd_signal": round(norm_macd_signal, 6),
        "norm_macd_diff": round(norm_macd_diff, 6),
        "bb_width": round(bb_width, 4),
        "bb_pct_b": round(bb_pct_b, 4),
        "spread_price_ema20": round(spread_price_ema20, 6),
        "spread_ema20_ema50": round(spread_ema20_ema50, 6),
        "spread_ema50_ema200": round(spread_ema50_ema200, 6),
        "vol_zscore": round(max(-5.0, min(5.0, vol_zscore)), 4),
        "vol_ratio": round(min(10.0, vol_ratio), 4),
        "time_sin": round(time_sin, 4),
        "time_cos": round(time_cos, 4),
        "day_sin": round(day_sin, 4),
        "day_cos": round(day_cos, 4),
        "is_morning_rush": is_morning_rush,
        "is_midday": is_midday,
        "is_power_hour": is_power_hour,
        "sentiment_score": round(sent_score, 4),
        "sentiment_delta": round(sent_delta, 4),
        "effective_sentiment": round(effective_sentiment, 4),
    }

    return features, features_asof


FEATURE_COLUMNS = [
    "ret_1", "ret_3", "ret_5", "ret_15",
    "norm_atr", "rolling_std_10", "body_pct", "upper_wick_pct", "lower_wick_pct",
    "norm_rsi", "norm_macd", "norm_macd_signal", "norm_macd_diff",
    "bb_width", "bb_pct_b",
    "spread_price_ema20", "spread_ema20_ema50", "spread_ema50_ema200",
    "vol_zscore", "vol_ratio",
    "time_sin", "time_cos", "day_sin", "day_cos",
    "is_morning_rush", "is_midday", "is_power_hour",
    "sentiment_score", "sentiment_delta", "effective_sentiment",
]
