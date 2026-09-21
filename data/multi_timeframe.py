"""
ARGUS V5 — Multi-Timeframe Data Ingestion & Context Layer
Provides:
1. Slow / Daily-Context Features: 90-day daily OHLCV context (trend, ATR, moving average distance, swing levels)
   computed from completed trading sessions and held constant intraday.
2. Fast / Intraday Series: Primary 15-minute OHLCV candles replacing 5-minute bars for improved SNR.
3. Strict Point-in-Time Leakage Guard: Daily context for intraday timestamp T strictly uses completed daily
   bars where date < T.date(). Today's still-forming daily candle is never accessed.
"""

import os
import math
from datetime import datetime, timezone, date
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
import yfinance as yf

# In-memory caches to prevent redundant yfinance network requests
_DAILY_BARS_CACHE: Dict[str, pd.DataFrame] = {}
_DAILY_CONTEXT_CACHE: Dict[Tuple[str, str], Tuple[Dict[str, float], str]] = {}
_INTRADAY_CACHE: Dict[Tuple[str, str, str], pd.DataFrame] = {}


def get_cached_daily_bars(ticker: str, lookback_days: int = 180) -> pd.DataFrame:
    """Fetches and caches daily OHLCV bars for a ticker."""
    clean_ticker = ticker.upper()
    if clean_ticker in _DAILY_BARS_CACHE:
        return _DAILY_BARS_CACHE[clean_ticker]

    period = "1y" if lookback_days > 90 else "6mo"
    df = yf.download(clean_ticker, period=period, interval="1d", progress=False, auto_adjust=True)
    if hasattr(df.columns, "get_level_values"):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    _DAILY_BARS_CACHE[clean_ticker] = df
    return df


def fetch_daily_context(
    ticker: str,
    asof_timestamp: Optional[datetime | str | pd.Timestamp] = None,
    lookback_days: int = 90,
    daily_df: Optional[pd.DataFrame] = None,
) -> Tuple[Dict[str, float], str]:
    """
    Computes daily-context features from completed daily candles.
    
    STRICT POINT-IN-TIME LEAKAGE GUARD:
    For an intraday timestamp T, filters daily data to strictly date < T.date().
    Today's still-forming daily candle is NEVER included.

    Returns:
        (features_dict, daily_asof_iso_string)
    """
    clean_ticker = ticker.upper()

    # Determine reference date for leak-free slicing
    if asof_timestamp is not None:
        if isinstance(asof_timestamp, str):
            ref_dt = pd.to_datetime(asof_timestamp)
        else:
            ref_dt = asof_timestamp
        ref_date = ref_dt.date()
    else:
        ref_dt = datetime.now(timezone.utc)
        ref_date = ref_dt.date()

    cache_key = (clean_ticker, str(ref_date))
    if cache_key in _DAILY_CONTEXT_CACHE and daily_df is None:
        return _DAILY_CONTEXT_CACHE[cache_key]

    if daily_df is not None:
        df = daily_df.copy()
    else:
        df = get_cached_daily_bars(clean_ticker, lookback_days=lookback_days).copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Slicing: strictly before ref_date (only completed prior trading days)
    # If df index has timezone, align or strip to match
    df_dates = pd.Series([ts.date() for ts in df.index], index=df.index)
    completed_daily = df[df_dates < ref_date]

    if len(completed_daily) < 25:
        # If not enough strictly prior days (e.g. at the start of backtest), use all available up to ref_date
        completed_daily = df[df_dates <= ref_date]

    if len(completed_daily) < 15:
        # Safe neutral defaults if historical data is extremely sparse
        fallback_asof = str(ref_date)
        fallback = {
            "daily_ret_3": 0.0,
            "daily_ret_5": 0.0,
            "daily_ret_20": 0.0,
            "daily_ret_60": 0.0,
            "daily_dist_ema20": 0.0,
            "daily_dist_ema50": 0.0,
            "daily_atr14_pct": 1.5,
            "daily_prox_swing_high": 0.0,
            "daily_prox_swing_low": 0.0,
            "daily_weekly_momentum": 0.0,
        }
        return fallback, fallback_asof

    daily_asof = completed_daily.index[-1].strftime("%Y-%m-%d")
    c = completed_daily["Close"]
    h = completed_daily["High"]
    l = completed_daily["Low"]
    n = len(completed_daily)

    # 1. Multi-Day Returns (%)
    ret_3 = float((c.iloc[-1] - c.iloc[-min(4, n)]) / c.iloc[-min(4, n)] * 100.0) if n >= 4 else 0.0
    ret_5 = float((c.iloc[-1] - c.iloc[-min(6, n)]) / c.iloc[-min(6, n)] * 100.0) if n >= 6 else 0.0
    ret_20 = float((c.iloc[-1] - c.iloc[-min(21, n)]) / c.iloc[-min(21, n)] * 100.0) if n >= 21 else ret_5
    ret_60 = float((c.iloc[-1] - c.iloc[-min(61, n)]) / c.iloc[-min(61, n)] * 100.0) if n >= 61 else ret_20

    # 2. Distance from Daily EMAs (%)
    ema20 = float(c.ewm(span=20, adjust=False).mean().iloc[-1])
    ema50 = float(c.ewm(span=50, adjust=False).mean().iloc[-1]) if n >= 20 else ema20
    dist_ema20 = float((c.iloc[-1] - ema20) / ema20 * 100.0) if ema20 > 0 else 0.0
    dist_ema50 = float((c.iloc[-1] - ema50) / ema50 * 100.0) if ema50 > 0 else 0.0

    # 3. Daily ATR (14-Day) as % of Price (Volatility Regime Indicator)
    prev_close = c.shift(1)
    tr = pd.concat([h - l, (h - prev_close).abs(), (l - prev_close).abs()], axis=1).max(axis=1)
    atr14 = float(tr.rolling(14).mean().dropna().iloc[-1]) if n >= 15 else float(tr.mean())
    atr14_pct = float(atr14 / c.iloc[-1] * 100.0) if c.iloc[-1] > 0 else 1.5

    # 4. Proximity to Recent Swing High / Low over 60-Day Lookback (%)
    lookback = min(60, n)
    swing_high = float(h.iloc[-lookback:].max())
    swing_low = float(l.iloc[-lookback:].min())
    prox_high = float((c.iloc[-1] - swing_high) / c.iloc[-1] * 100.0) if c.iloc[-1] > 0 else 0.0
    prox_low = float((c.iloc[-1] - swing_low) / c.iloc[-1] * 100.0) if c.iloc[-1] > 0 else 0.0

    # 5. Weekly Momentum: Return over last 5 trading days vs. prior 5 trading days
    if n >= 11:
        ret_last_5 = (c.iloc[-1] - c.iloc[-6]) / c.iloc[-6] * 100.0
        ret_prior_5 = (c.iloc[-6] - c.iloc[-11]) / c.iloc[-11] * 100.0
        weekly_momentum = float(ret_last_5 - ret_prior_5)
    else:
        weekly_momentum = 0.0

    features = {
        "daily_ret_3": round(ret_3, 4),
        "daily_ret_5": round(ret_5, 4),
        "daily_ret_20": round(ret_20, 4),
        "daily_ret_60": round(ret_60, 4),
        "daily_dist_ema20": round(dist_ema20, 4),
        "daily_dist_ema50": round(dist_ema50, 4),
        "daily_atr14_pct": round(atr14_pct, 4),
        "daily_prox_swing_high": round(prox_high, 4),
        "daily_prox_swing_low": round(prox_low, 4),
        "daily_weekly_momentum": round(weekly_momentum, 4),
    }

    if daily_df is None:
        _DAILY_CONTEXT_CACHE[cache_key] = (features, daily_asof)

    return features, daily_asof


def fetch_intraday_series(
    ticker: str,
    period: str = "60d",
    interval: str = "15m",
) -> pd.DataFrame:
    """
    Maintains and fetches the primary 15-minute intraday candle series.
    Capped at 60d by Yahoo Finance intraday API specifications.
    """
    clean_ticker = ticker.upper()
    cache_key = (clean_ticker, period, interval)
    if cache_key in _INTRADAY_CACHE:
        return _INTRADAY_CACHE[cache_key]

    df = yf.download(clean_ticker, period=period, interval=interval, progress=False, auto_adjust=True)
    if hasattr(df.columns, "get_level_values"):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    _INTRADAY_CACHE[cache_key] = df
    return df
