"""
ARGUS V4 — Technical Indicators
Computed from real OHLCV candle data via pandas-ta.
Replaces the pixel-brightness heuristics in vision/analyzer.py.
"""

import warnings
warnings.filterwarnings("ignore")

from dataclasses import dataclass, field
from typing import Optional
import pandas as pd

try:
    import pandas_ta as ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False


@dataclass
class IndicatorResult:
    # Price levels
    current_price:  float = 0.0
    open_price:     float = 0.0
    high_today:     float = 0.0
    low_today:      float = 0.0
    prev_close:     float = 0.0

    # EMAs
    ema_20:   Optional[float] = None
    ema_50:   Optional[float] = None
    ema_200:  Optional[float] = None
    price_vs_ema20: str = "unknown"   # "above" / "below"
    ema_stack:      str = "unknown"   # "bullish" / "bearish" / "mixed"

    # RSI
    rsi:       Optional[float] = None
    rsi_state: str = "neutral"   # "overbought" / "oversold" / "neutral"

    # MACD
    macd:        Optional[float] = None
    macd_signal: Optional[float] = None
    macd_hist:   Optional[float] = None
    macd_cross:  str = "none"    # "bullish_cross" / "bearish_cross" / "none"

    # Volume
    volume_current: int   = 0
    volume_avg_20:  float = 0.0
    volume_signal:  str   = "normal"  # "high" / "low" / "normal"
    volume_ratio:   float = 1.0

    # Trend structure
    trend:          str = "sideways"   # "bullish" / "bearish" / "sideways"
    strength:       str = "weak"       # "strong" / "moderate" / "weak"
    momentum:       str = "neutral"
    higher_lows:    bool = False
    lower_highs:    bool = False

    # Support / Resistance (from recent swing points)
    support:    Optional[float] = None
    resistance: Optional[float] = None

    # Patterns detected
    patterns:   list = field(default_factory=list)
    warnings:   list = field(default_factory=list)

    # Session context
    session:    str = "unknown"   # "morning" / "midday" / "power_hour" / "closing"
    session_confidence_mult: float = 1.0


def calculate(df: pd.DataFrame) -> IndicatorResult:
    """
    Main entry point. Pass a yfinance OHLCV DataFrame.
    Returns a fully populated IndicatorResult.
    """
    if df is None or len(df) < 5:
        return IndicatorResult()

    res = IndicatorResult()

    # ── Price basics ──────────────────────────────────────────────────────────
    res.current_price = round(float(df["Close"].iloc[-1]), 2)
    res.open_price    = round(float(df["Open"].iloc[0]), 2)
    res.high_today    = round(float(df["High"].max()), 2)
    res.low_today     = round(float(df["Low"].min()), 2)
    res.prev_close    = round(float(df["Close"].iloc[-2]), 2) if len(df) > 1 else res.current_price
    res.volume_current = int(df["Volume"].iloc[-1])

    # ── EMAs ─────────────────────────────────────────────────────────────────
    close = df["Close"]
    if len(close) >= 20:
        res.ema_20 = round(float(close.ewm(span=20, adjust=False).mean().iloc[-1]), 2)
    if len(close) >= 50:
        res.ema_50 = round(float(close.ewm(span=50, adjust=False).mean().iloc[-1]), 2)
    if len(close) >= 100:
        res.ema_200 = round(float(close.ewm(span=100, adjust=False).mean().iloc[-1]), 2)

    if res.ema_20:
        res.price_vs_ema20 = "above" if res.current_price > res.ema_20 else "below"

    if res.ema_20 and res.ema_50:
        if res.ema_20 > res.ema_50:
            res.ema_stack = "bullish" if (not res.ema_200 or res.ema_50 > res.ema_200) else "mixed"
        else:
            res.ema_stack = "bearish" if (not res.ema_200 or res.ema_50 < res.ema_200) else "mixed"

    # ── RSI ───────────────────────────────────────────────────────────────────
    if len(close) >= 14:
        delta  = close.diff()
        gain   = delta.clip(lower=0).rolling(14).mean()
        loss   = (-delta.clip(upper=0)).rolling(14).mean()
        rs     = gain / loss.replace(0, 1e-9)
        rsi_s  = 100 - (100 / (1 + rs))
        res.rsi = round(float(rsi_s.iloc[-1]), 1)
        if res.rsi >= 70:   res.rsi_state = "overbought"
        elif res.rsi <= 30: res.rsi_state = "oversold"
        else:               res.rsi_state = "neutral"

    # ── MACD ──────────────────────────────────────────────────────────────────
    if len(close) >= 26:
        ema12      = close.ewm(span=12, adjust=False).mean()
        ema26      = close.ewm(span=26, adjust=False).mean()
        macd_line  = ema12 - ema26
        signal_line= macd_line.ewm(span=9, adjust=False).mean()
        hist       = macd_line - signal_line
        res.macd        = round(float(macd_line.iloc[-1]), 4)
        res.macd_signal = round(float(signal_line.iloc[-1]), 4)
        res.macd_hist   = round(float(hist.iloc[-1]), 4)
        # Detect fresh cross
        if len(hist) >= 2:
            prev_h = float(hist.iloc[-2])
            curr_h = float(hist.iloc[-1])
            if prev_h < 0 and curr_h > 0:
                res.macd_cross = "bullish_cross"
            elif prev_h > 0 and curr_h < 0:
                res.macd_cross = "bearish_cross"

    # ── Volume ────────────────────────────────────────────────────────────────
    vol_series    = df["Volume"]
    vol_avg       = float(vol_series.rolling(min(20, len(vol_series))).mean().iloc[-1])
    res.volume_avg_20 = round(vol_avg, 0)
    res.volume_ratio  = round(res.volume_current / vol_avg, 2) if vol_avg > 0 else 1.0
    if res.volume_ratio > 1.5:   res.volume_signal = "high"
    elif res.volume_ratio < 0.6: res.volume_signal = "low"
    else:                        res.volume_signal = "normal"

    # ── Trend structure ───────────────────────────────────────────────────────
    _detect_trend(df, res)

    # ── Support / Resistance ──────────────────────────────────────────────────
    _detect_sr(df, res)

    # ── Pattern detection ─────────────────────────────────────────────────────
    _detect_patterns(df, res)

    # ── Session clock ─────────────────────────────────────────────────────────
    _session_clock(df, res)

    return res


def _detect_trend(df: pd.DataFrame, res: IndicatorResult):
    close = df["Close"]
    n     = min(20, len(close))
    recent = close.iloc[-n:]

    # Simple linear slope
    import numpy as np
    x     = range(len(recent))
    slope = float(pd.Series(recent.values).diff().mean())

    if slope > 0.05:
        res.trend    = "bullish"
        res.momentum = "positive"
        res.strength = "strong" if slope > 0.2 else "moderate"
    elif slope < -0.05:
        res.trend    = "bearish"
        res.momentum = "negative"
        res.strength = "strong" if slope < -0.2 else "moderate"
    else:
        res.trend    = "sideways"
        res.momentum = "neutral"
        res.strength = "weak"

    # Higher lows check (last 6 lows)
    lows = df["Low"].iloc[-12:]
    local_lows = []
    for i in range(1, len(lows)-1):
        if lows.iloc[i] < lows.iloc[i-1] and lows.iloc[i] < lows.iloc[i+1]:
            local_lows.append(float(lows.iloc[i]))
    if len(local_lows) >= 2:
        res.higher_lows = local_lows[-1] > local_lows[-2]
        res.lower_highs = False

    # Lower highs check (last 6 highs)
    highs = df["High"].iloc[-12:]
    local_highs = []
    for i in range(1, len(highs)-1):
        if highs.iloc[i] > highs.iloc[i-1] and highs.iloc[i] > highs.iloc[i+1]:
            local_highs.append(float(highs.iloc[i]))
    if len(local_highs) >= 2:
        res.lower_highs = local_highs[-1] < local_highs[-2]


def _detect_sr(df: pd.DataFrame, res: IndicatorResult):
    """Recent swing high/low as resistance/support."""
    window = min(30, len(df))
    recent = df.iloc[-window:]
    res.resistance = round(float(recent["High"].max()), 2)
    res.support    = round(float(recent["Low"].min()), 2)

    # Tighten to meaningful nearby levels
    price = res.current_price
    if res.resistance and res.resistance < price * 0.98:
        res.resistance = None   # not above price
    if res.support and res.support > price * 1.02:
        res.support = None      # not below price


def _detect_patterns(df: pd.DataFrame, res: IndicatorResult):
    patterns = []
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]
    op    = df["Open"]

    if len(df) < 3:
        res.patterns = patterns
        return

    # Last candle
    c1  = float(close.iloc[-1])
    o1  = float(op.iloc[-1])
    h1  = float(high.iloc[-1])
    l1  = float(low.iloc[-1])
    body = abs(c1 - o1)
    rng  = h1 - l1

    # Doji
    if rng > 0 and body / rng < 0.1:
        patterns.append("doji")

    # Hammer / hanging man
    lower_wick = min(o1, c1) - l1
    upper_wick = h1 - max(o1, c1)
    if rng > 0 and lower_wick > body * 2 and upper_wick < body * 0.5:
        patterns.append("hammer" if res.trend == "bearish" else "hanging_man")

    # Shooting star
    if rng > 0 and upper_wick > body * 2 and lower_wick < body * 0.5:
        patterns.append("shooting_star")

    # Engulfing
    if len(df) >= 2:
        c2 = float(close.iloc[-2])
        o2 = float(op.iloc[-2])
        if c1 > o1 and c2 < o2 and c1 > o2 and o1 < c2:
            patterns.append("bullish_engulfing")
        elif c1 < o1 and c2 > o2 and c1 < o2 and o1 > c2:
            patterns.append("bearish_engulfing")

    # Lilliput (volatility contraction)
    avg_rng = float((high - low).iloc[-10:-1].mean()) if len(df) >= 10 else rng
    if avg_rng > 0 and rng < avg_rng * 0.4:
        patterns.append("volatility_contraction")

    # Lower highs (from trend detection)
    if res.lower_highs:
        patterns.append("lower_highs")
    if res.higher_lows:
        patterns.append("higher_lows")

    # MACD warnings
    if res.macd_cross == "bearish_cross":
        res.warnings.append("MACD bearish cross — momentum shifting down")
    elif res.macd_cross == "bullish_cross":
        patterns.append("macd_bullish_cross")

    res.patterns = patterns


def _session_clock(df: pd.DataFrame, res: IndicatorResult):
    """
    Determine market session from last candle timestamp.
    Applies confidence multiplier based on session.
    """
    try:
        last_ts = df.index[-1]
        # Convert to IST if timezone-aware
        if hasattr(last_ts, 'hour'):
            h = last_ts.hour
            m = last_ts.minute
            # yfinance returns UTC; IST = UTC+5:30
            # Approximate: add 5.5 hours
            h_ist = (h + 5) % 24
            m_ist = m + 30
            if m_ist >= 60:
                h_ist += 1
                m_ist -= 60
            total_min = h_ist * 60 + m_ist
        else:
            total_min = -1

        if 555 <= total_min < 630:        # 9:15 – 10:30
            res.session = "morning"
            res.session_confidence_mult = 0.85   # volatile open
        elif 630 <= total_min < 810:      # 10:30 – 13:30
            res.session = "midday"
            res.session_confidence_mult = 1.0
        elif 810 <= total_min < 900:      # 13:30 – 15:00
            res.session = "power_hour"
            res.session_confidence_mult = 1.1    # highest reliability
        elif 900 <= total_min < 915:      # 15:00 – 15:15
            res.session = "closing"
            res.session_confidence_mult = 0.5    # exit time, no new entries
            res.warnings.append("Market closing in <15 min — exit positions, no new entries")
        else:
            res.session = "pre_market"
            res.session_confidence_mult = 0.7
    except Exception:
        res.session = "unknown"
        res.session_confidence_mult = 1.0
