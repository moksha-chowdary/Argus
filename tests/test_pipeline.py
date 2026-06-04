"""
ARGUS Test Suite
Tests intelligence layer with synthetic data (no image required).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from vision.cv_analyzer import CVAnalysisResult
from vision.ocr_pipeline import OCRResult
from intelligence.market_summary import MarketSummaryBuilder
from intelligence.signal_generator import SignalGenerator


def make_cv(trend="bullish", strength="strong", rsi_structure=None) -> CVAnalysisResult:
    return CVAnalysisResult(
        trend_direction=trend,
        trend_strength=strength,
        support_zones=[0.7, 0.75],
        resistance_zones=[0.2, 0.25],
        candle_bodies=[{"x": i*10, "y": 50-i, "w": 6, "h": 15, "type": "bull"} for i in range(10)],
        volume_profile="increasing",
        volatility="moderate",
        breakout_structure="ascending_triangle",
        ema_cross=None,
        consolidation_detected=False,
    )


def make_ocr(rsi=58.0, prices=None) -> OCRResult:
    return OCRResult(
        raw_text="BTCUSDT 4h RSI(14) 58.0",
        price_levels=prices or [2820.0, 2850.0, 2870.0, 2890.0],
        rsi_value=rsi,
        macd_value=12.5,
        volume_labels=["1.2M", "1.5M"],
        axis_labels=["2820", "2850", "2890"],
        indicator_labels=["RSI 58"],
        timeframe="4h",
        ticker="BTCUSDT",
    )


def test_market_summary_builds():
    builder = MarketSummaryBuilder()
    summary = builder.build(make_cv(), make_ocr())
    assert summary.trend == "bullish"
    assert summary.rsi == 58.0
    assert summary.ticker == "BTCUSDT"
    assert summary.timeframe == "4h"
    assert summary.rsi_state == "neutral"
    assert summary.support is not None
    assert summary.resistance is not None


def test_signal_buy_on_bullish():
    builder = MarketSummaryBuilder()
    summary = builder.build(make_cv(trend="bullish", strength="strong"), make_ocr(rsi=55.0))
    gen = SignalGenerator()
    signal = gen.generate(summary)
    assert signal.action == "BUY"
    assert signal.conviction in ("high", "moderate")


def test_signal_sell_on_bearish():
    builder = MarketSummaryBuilder()
    summary = builder.build(make_cv(trend="bearish", strength="strong"), make_ocr(rsi=72.0))
    gen = SignalGenerator()
    signal = gen.generate(summary)
    assert signal.action in ("SELL", "WAIT")  # overbought may also trigger WAIT


def test_signal_wait_on_overbought_resistance():
    builder = MarketSummaryBuilder()
    cv = make_cv(trend="bullish", strength="weak")
    cv.consolidation_detected = True
    ocr = make_ocr(rsi=73.0)
    summary = builder.build(cv, ocr)
    summary.momentum = "weakening"
    gen = SignalGenerator()
    signal = gen.generate(summary)
    assert signal.action == "WAIT"


def test_signal_has_reasoning():
    builder = MarketSummaryBuilder()
    summary = builder.build(make_cv(), make_ocr())
    gen = SignalGenerator()
    signal = gen.generate(summary)
    assert len(signal.reasoning_points) > 0
    assert len(signal.warnings) > 0


def test_confidence_score():
    builder = MarketSummaryBuilder()
    summary = builder.build(make_cv(), make_ocr())
    assert 0.0 <= summary.confidence <= 1.0
    assert summary.confidence > 0.4  # well-populated data should score decently


def test_to_json_serializable():
    import json
    builder = MarketSummaryBuilder()
    summary = builder.build(make_cv(), make_ocr())
    json_str = summary.to_json()
    data = json.loads(json_str)
    assert "trend" in data
    assert "rsi" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
