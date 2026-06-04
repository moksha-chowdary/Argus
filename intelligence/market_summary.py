"""
ARGUS Intelligence Layer — Market Summary Builder
Merges CV analysis + OCR extraction into a structured MarketSummary
that feeds the LLM reasoning pipeline.
"""
from dataclasses import dataclass, field, asdict
from typing import Optional
import json

from vision.cv_analyzer import CVAnalysisResult
from vision.ocr_pipeline import OCRResult


@dataclass
class MarketSummary:
    """
    Structured representation of a chart's technical state.
    This is the contract between the Vision layer and the LLM layer.
    """
    # Price action
    trend: str                        # "bullish" | "bearish" | "sideways"
    trend_strength: str               # "strong" | "moderate" | "weak"
    chart_pattern: Optional[str]      # "ascending_triangle" | "bull_flag" | ...
    momentum: str                     # "strong" | "moderate" | "weakening" | "weak"

    # Key levels (actual prices if OCR succeeded, else normalized positions)
    support: Optional[float]
    resistance: Optional[float]
    support_zones: list[float]
    resistance_zones: list[float]

    # Indicators
    rsi: Optional[float]
    rsi_state: str                    # "overbought" | "oversold" | "neutral"
    ema_alignment: str                # "bullish" | "bearish" | "mixed"
    macd_state: str                   # "bullish_crossover" | "bearish_crossover" | "above_zero" | "below_zero" | "unknown"

    # Volume & volatility
    volume_behavior: str              # "increasing" | "decreasing" | "flat"
    volatility: str                   # "high" | "moderate" | "low" | "compressed"
    consolidation: bool

    # Meta
    ticker: Optional[str]
    timeframe: Optional[str]
    breakout_probability: str         # "high" | "moderate" | "low"
    confidence: float                 # 0.0 – 1.0 overall extraction confidence

    # Raw source data
    raw_cv: dict = field(default_factory=dict)
    raw_ocr: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("raw_cv", None)
        d.pop("raw_ocr", None)
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


class MarketSummaryBuilder:
    """
    Fuses CVAnalysisResult + OCRResult into a MarketSummary.
    """

    def build(self, cv: CVAnalysisResult, ocr: OCRResult) -> MarketSummary:
        support_price, resistance_price = self._resolve_prices(cv, ocr)
        rsi_state = self._classify_rsi(ocr.rsi_value)
        momentum = self._infer_momentum(cv, ocr)
        macd_state = self._classify_macd(ocr.macd_value, cv.trend_direction)
        ema_alignment = self._infer_ema_alignment(cv)
        breakout_prob = self._estimate_breakout_probability(cv, ocr)
        confidence = self._estimate_confidence(cv, ocr)

        return MarketSummary(
            trend=cv.trend_direction,
            trend_strength=cv.trend_strength,
            chart_pattern=cv.breakout_structure,
            momentum=momentum,
            support=support_price,
            resistance=resistance_price,
            support_zones=cv.support_zones,
            resistance_zones=cv.resistance_zones,
            rsi=ocr.rsi_value,
            rsi_state=rsi_state,
            ema_alignment=ema_alignment,
            macd_state=macd_state,
            volume_behavior=cv.volume_profile,
            volatility=cv.volatility,
            consolidation=cv.consolidation_detected,
            ticker=ocr.ticker,
            timeframe=ocr.timeframe,
            breakout_probability=breakout_prob,
            confidence=confidence,
            raw_cv={"trend": cv.trend_direction, "candles_detected": len(cv.candle_bodies)},
            raw_ocr={"prices_found": len(ocr.price_levels), "raw_text_len": len(ocr.raw_text)},
        )

    # ─── Helpers ──────────────────────────────────────────────────────────────

    def _resolve_prices(
        self, cv: CVAnalysisResult, ocr: OCRResult
    ) -> tuple[Optional[float], Optional[float]]:
        """Map normalized CV zones to actual prices using OCR price levels."""
        prices = sorted(ocr.price_levels)
        if len(prices) < 2:
            return None, None
        price_range = prices[-1] - prices[0]
        if price_range == 0:
            return None, None

        def norm_to_price(norm_y: float) -> float:
            # y=0 is top of chart (highest price), y=1 is bottom (lowest)
            return prices[-1] - norm_y * price_range

        support = None
        if cv.support_zones:
            support = round(norm_to_price(cv.support_zones[0]), 2)

        resistance = None
        if cv.resistance_zones:
            resistance = round(norm_to_price(cv.resistance_zones[0]), 2)

        return support, resistance

    def _classify_rsi(self, rsi: Optional[float]) -> str:
        if rsi is None:
            return "unknown"
        if rsi >= 70:
            return "overbought"
        if rsi <= 30:
            return "oversold"
        return "neutral"

    def _infer_momentum(self, cv: CVAnalysisResult, ocr: OCRResult) -> str:
        rsi = ocr.rsi_value
        if rsi is not None:
            if rsi > 65 and cv.volume_profile == "decreasing":
                return "weakening"
            if rsi > 70:
                return "overbought"
            if rsi < 35:
                return "oversold"
        if cv.trend_strength == "strong":
            return "strong"
        if cv.trend_strength == "moderate":
            return "moderate"
        return "weak"

    def _classify_macd(self, macd: Optional[float], trend: str) -> str:
        if macd is None:
            return "unknown"
        if macd > 0:
            return "above_zero" if trend == "bullish" else "bearish_divergence"
        return "below_zero" if trend == "bearish" else "bullish_divergence"

    def _infer_ema_alignment(self, cv: CVAnalysisResult) -> str:
        """
        In absence of direct EMA color detection, infer from trend + candle data.
        A proper implementation would segment EMA line colors from the chart.
        """
        if cv.trend_direction == "bullish" and cv.trend_strength in ("strong", "moderate"):
            return "bullish"
        if cv.trend_direction == "bearish" and cv.trend_strength in ("strong", "moderate"):
            return "bearish"
        return "mixed"

    def _estimate_breakout_probability(
        self, cv: CVAnalysisResult, ocr: OCRResult
    ) -> str:
        score = 0
        # Volume building near resistance = positive
        if cv.volume_profile == "increasing":
            score += 2
        # Compression often precedes breakout
        if cv.volatility == "compressed":
            score += 2
        # Pattern type matters
        if cv.breakout_structure in ("ascending_triangle", "bull_flag"):
            score += 2
        if cv.breakout_structure in ("descending_triangle", "bear_flag"):
            score -= 1
        # RSI momentum
        if ocr.rsi_value and 50 < ocr.rsi_value < 70:
            score += 1
        if ocr.rsi_value and ocr.rsi_value > 70:
            score -= 1

        if score >= 4:
            return "high"
        if score >= 2:
            return "moderate"
        return "low"

    def _estimate_confidence(self, cv: CVAnalysisResult, ocr: OCRResult) -> float:
        score = 0.0
        if len(cv.candle_bodies) >= 5:
            score += 0.2
        if len(ocr.price_levels) >= 3:
            score += 0.2
        if ocr.rsi_value is not None:
            score += 0.15
        if ocr.ticker:
            score += 0.1
        if ocr.timeframe:
            score += 0.1
        if cv.breakout_structure:
            score += 0.15
        if cv.support_zones and cv.resistance_zones:
            score += 0.1
        return round(min(score, 1.0), 2)
