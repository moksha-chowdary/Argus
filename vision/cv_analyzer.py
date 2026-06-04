"""
ARGUS Vision Layer — CV Analyzer
Detects candlestick structure, trend direction, support/resistance,
and technical patterns using computer vision techniques.
"""
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from vision.preprocessor import PreprocessedImage


@dataclass
class CVAnalysisResult:
    trend_direction: str          # "bullish" | "bearish" | "sideways"
    trend_strength: str           # "strong" | "moderate" | "weak"
    support_zones: list[float]    # normalized 0.0-1.0 (vertical position)
    resistance_zones: list[float]
    candle_bodies: list[dict]     # [{x, y, w, h, type: "bull"/"bear"}]
    volume_profile: str           # "increasing" | "decreasing" | "flat"
    volatility: str               # "high" | "moderate" | "low" | "compressed"
    breakout_structure: Optional[str]  # "ascending_triangle" | "wedge" | etc.
    ema_cross: Optional[str]      # "golden_cross" | "death_cross" | None
    consolidation_detected: bool
    extra: dict = field(default_factory=dict)


class CVAnalyzer:
    """
    Computer vision analysis of chart structure.

    Uses:
    - Horizontal line detection for support/resistance
    - Contour analysis for candlestick body detection
    - Slope analysis of price extremes for trend detection
    - Column-wise intensity for volume behavior
    """

    def analyze(self, processed: PreprocessedImage) -> CVAnalysisResult:
        chart_region = self._get_chart_region(processed)
        volume_region = self._get_volume_region(processed)

        support, resistance = self._detect_horizontal_levels(chart_region, processed)
        candles = self._detect_candle_bodies(chart_region)
        trend_dir, trend_str = self._detect_trend(chart_region, candles)
        vol_behavior = self._analyze_volume(volume_region)
        volatility = self._estimate_volatility(chart_region)
        consolidation = self._detect_consolidation(chart_region)
        breakout_struct = self._detect_chart_pattern(chart_region, trend_dir)

        return CVAnalysisResult(
            trend_direction=trend_dir,
            trend_strength=trend_str,
            support_zones=support,
            resistance_zones=resistance,
            candle_bodies=candles,
            volume_profile=vol_behavior,
            volatility=volatility,
            breakout_structure=breakout_struct,
            ema_cross=None,  # populated by indicator extractor
            consolidation_detected=consolidation,
        )

    # ─── Region extraction ────────────────────────────────────────────────────

    def _get_chart_region(self, img: PreprocessedImage) -> np.ndarray:
        h, w = img.gray.shape
        return img.gray[int(h * 0.06):int(h * 0.65), :int(w * 0.92)]

    def _get_volume_region(self, img: PreprocessedImage) -> np.ndarray:
        h, w = img.gray.shape
        return img.gray[int(h * 0.75):, :int(w * 0.92)]

    # ─── Horizontal level detection (support / resistance) ───────────────────

    def _detect_horizontal_levels(
        self, chart: np.ndarray, processed: PreprocessedImage
    ) -> tuple[list[float], list[float]]:
        """
        Detect significant horizontal price levels via Hough lines on edges.
        Returns normalized y-positions (0=top, 1=bottom).
        """
        edges = cv2.Canny(chart, 30, 100)
        lines = cv2.HoughLinesP(
            edges, rho=1, theta=np.pi / 180,
            threshold=80, minLineLength=int(chart.shape[1] * 0.2),
            maxLineGap=20
        )

        h_lines: list[float] = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(y2 - y1) < 5:  # near-horizontal
                    norm_y = ((y1 + y2) / 2) / chart.shape[0]
                    h_lines.append(round(norm_y, 3))

        h_lines = self._cluster_levels(h_lines)
        if len(h_lines) < 2:
            return [], []

        # Upper half = resistance, lower half = support
        midpoint = 0.5
        resistance = sorted([y for y in h_lines if y < midpoint])
        support = sorted([y for y in h_lines if y >= midpoint])
        return support[:3], resistance[:3]

    def _cluster_levels(self, levels: list[float], tolerance: float = 0.03) -> list[float]:
        if not levels:
            return []
        levels = sorted(levels)
        clusters: list[list[float]] = [[levels[0]]]
        for lv in levels[1:]:
            if lv - clusters[-1][-1] < tolerance:
                clusters[-1].append(lv)
            else:
                clusters.append([lv])
        return [round(sum(c) / len(c), 3) for c in clusters]

    # ─── Candlestick body detection ───────────────────────────────────────────

    def _detect_candle_bodies(self, chart: np.ndarray) -> list[dict]:
        """
        Detect candlestick bodies as rectangular contours.
        Classifies each as bullish (light body) or bearish (dark body).
        """
        _, thresh = cv2.threshold(chart, 160, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candles = []
        h, w = chart.shape
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect = bh / bw if bw > 0 else 0
            # Candle bodies are roughly 2-10x taller than wide, small relative to chart
            if 1.5 < aspect < 12 and bw < w * 0.03 and bh < h * 0.5 and bh > 3:
                roi = chart[y:y+bh, x:x+bw]
                mean_val = roi.mean()
                candle_type = "bear" if mean_val < 127 else "bull"
                candles.append({"x": x, "y": y, "w": bw, "h": bh, "type": candle_type})

        # Sort left to right (chronological)
        return sorted(candles, key=lambda c: c["x"])

    # ─── Trend detection ──────────────────────────────────────────────────────

    def _detect_trend(
        self, chart: np.ndarray, candles: list[dict]
    ) -> tuple[str, str]:
        """
        Estimate trend from the slope of candle body midpoints.
        Falls back to edge-based regression if no candles detected.
        """
        if len(candles) >= 5:
            xs = np.array([c["x"] + c["w"] / 2 for c in candles], dtype=float)
            ys = np.array([c["y"] + c["h"] / 2 for c in candles], dtype=float)
            slope, _ = np.polyfit(xs, ys, 1)
            # Negative slope = candle midpoints moving up (y inverted in image)
            normalized = slope / chart.shape[0]
            if abs(normalized) < 0.0005:
                direction = "sideways"
                strength = "moderate"
            elif normalized < 0:
                direction = "bullish"
                strength = "strong" if normalized < -0.001 else "moderate"
            else:
                direction = "bearish"
                strength = "strong" if normalized > 0.001 else "moderate"
            return direction, strength

        # Fallback: use overall image brightness gradient
        top_half_mean = chart[:chart.shape[0] // 2, :].mean()
        bottom_half_mean = chart[chart.shape[0] // 2:, :].mean()
        if abs(top_half_mean - bottom_half_mean) < 5:
            return "sideways", "weak"
        elif top_half_mean > bottom_half_mean:
            return "bullish", "weak"
        else:
            return "bearish", "weak"

    # ─── Volume analysis ──────────────────────────────────────────────────────

    def _analyze_volume(self, volume_region: np.ndarray) -> str:
        if volume_region.size == 0:
            return "unknown"
        h, w = volume_region.shape
        if w < 10:
            return "unknown"
        # Compare average bar height in left vs right thirds
        third = w // 3
        left_mean = volume_region[:, :third].mean()
        right_mean = volume_region[:, 2*third:].mean()
        diff = right_mean - left_mean
        if abs(diff) < 5:
            return "flat"
        return "increasing" if diff > 0 else "decreasing"

    # ─── Volatility estimation ────────────────────────────────────────────────

    def _estimate_volatility(self, chart: np.ndarray) -> str:
        """
        Estimate volatility by measuring the standard deviation of
        horizontal brightness variation (proxy for candle range).
        """
        col_means = np.std(chart, axis=0)
        mean_std = col_means.mean()
        if mean_std < 20:
            return "compressed"
        elif mean_std < 35:
            return "low"
        elif mean_std < 55:
            return "moderate"
        else:
            return "high"

    # ─── Consolidation detection ──────────────────────────────────────────────

    def _detect_consolidation(self, chart: np.ndarray) -> bool:
        """Flag if recent price action is in a tight horizontal range."""
        recent = chart[:, int(chart.shape[1] * 0.7):]
        row_range = recent.max(axis=1) - recent.min(axis=1)
        return float(row_range.std()) < 15.0

    # ─── Chart pattern recognition ────────────────────────────────────────────

    def _detect_chart_pattern(self, chart: np.ndarray, trend: str) -> Optional[str]:
        """
        Simple heuristic pattern classification.
        A proper implementation would use template matching or a CNN classifier.
        """
        edges = cv2.Canny(chart, 40, 120)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=60,
            minLineLength=int(chart.shape[1] * 0.15), maxLineGap=15
        )
        if lines is None:
            return None

        slopes = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = x2 - x1
            if dx != 0:
                slopes.append((y2 - y1) / dx)

        if not slopes:
            return None

        pos_slopes = [s for s in slopes if s > 0.05]
        neg_slopes = [s for s in slopes if s < -0.05]
        flat = [s for s in slopes if abs(s) <= 0.05]

        if trend == "bullish" and len(flat) > len(pos_slopes) and len(flat) > len(neg_slopes):
            return "ascending_triangle"
        if trend == "bearish" and len(flat) > len(pos_slopes) and len(flat) > len(neg_slopes):
            return "descending_triangle"
        if len(pos_slopes) > 2 and len(neg_slopes) > 2:
            return "symmetrical_triangle"
        if trend == "bullish" and len(neg_slopes) > len(pos_slopes):
            return "bull_flag"
        if trend == "bearish" and len(pos_slopes) > len(neg_slopes):
            return "bear_flag"
        return None
