"""
ARGUS Intelligence Layer — Signal Generator
Produces trading signals (BUY / SELL / WAIT) with risk assessment
from a MarketSummary. Deterministic rule-based layer that feeds
structured signal context to the LLM.
"""
from dataclasses import dataclass, field
from typing import Optional
from intelligence.market_summary import MarketSummary


@dataclass
class TradingSignal:
    action: str                    # "BUY" | "SELL" | "WAIT"
    conviction: str                # "high" | "moderate" | "low"
    risk_rating: str               # "low" | "moderate" | "high" | "very_high"
    risk_reward_ratio: Optional[float]
    entry_zone: Optional[str]
    stop_loss_zone: Optional[str]
    target_zone: Optional[str]
    reasoning_points: list[str]    # human-readable bullet points
    warnings: list[str]            # red flags / cautions
    confidence_score: float        # 0.0 – 1.0


class SignalGenerator:
    """
    Rule-based signal engine. Combines multiple technical factors
    into a weighted conviction score and generates a final signal.
    """

    def generate(self, summary: MarketSummary) -> TradingSignal:
        bull_score, bear_score = self._score(summary)
        action, conviction = self._decide(bull_score, bear_score, summary)
        risk = self._assess_risk(summary, action)
        rr_ratio = self._estimate_rr(summary, action)
        entry, stop, target = self._estimate_zones(summary, action)
        reasoning = self._build_reasoning(summary, bull_score, bear_score, action)
        warnings = self._build_warnings(summary)

        return TradingSignal(
            action=action,
            conviction=conviction,
            risk_rating=risk,
            risk_reward_ratio=rr_ratio,
            entry_zone=entry,
            stop_loss_zone=stop,
            target_zone=target,
            reasoning_points=reasoning,
            warnings=warnings,
            confidence_score=summary.confidence,
        )

    # ─── Scoring ──────────────────────────────────────────────────────────────

    def _score(self, s: MarketSummary) -> tuple[float, float]:
        bull = 0.0
        bear = 0.0

        # Trend
        if s.trend == "bullish":
            bull += 2.0 if s.trend_strength == "strong" else 1.0
        elif s.trend == "bearish":
            bear += 2.0 if s.trend_strength == "strong" else 1.0

        # RSI
        if s.rsi_state == "oversold":
            bull += 1.5
        elif s.rsi_state == "overbought":
            bear += 1.5
        elif s.rsi_state == "neutral" and s.rsi and s.rsi > 50:
            bull += 0.5
        elif s.rsi_state == "neutral" and s.rsi and s.rsi < 50:
            bear += 0.5

        # EMA alignment
        if s.ema_alignment == "bullish":
            bull += 1.0
        elif s.ema_alignment == "bearish":
            bear += 1.0

        # MACD
        if s.macd_state in ("above_zero", "bullish_divergence"):
            bull += 1.0
        elif s.macd_state in ("below_zero", "bearish_divergence"):
            bear += 1.0

        # Volume
        if s.volume_behavior == "increasing" and s.trend == "bullish":
            bull += 0.5
        elif s.volume_behavior == "increasing" and s.trend == "bearish":
            bear += 0.5
        elif s.volume_behavior == "decreasing":
            # Volume drying up = weakening trend
            bull -= 0.3
            bear -= 0.3

        # Momentum
        if s.momentum == "strong":
            bull += 0.5 if s.trend == "bullish" else 0.0
            bear += 0.5 if s.trend == "bearish" else 0.0
        elif s.momentum == "weakening":
            bull -= 0.5
            bear -= 0.5

        # Chart pattern
        bullish_patterns = {"ascending_triangle", "bull_flag", "cup_and_handle"}
        bearish_patterns = {"descending_triangle", "bear_flag", "head_and_shoulders"}
        if s.chart_pattern in bullish_patterns:
            bull += 1.0
        elif s.chart_pattern in bearish_patterns:
            bear += 1.0

        return round(bull, 2), round(bear, 2)

    # ─── Decision ─────────────────────────────────────────────────────────────

    def _decide(
        self, bull: float, bear: float, s: MarketSummary
    ) -> tuple[str, str]:
        diff = bull - bear
        total = bull + bear or 1.0

        # Force WAIT if near resistance with weakening momentum
        near_resistance = (
            s.resistance is not None
            and s.support is not None
            and s.resistance > s.support
            and s.momentum in ("weakening", "weak")
        )

        if near_resistance and abs(diff) <= 2.0:
            return "WAIT", "moderate"

        if diff >= 3.0:
            conviction = "high"
            action = "BUY"
        elif diff >= 1.5:
            conviction = "moderate"
            action = "BUY"
        elif diff <= -3.0:
            conviction = "high"
            action = "SELL"
        elif diff <= -1.5:
            conviction = "moderate"
            action = "SELL"
        else:
            action = "WAIT"
            conviction = "high" if abs(diff) < 0.5 else "moderate"

        # Downgrade conviction if confidence is low
        if s.confidence < 0.4 and conviction == "high":
            conviction = "moderate"

        return action, conviction

    # ─── Risk assessment ──────────────────────────────────────────────────────

    def _assess_risk(self, s: MarketSummary, action: str) -> str:
        if s.volatility == "high":
            return "very_high"
        if s.rsi_state == "overbought" and action == "BUY":
            return "high"
        if s.rsi_state == "oversold" and action == "SELL":
            return "high"
        if s.volume_behavior == "decreasing" and action != "WAIT":
            return "moderate"
        if s.volatility == "compressed":
            return "moderate"  # compression = potential explosive move either way
        return "low"

    def _estimate_rr(self, s: MarketSummary, action: str) -> Optional[float]:
        if s.support is None or s.resistance is None:
            return None
        if s.support >= s.resistance:
            return None
        zone_width = s.resistance - s.support
        if action == "BUY":
            # Entry near support, target = resistance, stop = below support
            potential_gain = zone_width * 0.9
            potential_loss = zone_width * 0.15
        elif action == "SELL":
            potential_gain = zone_width * 0.9
            potential_loss = zone_width * 0.15
        else:
            return None
        if potential_loss == 0:
            return None
        return round(potential_gain / potential_loss, 2)

    def _estimate_zones(
        self, s: MarketSummary, action: str
    ) -> tuple[Optional[str], Optional[str], Optional[str]]:
        if s.support is None or s.resistance is None:
            return None, None, None
        support = s.support
        resistance = s.resistance
        zone_width = resistance - support
        if action == "BUY":
            entry = f"~{support + zone_width * 0.05:.2f}"
            stop = f"~{support - zone_width * 0.1:.2f}"
            target = f"~{resistance - zone_width * 0.02:.2f}"
        elif action == "SELL":
            entry = f"~{resistance - zone_width * 0.05:.2f}"
            stop = f"~{resistance + zone_width * 0.1:.2f}"
            target = f"~{support + zone_width * 0.02:.2f}"
        else:
            return None, None, None
        return entry, stop, target

    # ─── Reasoning narrative ──────────────────────────────────────────────────

    def _build_reasoning(
        self, s: MarketSummary, bull: float, bear: float, action: str
    ) -> list[str]:
        points = []
        trend_verb = "maintaining" if s.trend_strength != "weak" else "attempting to hold"
        points.append(f"Trend is {s.trend} ({s.trend_strength}), {trend_verb} directional bias.")

        if s.rsi:
            points.append(f"RSI at {s.rsi:.1f} — {s.rsi_state} territory.")
        if s.ema_alignment != "mixed":
            points.append(f"EMA alignment is {s.ema_alignment}, confirming the trend.")
        if s.macd_state != "unknown":
            points.append(f"MACD is {s.macd_state.replace('_', ' ')}.")
        if s.volume_behavior == "increasing":
            points.append("Volume is rising, lending strength to the current move.")
        elif s.volume_behavior == "decreasing":
            points.append("Volume is declining — weakens conviction in the current direction.")
        if s.consolidation:
            points.append("Price is consolidating; a directional breakout may be forming.")
        if s.chart_pattern:
            points.append(f"Chart structure resembles a {s.chart_pattern.replace('_', ' ')} pattern.")
        if s.breakout_probability != "low":
            points.append(f"Breakout probability assessed as {s.breakout_probability}.")

        points.append(f"Bull score: {bull:.1f} | Bear score: {bear:.1f} → Signal: {action}")
        return points

    def _build_warnings(self, s: MarketSummary) -> list[str]:
        warnings = []
        if s.confidence < 0.4:
            warnings.append("Low extraction confidence — chart may be unclear or unsupported format.")
        if s.rsi_state == "overbought":
            warnings.append("RSI overbought: chasing long entries here carries elevated risk.")
        if s.rsi_state == "oversold":
            warnings.append("RSI oversold: short entries at this level carry elevated risk.")
        if s.volatility == "high":
            warnings.append("High volatility environment — widen stops and reduce position size.")
        if s.volume_behavior == "decreasing" and s.trend != "sideways":
            warnings.append("Trend lacks volume support — may be losing steam.")
        if s.volatility == "compressed":
            warnings.append("Volatility compression often precedes sharp moves in either direction.")
        if not warnings:
            warnings.append("No major structural warnings detected.")
        return warnings
