from dataclasses import dataclass, field, asdict
from typing import Optional
import json


@dataclass
class MarketSummary:
    trend: str
    strength: str
    pattern: Optional[str]
    momentum: str
    support: Optional[float]
    resistance: Optional[float]
    rsi: Optional[float]
    rsi_state: str
    ema_alignment: str
    macd_state: str
    volume: str
    volatility: str
    consolidation: bool
    ticker: Optional[str]
    timeframe: Optional[str]
    breakout_prob: str
    confidence: float

    def to_dict(self):
        return asdict(self)

    def to_json(self):
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class Signal:
    action: str           # BUY | SELL | WAIT
    conviction: str       # high | moderate | low
    risk: str             # low | moderate | high | very_high
    rr_ratio: Optional[float]
    entry: Optional[str]
    stop: Optional[str]
    target: Optional[str]
    reasons: list
    warnings: list
    confidence: float


# ── Builder ───────────────────────────────────────────────────────────────────

class SummaryBuilder:
    def build(self, cv, ocr) -> MarketSummary:
        sup, res = self._prices(cv, ocr)
        return MarketSummary(
            trend        = cv.trend,
            strength     = cv.strength,
            pattern      = cv.pattern,
            momentum     = self._momentum(cv, ocr),
            support      = sup,
            resistance   = res,
            rsi          = ocr.rsi_value,
            rsi_state    = self._rsi_state(ocr.rsi_value),
            ema_alignment= "bullish" if cv.trend == "bullish" and cv.strength != "weak"
                           else "bearish" if cv.trend == "bearish" and cv.strength != "weak"
                           else "mixed",
            macd_state   = self._macd(ocr.macd_value, cv.trend),
            volume       = cv.volume,
            volatility   = cv.volatility,
            consolidation= cv.consolidation,
            ticker       = ocr.ticker,
            timeframe    = ocr.timeframe,
            breakout_prob= self._breakout(cv, ocr),
            confidence   = self._confidence(cv, ocr),
        )

    def _prices(self, cv, ocr):
        prices = sorted(ocr.price_levels)
        if len(prices) < 2:
            return None, None
        r = prices[-1] - prices[0]
        if r == 0:
            return None, None
        def n2p(y):
            return round(prices[-1] - y * r, 2)
        sup = n2p(cv.support_norm[0])    if cv.support_norm    else None
        res = n2p(cv.resistance_norm[0]) if cv.resistance_norm else None
        return sup, res

    def _rsi_state(self, rsi):
        if rsi is None: return "unknown"
        if rsi >= 70:   return "overbought"
        if rsi <= 30:   return "oversold"
        return "neutral"

    def _momentum(self, cv, ocr):
        rsi = ocr.rsi_value
        if rsi and rsi > 65 and cv.volume == "decreasing": return "weakening"
        if rsi and rsi > 70: return "overbought"
        if rsi and rsi < 30: return "oversold"
        if cv.strength == "strong":   return "strong"
        if cv.strength == "moderate": return "moderate"
        return "weak"

    def _macd(self, macd, trend):
        if macd is None: return "unknown"
        if macd > 0: return "above_zero"
        return "below_zero"

    def _breakout(self, cv, ocr):
        score = 0
        if cv.volume == "increasing":                           score += 2
        if cv.volatility == "compressed":                       score += 2
        if cv.pattern in ("ascending_triangle","bull_flag"):    score += 2
        if ocr.rsi_value and 50 < ocr.rsi_value < 70:          score += 1
        if ocr.rsi_value and ocr.rsi_value > 70:               score -= 1
        if score >= 4: return "high"
        if score >= 2: return "moderate"
        return "low"

    def _confidence(self, cv, ocr):
        s = 0.0
        if len(cv.candles) >= 5:          s += 0.20
        if len(ocr.price_levels) >= 3:    s += 0.20
        if ocr.rsi_value is not None:     s += 0.15
        if ocr.ticker:                    s += 0.10
        if ocr.timeframe:                 s += 0.10
        if cv.pattern:                    s += 0.15
        if cv.support_norm:               s += 0.10
        return round(min(s, 1.0), 2)


# ── Signal Generator ──────────────────────────────────────────────────────────

class SignalGenerator:
    def generate(self, s: MarketSummary) -> Signal:
        bull, bear = self._score(s)
        action, conv = self._decide(bull, bear, s)
        risk = self._risk(s, action)
        rr   = self._rr(s, action)
        entry, stop, target = self._zones(s, action)
        return Signal(
            action=action, conviction=conv, risk=risk,
            rr_ratio=rr, entry=entry, stop=stop, target=target,
            reasons=self._reasons(s, bull, bear, action),
            warnings=self._warnings(s),
            confidence=s.confidence,
        )

    def _score(self, s):
        b = be = 0.0
        if s.trend == "bullish":
            b  += 2.0 if s.strength == "strong" else 1.0
        elif s.trend == "bearish":
            be += 2.0 if s.strength == "strong" else 1.0
        if s.rsi_state == "oversold":   b  += 1.5
        elif s.rsi_state == "overbought": be += 1.5
        elif s.rsi and s.rsi > 50: b  += 0.5
        elif s.rsi and s.rsi < 50: be += 0.5
        if s.ema_alignment == "bullish": b  += 1.0
        elif s.ema_alignment == "bearish":be += 1.0
        if s.macd_state == "above_zero": b  += 1.0
        elif s.macd_state == "below_zero":be += 1.0
        if s.volume == "increasing":
            b  += 0.5 if s.trend == "bullish" else 0.0
            be += 0.5 if s.trend == "bearish" else 0.0
        elif s.volume == "decreasing":
            b  -= 0.3; be -= 0.3
        if s.momentum in ("weakening","weak"):
            b -= 0.5; be -= 0.5
        if s.pattern in ("ascending_triangle","bull_flag"):   b  += 1.0
        if s.pattern in ("descending_triangle","bear_flag"):  be += 1.0
        return round(b, 2), round(be, 2)

    def _decide(self, bull, bear, s):
        diff = bull - bear
        near_res = (s.support and s.resistance and
                    s.resistance > s.support and
                    s.momentum in ("weakening","weak"))
        if near_res and abs(diff) < 2.0:
            return "WAIT", "moderate"
        if diff >= 3.0:   return "BUY",  "high"
        if diff >= 1.5:   return "BUY",  "moderate"
        if diff <= -3.0:  return "SELL", "high"
        if diff <= -1.5:  return "SELL", "moderate"
        return "WAIT", "high" if abs(diff) < 0.5 else "moderate"

    def _risk(self, s, action):
        if s.volatility == "high":                              return "very_high"
        if s.rsi_state == "overbought" and action == "BUY":    return "high"
        if s.rsi_state == "oversold"   and action == "SELL":   return "high"
        if s.volume == "decreasing" and action != "WAIT":      return "moderate"
        if s.volatility == "compressed":                        return "moderate"
        return "low"

    def _rr(self, s, action):
        if not s.support or not s.resistance or s.support >= s.resistance:
            return None
        z = s.resistance - s.support
        gain = z * 0.9
        loss = z * 0.15
        return round(gain / loss, 2) if loss else None

    def _zones(self, s, action):
        if not s.support or not s.resistance:
            return None, None, None
        z = s.resistance - s.support
        if action == "BUY":
            return (f"~{s.support  + z*.05:.2f}",
                    f"~{s.support  - z*.10:.2f}",
                    f"~{s.resistance - z*.02:.2f}")
        if action == "SELL":
            return (f"~{s.resistance - z*.05:.2f}",
                    f"~{s.resistance + z*.10:.2f}",
                    f"~{s.support    + z*.02:.2f}")
        return None, None, None

    def _reasons(self, s, bull, bear, action):
        pts = [
            f"Trend is {s.trend} ({s.strength}).",
        ]
        if s.rsi:
            pts.append(f"RSI at {s.rsi:.1f} — {s.rsi_state}.")
        if s.ema_alignment != "mixed":
            pts.append(f"EMA alignment: {s.ema_alignment}.")
        if s.volume == "increasing":
            pts.append("Volume rising — supports current move.")
        elif s.volume == "decreasing":
            pts.append("Volume declining — weakens conviction.")
        if s.consolidation:
            pts.append("Price consolidating — breakout may be forming.")
        if s.pattern:
            pts.append(f"Pattern: {s.pattern.replace('_',' ')}.")
        pts.append(f"Bull: {bull:.1f}  Bear: {bear:.1f}  →  {action}")
        return pts

    def _warnings(self, s):
        w = []
        if s.confidence < 0.4:
            w.append("Low extraction confidence — verify manually.")
        if s.rsi_state == "overbought":
            w.append("RSI overbought — chasing longs is risky.")
        if s.rsi_state == "oversold":
            w.append("RSI oversold — shorting here is risky.")
        if s.volatility == "high":
            w.append("High volatility — widen stops, reduce size.")
        if s.volatility == "compressed":
            w.append("Compressed volatility — sharp move imminent either way.")
        if s.volume == "decreasing" and s.trend != "sideways":
            w.append("Trend lacks volume confirmation.")
        if not w:
            w.append("No major structural warnings.")
        return w
