"""
ARGUS RULEBOOK V1
Extracted from training session — 8 samples, 100% accuracy.
All rules are evidence-based from NSE intraday backtesting.
Capital: ₹4,00,000 | Max risk: 1% = ₹4,000 | Style: Intraday only
"""

# ─── POSITION SIZING ────────────────────────────────────────────────────────

def position_size(entry: float, stop: float, capital: float = 400000, risk_pct: float = 0.01) -> dict:
    """Calculate shares, capital, max loss based on 1% risk rule."""
    max_risk = capital * risk_pct
    risk_per_share = abs(entry - stop)
    if risk_per_share == 0:
        return {}
    shares = int(max_risk / risk_per_share)
    capital_used = shares * entry
    return {
        "shares": shares,
        "capital_used": round(capital_used, 2),
        "max_loss": round(shares * risk_per_share, 2),
        "risk_pct": risk_pct * 100,
    }


# ─── PATTERN RULES ──────────────────────────────────────────────────────────

PATTERNS = {
    "PULLBACK_TO_MA": {
        "signal": "bullish",
        "reliability": "very_high",
        "conditions": [
            "Strong uptrend exists",
            "Price pulls back to 20MA or 50MA",
            "Bullish confirmation candle appears at MA",
            "Volume decreasing on pullback"
        ],
        "entry": "Above confirmation candle high",
        "stop": "Below swing low",
        "target": "Previous swing high",
        "timeframe": ["15m", "1h"],
        "confidence_base": 75,
    },
    "REJECTION_CANDLE": {
        "signal": "directional",
        "reliability": "high",
        "conditions": [
            "Long wick (tail > 2x body)",
            "Small body near one end",
            "Appears at clear support or resistance"
        ],
        "bullish": "Long lower wick at support → buy above body",
        "bearish": "Long upper wick at resistance → sell below body",
        "stop": "Beyond the wick extreme",
        "timeframe": ["15m", "1h"],
        "confidence_base": 72,
    },
    "GIANT_CANDLE_BREAKOUT": {
        "signal": "directional",
        "reliability": "high",
        "conditions": [
            "Candle body much larger than 5 previous candles",
            "Closes near its high (bullish) or low (bearish)",
            "Volume significantly above average"
        ],
        "entry": "Above high (bull) or below low (bear)",
        "stop": "Opposite end of giant candle",
        "timeframe": ["15m", "1h"],
        "confidence_base": 70,
        "penalty": "If stock already moved >10% that day, reduce confidence by 20 points"
    },
    "LILLIPUT_BREAKOUT": {
        "signal": "breakout",
        "reliability": "high",
        "conditions": [
            "Tiny candle(s) — body << surrounding candles",
            "Appears after a trend or at support/resistance",
            "Volatility contraction = energy accumulation"
        ],
        "entry_bull": "Above range high",
        "entry_bear": "Below range low",
        "stop": "Opposite side of range",
        "target": "1x-3x range height",
        "timeframe": ["15m"],
        "confidence_base": 68,
    },
    "BEARISH_CHANNEL": {
        "signal": "bearish",
        "reliability": "high",
        "conditions": [
            "Consistent lower highs",
            "Consistent lower lows",
            "Price bounces fail to break prior high"
        ],
        "entry": "Below last swing low",
        "stop": "Above last swing high",
        "timeframe": ["15m", "1h"],
        "confidence_base": 70,
    },
    "DEAD_CAT_BOUNCE": {
        "signal": "bearish",
        "reliability": "medium",
        "conditions": [
            "Extended downtrend",
            "Small green recovery candles",
            "Recovery lacks volume",
            "Cannot reclaim prior structure"
        ],
        "action": "Do not buy the bounce — wait for breakdown or ignore",
        "timeframe": ["15m", "1h"],
        "confidence_base": 62,
    },
    "BEAR_TRAP": {
        "signal": "bullish_reversal",
        "reliability": "medium",
        "conditions": [
            "Price breaks below support",
            "Immediate reversal — cannot sustain breakdown",
            "Large volume spike at lows",
            "Price reclaims support quickly"
        ],
        "action": "Watch for volume explosion + reclaim → potential long",
        "timeframe": ["15m"],
        "confidence_base": 60,
        "warning": "Never short after extended intraday decline without checking exhaustion"
    },
    "MA_ALIGNMENT_BULLISH": {
        "signal": "bullish",
        "reliability": "high",
        "conditions": [
            "20MA > 50MA > 200MA",
            "Price above all three MAs"
        ],
        "action": "Buy pullbacks to 20MA",
        "timeframe": ["1h", "daily"],
        "confidence_base": 73,
    },
    "MA_ALIGNMENT_BEARISH": {
        "signal": "bearish",
        "reliability": "high",
        "conditions": [
            "20MA < 50MA < 200MA",
            "Price below all three MAs"
        ],
        "action": "Sell rallies to 20MA",
        "timeframe": ["1h", "daily"],
        "confidence_base": 73,
    },
}


# ─── UPDATED RULES (from training feedback) ──────────────────────────────────

UPDATED_RULES = [
    {
        "id": "R01",
        "source": "Bharat Bijlee + Axis + Bajaj training",
        "rule": "Lower highs + weak bounce = bearish continuation. "
                "Default assumption when lower highs persist is BEARISH until proven otherwise.",
        "replaces": "Support + consolidation = buy",
        "confidence_delta": +15,  # add to bearish calls when this pattern present
    },
    {
        "id": "R02",
        "source": "Bajaj Finance training",
        "rule": "Support tested 3+ times becomes WEAKER. Each retest removes buyers. "
                "Third or fourth support test = higher breakdown probability.",
        "confidence_delta": +10,  # add to bearish on 3rd+ support test
    },
    {
        "id": "R03",
        "source": "Muthoot Finance training",
        "rule": "Never short after extended intraday decline without exhaustion check. "
                "Large volume spike at support could be absorption, not continuation.",
        "checklist": [
            "Has price already fallen significantly today?",
            "Is volume expanding at the lows?",
            "Are candles losing downside momentum?",
        ],
        "action": "If all 3 YES → reduce bearish confidence by 15 points, consider WAIT",
    },
    {
        "id": "R04",
        "source": "Netweb Technologies training",
        "rule": "Stock already moved >20% in recent sessions + breaking out again = reduce confidence 15-20 points. "
                "Breakout + large red volume near highs = distribution until proven otherwise.",
        "confidence_delta": -18,  # penalty for extended breakout chasing
    },
    {
        "id": "R05",
        "source": "Reliance training",
        "rule": "Relief rally with no volume expansion, no higher-high structure = trap. "
                "Do not buy support blindly. Support is NOT bullish — "
                "it becomes bullish only when higher lows appear and resistance gets reclaimed.",
    },
    {
        "id": "R06",
        "source": "General learning",
        "rule": "Stock underperforming index on a strong market day = active selling. "
                "If NIFTY up >1% and stock is flat/red = skip or short, do not long.",
    },
    {
        "id": "R07",
        "source": "General learning",
        "rule": "Opening range: Do not enter in first 15 minutes. "
                "Wait for 2 × 15m candles to confirm direction. "
                "9:15 → watch. 9:30 → candle 1 result. 9:45 → candle 2 result. 9:45-9:50 → enter.",
    },
    {
        "id": "R08",
        "source": "General learning",
        "rule": "Low volume recovery after high volume selloff = yellow flag (dead cat). "
                "High volume recovery after low volume selloff = genuine buying.",
    },
]


# ─── GRADE SYSTEM ────────────────────────────────────────────────────────────

GRADE_THRESHOLDS = {
    "A+": {"min_confidence": 80, "min_rr": 2.5, "description": "Exceptional — deploy full 1% risk"},
    "A":  {"min_confidence": 70, "min_rr": 2.0, "description": "Strong — deploy full 1% risk"},
    "B+": {"min_confidence": 60, "min_rr": 1.5, "description": "Good — deploy 0.75% risk"},
    "B":  {"min_confidence": 55, "min_rr": 1.25, "description": "Tradable — deploy 0.5% risk"},
    "C":  {"min_confidence": 45, "min_rr": 1.0,  "description": "Weak — skip unless nothing better"},
    "D":  {"min_confidence": 0,  "min_rr": 0,    "description": "Avoid — no edge"},
}

def grade_signal(confidence: float, rr_ratio: float) -> str:
    for grade, thresh in GRADE_THRESHOLDS.items():
        if confidence >= thresh["min_confidence"] and rr_ratio >= thresh["min_rr"]:
            return grade
    return "D"


# ─── CONFIDENCE CALIBRATION ──────────────────────────────────────────────────

MAX_CONFIDENCE = 85  # Never exceed this — prevents overconfidence

def calibrate_confidence(base: float, modifiers: list) -> float:
    """Apply modifiers (positive or negative) to base confidence."""
    final = base + sum(modifiers)
    return max(20, min(MAX_CONFIDENCE, final))


# ─── INTRADAY TIMING RULES ───────────────────────────────────────────────────

TIMING_RULES = {
    "avoid_entry": ["09:15-09:30"],  # Opening volatility
    "best_entry": ["09:45-11:30", "13:00-14:30"],
    "exit_by": "15:00",  # Square off by 3 PM
    "hard_exit": "15:15",  # Broker auto-squares at 3:20
}
