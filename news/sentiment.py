"""
ARGUS V3 — News Sentiment Engine
Analyzes headlines → sector sentiment scores → trading bias.
No ML model needed — keyword-based with LLM refinement.
"""
import re
from dataclasses import dataclass, field
from config import SECTORS

# ── Sentiment keywords ────────────────────────────────────────────────────────

BULLISH_WORDS = {
    "strong", "surge", "rally", "jump", "soar", "gain", "rise", "up",
    "beat", "record", "high", "growth", "profit", "positive", "boost",
    "upgrade", "buy", "outperform", "bullish", "recovery", "rebound",
    "breakout", "expand", "revenue", "earnings beat", "q4 profit",
    "dividend", "acquisition", "order", "wins", "contract", "approved",
    "launches", "milestone", "fii buying", "dii buying", "inflow",
}

BEARISH_WORDS = {
    "fall", "drop", "crash", "decline", "down", "loss", "weak", "miss",
    "cut", "downgrade", "sell", "underperform", "bearish", "concern",
    "risk", "warning", "slow", "disappoint", "profit warning", "below",
    "outflow", "fii selling", "dii selling", "ban", "penalty", "fine",
    "recall", "fraud", "scam", "default", "debt", "pressure", "caution",
    "volatile", "uncertainty", "inflation", "rate hike", "recession",
}

# Sector-specific keywords to detect which sector a headline belongs to
SECTOR_KEYWORDS = {
    "banking":   ["bank", "nbfc", "rbi", "repo", "credit", "loan", "npa", "lending", "deposit"],
    "pharma":    ["pharma", "drug", "fda", "medicine", "hospital", "healthcare", "usfda", "api"],
    "it":        ["software", "it ", "tech", "digital", "cloud", "ai", "infosys", "wipro", "tcs"],
    "auto":      ["auto", "vehicle", "car", "ev", "electric vehicle", "two-wheeler", "suv", "tractor"],
    "energy":    ["oil", "gas", "power", "energy", "reliance", "ongc", "fuel", "electricity", "solar"],
    "metals":    ["steel", "metal", "aluminium", "copper", "iron", "ore", "mining", "coal"],
    "fmcg":      ["fmcg", "consumer", "food", "beverage", "rural", "demand", "inflation", "pricing"],
    "realty":    ["real estate", "realty", "housing", "property", "construction", "cement", "dlf"],
    "finance":   ["finance", "bajaj finance", "muthoot", "gold loan", "asset", "npa"],
    "market":    ["nifty", "sensex", "market", "index", "fii", "dii", "sebi", "ipo", "equity"],
}


@dataclass
class SectorSentiment:
    sector: str
    score: float          # -1.0 (very bearish) to +1.0 (very bullish)
    signal: str           # "bullish" | "bearish" | "neutral"
    headline_count: int
    key_headlines: list   # top 3 relevant headlines
    stocks: list          # stocks in this sector


@dataclass
class MarketSentiment:
    date: str
    overall_score: float
    overall_signal: str
    sectors: dict         # sector_name -> SectorSentiment
    top_bullish: list     # top 3 bullish headlines
    top_bearish: list     # top 3 bearish headlines
    summary: str          # one-line market mood


class SentimentAnalyzer:
    """Keyword-based sentiment engine for NSE market news."""

    def analyze(self, news_items: list) -> MarketSentiment:
        from datetime import date
        sector_data = {s: {"bull": 0, "bear": 0, "headlines": []} for s in SECTOR_KEYWORDS}

        all_bull = []
        all_bear = []

        for item in news_items:
            text = (item.title + " " + item.summary).lower()
            bull_score = sum(1 for w in BULLISH_WORDS if w in text)
            bear_score = sum(1 for w in BEARISH_WORDS if w in text)

            # Detect sector
            for sector, keywords in SECTOR_KEYWORDS.items():
                if any(kw in text for kw in keywords):
                    sector_data[sector]["headlines"].append(item.title)
                    sector_data[sector]["bull"] += bull_score
                    sector_data[sector]["bear"] += bear_score

            # Track top headlines
            if bull_score > bear_score and bull_score >= 2:
                all_bull.append((bull_score, item.title))
            elif bear_score > bull_score and bear_score >= 2:
                all_bear.append((bear_score, item.title))

        # Build sector sentiments
        sectors = {}
        total_score = 0.0
        for sector, data in sector_data.items():
            if sector == "market":
                continue
            total = data["bull"] + data["bear"]
            if total == 0:
                score = 0.0
            else:
                score = round((data["bull"] - data["bear"]) / total, 2)
            signal = "bullish" if score > 0.1 else "bearish" if score < -0.1 else "neutral"
            sectors[sector] = SectorSentiment(
                sector=sector,
                score=score,
                signal=signal,
                headline_count=len(data["headlines"]),
                key_headlines=data["headlines"][:3],
                stocks=SECTORS.get(sector, []),
            )
            total_score += score

        overall = round(total_score / max(len(sectors), 1), 2)
        overall_signal = "bullish" if overall > 0.05 else "bearish" if overall < -0.05 else "neutral"

        # Sort top headlines
        top_bull = [h for _, h in sorted(all_bull, reverse=True)[:3]]
        top_bear = [h for _, h in sorted(all_bear, reverse=True)[:3]]

        summary = self._one_liner(overall_signal, sectors)

        return MarketSentiment(
            date=date.today().isoformat(),
            overall_score=overall,
            overall_signal=overall_signal,
            sectors=sectors,
            top_bullish=top_bull,
            top_bearish=top_bear,
            summary=summary,
        )

    def get_stock_sentiment(self, ticker: str, sentiment: MarketSentiment) -> tuple[str, float]:
        """Return (signal, score) for a specific stock ticker."""
        ticker = ticker.upper()
        for sector_name, sec in sentiment.sectors.items():
            if ticker in [s.upper() for s in sec.stocks]:
                return sec.signal, sec.score
        return "neutral", 0.0

    def _one_liner(self, overall: str, sectors: dict) -> str:
        bull_sectors = [s for s, d in sectors.items() if d.signal == "bullish"]
        bear_sectors = [s for s, d in sectors.items() if d.signal == "bearish"]
        if overall == "bullish":
            return f"Market tone bullish. Strong sectors: {', '.join(bull_sectors[:3]) or 'broad'}."
        if overall == "bearish":
            return f"Market tone bearish. Weak sectors: {', '.join(bear_sectors[:3]) or 'broad'}."
        return f"Mixed market. Bullish: {', '.join(bull_sectors[:2])}. Bearish: {', '.join(bear_sectors[:2])}."
