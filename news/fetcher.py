"""
ARGUS V4 — News Fetcher + Sentiment Engine
Pulls from free RSS feeds. No API key needed.
Scores sectors as bull/bear based on keyword matching.
"""

import feedparser, re, json
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional
from config import NEWS_FEEDS


# ─── Sector keyword maps ──────────────────────────────────────────────────────

SECTOR_KEYWORDS = {
    "BANKING":  ["bank", "rbi", "repo rate", "credit", "npa", "lending", "hdfc", "icici", "kotak", "axis"],
    "PHARMA":   ["pharma", "drug", "fda", "medicine", "hospital", "healthcare", "sun pharma", "cipla", "divi"],
    "IT":       ["it sector", "software", "infosys", "tcs", "wipro", "tech mahindra", "hcl", "export", "rupee"],
    "AUTO":     ["auto", "ev", "electric vehicle", "tata motors", "maruti", "tvs", "bajaj auto", "m&m"],
    "ENERGY":   ["oil", "gas", "power", "reliance", "adani", "ongc", "ntpc", "coal", "renewable"],
    "NBFC":     ["nbfc", "bajaj finance", "muthoot", "chola", "pfc", "recl", "gold loan"],
    "METAL":    ["steel", "metal", "aluminium", "tata steel", "jsw", "hindalco", "vedanta"],
    "FMCG":     ["fmcg", "consumer", "itc", "hindustan unilever", "nestle", "dabur", "britannia"],
}

BULLISH_WORDS = [
    "surge", "rally", "gain", "rise", "jump", "strong", "growth", "profit",
    "beat", "record", "upgrade", "buy", "positive", "outperform", "bullish",
    "expansion", "contract", "increase", "recovery", "boost", "upside",
]
BEARISH_WORDS = [
    "fall", "drop", "decline", "loss", "weak", "miss", "downgrade", "sell",
    "risk", "concern", "pressure", "bearish", "cut", "slowdown", "warn",
    "crash", "tumble", "plunge", "deficit", "debt", "default",
]


@dataclass
class SectorSentiment:
    sector: str
    score:  float          # -1.0 (very bearish) to +1.0 (very bullish)
    signal: str            # "bullish" / "bearish" / "neutral"
    headlines: list[str]   = field(default_factory=list)


@dataclass
class MarketSentiment:
    overall_signal: str                     # "bullish" / "bearish" / "neutral"
    overall_score:  float
    sectors:        dict[str, SectorSentiment] = field(default_factory=dict)
    top_headlines:  list[str]               = field(default_factory=list)
    fetched_at:     str                     = ""
    summary:        str                     = ""


class NewsFetcher:
    def __init__(self):
        self._cache: Optional[MarketSentiment] = None
        self._cache_time: Optional[datetime]   = None
        self._cache_ttl = timedelta(minutes=30)

    def fetch(self, force: bool = False) -> MarketSentiment:
        """Fetch + score headlines. Cached for 30 min unless force=True."""
        if (not force and self._cache and self._cache_time
                and datetime.now() - self._cache_time < self._cache_ttl):
            return self._cache

        headlines = self._pull_headlines()
        sentiment = self._score(headlines)
        self._cache = sentiment
        self._cache_time = datetime.now()
        return sentiment

    # ── Internal ──────────────────────────────────────────────────────────────

    def _pull_headlines(self) -> list[str]:
        headlines = []
        for url in NEWS_FEEDS:
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:15]:
                    title = entry.get("title", "")
                    if title:
                        headlines.append(title.lower().strip())
            except Exception:
                pass
        return headlines

    def _score(self, headlines: list[str]) -> MarketSentiment:
        sector_hits: dict[str, list[str]] = {s: [] for s in SECTOR_KEYWORDS}

        # Match headlines to sectors
        for h in headlines:
            for sector, keywords in SECTOR_KEYWORDS.items():
                if any(kw in h for kw in keywords):
                    sector_hits[sector].append(h)

        # Score each sector
        sectors: dict[str, SectorSentiment] = {}
        for sector, matched in sector_hits.items():
            if not matched:
                continue
            score = 0.0
            for h in matched:
                bull = sum(1 for w in BULLISH_WORDS if w in h)
                bear = sum(1 for w in BEARISH_WORDS if w in h)
                score += (bull - bear)
            norm = score / max(len(matched), 1)
            norm = max(-1.0, min(1.0, norm))
            signal = "bullish" if norm > 0.1 else ("bearish" if norm < -0.1 else "neutral")
            sectors[sector] = SectorSentiment(
                sector=sector,
                score=round(norm, 2),
                signal=signal,
                headlines=matched[:3],
            )

        # Overall
        if sectors:
            overall = sum(s.score for s in sectors.values()) / len(sectors)
        else:
            overall = 0.0

        overall_signal = "bullish" if overall > 0.1 else ("bearish" if overall < -0.1 else "neutral")

        # Top 5 headlines
        top = headlines[:5]

        # Human-readable summary
        bull_sectors = [s for s, v in sectors.items() if v.signal == "bullish"]
        bear_sectors = [s for s, v in sectors.items() if v.signal == "bearish"]
        parts = []
        if bull_sectors:
            parts.append(f"Bullish: {', '.join(bull_sectors)}")
        if bear_sectors:
            parts.append(f"Bearish: {', '.join(bear_sectors)}")
        summary = " | ".join(parts) if parts else "No strong sector signals today"

        return MarketSentiment(
            overall_signal=overall_signal,
            overall_score=round(overall, 2),
            sectors=sectors,
            top_headlines=top,
            fetched_at=datetime.now().strftime("%H:%M"),
            summary=summary,
        )
