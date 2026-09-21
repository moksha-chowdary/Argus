"""
ARGUS V5 — Institutional RSS News Scraper & Deduplicator
Pulls live headlines from premier Indian financial news RSS feeds,
deduplicates via SHA-256 hash in SQLite, classifies by sector/ticker,
and writes point-in-time FinBERT sentiment into the FeatureStore.
"""

import hashlib
import re
import feedparser
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple

from config import WATCHLIST
from data.feature_store import FeatureStore
from news.sentiment_analyzer import FinBERTSentimentAnalyzer

RSS_FEEDS = {
    "EconomicTimes": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "Moneycontrol": "https://www.moneycontrol.com/rss/MCtopnews.xml",
    "BusinessStandard": "https://www.business-standard.com/rss/markets-106.rss",
    "Livemint": "https://www.livemint.com/rss/markets",
}

SECTOR_MAP = {
    "BANKING": ["bank", "rbi", "repo", "credit", "npa", "lending", "hdfc", "icici", "kotak", "axis", "sbin", "pnb"],
    "IT": ["it sector", "software", "infosys", "tcs", "wipro", "hcl tech", "tech mahindra", "cognizant"],
    "AUTO": ["auto", "ev", "electric vehicle", "tata motors", "maruti", "tvs", "bajaj auto", "mahindra", "m&m"],
    "PHARMA": ["pharma", "drug", "fda", "usfda", "sun pharma", "cipla", "dr reddy", "divis", "lupin"],
    "ENERGY": ["oil", "crude", "gas", "power", "reliance", "ongc", "ntpc", "adani power", "coal india", "bpcl"],
    "METALS": ["steel", "metal", "aluminium", "tata steel", "jsw", "hindalco", "vedanta", "sail"],
    "FMCG": ["fmcg", "consumer", "itc", "hindustan unilever", "hul", "nestle", "dabur", "marico"],
    "NBFC": ["nbfc", "bajaj finance", "muthoot", "chola", "shriram", "pfc", "rec"],
}

TICKER_KEYWORD_MAP = {
    "RELIANCE": ["reliance", "ril", "jio"],
    "TCS": ["tcs", "tata consultancy"],
    "INFY": ["infosys", "infy"],
    "HDFCBANK": ["hdfc bank", "hdfc"],
    "ICICIBANK": ["icici bank", "icici"],
    "AXISBANK": ["axis bank", "axis"],
    "KOTAKBANK": ["kotak bank", "kotak"],
    "SBIN": ["sbi", "state bank of india"],
    "BAJFINANCE": ["bajaj finance", "bajfinance"],
    "TATAMOTORS": ["tata motors", "tatamotors"],
    "WIPRO": ["wipro"],
    "SUNPHARMA": ["sun pharma", "sunpharma"],
    "ADANIENT": ["adani enterprises", "adanient"],
    "MARUTI": ["maruti", "maruti suzuki"],
    "LT": ["larsen", "l&t"],
}


class RSSNewsScraper:
    """
    Scrapes Indian financial RSS feeds, dedupes articles, tags sectors/tickers,
    and publishes sentiment updates to the FeatureStore.
    """

    def __init__(
        self,
        feature_store: Optional[FeatureStore] = None,
        sentiment_analyzer: Optional[FinBERTSentimentAnalyzer] = None,
    ):
        self.feature_store = feature_store or FeatureStore()
        self.sentiment_analyzer = sentiment_analyzer or FinBERTSentimentAnalyzer()

    def _hash_article(self, title: str, source: str) -> str:
        """Create a deterministic SHA-256 hash for deduplication."""
        clean = (title.strip().lower() + source.strip().lower()).encode("utf-8")
        return hashlib.sha256(clean).hexdigest()

    def _tag_sectors_and_tickers(self, text: str) -> Tuple[List[str], List[str]]:
        """Identify which sectors and tickers a headline refers to."""
        clean = text.lower()
        matched_sectors = []
        for sector, keywords in SECTOR_MAP.items():
            if any(re.search(rf"\b{re.escape(k)}\b", clean) for k in keywords):
                matched_sectors.append(sector)

        matched_tickers = []
        for ticker, keywords in TICKER_KEYWORD_MAP.items():
            if any(re.search(rf"\b{re.escape(k)}\b", clean) for k in keywords):
                matched_tickers.append(ticker)

        return matched_sectors, matched_tickers

    def scrape_and_process(self) -> Dict[str, Any]:
        """
        Executes a complete scrape cycle:
        1. Ingest RSS entries from all feeds
        2. Deduplicate against seen_articles table
        3. Score new headlines with FinBERT
        4. Group by ticker / sector and compute sentiment delta
        5. Write point-in-time records to feature store
        """
        now_ts = datetime.now(timezone.utc).isoformat()
        new_articles = []

        # 1. Fetch feeds
        for source_name, feed_url in RSS_FEEDS.items():
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:25]:
                    title = entry.get("title", "").strip()
                    summary = entry.get("summary", "").strip()
                    published = entry.get("published", "")

                    if not title:
                        continue

                    art_hash = self._hash_article(title, source_name)
                    # Check SQLite seen cache
                    if self.feature_store.is_article_seen(art_hash):
                        continue

                    # Mark as seen
                    self.feature_store.mark_article_seen(
                        article_hash=art_hash,
                        title=title,
                        source=source_name,
                        published_at=published,
                        scraped_at=now_ts,
                    )

                    combined_text = f"{title}. {summary}" if summary else title
                    sectors, tickers = self._tag_sectors_and_tickers(combined_text)

                    new_articles.append({
                        "hash": art_hash,
                        "title": title,
                        "source": source_name,
                        "text": combined_text,
                        "sectors": sectors,
                        "tickers": tickers,
                        "published": published,
                    })
            except Exception:
                continue

        if not new_articles:
            return {
                "scraped_at": now_ts,
                "new_articles_count": 0,
                "tickers_updated": 0,
                "sectors_updated": 0,
            }

        # 2. Score new articles with FinBERT
        scored_articles = []
        for art in new_articles:
            sent = self.sentiment_analyzer.score_text(art["text"])
            scored_articles.append({**art, "sentiment": sent})

        # 3. Aggregate by ticker and update FeatureStore
        ticker_scores: Dict[str, List[float]] = {}
        for art in scored_articles:
            score = art["sentiment"]["compound"]
            for t in art["tickers"]:
                ticker_scores.setdefault(t, []).append(score)

        for ticker, scores in ticker_scores.items():
            avg_score = round(float(sum(scores) / len(scores)), 4)
            # Retrieve previous score to compute delta
            prev_sent = self.feature_store.get_latest_sentiment(ticker=ticker, asof_time=now_ts)
            prev_score = prev_sent.get("sentiment_score", 0.0)
            delta = round(avg_score - prev_score, 4)

            self.feature_store.save_sentiment(
                timestamp=now_ts,
                ticker=ticker,
                sector=None,
                sentiment_score=avg_score,
                sentiment_delta=delta,
                article_count=len(scores),
            )

        # 4. Aggregate by sector and update FeatureStore
        sector_scores: Dict[str, List[float]] = {}
        for art in scored_articles:
            score = art["sentiment"]["compound"]
            for s in art["sectors"]:
                sector_scores.setdefault(s, []).append(score)

        for sector, scores in sector_scores.items():
            avg_score = round(float(sum(scores) / len(scores)), 4)
            prev_sent = self.feature_store.get_latest_sentiment(sector=sector, asof_time=now_ts)
            prev_score = prev_sent.get("sentiment_score", 0.0)
            delta = round(avg_score - prev_score, 4)

            self.feature_store.save_sentiment(
                timestamp=now_ts,
                ticker=None,
                sector=sector,
                sentiment_score=avg_score,
                sentiment_delta=delta,
                article_count=len(scores),
            )

        return {
            "scraped_at": now_ts,
            "new_articles_count": len(new_articles),
            "tickers_updated": len(ticker_scores),
            "sectors_updated": len(sector_scores),
            "sample_headlines": [a["title"] for a in new_articles[:3]],
        }
