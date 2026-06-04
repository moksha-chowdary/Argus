"""
ARGUS V3 — News Fetcher
Scrapes RSS feeds from Moneycontrol, ET, Business Standard, Mint.
No API key required.
"""
import json
import time
import hashlib
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional

import requests
from config import NEWS_FEEDS, CACHE_DIR, NEWS_CACHE_HOURS


@dataclass
class NewsItem:
    title: str
    summary: str
    source: str
    url: str
    published: str
    fetched_at: str = ""

    def __post_init__(self):
        if not self.fetched_at:
            self.fetched_at = datetime.now().isoformat()

    @property
    def full_text(self) -> str:
        return f"{self.title}. {self.summary}"


class NewsFetcher:
    """Fetches and caches NSE market news from free RSS feeds."""

    CACHE_FILE = CACHE_DIR / "news_cache.json"
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    def get_news(self, force_refresh: bool = False) -> list[NewsItem]:
        """Return cached news or fetch fresh if stale."""
        if not force_refresh and self._cache_valid():
            return self._load_cache()
        items = self._fetch_all()
        self._save_cache(items)
        return items

    def _cache_valid(self) -> bool:
        if not self.CACHE_FILE.exists():
            return False
        try:
            data = json.loads(self.CACHE_FILE.read_text(encoding="utf-8"))
            fetched = datetime.fromisoformat(data.get("fetched_at", "2000-01-01"))
            return datetime.now() - fetched < timedelta(hours=NEWS_CACHE_HOURS)
        except Exception:
            return False

    def _load_cache(self) -> list[NewsItem]:
        try:
            data = json.loads(self.CACHE_FILE.read_text(encoding="utf-8"))
            return [NewsItem(**item) for item in data.get("items", [])]
        except Exception:
            return []

    def _save_cache(self, items: list[NewsItem]):
        data = {
            "fetched_at": datetime.now().isoformat(),
            "items": [asdict(i) for i in items],
        }
        self.CACHE_FILE.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def _fetch_all(self) -> list[NewsItem]:
        all_items = []
        seen = set()
        for url in NEWS_FEEDS:
            try:
                items = self._fetch_feed(url)
                for item in items:
                    h = hashlib.md5(item.title.encode()).hexdigest()
                    if h not in seen:
                        seen.add(h)
                        all_items.append(item)
            except Exception:
                continue
        return all_items[:80]  # keep top 80 headlines

    def _fetch_feed(self, url: str) -> list[NewsItem]:
        try:
            r = requests.get(url, headers=self.HEADERS, timeout=10)
            r.raise_for_status()
            root = ET.fromstring(r.content)
        except Exception:
            return []

        source = self._source_name(url)
        items = []

        # Handle both RSS 2.0 and Atom
        entries = root.findall(".//item") or root.findall(".//{http://www.w3.org/2005/Atom}entry")
        for entry in entries[:20]:
            title   = self._text(entry, ["title"])
            summary = self._text(entry, ["description", "summary", "{http://www.w3.org/2005/Atom}summary"])
            link    = self._text(entry, ["link", "guid"])
            pubdate = self._text(entry, ["pubDate", "published", "updated"])

            if not title:
                continue

            items.append(NewsItem(
                title     = title.strip()[:200],
                summary   = (summary or "").strip()[:400],
                source    = source,
                url       = (link or "").strip(),
                published = (pubdate or "").strip(),
            ))
        return items

    def _text(self, el, tags: list[str]) -> Optional[str]:
        for tag in tags:
            child = el.find(tag)
            if child is not None and child.text:
                return child.text.strip()
        return None

    def _source_name(self, url: str) -> str:
        if "moneycontrol" in url: return "Moneycontrol"
        if "economictimes" in url: return "Economic Times"
        if "business-standard" in url: return "Business Standard"
        if "livemint" in url: return "Mint"
        return "News"
