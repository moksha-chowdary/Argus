"""
ARGUS V4 — Central Config
Edit this file to change behaviour without touching any other code.
"""

import os

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
RULEBOOK_PATH = os.path.join(BASE_DIR, "data", "rulebook.json")
MEMORY_DIR    = os.path.join(BASE_DIR, "data", "memory")
LOG_PATH      = os.path.join(BASE_DIR, "data", "trade_log.json")
STATE_PATH    = os.path.join(BASE_DIR, "data", "state.json")   # persists live_feed toggle

# ─── Ollama ───────────────────────────────────────────────────────────────────
OLLAMA_URL    = "http://localhost:11434"
OLLAMA_MODEL  = "gemma4:latest"
MAX_TOKENS    = 500
CTX_WINDOW    = 2048
TEMPERATURE   = 0.3

# ─── Capital ──────────────────────────────────────────────────────────────────
CAPITAL       = 400_000
MAX_RISK_PCT  = 0.01          # 1 % per trade = ₹4,000

# ─── Live feed (can be toggled at runtime via 'livefeed on/off') ──────────────
LIVE_FEED_DEFAULT = True      # starts ON — user can toggle
PRICE_REFRESH_SEC = 30        # how often to refresh live prices
NEWS_REFRESH_MIN  = 30        # how often to re-fetch news (minutes)

# ─── NSE tickers to watch (auto-fetched on startup if live feed is ON) ────────
WATCHLIST = [
    "RELIANCE.NS", "ICICIBANK.NS", "HDFCBANK.NS", "AXISBANK.NS",
    "BAJFINANCE.NS", "TATAMOTORS.NS", "WIPRO.NS", "SUNPHARMA.NS",
    "ADANIENT.NS", "KOTAKBANK.NS", "INFY.NS", "TCS.NS",
]

# ─── NIFTY / BankNifty indices ────────────────────────────────────────────────
NIFTY_TICKER     = "^NSEI"
BANKNIFTY_TICKER = "^NSEBANK"

# ─── News sources (RSS, no key needed) ───────────────────────────────────────
NEWS_FEEDS = [
    "https://www.moneycontrol.com/rss/MCtopnews.xml",
    "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "https://www.business-standard.com/rss/markets-106.rss",
]

# ─── Tesseract path (Windows) ─────────────────────────────────────────────────
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ─── Display ──────────────────────────────────────────────────────────────────
BANNER = r"""
    ___    ____  ______  __  _______
   /   |  / __ \/ ____/ / / / / ___/
  / /| | / /_/ / / __  / / / /\__ \ 
 / ___ |/ _, _/ /_/ / / /_/ /___/ / 
/_/  |_/_/ |_|\____/  \____//____/  
              V4  —  NSE Intelligence
"""
