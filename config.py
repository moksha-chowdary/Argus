"""ARGUS V4 — Central Config"""

import os, logging, warnings

# Suppress noise globally
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("peewee").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore")

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
RULEBOOK_PATH = os.path.join(BASE_DIR, "data", "rulebook.json")
MEMORY_DIR    = os.path.join(BASE_DIR, "data", "memory")
LOG_PATH      = os.path.join(BASE_DIR, "data", "trade_log.json")
STATE_PATH    = os.path.join(BASE_DIR, "data", "state.json")

# Default watch folder — charts dropped here are auto-analyzed
WATCH_FOLDER  = os.path.join(BASE_DIR, "charts")

# Ollama
OLLAMA_URL    = "http://localhost:11434"
OLLAMA_MODEL  = "gemma4:latest"
MAX_TOKENS    = 500
CTX_WINDOW    = 2048
TEMPERATURE   = 0.3

# Capital
CAPITAL       = 400_000
MAX_RISK_PCT  = 0.01

# Live feed
LIVE_FEED_DEFAULT = True
PRICE_REFRESH_SEC = 30
NEWS_REFRESH_MIN  = 30

# Watchlist — verified NSE tickers for yfinance
WATCHLIST = [
    "RELIANCE.NS",  "ICICIBANK.NS",  "HDFCBANK.NS",  "AXISBANK.NS",
    "BAJFINANCE.NS","TATAMOTORS.NS", "WIPRO.NS",      "SUNPHARMA.NS",
    "ADANIENT.NS",  "KOTAKBANK.NS",  "INFY.NS",       "TCS.NS",
]

NIFTY_TICKER     = "^NSEI"
BANKNIFTY_TICKER = "^NSEBANK"

NEWS_FEEDS = [
    "https://www.moneycontrol.com/rss/MCtopnews.xml",
    "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "https://www.business-standard.com/rss/markets-106.rss",
]

# Tesseract — Windows path
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

BANNER = r"""
    ___    ____  ______  __  _______
   /   |  / __ \/ ____/ / / / / ___/
  / /| | / /_/ / / __  / / / /\__ \ 
 / ___ |/ _, _/ /_/ / / /_/ /___/ / 
/_/  |_/_/ |_|\____/  \____//____/  
              V5  —  ML/DL Trading Intelligence
"""
