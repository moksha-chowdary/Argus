"""
ARGUS V3 — Central Configuration
Edit this file to customize your setup.
"""
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent
DATA_DIR    = ROOT / "data"
TRADES_DIR  = DATA_DIR / "trades"
CACHE_DIR   = DATA_DIR / "news_cache"
MEMORY_DIR  = DATA_DIR / "memory"

for d in [DATA_DIR, TRADES_DIR, CACHE_DIR, MEMORY_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Ollama ────────────────────────────────────────────────────────────────────
OLLAMA_URL   = "http://localhost:11434"
OLLAMA_MODEL = "gemma4:latest"

# ── Trading ───────────────────────────────────────────────────────────────────
CAPITAL          = 400000   # ₹4,00,000
MAX_RISK_PCT     = 0.01     # 1% per trade = ₹4,000
MAX_ALLOCATION   = 0.30     # max 30% capital in one stock

# ── News ──────────────────────────────────────────────────────────────────────
NEWS_CACHE_HOURS = 2        # re-fetch after 2 hours
NEWS_FEEDS = [
    # Moneycontrol
    "https://www.moneycontrol.com/rss/latestnews.xml",
    "https://www.moneycontrol.com/rss/marketreports.xml",
    # Economic Times
    "https://economictimes.indiatimes.com/markets/stocks/rss.cms",
    "https://economictimes.indiatimes.com/markets/rss.cms",
    # Business Standard
    "https://www.business-standard.com/rss/markets-106.rss",
    # Mint
    "https://www.livemint.com/rss/markets",
]

# ── NSE Sectors ───────────────────────────────────────────────────────────────
SECTORS = {
    "banking":    ["HDFCBANK","ICICIBANK","AXISBANK","SBIN","KOTAKBANK","INDUSINDBK","BANDHANBNK"],
    "pharma":     ["SUNPHARMA","DRREDDY","CIPLA","DIVISLAB","APOLLOHOSP","TORNTPHARM"],
    "it":         ["TCS","INFY","WIPRO","HCLTECH","TECHM","LTIM","PERSISTENT"],
    "auto":       ["TATAMOTORS","MARUTI","BAJAJ-AUTO","HEROMOTOCO","TVSMOTOR","EICHERMOT"],
    "energy":     ["RELIANCE","ONGC","NTPC","POWERGRID","BPCL","IOC"],
    "metals":     ["TATASTEEL","JSWSTEEL","HINDALCO","COALINDIA","VEDL","NMDC"],
    "fmcg":       ["HINDUNILVR","ITC","NESTLEIND","BRITANNIA","DABUR","MARICO"],
    "realty":     ["DLF","GODREJPROP","OBEROIRLTY","PRESTIGE","BRIGADE"],
    "finance":    ["BAJFINANCE","BAJAJFINSV","MUTHOOTFIN","CHOLAFIN","M&MFIN"],
}

# ── Signal thresholds ─────────────────────────────────────────────────────────
MIN_RR_RATIO     = 1.5      # minimum risk/reward to recommend entry
MIN_CONFIDENCE   = 0.35     # minimum vision confidence to trust signal
NEWS_WEIGHT      = 0.25     # how much news sentiment influences final signal
