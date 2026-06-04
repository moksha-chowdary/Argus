"""
ARGUS Live Data Layer
Primary: Zerodha Kite API (real-time, free with Zerodha account)
Fallback: yfinance (15min delayed, no account needed)
NSE India only.
"""

import os
import time
import threading
from datetime import datetime, timedelta
from typing import Optional
import yfinance as yf

# Try importing Kite — optional dependency
try:
    from kiteconnect import KiteConnect, KiteTicker
    KITE_AVAILABLE = True
except ImportError:
    KITE_AVAILABLE = False


# ─── CONFIG ──────────────────────────────────────────────────────────────────

KITE_API_KEY    = os.getenv("KITE_API_KEY", "")
KITE_API_SECRET = os.getenv("KITE_API_SECRET", "")
KITE_ACCESS_TOKEN = os.getenv("KITE_ACCESS_TOKEN", "")

# NSE symbol suffix for yfinance
def nse_symbol(ticker: str) -> str:
    ticker = ticker.upper().strip()
    if not ticker.endswith(".NS"):
        return ticker + ".NS"
    return ticker


# ─── QUOTE ───────────────────────────────────────────────────────────────────

class LiveQuote:
    def __init__(self):
        self.kite = None
        self._init_kite()

    def _init_kite(self):
        if KITE_AVAILABLE and KITE_API_KEY and KITE_ACCESS_TOKEN:
            try:
                self.kite = KiteConnect(api_key=KITE_API_KEY)
                self.kite.set_access_token(KITE_ACCESS_TOKEN)
            except Exception:
                self.kite = None

    def get_quote(self, symbol: str) -> dict:
        """Get live quote. Falls back to yfinance if Kite not configured."""
        if self.kite:
            return self._kite_quote(symbol)
        return self._yfinance_quote(symbol)

    def _kite_quote(self, symbol: str) -> dict:
        try:
            data = self.kite.quote([f"NSE:{symbol}"])
            q = data[f"NSE:{symbol}"]
            return {
                "symbol": symbol,
                "price": q["last_price"],
                "open": q["ohlc"]["open"],
                "high": q["ohlc"]["high"],
                "low": q["ohlc"]["low"],
                "close": q["ohlc"]["close"],
                "volume": q["volume"],
                "change": q["net_change"],
                "change_pct": round(q["net_change"] / q["ohlc"]["close"] * 100, 2),
                "source": "kite_live",
                "timestamp": datetime.now().strftime("%H:%M:%S"),
            }
        except Exception as e:
            return self._yfinance_quote(symbol)

    def _yfinance_quote(self, symbol: str) -> dict:
        try:
            ticker = yf.Ticker(nse_symbol(symbol))
            info = ticker.fast_info
            hist = ticker.history(period="2d", interval="1m")
            if hist.empty:
                hist = ticker.history(period="5d", interval="5m")

            price = float(info.last_price) if hasattr(info, 'last_price') else float(hist["Close"].iloc[-1])
            prev_close = float(info.previous_close) if hasattr(info, 'previous_close') else float(hist["Close"].iloc[-2]) if len(hist) > 1 else price

            today = hist[hist.index.date == hist.index[-1].date()] if not hist.empty else hist
            return {
                "symbol": symbol,
                "price": round(price, 2),
                "open": round(float(today["Open"].iloc[0]), 2) if not today.empty else 0,
                "high": round(float(today["High"].max()), 2) if not today.empty else 0,
                "low": round(float(today["Low"].min()), 2) if not today.empty else 0,
                "close": round(prev_close, 2),
                "volume": int(today["Volume"].sum()) if not today.empty else 0,
                "change": round(price - prev_close, 2),
                "change_pct": round((price - prev_close) / prev_close * 100, 2) if prev_close else 0,
                "source": "yfinance_delayed",
                "timestamp": datetime.now().strftime("%H:%M:%S"),
            }
        except Exception as e:
            return {"symbol": symbol, "error": str(e), "price": 0}

    def get_multiple(self, symbols: list) -> dict:
        results = {}
        for sym in symbols:
            results[sym] = self.get_quote(sym)
            time.sleep(0.2)  # Rate limit
        return results


# ─── HISTORICAL DATA ──────────────────────────────────────────────────────────

class HistoricalData:
    def __init__(self, live_quote: LiveQuote = None):
        self.lq = live_quote or LiveQuote()

    def get_daily(self, symbol: str, months: int = 4) -> "pd.DataFrame":
        """Get daily OHLCV — for trend analysis (3-4 month chart)."""
        import pandas as pd
        try:
            ticker = yf.Ticker(nse_symbol(symbol))
            df = ticker.history(period=f"{months}mo", interval="1d")
            df.index = df.index.tz_localize(None)
            return df[["Open", "High", "Low", "Close", "Volume"]]
        except Exception as e:
            return pd.DataFrame()

    def get_intraday(self, symbol: str, days: int = 3) -> "pd.DataFrame":
        """Get 15min OHLCV — for execution chart (3-day chart)."""
        import pandas as pd
        try:
            ticker = yf.Ticker(nse_symbol(symbol))
            df = ticker.history(period=f"{days}d", interval="15m")
            df.index = df.index.tz_localize(None)
            return df[["Open", "High", "Low", "Close", "Volume"]]
        except Exception as e:
            return pd.DataFrame()

    def get_nifty(self) -> dict:
        """Get Nifty 50 current status."""
        return self.lq.get_quote("^NSEI")

    def get_banknifty(self) -> dict:
        return self.lq.get_quote("^NSEBANK")


# ─── WATCHLIST SCREENER ───────────────────────────────────────────────────────

TOP_NSE_STOCKS = [
    "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
    "AXISBANK", "SBIN", "KOTAKBANK", "BAJFINANCE", "WIPRO",
    "SUNPHARMA", "TATAMOTORS", "ADANIENT", "MARUTI", "LT",
    "NTPC", "POWERGRID", "COALINDIA", "ONGC", "ITC",
]

class MarketScreener:
    def __init__(self):
        self.lq = LiveQuote()

    def morning_scan(self, watchlist: list = None) -> list:
        """
        Run morning scan — returns stocks sorted by gap + volume signal.
        Call this at market open (9:15 AM).
        """
        symbols = watchlist or TOP_NSE_STOCKS[:10]  # Limit to avoid rate limits
        quotes = self.lq.get_multiple(symbols)

        results = []
        for sym, q in quotes.items():
            if "error" in q or q.get("price", 0) == 0:
                continue
            score = 0
            # Gap scoring
            gap = q.get("change_pct", 0)
            if abs(gap) > 2:
                score += 2
            elif abs(gap) > 1:
                score += 1
            # Directional
            direction = "bullish" if gap > 0 else "bearish" if gap < 0 else "neutral"
            results.append({
                "symbol": sym,
                "price": q["price"],
                "change_pct": gap,
                "direction": direction,
                "volume": q.get("volume", 0),
                "score": score,
            })

        results.sort(key=lambda x: abs(x["change_pct"]), reverse=True)
        return results

    def index_health(self) -> dict:
        """Check if market is bullish or bearish today."""
        nifty = self.lq.get_quote("^NSEI")
        banknifty = self.lq.get_quote("^NSEBANK")
        nifty_chg = nifty.get("change_pct", 0)
        bank_chg = banknifty.get("change_pct", 0)

        if nifty_chg > 1.0:
            bias = "STRONG_BULL"
        elif nifty_chg > 0.3:
            bias = "MILD_BULL"
        elif nifty_chg < -1.0:
            bias = "STRONG_BEAR"
        elif nifty_chg < -0.3:
            bias = "MILD_BEAR"
        else:
            bias = "NEUTRAL"

        return {
            "nifty": nifty.get("price", 0),
            "nifty_change": nifty_chg,
            "banknifty": banknifty.get("price", 0),
            "banknifty_change": bank_chg,
            "bias": bias,
            "trading_advice": _bias_advice(bias),
        }


def _bias_advice(bias: str) -> str:
    return {
        "STRONG_BULL": "Green day — favor longs. Avoid shorts unless stock-specific weakness.",
        "MILD_BULL": "Mild upside. Be selective. Only A/A+ long setups.",
        "NEUTRAL": "No edge from market. Need stock-specific setups.",
        "MILD_BEAR": "Cautious. Prefer shorts or cash. Avoid weak longs.",
        "STRONG_BEAR": "Bear day — favor shorts. No aggressive longs.",
    }.get(bias, "Check market manually.")
