"""
ARGUS V4 — Live Price Tracker
Fetches NSE prices via yfinance. Respects the live_feed toggle.
"""

import threading, time, json
from datetime import datetime
from typing import Optional
import yfinance as yf
from config import WATCHLIST, NIFTY_TICKER, BANKNIFTY_TICKER, PRICE_REFRESH_SEC


class PriceTracker:
    def __init__(self):
        self._prices: dict  = {}          # ticker → {price, change_pct, high, low, volume}
        self._lock          = threading.Lock()
        self._running       = False
        self._thread: Optional[threading.Thread] = None
        self._last_update   = None

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self):
        """Start background polling thread."""
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop polling (called when live_feed toggled OFF)."""
        self._running = False

    def get(self, ticker: str) -> Optional[dict]:
        """Return latest data for a ticker (e.g. 'ICICIBANK.NS')."""
        with self._lock:
            return self._prices.get(ticker)

    def get_nifty(self) -> Optional[dict]:
        with self._lock:
            return self._prices.get(NIFTY_TICKER)

    def get_banknifty(self) -> Optional[dict]:
        with self._lock:
            return self._prices.get(BANKNIFTY_TICKER)

    def snapshot(self) -> dict:
        """Return all current prices as dict."""
        with self._lock:
            return dict(self._prices)

    def last_update(self) -> Optional[str]:
        return self._last_update

    def fetch_once(self, tickers: list[str]) -> dict:
        """One-shot fetch (used when live feed is OFF but user explicitly asks)."""
        return self._fetch(tickers)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _loop(self):
        all_tickers = WATCHLIST + [NIFTY_TICKER, BANKNIFTY_TICKER]
        while self._running:
            try:
                data = self._fetch(all_tickers)
                with self._lock:
                    self._prices.update(data)
                self._last_update = datetime.now().strftime("%H:%M:%S")
            except Exception:
                pass
            time.sleep(PRICE_REFRESH_SEC)

    def _fetch(self, tickers: list[str]) -> dict:
        result = {}
        try:
            raw = yf.download(
                tickers,
                period="1d",
                interval="1m",
                progress=False,
                threads=True,
                auto_adjust=True,
            )
            # yfinance returns MultiIndex when multiple tickers
            for t in tickers:
                try:
                    if len(tickers) == 1:
                        close  = float(raw["Close"].iloc[-1])
                        open_  = float(raw["Open"].iloc[0])
                        high   = float(raw["High"].max())
                        low    = float(raw["Low"].min())
                        vol    = int(raw["Volume"].sum())
                    else:
                        close  = float(raw["Close"][t].iloc[-1])
                        open_  = float(raw["Open"][t].iloc[0])
                        high   = float(raw["High"][t].max())
                        low    = float(raw["Low"][t].min())
                        vol    = int(raw["Volume"][t].sum())

                    chg_pct = round((close - open_) / open_ * 100, 2) if open_ else 0.0
                    result[t] = {
                        "price":      round(close, 2),
                        "change_pct": chg_pct,
                        "high":       round(high, 2),
                        "low":        round(low, 2),
                        "volume":     vol,
                        "open":       round(open_, 2),
                    }
                except Exception:
                    pass
        except Exception:
            pass
        return result
