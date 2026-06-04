"""
ARGUS V4 — Live Price Tracker + OHLCV candle fetcher
- Background polling for watchlist prices
- Per-stock 15m candle data for indicator calculation
- yfinance error spam suppressed completely
"""

import threading, time, logging, warnings, contextlib, io
from datetime import datetime
from typing import Optional
import pandas as pd

# Suppress ALL yfinance/peewee noise before import
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("peewee").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore")

import yfinance as yf
from config import WATCHLIST, NIFTY_TICKER, BANKNIFTY_TICKER, PRICE_REFRESH_SEC


class PriceTracker:
    def __init__(self):
        self._prices: dict = {}
        self._lock         = threading.Lock()
        self._running      = False
        self._thread: Optional[threading.Thread] = None
        self._last_update  = None

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def get(self, ticker: str) -> Optional[dict]:
        with self._lock:
            return self._prices.get(ticker)

    def get_nifty(self) -> Optional[dict]:
        with self._lock:
            return self._prices.get(NIFTY_TICKER)

    def get_banknifty(self) -> Optional[dict]:
        with self._lock:
            return self._prices.get(BANKNIFTY_TICKER)

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._prices)

    def last_update(self) -> Optional[str]:
        return self._last_update

    def fetch_candles(self, ticker: str, interval: str = "15m", period: str = "5d") -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV candles for a specific stock.
        Used by the brain for real indicator calculation.
        Returns a DataFrame with columns: Open, High, Low, Close, Volume
        """
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                df = yf.download(
                    ticker, period=period, interval=interval,
                    progress=False, auto_adjust=True,
                )
            if df is None or df.empty:
                return None
            # Flatten MultiIndex columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.dropna()
            return df if len(df) >= 5 else None
        except Exception:
            return None

    def fetch_once(self, tickers: list) -> dict:
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

    def _fetch(self, tickers: list) -> dict:
        result = {}
        if not tickers:
            return result
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                raw = yf.download(
                    tickers, period="1d", interval="1m",
                    progress=False, threads=True, auto_adjust=True,
                )
            if raw is None or raw.empty:
                return result

            # Handle single vs multi-ticker response
            is_multi = isinstance(raw.columns, pd.MultiIndex)

            for t in tickers:
                try:
                    if is_multi:
                        close  = float(raw["Close"][t].dropna().iloc[-1])
                        open_  = float(raw["Open"][t].dropna().iloc[0])
                        high   = float(raw["High"][t].dropna().max())
                        low    = float(raw["Low"][t].dropna().min())
                        vol    = int(raw["Volume"][t].dropna().sum())
                    else:
                        # Single ticker — no secondary index
                        close  = float(raw["Close"].dropna().iloc[-1])
                        open_  = float(raw["Open"].dropna().iloc[0])
                        high   = float(raw["High"].dropna().max())
                        low    = float(raw["Low"].dropna().min())
                        vol    = int(raw["Volume"].dropna().sum())

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
                    continue
        except Exception:
            pass
        return result
