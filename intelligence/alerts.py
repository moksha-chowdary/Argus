"""
ARGUS V4 — Price Alert System
Set price triggers → ARGUS beeps + prints when crossed.
Usage: alert ICICIBANK above 1320
       alert RELIANCE below 1300
"""

import threading, time
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class PriceAlert:
    id:        str
    ticker:    str          # e.g. "ICICIBANK.NS"
    direction: str          # "above" / "below"
    price:     float
    triggered: bool = False
    note:      str  = ""


class AlertManager:
    def __init__(self, tracker, on_trigger: Callable[[PriceAlert, float], None]):
        self._tracker    = tracker
        self._on_trigger = on_trigger
        self._alerts: list[PriceAlert] = []
        self._lock    = threading.Lock()
        self._running = False
        self._thread  = None

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def add(self, ticker: str, direction: str, price: float, note: str = "") -> str:
        import uuid
        aid = str(uuid.uuid4())[:6]
        # Normalize ticker
        ns = ticker.upper()
        if not ns.endswith(".NS") and not ns.startswith("^"):
            ns += ".NS"
        alert = PriceAlert(id=aid, ticker=ns, direction=direction,
                           price=price, note=note)
        with self._lock:
            self._alerts.append(alert)
        return aid

    def remove(self, alert_id: str) -> bool:
        with self._lock:
            before = len(self._alerts)
            self._alerts = [a for a in self._alerts if a.id != alert_id]
            return len(self._alerts) < before

    def list_alerts(self) -> list:
        with self._lock:
            return [a for a in self._alerts if not a.triggered]

    def clear(self):
        with self._lock:
            self._alerts = []

    def _loop(self):
        while self._running:
            try:
                with self._lock:
                    pending = [a for a in self._alerts if not a.triggered]

                for alert in pending:
                    data = self._tracker.get(alert.ticker)
                    if not data:
                        continue
                    price = data["price"]
                    hit   = False
                    if alert.direction == "above" and price >= alert.price:
                        hit = True
                    elif alert.direction == "below" and price <= alert.price:
                        hit = True

                    if hit:
                        with self._lock:
                            alert.triggered = True
                        self._on_trigger(alert, price)
            except Exception:
                pass
            time.sleep(3)   # check every 3 seconds


def _beep():
    """Cross-platform terminal bell."""
    try:
        import sys
        sys.stdout.write("\a")
        sys.stdout.flush()
    except Exception:
        pass
