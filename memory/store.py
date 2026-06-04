"""
ARGUS V4 — Persistent Memory (ChromaDB)
Stores every analysis + outcome. Enables per-stock learning.
"""

import json, uuid, os
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional
import chromadb
from config import MEMORY_DIR, RULEBOOK_PATH


@dataclass
class TradeRecord:
    id:           str
    stock:        str
    date:         str
    action:       str
    entry:        float
    stop:         float
    target:       float
    confidence:   float
    patterns:     list
    news_signal:  str
    outcome:      str
    exit_price:   float = 0.0
    pnl:          float = 0.0
    notes:        str   = ""


class ArgusMemory:
    def __init__(self):
        os.makedirs(MEMORY_DIR, exist_ok=True)
        self._client     = chromadb.PersistentClient(path=MEMORY_DIR)
        self._collection = self._client.get_or_create_collection(
            name="argus_trades",
            metadata={"hnsw:space": "cosine"},
        )
        self._rulebook = self._load_rulebook()

    def save_trade(self, record: TradeRecord) -> str:
        doc = json.dumps(asdict(record))
        self._collection.add(
            documents=[doc],
            ids=[record.id],
            metadatas=[{
                "stock":   record.stock,
                "action":  record.action,
                "outcome": record.outcome,
                "date":    record.date,
            }],
        )
        return record.id

    def record_outcome(self, trade_id: str, outcome: str, exit_price: float = 0.0) -> bool:
        try:
            res = self._collection.get(ids=[trade_id])
            if not res["documents"]:
                return False
            record = json.loads(res["documents"][0])
            record["outcome"]    = outcome
            record["exit_price"] = exit_price
            if exit_price and record.get("entry"):
                mult = -1 if record["action"] == "SELL" else 1
                record["pnl"] = round((exit_price - record["entry"]) * mult, 2)
            self._collection.update(
                ids=[trade_id],
                documents=[json.dumps(record)],
                metadatas=[{
                    "stock":   record["stock"],
                    "action":  record["action"],
                    "outcome": outcome,
                    "date":    record["date"],
                }],
            )
            self._update_stock_profile(record["stock"], outcome)
            return True
        except Exception:
            return False

    def recall(self, stock: str, limit: int = 5) -> list:
        try:
            res = self._collection.get(where={"stock": stock}, limit=limit)
            return [json.loads(d) for d in res["documents"]]
        except Exception:
            return []

    def stock_accuracy(self, stock: str) -> dict:
        trades   = self.recall(stock, limit=50)
        resolved = [t for t in trades if t["outcome"] in ("win", "loss")]
        if not resolved:
            return {"win_rate": 0.5, "samples": 0, "note": "no data yet"}
        wins = sum(1 for t in resolved if t["outcome"] == "win")
        return {
            "win_rate": round(wins / len(resolved), 2),
            "samples":  len(resolved),
            "wins":     wins,
            "losses":   len(resolved) - wins,
        }

    def memory_score(self, stock: str, action: str) -> float:
        acc = self.stock_accuracy(stock)
        if acc["samples"] < 3:
            return 0.0
        wr   = acc["win_rate"]
        base = (wr - 0.5) * 0.6
        return round(base, 3)

    def stats(self) -> dict:
        try:
            res    = self._collection.get()
            trades = [json.loads(d) for d in res["documents"]]
        except Exception:
            trades = []
        resolved  = [t for t in trades if t["outcome"] in ("win", "loss")]
        wins      = [t for t in resolved if t["outcome"] == "win"]
        total_pnl = sum(t.get("pnl", 0) for t in resolved)
        by_stock  = {}
        for t in resolved:
            s = t["stock"]
            if s not in by_stock:
                by_stock[s] = {"wins": 0, "losses": 0}
            if t["outcome"] == "win":
                by_stock[s]["wins"] += 1
            else:
                by_stock[s]["losses"] += 1
        return {
            "total_trades": len(trades),
            "resolved":     len(resolved),
            "wins":         len(wins),
            "losses":       len(resolved) - len(wins),
            "win_rate":     round(len(wins) / max(len(resolved), 1), 2),
            "total_pnl":    round(total_pnl, 2),
            "by_stock":     by_stock,
        }

    def pending_trades(self) -> list:
        try:
            res = self._collection.get(where={"outcome": "pending"})
            return [json.loads(d) for d in res["documents"]]
        except Exception:
            return []

    def _load_rulebook(self) -> dict:
        try:
            with open(RULEBOOK_PATH) as f:
                return json.load(f)
        except Exception:
            return {}

    def _update_stock_profile(self, stock: str, outcome: str):
        try:
            acc      = self.stock_accuracy(stock)
            profiles = self._rulebook.setdefault("stock_profiles", {})
            existing = profiles.get(stock, {})
            existing["samples"] = acc["samples"]
            delta = 1.0 if outcome == "win" else 0.0
            existing["buy_reliability"]  = round(existing.get("buy_reliability", 0.5) * 0.7 + 0.3 * delta, 2)
            existing["sell_reliability"] = round(existing.get("sell_reliability", 0.5) * 0.7 + 0.3 * delta, 2)
            profiles[stock] = existing
            with open(RULEBOOK_PATH, "w") as f:
                json.dump(self._rulebook, f, indent=2)
        except Exception:
            pass


def new_trade_id() -> str:
    return str(uuid.uuid4())[:8]
