"""
ARGUS V3 — Memory Store
Persistent memory using ChromaDB.
Stores every analysis + outcome so ARGUS gets smarter over time.
"""
import json
import uuid
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import Optional
from pathlib import Path
from config import MEMORY_DIR


@dataclass
class TradeMemory:
    """One complete trade record — prediction + outcome."""
    id: str
    date: str
    ticker: str
    timeframe: str
    signal: str               # BUY | SELL | WAIT
    conviction: str
    entry_price: Optional[float]
    stop_price: Optional[float]
    target_price: Optional[float]
    chart_summary: dict       # vision layer output
    news_sentiment: str       # bullish | bearish | neutral
    news_score: float
    llm_reasoning: str

    # Outcome (filled in after trade)
    outcome: Optional[str] = None      # "win" | "loss" | "breakeven" | "skipped"
    actual_entry: Optional[float] = None
    actual_exit: Optional[float] = None
    pnl: Optional[float] = None
    feedback: Optional[str] = None    # user's note
    correct_prediction: Optional[bool] = None

    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.id:
            self.id = str(uuid.uuid4())[:8]


class MemoryStore:
    """
    ChromaDB-backed memory store.
    Falls back to JSON if ChromaDB not installed.
    """

    def __init__(self):
        self._db = None
        self._collection = None
        self._json_path = MEMORY_DIR / "trades.json"
        self._use_chroma = self._init_chroma()

    def _init_chroma(self) -> bool:
        try:
            import chromadb
            self._db = chromadb.PersistentClient(path=str(MEMORY_DIR))
            self._collection = self._db.get_or_create_collection(
                name="argus_trades",
                metadata={"hnsw:space": "cosine"}
            )
            return True
        except ImportError:
            return False
        except Exception:
            return False

    # ── Save ──────────────────────────────────────────────────────────────────

    def save(self, memory: TradeMemory):
        memory.updated_at = datetime.now().isoformat()
        if self._use_chroma:
            self._chroma_save(memory)
        self._json_save(memory)  # always save to JSON as backup

    def _chroma_save(self, m: TradeMemory):
        doc = f"{m.ticker} {m.signal} {m.date} {m.llm_reasoning[:200]}"
        meta = {k: str(v) for k, v in asdict(m).items() if v is not None}
        try:
            self._collection.upsert(
                ids=[m.id],
                documents=[doc],
                metadatas=[meta],
            )
        except Exception:
            pass

    def _json_save(self, m: TradeMemory):
        trades = self._json_load_all()
        trades[m.id] = asdict(m)
        self._json_path.write_text(
            json.dumps(trades, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # ── Load ──────────────────────────────────────────────────────────────────

    def load(self, trade_id: str) -> Optional[TradeMemory]:
        trades = self._json_load_all()
        data = trades.get(trade_id)
        if data:
            return TradeMemory(**data)
        return None

    def all_trades(self) -> list[TradeMemory]:
        trades = self._json_load_all()
        return [TradeMemory(**v) for v in trades.values()]

    def recent_trades(self, n: int = 10) -> list[TradeMemory]:
        all_t = self.all_trades()
        return sorted(all_t, key=lambda t: t.created_at, reverse=True)[:n]

    def trades_for_ticker(self, ticker: str) -> list[TradeMemory]:
        return [t for t in self.all_trades() if t.ticker.upper() == ticker.upper()]

    # ── Update outcome ────────────────────────────────────────────────────────

    def record_outcome(
        self, trade_id: str,
        outcome: str,
        actual_entry: float = None,
        actual_exit: float = None,
        feedback: str = None,
    ):
        m = self.load(trade_id)
        if not m:
            return False
        m.outcome = outcome
        m.actual_entry = actual_entry
        m.actual_exit = actual_exit
        m.feedback = feedback
        if actual_entry and actual_exit:
            m.pnl = round(actual_exit - actual_entry, 2)
            if m.signal == "BUY":
                m.correct_prediction = actual_exit > actual_entry
            elif m.signal == "SELL":
                m.correct_prediction = actual_exit < actual_entry
        self.save(m)
        return True

    # ── Stats ─────────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        trades = [t for t in self.all_trades() if t.outcome is not None]
        if not trades:
            return {"total": 0, "wins": 0, "losses": 0, "win_rate": 0.0, "avg_pnl": 0.0}

        wins   = sum(1 for t in trades if t.outcome == "win")
        losses = sum(1 for t in trades if t.outcome == "loss")
        pnls   = [t.pnl for t in trades if t.pnl is not None]

        return {
            "total":    len(trades),
            "wins":     wins,
            "losses":   losses,
            "skipped":  sum(1 for t in trades if t.outcome == "skipped"),
            "win_rate": round(wins / len(trades) * 100, 1),
            "avg_pnl":  round(sum(pnls) / len(pnls), 2) if pnls else 0.0,
            "total_pnl":round(sum(pnls), 2) if pnls else 0.0,
        }

    def get_signal_accuracy(self) -> dict:
        """How accurate is each signal type?"""
        trades = [t for t in self.all_trades() if t.correct_prediction is not None]
        result = {"BUY": {"correct": 0, "total": 0}, "SELL": {"correct": 0, "total": 0}}
        for t in trades:
            if t.signal in result:
                result[t.signal]["total"] += 1
                if t.correct_prediction:
                    result[t.signal]["correct"] += 1
        for sig in result:
            total = result[sig]["total"]
            result[sig]["accuracy"] = round(
                result[sig]["correct"] / total * 100, 1
            ) if total > 0 else 0.0
        return result

    def get_ticker_history(self, ticker: str) -> dict:
        """What's the track record for a specific stock?"""
        trades = self.trades_for_ticker(ticker)
        completed = [t for t in trades if t.outcome is not None]
        if not completed:
            return {"ticker": ticker, "trades": 0, "note": "No history yet"}
        wins = sum(1 for t in completed if t.outcome == "win")
        return {
            "ticker":   ticker,
            "trades":   len(completed),
            "wins":     wins,
            "win_rate": round(wins / len(completed) * 100, 1),
            "last_signal": trades[0].signal if trades else "None",
            "last_outcome": trades[0].outcome if completed else "None",
        }

    # ── Context for LLM ───────────────────────────────────────────────────────

    def get_context_for_ticker(self, ticker: str, max_trades: int = 5) -> str:
        """Return formatted past trade history for a ticker — injected into LLM prompt."""
        trades = self.trades_for_ticker(ticker)[:max_trades]
        if not trades:
            return f"No previous trades recorded for {ticker}."
        lines = [f"Past {ticker} trades:"]
        for t in trades:
            outcome_str = t.outcome or "pending"
            pnl_str = f"  PnL: ₹{t.pnl}" if t.pnl else ""
            lines.append(
                f"  [{t.date}] Signal: {t.signal} ({t.conviction}) → {outcome_str}{pnl_str}"
            )
            if t.feedback:
                lines.append(f"    Note: {t.feedback}")
        return "\n".join(lines)

    # ── JSON helpers ──────────────────────────────────────────────────────────

    def _json_load_all(self) -> dict:
        if not self._json_path.exists():
            return {}
        try:
            return json.loads(self._json_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
