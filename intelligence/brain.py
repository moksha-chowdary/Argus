"""
ARGUS V3 — Intelligence Brain
Fuses chart analysis + news sentiment + memory → best possible signal.
This is what makes ARGUS intelligent, not just a chart reader.
"""
from dataclasses import dataclass
from typing import Optional
from config import CAPITAL, MAX_RISK_PCT, MAX_ALLOCATION, MIN_RR_RATIO, NEWS_WEIGHT


@dataclass
class IntelligentSignal:
    # Core signal
    action: str              # BUY | SELL | WAIT
    conviction: str          # high | moderate | low
    risk: str                # low | moderate | high | very_high

    # Position sizing (calculated for your ₹4L capital)
    entry_price: Optional[float]
    stop_price: Optional[float]
    target_price: Optional[float]
    quantity: Optional[int]
    capital_required: Optional[float]
    max_loss: Optional[float]
    potential_gain: Optional[float]
    rr_ratio: Optional[float]

    # Intelligence layers
    chart_score: float       # from vision layer
    news_score: float        # from news sentiment
    memory_score: float      # from past trades
    combined_score: float    # final weighted score

    # Reasoning
    chart_reasons: list
    news_reasons: list
    memory_reasons: list
    warnings: list
    final_advice: str        # one clear sentence

    # Trade ID for tracking
    trade_id: str = ""


class IntelligenceBrain:
    """
    The core intelligence layer.
    Takes outputs from all 3 sources and produces one unified signal.
    """

    def analyze(
        self,
        chart_summary,      # MarketSummary from vision
        chart_signal,       # Signal from signal generator
        news_sentiment,     # MarketSentiment from news
        memory_store,       # MemoryStore instance
        ticker: str = None,
    ) -> IntelligentSignal:

        import uuid
        trade_id = str(uuid.uuid4())[:8]
        ticker = ticker or chart_summary.ticker or "UNKNOWN"

        # ── Layer 1: Chart score ───────────────────────────────────────────────
        chart_score = self._chart_score(chart_summary, chart_signal)

        # ── Layer 2: News score ────────────────────────────────────────────────
        news_score, news_reasons = self._news_score(ticker, news_sentiment)

        # ── Layer 3: Memory score ──────────────────────────────────────────────
        memory_score, memory_reasons = self._memory_score(ticker, memory_store)

        # ── Combine scores ─────────────────────────────────────────────────────
        combined = self._combine(chart_score, news_score, memory_score)

        # ── Final decision ─────────────────────────────────────────────────────
        action, conviction = self._decide(combined, chart_summary, chart_signal)

        # ── Position sizing ────────────────────────────────────────────────────
        entry, stop, target, qty, cap, loss, gain, rr = self._position_size(
            chart_summary, chart_signal, action
        )

        # ── Risk assessment ────────────────────────────────────────────────────
        risk = self._risk(chart_summary, chart_signal, news_score, action)

        # ── Warnings ──────────────────────────────────────────────────────────
        warnings = self._warnings(
            chart_summary, chart_signal, news_score, memory_score, rr, action
        )

        # ── Final advice ───────────────────────────────────────────────────────
        advice = self._advice(action, conviction, entry, stop, target, qty, rr, ticker)

        return IntelligentSignal(
            action=action,
            conviction=conviction,
            risk=risk,
            entry_price=entry,
            stop_price=stop,
            target_price=target,
            quantity=qty,
            capital_required=cap,
            max_loss=loss,
            potential_gain=gain,
            rr_ratio=rr,
            chart_score=chart_score,
            news_score=news_score,
            memory_score=memory_score,
            combined_score=combined,
            chart_reasons=chart_signal.reasons if chart_signal else [],
            news_reasons=news_reasons,
            memory_reasons=memory_reasons,
            warnings=warnings,
            final_advice=advice,
            trade_id=trade_id,
        )

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _chart_score(self, summary, signal) -> float:
        """Convert chart signal to -1.0 to +1.0 score."""
        base = {"BUY": 0.6, "SELL": -0.6, "WAIT": 0.0}.get(signal.action, 0.0)
        mult = {"high": 1.0, "moderate": 0.7, "low": 0.4}.get(signal.conviction, 0.5)
        return round(base * mult, 2)

    def _news_score(self, ticker: str, sentiment) -> tuple[float, list]:
        """Get news sentiment score for this ticker's sector."""
        reasons = []
        if sentiment is None:
            return 0.0, ["News data unavailable."]

        # Overall market
        overall = sentiment.overall_score
        reasons.append(f"Market sentiment: {sentiment.overall_signal} ({overall:+.2f})")

        # Sector-specific
        from news.sentiment import SentimentAnalyzer
        analyzer = SentimentAnalyzer()
        stock_signal, stock_score = analyzer.get_stock_sentiment(ticker, sentiment)

        if stock_signal != "neutral":
            reasons.append(f"Sector sentiment for {ticker}: {stock_signal} ({stock_score:+.2f})")

        # Top headlines relevant to this ticker
        ticker_lower = ticker.lower()
        for headline in sentiment.top_bullish[:2]:
            if ticker_lower in headline.lower() or any(
                kw in headline.lower() for kw in [ticker_lower[:4]]
            ):
                reasons.append(f"Bullish news: {headline[:80]}")

        # Combined score: 70% sector, 30% overall
        combined = round(stock_score * 0.7 + overall * 0.3, 2)
        return combined, reasons

    def _memory_score(self, ticker: str, memory_store) -> tuple[float, list]:
        """Score based on past trade performance for this ticker."""
        reasons = []
        history = memory_store.get_ticker_history(ticker)

        if history.get("trades", 0) == 0:
            return 0.0, [f"No previous trades for {ticker} — no memory bias."]

        win_rate = history.get("win_rate", 50.0)
        total = history.get("trades", 0)

        # Convert win rate to score (-1 to +1)
        score = round((win_rate - 50) / 50, 2)  # 100% wins = +1.0, 0% wins = -1.0

        reasons.append(
            f"Past {ticker} trades: {total} total, {win_rate:.0f}% win rate"
        )
        reasons.append(
            f"Last signal: {history.get('last_signal')} → {history.get('last_outcome')}"
        )

        # Recent accuracy
        accuracy = memory_store.get_signal_accuracy()
        buy_acc = accuracy.get("BUY", {}).get("accuracy", 0)
        sell_acc = accuracy.get("SELL", {}).get("accuracy", 0)
        if buy_acc or sell_acc:
            reasons.append(f"Overall accuracy — BUY: {buy_acc:.0f}%  SELL: {sell_acc:.0f}%")

        return score, reasons

    def _combine(self, chart: float, news: float, memory: float) -> float:
        """Weighted combination of all three scores."""
        # Chart is most important (60%), news (25%), memory (15%)
        return round(chart * 0.60 + news * NEWS_WEIGHT + memory * 0.15, 2)

    # ── Decision ──────────────────────────────────────────────────────────────

    def _decide(self, combined: float, summary, signal) -> tuple[str, str]:
        # If chart says WAIT and combined score is ambiguous, respect it
        if signal.action == "WAIT" and abs(combined) < 0.3:
            return "WAIT", "high"
        if combined >= 0.4:
            return "BUY", "high" if combined >= 0.6 else "moderate"
        if combined <= -0.4:
            return "SELL", "high" if combined <= -0.6 else "moderate"
        if combined >= 0.2:
            return "BUY", "low"
        if combined <= -0.2:
            return "SELL", "low"
        return "WAIT", "high"

    # ── Position sizing ────────────────────────────────────────────────────────

    def _position_size(self, summary, signal, action):
        if action == "WAIT" or not summary.support or not summary.resistance:
            return None, None, None, None, None, None, None, None

        sup = summary.support
        res = summary.resistance
        if sup >= res:
            return None, None, None, None, None, None, None, None

        zone = res - sup
        if action == "BUY":
            entry  = round(sup + zone * 0.05, 2)
            stop   = round(sup - zone * 0.10, 2)
            target = round(res - zone * 0.02, 2)
        else:  # SELL
            entry  = round(res - zone * 0.05, 2)
            stop   = round(res + zone * 0.10, 2)
            target = round(sup + zone * 0.02, 2)

        risk_per_share = abs(entry - stop)
        if risk_per_share == 0:
            return entry, stop, target, None, None, None, None, None

        max_risk = CAPITAL * MAX_RISK_PCT
        qty = int(max_risk / risk_per_share)
        if qty == 0:
            qty = 1

        cap_required = round(qty * entry, 2)
        # Don't exceed max allocation
        if cap_required > CAPITAL * MAX_ALLOCATION:
            qty = int((CAPITAL * MAX_ALLOCATION) / entry)
            cap_required = round(qty * entry, 2)

        max_loss     = round(qty * risk_per_share, 2)
        potential    = round(qty * abs(target - entry), 2)
        rr           = round(abs(target - entry) / risk_per_share, 2) if risk_per_share else None

        return entry, stop, target, qty, cap_required, max_loss, potential, rr

    # ── Risk ──────────────────────────────────────────────────────────────────

    def _risk(self, summary, signal, news_score, action) -> str:
        if summary.volatility == "high":                            return "very_high"
        if summary.rsi_state == "overbought" and action == "BUY":  return "high"
        if summary.rsi_state == "oversold" and action == "SELL":   return "high"
        if abs(news_score) > 0.6:                                  return "moderate"
        if summary.volatility == "compressed":                     return "moderate"
        return "low"

    # ── Warnings ──────────────────────────────────────────────────────────────

    def _warnings(self, summary, signal, news_score, memory_score, rr, action) -> list:
        w = []
        if summary.confidence < 0.35:
            w.append("Low chart extraction confidence — verify manually before trading.")
        if rr and rr < MIN_RR_RATIO:
            w.append(f"R/R ratio {rr:.1f} is below minimum {MIN_RR_RATIO} — not ideal.")
        if summary.rsi_state == "overbought" and action == "BUY":
            w.append("RSI overbought — chasing longs is risky here.")
        if summary.rsi_state == "oversold" and action == "SELL":
            w.append("RSI oversold — shorting here is dangerous.")
        if summary.volatility == "high":
            w.append("High volatility — use smaller position size than suggested.")
        if summary.volume == "decreasing" and action != "WAIT":
            w.append("Volume declining — trend lacks confirmation.")
        if news_score < -0.3 and action == "BUY":
            w.append("News sentiment bearish — headwind against long position.")
        if news_score > 0.3 and action == "SELL":
            w.append("News sentiment bullish — headwind against short position.")
        if memory_score < -0.3:
            w.append("Poor historical accuracy on this stock — trade smaller.")
        if not w:
            w.append("No major warnings. Still — always use a stop loss.")
        return w

    # ── Final advice ──────────────────────────────────────────────────────────

    def _advice(self, action, conviction, entry, stop, target, qty, rr, ticker) -> str:
        if action == "WAIT":
            return f"WAIT — no high-probability setup on {ticker} right now. Preserve capital."
        direction = "BUY" if action == "BUY" else "SELL/SHORT"
        if entry and stop and target and qty:
            return (
                f"{direction} {ticker}: {qty} shares near ₹{entry:.2f} | "
                f"Stop ₹{stop:.2f} | Target ₹{target:.2f} | "
                f"R/R {rr:.1f}x | Max loss ₹{int(qty * abs(entry - stop))}"
            )
        return f"{direction} {ticker} — confirm entry with price action. Use stop loss."
