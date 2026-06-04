"""
ARGUS V4 — Intelligence Brain
Fuses chart score + news score + memory score using rulebook weights.
Outputs a final TradeSignal with entry, stop, target, position size.
"""

import json, math
from dataclasses import dataclass, field
from typing import Optional
from config import RULEBOOK_PATH, CAPITAL, MAX_RISK_PCT


@dataclass
class TradeSignal:
    action:           str         # BUY / SELL / WAIT
    conviction:       str         # high / medium / low
    confidence:       int         # 0-100
    chart_score:      float
    news_score:       float
    memory_score:     float
    combined_score:   float
    entry_price:      float       = 0.0
    stop_price:       float       = 0.0
    target_price:     float       = 0.0
    risk_per_share:   float       = 0.0
    quantity:         int         = 0
    capital_required: float       = 0.0
    max_loss:         float       = 0.0
    rr_ratio:         str         = ""
    grade:            str         = "B"
    warnings:         list        = field(default_factory=list)
    matched_rules:    list        = field(default_factory=list)
    final_advice:     str         = ""
    live_price:       float       = 0.0
    live_change_pct:  float       = 0.0


class ArgusBrain:
    def __init__(self):
        self._rulebook = self._load_rulebook()

    def reload_rulebook(self):
        self._rulebook = self._load_rulebook()

    # ── Main entry point ──────────────────────────────────────────────────────

    def generate_signal(
        self,
        chart_summary,          # MarketSummary from vision layer
        news_sentiment,         # MarketSentiment (can be None)
        memory_context: dict,   # from ArgusMemory.stock_accuracy()
        live_price: dict = None, # from PriceTracker.get()
        live_feed_on: bool = True,
    ) -> TradeSignal:

        w    = self._rulebook.get("scoring_weights", {})
        cw   = float(w.get("chart_weight",  0.50))
        nw   = float(w.get("news_weight",   0.25))
        mw   = float(w.get("memory_weight", 0.25))
        caps = self._rulebook.get("confidence_caps", {})

        # ── 1. Chart score ────────────────────────────────────────────────────
        chart_score, matched_rules, warnings = self._score_chart(chart_summary)

        # ── 2. News score ─────────────────────────────────────────────────────
        news_score = 0.0
        if news_sentiment and live_feed_on:
            ns = news_sentiment.overall_score
            # sector-specific override
            ticker = (chart_summary.ticker or "").upper()
            for sector, keywords in _SECTOR_TICKER_MAP.items():
                if any(k in ticker for k in keywords):
                    sec = news_sentiment.sectors.get(sector)
                    if sec:
                        ns = sec.score
                    break
            news_score = ns

        # ── 3. Memory score ───────────────────────────────────────────────────
        mem_score = 0.0
        if memory_context and memory_context.get("samples", 0) >= 3:
            wr        = memory_context["win_rate"]
            mem_score = round((wr - 0.5) * 0.6, 3)

        # ── 4. Combined weighted score ────────────────────────────────────────
        if live_feed_on:
            combined = chart_score * cw + news_score * nw + mem_score * mw
        else:
            # Without live feed, rebalance weights to chart + memory only
            total_w  = cw + mw
            combined = (chart_score * cw + mem_score * mw) / total_w

        # ── 5. Live price bonus ───────────────────────────────────────────────
        live_chg = 0.0
        live_px  = 0.0
        if live_price and live_feed_on:
            live_px  = live_price.get("price", 0)
            live_chg = live_price.get("change_pct", 0)
            bonus    = float(w.get("live_price_bonus", 0.05))
            if live_chg > 1.0 and combined > 0:
                combined += bonus
            elif live_chg < -1.0 and combined < 0:
                combined -= bonus

        # ── 6. Confidence ─────────────────────────────────────────────────────
        max_conf = int(caps.get("max_confidence", 82))
        conf     = int(min(max_conf, 50 + abs(combined) * 50))

        # Apply penalties
        if any("extended" in w.lower() for w in warnings):
            conf -= int(caps.get("extended_stock_penalty", 15))
        if any("support" in w.lower() for w in warnings):
            conf -= int(caps.get("multiple_support_retest_penalty", 10))
        if any("volume" in w.lower() for w in warnings):
            conf -= int(caps.get("weak_volume_recovery_penalty", 12))

        # Apply bonuses
        if news_sentiment and news_sentiment.overall_signal == "bullish" and combined > 0:
            conf += int(caps.get("sector_alignment_bonus", 6))
        conf = max(30, min(max_conf, conf))

        # ── 7. Action ─────────────────────────────────────────────────────────
        if   combined >  0.15: action = "BUY"
        elif combined < -0.15: action = "SELL"
        else:                  action = "WAIT"

        conviction = "high" if conf >= 70 else ("medium" if conf >= 55 else "low")
        grade      = self._grade(conf, combined)

        # ── 8. Position sizing ────────────────────────────────────────────────
        entry = stop = target = 0.0
        qty   = cap_req = max_loss = 0
        rr    = ""

        if action != "WAIT" and chart_summary:
            entry  = float(chart_summary.support  or live_px or 0)
            stop   = float(chart_summary.support  or live_px or 0) * 0.97
            target = float(chart_summary.resistance or live_px or 0) * 1.03

            # Prefer chart-extracted values
            if action == "BUY":
                entry  = live_px or entry
                stop   = entry * 0.97
                target = entry * 1.05
            else:
                entry  = live_px or entry
                stop   = entry * 1.03
                target = entry * 0.95

            risk_ps  = abs(entry - stop)
            if risk_ps > 0:
                max_risk = CAPITAL * MAX_RISK_PCT
                qty      = max(1, int(max_risk / risk_ps))
                cap_req  = round(qty * entry, 2)
                max_loss = round(qty * risk_ps, 2)
                reward   = abs(target - entry) * qty
                rr_num   = round(reward / max_loss, 1) if max_loss else 0
                rr       = f"1:{rr_num}"

        advice = self._advice(action, conviction, conf, chart_summary, news_sentiment, live_feed_on)

        return TradeSignal(
            action=action, conviction=conviction, confidence=conf,
            chart_score=round(chart_score, 3), news_score=round(news_score, 3),
            memory_score=round(mem_score, 3), combined_score=round(combined, 3),
            entry_price=round(entry, 2), stop_price=round(stop, 2),
            target_price=round(target, 2), risk_per_share=round(abs(entry-stop), 2),
            quantity=qty, capital_required=cap_req, max_loss=max_loss,
            rr_ratio=rr, grade=grade, warnings=warnings,
            matched_rules=matched_rules, final_advice=advice,
            live_price=live_px, live_change_pct=live_chg,
        )

    # ── Rulebook scoring ──────────────────────────────────────────────────────

    def _score_chart(self, summary) -> tuple:
        if not summary:
            return 0.0, [], []

        rules    = self._rulebook.get("rules", [])
        score    = 0.0
        matched  = []
        warnings = []

        trend = (summary.trend or "").lower()
        vol   = (summary.volume or "").lower()
        pat   = (summary.pattern or "").lower()
        mom   = (summary.momentum or "").lower()

        # Rule 1/2 — MA alignment (inferred from trend strength)
        if "bullish" in trend:
            score += 0.20
            matched.append("Bullish trend")
        elif "bearish" in trend:
            score -= 0.20
            matched.append("Bearish trend")

        # Rule 3 — lower high structure
        if "lower high" in pat or "bearish channel" in pat:
            score -= 0.30
            matched.append("Rule 3: Lower high structure — default bearish")

        # Rule 5 — rejection candle
        if "rejection" in pat or "wick" in pat:
            score += 0.15 if "bullish" in pat else -0.15
            matched.append("Rule 5: Rejection candle")

        # Rule 6 — giant candle
        if "giant" in pat or "breakout" in pat:
            score += 0.18 if "bullish" in pat else -0.18
            matched.append("Rule 6: Giant candle momentum")

        # Rule 7 — lilliput
        if "lilliput" in pat or "contraction" in pat or "squeeze" in pat:
            matched.append("Rule 7: Volatility contraction — watch for breakout")

        # Rule 8 — breakdown retest
        if "breakdown" in pat and ("retest" in pat or "weak" in vol):
            score -= 0.28
            matched.append("Rule 8: Breakdown + weak retest — strong short")

        # Rule 9 — exhaustion warning
        if "exhaustion" in pat or ("bearish" in trend and "oversold" in (summary.rsi_state or "")):
            warnings.append("Exhaustion possible — rule 9 applies")
            score += 0.10

        # Rule 10 — extended stock penalty
        if summary.strength and "strong" in str(summary.strength).lower():
            if "overbought" in (summary.rsi_state or "").lower():
                warnings.append("Extended stock — rule 10 penalty applied")
                score -= 0.15

        # Volume confirmation
        if "strong" in vol or "high" in vol:
            score *= 1.15
            matched.append("Volume confirmation")
        elif "weak" in vol or "low" in vol:
            score *= 0.85
            warnings.append("Weak volume — reduce confidence")

        return round(max(-1.0, min(1.0, score)), 3), matched, warnings

    def _grade(self, conf: int, score: float) -> str:
        if conf >= 75 and abs(score) > 0.25: return "A+"
        if conf >= 65 and abs(score) > 0.18: return "A"
        if conf >= 55 and abs(score) > 0.10: return "B+"
        if conf >= 45:                        return "B"
        return "C"

    def _advice(self, action, conviction, conf, summary, news, live_feed_on) -> str:
        parts = []
        if action == "WAIT":
            parts.append("No clean edge. Wait for breakout or breakdown confirmation.")
        elif conviction == "high":
            parts.append(f"Strong {action} setup. Execute with defined stop.")
        else:
            parts.append(f"Weak {action} signal. Reduce position size.")
        if not live_feed_on:
            parts.append("[Live feed OFF — no real-time price or news context]")
        return " ".join(parts)

    def _load_rulebook(self) -> dict:
        try:
            with open(RULEBOOK_PATH) as f:
                return json.load(f)
        except Exception:
            return {}


# Sector → ticker fragment mapping for news alignment
_SECTOR_TICKER_MAP = {
    "BANKING": ["BANK", "HDFC", "ICICI", "AXIS", "KOTAK"],
    "PHARMA":  ["PHARMA", "SUN", "CIPLA", "DIVI"],
    "IT":      ["INFY", "TCS", "WIPRO", "HCL"],
    "AUTO":    ["TATA", "TVS", "MARUTI", "BAJAJ"],
    "ENERGY":  ["RELIANCE", "ADANI", "ONGC"],
    "NBFC":    ["BAJFIN", "MUTHOOT", "CHOLA"],
}
