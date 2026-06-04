"""
ARGUS V4 — Intelligence Brain
Fuses: real OHLCV indicators + chart vision + news sentiment + memory scores.
Fixes:
  - Entry/stop/target now use REAL live price, not pixel-guessed values
  - Each analysis is fully isolated — no cross-stock bleed
  - Session clock multiplier applied to confidence
"""

import json
from dataclasses import dataclass, field
from typing import Optional
from config import RULEBOOK_PATH, CAPITAL, MAX_RISK_PCT


@dataclass
class TradeSignal:
    action:           str
    conviction:       str
    confidence:       int
    chart_score:      float
    news_score:       float
    memory_score:     float
    indicator_score:  float
    combined_score:   float
    entry_price:      float = 0.0
    stop_price:       float = 0.0
    target_price:     float = 0.0
    risk_per_share:   float = 0.0
    quantity:         int   = 0
    capital_required: float = 0.0
    max_loss:         float = 0.0
    rr_ratio:         str   = ""
    grade:            str   = "B"
    warnings:         list  = field(default_factory=list)
    matched_rules:    list  = field(default_factory=list)
    final_advice:     str   = ""
    live_price:       float = 0.0
    live_change_pct:  float = 0.0
    session:          str   = "unknown"
    # For display
    ema_20:   Optional[float] = None
    ema_50:   Optional[float] = None
    rsi:      Optional[float] = None
    macd_cross: str = "none"
    patterns: list  = field(default_factory=list)


class ArgusBrain:
    def __init__(self):
        self._rulebook = self._load_rulebook()

    def reload_rulebook(self):
        self._rulebook = self._load_rulebook()

    def generate_signal(
        self,
        chart_summary,          # MarketSummary from vision layer (can be None)
        news_sentiment,         # MarketSentiment (can be None)
        memory_context: dict,   # from ArgusMemory.stock_accuracy()
        live_price: dict = None,
        indicators=None,        # IndicatorResult from indicators.py (NEW)
        live_feed_on: bool = True,
        ticker: str = "",
    ) -> TradeSignal:

        w    = self._rulebook.get("scoring_weights", {})
        cw   = float(w.get("chart_weight",  0.50))
        nw   = float(w.get("news_weight",   0.25))
        mw   = float(w.get("memory_weight", 0.25))
        caps = self._rulebook.get("confidence_caps", {})

        warnings_out = []
        matched_rules = []

        # ── 1. Indicator score (real OHLCV — primary signal) ─────────────────
        ind_score = 0.0
        ind_patterns = []
        session = "unknown"
        session_mult = 1.0
        ema20 = ema50 = rsi = None
        macd_cross = "none"

        if indicators:
            ind_score, ind_matched, ind_warnings = self._score_indicators(indicators)
            matched_rules.extend(ind_matched)
            warnings_out.extend(ind_warnings)
            ind_patterns = indicators.patterns
            session      = indicators.session
            session_mult = indicators.session_confidence_mult
            ema20        = indicators.ema_20
            ema50        = indicators.ema_50
            rsi          = indicators.rsi
            macd_cross   = indicators.macd_cross

        # ── 2. Chart/vision score (fallback when no live data) ────────────────
        chart_score = 0.0
        if chart_summary:
            chart_score, c_matched, c_warnings = self._score_chart(chart_summary)
            matched_rules.extend(c_matched)
            warnings_out.extend(c_warnings)

        # Blend: if we have real indicators, they dominate
        if indicators:
            blended_chart = ind_score * 0.75 + chart_score * 0.25
        else:
            blended_chart = chart_score

        # ── 3. News score ─────────────────────────────────────────────────────
        news_score = 0.0
        if news_sentiment and live_feed_on:
            ns = news_sentiment.overall_score
            t  = ticker.upper()
            for sector, keywords in _SECTOR_TICKER_MAP.items():
                if any(k in t for k in keywords):
                    sec = news_sentiment.sectors.get(sector)
                    if sec:
                        ns = sec.score
                    break
            news_score = ns

        # ── 4. Memory score ───────────────────────────────────────────────────
        mem_score = 0.0
        if memory_context and memory_context.get("samples", 0) >= 3:
            wr        = memory_context["win_rate"]
            mem_score = round((wr - 0.5) * 0.6, 3)

        # ── 5. Combined weighted score ────────────────────────────────────────
        if live_feed_on:
            combined = blended_chart * cw + news_score * nw + mem_score * mw
        else:
            total_w  = cw + mw
            combined = (blended_chart * cw + mem_score * mw) / (total_w or 1)

        # Live price momentum bonus
        live_px  = 0.0
        live_chg = 0.0
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
        if any("extended" in w_ for w_ in warnings_out):
            conf -= int(caps.get("extended_stock_penalty", 15))
        if any("closing" in w_ for w_ in warnings_out):
            conf = min(conf, 40)   # hard cap near close

        # Apply bonuses
        if news_sentiment and news_sentiment.overall_signal == "bullish" and combined > 0:
            conf += int(caps.get("sector_alignment_bonus", 6))

        # Session multiplier
        conf = int(conf * session_mult)
        conf = max(25, min(max_conf, conf))

        # ── 7. Action ─────────────────────────────────────────────────────────
        if   combined >  0.15: action = "BUY"
        elif combined < -0.15: action = "SELL"
        else:                  action = "WAIT"

        conviction = "high" if conf >= 70 else ("medium" if conf >= 55 else "low")
        grade      = self._grade(conf, combined)

        # ── 8. Position sizing — use REAL live price ──────────────────────────
        entry = stop = target = 0.0
        qty = cap_req = max_loss_val = 0
        rr  = ""

        # Best price source: live → indicators → chart
        base_price = (live_px or
                      (indicators.current_price if indicators else 0) or
                      (chart_summary.support if chart_summary else 0) or 0)

        if action != "WAIT" and base_price > 0:
            if action == "BUY":
                entry  = round(base_price * 1.001, 2)   # tiny slippage
                stop   = round(base_price * 0.97,  2)   # 3% stop
                target = round(base_price * 1.05,  2)   # 5% target

                # If we have real S/R, use them
                if indicators and indicators.support:
                    stop = round(indicators.support * 0.995, 2)
                if indicators and indicators.resistance:
                    target = round(indicators.resistance * 0.995, 2)
            else:  # SELL
                entry  = round(base_price * 0.999, 2)
                stop   = round(base_price * 1.03,  2)
                target = round(base_price * 0.95,  2)

                if indicators and indicators.resistance:
                    stop = round(indicators.resistance * 1.005, 2)
                if indicators and indicators.support:
                    target = round(indicators.support * 1.005, 2)

            risk_ps  = abs(entry - stop)
            if risk_ps > 0:
                max_risk = CAPITAL * MAX_RISK_PCT
                qty      = max(1, int(max_risk / risk_ps))
                cap_req  = round(qty * entry, 2)
                max_loss_val = round(qty * risk_ps, 2)
                reward   = abs(target - entry) * qty
                rr_num   = round(reward / max_loss_val, 1) if max_loss_val else 0
                rr       = f"1:{rr_num}"

        advice = self._advice(action, conviction, conf, session, live_feed_on, warnings_out)

        return TradeSignal(
            action=action, conviction=conviction, confidence=conf,
            chart_score=round(blended_chart, 3),
            news_score=round(news_score, 3),
            memory_score=round(mem_score, 3),
            indicator_score=round(ind_score, 3),
            combined_score=round(combined, 3),
            entry_price=entry, stop_price=stop, target_price=target,
            risk_per_share=round(abs(entry - stop), 2) if entry and stop else 0,
            quantity=qty, capital_required=cap_req, max_loss=max_loss_val,
            rr_ratio=rr, grade=grade,
            warnings=list(set(warnings_out)),   # deduplicate
            matched_rules=matched_rules,
            final_advice=advice,
            live_price=live_px, live_change_pct=live_chg,
            session=session,
            ema_20=ema20, ema_50=ema50, rsi=rsi,
            macd_cross=macd_cross, patterns=ind_patterns,
        )

    # ── Indicator scoring ─────────────────────────────────────────────────────

    def _score_indicators(self, ind) -> tuple:
        score    = 0.0
        matched  = []
        warnings = []

        # EMA stack (Rule 1/2)
        if ind.ema_stack == "bullish":
            score += 0.25
            matched.append("EMA bullish stack (20>50>200)")
        elif ind.ema_stack == "bearish":
            score -= 0.25
            matched.append("EMA bearish stack (20<50<200)")

        # Price vs EMA20 (Rule 1/2)
        if ind.price_vs_ema20 == "above":
            score += 0.10
        elif ind.price_vs_ema20 == "below":
            score -= 0.10

        # Lower highs (Rule 3 — critical)
        if ind.lower_highs:
            score -= 0.30
            matched.append("Rule 3: Lower highs confirmed — default bearish")

        # Higher lows (bullish structure)
        if ind.higher_lows:
            score += 0.20
            matched.append("Higher lows — bullish structure")

        # RSI
        if ind.rsi:
            if ind.rsi_state == "overbought":
                score -= 0.10
                warnings.append(f"RSI {ind.rsi} overbought — rule 10 penalty")
            elif ind.rsi_state == "oversold":
                score += 0.10
                matched.append(f"RSI {ind.rsi} oversold — potential bounce")

        # MACD cross
        if ind.macd_cross == "bullish_cross":
            score += 0.18
            matched.append("MACD bullish cross")
        elif ind.macd_cross == "bearish_cross":
            score -= 0.18
            matched.append("MACD bearish cross")
        elif ind.macd_hist and ind.macd_hist > 0:
            score += 0.05
        elif ind.macd_hist and ind.macd_hist < 0:
            score -= 0.05

        # Volume confirmation
        if ind.volume_signal == "high":
            score *= 1.2
            matched.append(f"Volume {ind.volume_ratio:.1f}x avg — strong participation")
        elif ind.volume_signal == "low":
            score *= 0.8
            warnings.append("Weak volume — reduce confidence")

        # Patterns
        for p in ind.patterns:
            if p in ("bullish_engulfing", "hammer", "macd_bullish_cross", "higher_lows"):
                score += 0.12
                matched.append(f"Pattern: {p}")
            elif p in ("bearish_engulfing", "shooting_star", "lower_highs"):
                score -= 0.12
                matched.append(f"Pattern: {p}")
            elif p == "volatility_contraction":
                matched.append("Volatility contraction — watch for breakout")

        # Session warnings
        warnings.extend(ind.warnings)

        return round(max(-1.0, min(1.0, score)), 3), matched, warnings

    # ── Vision/chart scoring (fallback) ──────────────────────────────────────

    def _score_chart(self, summary) -> tuple:
        if not summary:
            return 0.0, [], []
        score    = 0.0
        matched  = []
        warnings = []
        trend    = (summary.trend or "").lower()
        vol      = (summary.volume or "").lower()
        pat      = (summary.pattern or "").lower()

        if "bullish" in trend:
            score += 0.20
            matched.append("Chart: bullish trend")
        elif "bearish" in trend:
            score -= 0.20
            matched.append("Chart: bearish trend")

        if "lower high" in pat or "bearish channel" in pat:
            score -= 0.25
            matched.append("Chart: lower high structure")

        if "breakdown" in pat and "weak" in vol:
            score -= 0.20
            matched.append("Chart: breakdown + weak retest")

        if "bullish" in pat and ("breakout" in pat or "higher" in pat):
            score += 0.15
            matched.append("Chart: bullish breakout pattern")

        if "high" in vol or "strong" in vol:
            score *= 1.1
        elif "low" in vol or "weak" in vol:
            score *= 0.9
            warnings.append("Weak volume on chart")

        return round(max(-1.0, min(1.0, score)), 3), matched, warnings

    def _grade(self, conf: int, score: float) -> str:
        if conf >= 75 and abs(score) > 0.25: return "A+"
        if conf >= 65 and abs(score) > 0.18: return "A"
        if conf >= 55 and abs(score) > 0.10: return "B+"
        if conf >= 45:                        return "B"
        return "C"

    def _advice(self, action, conviction, conf, session, live_feed_on, warnings) -> str:
        parts = []
        if session == "closing":
            parts.append("CLOSING TIME — exit all open positions before 3:15 PM.")
        elif action == "WAIT":
            parts.append("No clean edge. Wait for confirmation.")
        elif conviction == "high":
            parts.append(f"Strong {action} setup. Execute with defined stop.")
        else:
            parts.append(f"Weak {action} signal — reduce size by 50%.")
        if session == "morning":
            parts.append("Morning session: wait for 9:45 AM (2 candle confirmation).")
        if not live_feed_on:
            parts.append("[Live feed OFF — no real-time price context]")
        return " ".join(parts)

    def _load_rulebook(self) -> dict:
        try:
            with open(RULEBOOK_PATH) as f:
                return json.load(f)
        except Exception:
            return {}


_SECTOR_TICKER_MAP = {
    "BANKING": ["BANK", "HDFC", "ICICI", "AXIS", "KOTAK"],
    "PHARMA":  ["PHARMA", "SUN", "CIPLA", "DIVI"],
    "IT":      ["INFY", "TCS", "WIPRO", "HCL"],
    "AUTO":    ["TATA", "TVS", "MARUTI", "BAJAJ"],
    "ENERGY":  ["RELIANCE", "ADANI", "ONGC"],
    "NBFC":    ["BAJFIN", "MUTHOOT", "CHOLA"],
}
