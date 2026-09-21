"""
ARGUS V5 — Continuous Learning Intelligence Brain
Fuses:
  - River Hoeffding Adaptive Tree (Continuous Online Learner)
  - PyTorch Rolling-Window LSTM (Periodic Deep Learning Retrainer)
  - ADWIN Concept Drift Detector
  - FinBERT News Sentiment + Feature Store (Zero-Lookahead Leakage Guard)
  - Vision/OCR Chart Context (Legacy/Fallback)
"""

import json
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd

from config import RULEBOOK_PATH, CAPITAL, MAX_RISK_PCT
from data.feature_store import FeatureStore
from intelligence.features import calculate_numeric_features, FEATURE_COLUMNS
from intelligence.online_learner import OnlineLearner
from intelligence.dl_retrainer import DeepLearningRetrainer
from intelligence.outcome_tracker import OutcomeTracker


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
    # Display indicators
    ema_20:   Optional[float] = None
    ema_50:   Optional[float] = None
    rsi:      Optional[float] = None
    macd_cross: str = "none"
    patterns: list  = field(default_factory=list)
    # V5 ML/DL additions
    prob_online:    float = 0.50
    prob_dl:        float = 0.50
    prediction_id:  str   = ""
    features_asof:  str   = ""
    drift_status:   str   = "stable"


class ArgusBrain:
    """
    ARGUS V5 Central Decision Engine.
    Replaces hand-coded scoring ladder with probabilistic machine learning.
    """

    def __init__(
        self,
        feature_store: Optional[FeatureStore] = None,
        online_learner: Optional[OnlineLearner] = None,
        dl_retrainer: Optional[DeepLearningRetrainer] = None,
        outcome_tracker: Optional[OutcomeTracker] = None,
    ):
        self.feature_store = feature_store or FeatureStore()
        self.online_learner = online_learner or OnlineLearner(feature_store=self.feature_store)
        self.dl_retrainer = dl_retrainer or DeepLearningRetrainer(feature_store=self.feature_store)
        self.outcome_tracker = outcome_tracker or OutcomeTracker(
            online_learner=self.online_learner,
            feature_store=self.feature_store,
        )
        self._rulebook = self._load_rulebook()

    def reload_rulebook(self):
        self._rulebook = self._load_rulebook()

    def generate_signal(
        self,
        chart_summary=None,          # MarketSummary from vision layer
        news_sentiment=None,         # MarketSentiment
        memory_context: dict = None, # historical accuracy context
        live_price: dict = None,
        indicators=None,             # IndicatorResult
        live_feed_on: bool = True,
        ticker: str = "",
        candle_df: Optional[pd.DataFrame] = None,
    ) -> TradeSignal:
        ticker_clean = ticker.replace(".NS", "").upper() if ticker else "UNKNOWN"
        warnings_out = []
        matched_rules = []

        # ── 1. Live Price & Session Extraction ────────────────────────────────
        live_px = 0.0
        live_chg = 0.0
        session = "midday"
        session_mult = 1.0
        ema20 = ema50 = rsi = None
        macd_cross = "none"
        patterns = []

        if live_price:
            live_px = float(live_price.get("price", 0.0))
            live_chg = float(live_price.get("change_pct", 0.0))

        if indicators:
            if not live_px:
                live_px = indicators.current_price
            ema20 = indicators.ema_20
            ema50 = indicators.ema_50
            rsi = indicators.rsi
            macd_cross = indicators.macd_cross
            patterns = indicators.patterns
            session = indicators.session
            session_mult = indicators.session_confidence_mult
            warnings_out.extend(indicators.warnings)

        base_price = live_px or (chart_summary.support if chart_summary else 100.0) or 100.0

        # ── 2. Continuous ML Feature Extraction & Prediction ──────────────────
        prob_online = 0.50
        prob_dl = 0.50
        features_asof = "N/A"
        pred_id = ""

        if candle_df is not None and len(candle_df) >= 20:
            try:
                numeric_feats, features_asof = calculate_numeric_features(
                    df=candle_df,
                    ticker=ticker,
                    feature_store=self.feature_store,
                )

                # Log prediction with time-leakage guard in outcome tracker
                pred_rec = self.outcome_tracker.record_prediction(
                    ticker=ticker_clean,
                    features=numeric_feats,
                    features_asof=features_asof,
                    base_price=base_price,
                )
                pred_id = pred_rec["prediction_id"]
                prob_online = pred_rec["prob_online"]

                # PyTorch LSTM inference (using feature vector)
                feat_vec = [numeric_feats.get(c, 0.0) for c in FEATURE_COLUMNS]
                prob_dl = self.dl_retrainer.predict_proba(np.array(feat_vec, dtype=np.float32))

                matched_rules.append(f"River Online HAT: P(Up) = {prob_online:.1%}")
                matched_rules.append(f"PyTorch LSTM: P(Up) = {prob_dl:.1%}")
            except Exception as e:
                warnings_out.append(f"ML feature extraction fallback: {e}")

        # If no candles, use technical indicators to estimate prior probability
        if prob_online == 0.50 and indicators:
            bull_score = 0.0
            if indicators.ema_stack == "bullish": bull_score += 0.08
            elif indicators.ema_stack == "bearish": bull_score -= 0.08
            if indicators.rsi and indicators.rsi < 35: bull_score += 0.06
            elif indicators.rsi and indicators.rsi > 65: bull_score -= 0.06
            if "bullish" in indicators.macd_cross: bull_score += 0.07
            elif "bearish" in indicators.macd_cross: bull_score -= 0.07
            prob_online = round(float(np.clip(0.50 + bull_score, 0.20, 0.80)), 3)
            prob_dl = prob_online

        # ── 3. Calibrated Probabilistic Decision ───────────────────────────────
        action, conviction, conf_score = self.outcome_tracker.decide_action(prob_online)

        # ADWIN Drift Status
        drift_events = self.feature_store.get_drift_events(limit=1)
        drift_status = "drift_detected" if (drift_events and self.online_learner.drift_count > 0) else "stable"
        if drift_status == "drift_detected":
            warnings_out.append("ADWIN detected market distribution shift — confidence dampened")
            conf_score *= 0.85

        # Scaled integer confidence (0 to 100)
        conf_int = int(min(90, max(20, conf_score * 85 * session_mult)))

        # ── 4. Position Sizing & Risk Management ──────────────────────────────
        entry = stop = target = 0.0
        qty = cap_req = max_loss_val = 0
        rr_str = ""

        if action != "WAIT" and base_price > 0:
            if action == "BUY":
                entry = round(base_price * 1.0005, 2)
                # Use real ATR or 2% stop loss
                stop_dist = base_price * 0.02
                if indicators and indicators.support and indicators.support < base_price:
                    stop = round(indicators.support * 0.998, 2)
                else:
                    stop = round(base_price - stop_dist, 2)
                target = round(entry + (entry - stop) * 2.0, 2)
            else:  # SELL
                entry = round(base_price * 0.9995, 2)
                stop_dist = base_price * 0.02
                if indicators and indicators.resistance and indicators.resistance > base_price:
                    stop = round(indicators.resistance * 1.002, 2)
                else:
                    stop = round(base_price + stop_dist, 2)
                target = round(entry - (stop - entry) * 2.0, 2)

            risk_ps = abs(entry - stop)
            if risk_ps > 0:
                max_risk = CAPITAL * MAX_RISK_PCT  # 1% capital risk
                qty = max(1, int(max_risk / risk_ps))
                cap_req = round(qty * entry, 2)
                max_loss_val = round(qty * risk_ps, 2)
                reward = abs(target - entry) * qty
                rr_ratio = round(reward / max_loss_val, 1) if max_loss_val else 0
                rr_str = f"1:{rr_ratio}"

        # ── 5. Reasoning & Advice ─────────────────────────────────────────────
        advice_parts = []
        if session == "closing":
            advice_parts.append("CLOSING TIME: Exit intraday positions before 3:15 PM.")
        elif action == "WAIT":
            advice_parts.append(f"Neutral probability ({prob_online:.1%}). No statistical edge. Standing aside.")
        else:
            advice_parts.append(
                f"Calibrated edge: {prob_online:.1%} upward probability with {conviction} conviction."
            )

        news_score = 0.0
        if news_sentiment and live_feed_on:
            news_score = news_sentiment.overall_score

        mem_score = 0.0
        if memory_context and memory_context.get("samples", 0) >= 3:
            wr = memory_context["win_rate"]
            mem_score = round((wr - 0.5) * 0.6, 3)

        combined_score = round(prob_online - 0.5, 3)

        return TradeSignal(
            action=action,
            conviction=conviction,
            confidence=conf_int,
            chart_score=round(prob_online - 0.5, 3),
            news_score=round(news_score, 3),
            memory_score=round(mem_score, 3),
            indicator_score=round(prob_dl - 0.5, 3),
            combined_score=combined_score,
            entry_price=entry,
            stop_price=stop,
            target_price=target,
            risk_per_share=round(abs(entry - stop), 2) if entry and stop else 0.0,
            quantity=qty,
            capital_required=cap_req,
            max_loss=max_loss_val,
            rr_ratio=rr_str,
            grade="A+" if conf_int >= 75 else ("A" if conf_int >= 60 else "B"),
            warnings=list(set(warnings_out)),
            matched_rules=matched_rules,
            final_advice=" ".join(advice_parts),
            live_price=live_px,
            live_change_pct=live_chg,
            session=session,
            ema_20=ema20,
            ema_50=ema50,
            rsi=rsi,
            macd_cross=macd_cross,
            patterns=patterns,
            prob_online=prob_online,
            prob_dl=prob_dl,
            prediction_id=pred_id,
            features_asof=features_asof,
            drift_status=drift_status,
        )

    def _load_rulebook(self) -> dict:
        try:
            with open(RULEBOOK_PATH) as f:
                return json.load(f)
        except Exception:
            return {}
