"""
ARGUS V5 — Outcome Tracker & Continuous Learning Loop Coordinator
Binds prediction at time T with realized outcome at time T+5min.
Enforces strict time-based leakage prevention and feeds verified (features, label) pairs
directly into the online incremental learner.
"""

import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List, Callable, Tuple
import pandas as pd

from data.feature_store import FeatureStore
from intelligence.online_learner import OnlineLearner


class OutcomeTracker:
    """
    Coordinates the 5-minute prediction-outcome cycle:
    1. At Time T:
       - Records prediction with auditable 'features_asof'
       - Maps calibrated probability to BUY / SELL / WAIT using confidence bands
       - Persists to FeatureStore
    2. At Time T+5min:
       - Fetches realized market price
       - Computes ground truth binary label y in {0, 1}
       - Updates OnlineLearner via learn_one()
       - Evaluates concept drift via ADWIN
       - Persists outcome to FeatureStore
    """

    def __init__(
        self,
        online_learner: Optional[OnlineLearner] = None,
        feature_store: Optional[FeatureStore] = None,
        buy_threshold: float = 0.58,
        sell_threshold: float = 0.42,
        horizon_minutes: int = 5,
    ):
        self.feature_store = feature_store or FeatureStore()
        self.online_learner = online_learner or OnlineLearner(feature_store=self.feature_store)
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.horizon_minutes = horizon_minutes

    # ── Prediction Phase (Time T) ─────────────────────────────────────────────

    def decide_action(self, prob_up: float) -> Tuple[str, str, float]:
        """
        Converts a continuous calibrated probability of upward movement into an action.
        Only emits BUY / SELL when crossing confidence thresholds; otherwise WAIT.

        Returns: (action, conviction, confidence_score)
        """
        if prob_up >= self.buy_threshold:
            action = "BUY"
            # Distance from 0.5 scaled to 0..1
            conf = min(1.0, (prob_up - 0.5) / 0.5)
            conviction = "high" if prob_up >= 0.68 else "moderate"
        elif prob_up <= self.sell_threshold:
            action = "SELL"
            conf = min(1.0, (0.5 - prob_up) / 0.5)
            conviction = "high" if prob_up <= 0.32 else "moderate"
        else:
            action = "WAIT"
            conf = 1.0 - (abs(prob_up - 0.5) / (self.buy_threshold - 0.5 + 1e-6))
            conviction = "neutral"

        return action, conviction, round(conf, 3)

    def record_prediction(
        self,
        ticker: str,
        features: Dict[str, Any],
        features_asof: str,
        base_price: float,
        timestamp: Optional[str] = None,
        dl_prob: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Logs a prediction at time T with strict leakage prevention audit metadata.
        """
        pred_id = f"pred_{str(uuid.uuid4())[:10]}"
        now_ts = timestamp or datetime.now(timezone.utc).isoformat()

        # Online learner calibrated probability
        prob_online = self.online_learner.predict_up_probability(features)
        action_online, conviction_online, conf_online = self.decide_action(prob_online)

        action_dl = None
        if dl_prob is not None:
            action_dl, _, _ = self.decide_action(dl_prob)

        # Persist to feature store
        self.feature_store.save_prediction(
            pred_id=pred_id,
            timestamp=now_ts,
            ticker=ticker,
            features_asof=features_asof,
            features=features,
            prob_online=prob_online,
            action_online=action_online,
            base_price=base_price,
            prob_dl=dl_prob,
            action_dl=action_dl,
        )

        return {
            "prediction_id": pred_id,
            "timestamp": now_ts,
            "ticker": ticker.upper(),
            "features_asof": features_asof,
            "base_price": base_price,
            "prob_online": prob_online,
            "action_online": action_online,
            "conviction": conviction_online,
            "confidence": conf_online,
            "prob_dl": dl_prob,
            "action_dl": action_dl,
        }

    # ── Outcome Phase (Time T+5min) ───────────────────────────────────────────

    def resolve_prediction(
        self,
        pred_id: str,
        realized_price: float,
        realized_timestamp: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Resolves a prediction at T+5min:
        - Computes true label y in {0, 1}
        - Updates OnlineLearner via learn_one()
        - Updates FeatureStore outcomes table
        """
        now_ts = realized_timestamp or datetime.now(timezone.utc).isoformat()

        # Update in database and compute pnl / accuracy
        res = self.feature_store.resolve_prediction(
            pred_id=pred_id,
            realized_timestamp=now_ts,
            realized_price=realized_price,
        )
        if not res:
            return None

        # Feed completed (features, realized_label) to online learner
        features = res["features"]
        realized_label = res["realized_label"]
        update_info = self.online_learner.learn_one(
            features=features,
            target=realized_label,
            timestamp=now_ts,
            pred_id=pred_id,
        )

        res["online_learner_metrics"] = update_info
        return res

    def check_and_resolve_pending(
        self,
        price_lookup_fn: Callable[[str], Optional[float]],
        current_timestamp: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """
        Scans all pending predictions. Any prediction older than horizon_minutes
        is fetched, compared against realized market price, and resolved.
        """
        now = current_timestamp or datetime.now(timezone.utc)
        pending = self.feature_store.get_pending_predictions()
        resolved_list = []

        for p in pending:
            try:
                pred_time = pd.to_datetime(p["timestamp"])
                if pred_time.tzinfo is None and now.tzinfo is not None:
                    pred_time = pred_time.replace(tzinfo=now.tzinfo)

                elapsed_minutes = (now - pred_time).total_seconds() / 60.0
                if elapsed_minutes >= self.horizon_minutes:
                    ticker = p["ticker"]
                    curr_price = price_lookup_fn(ticker)
                    if curr_price and curr_price > 0:
                        resolved = self.resolve_prediction(
                            pred_id=p["id"],
                            realized_price=curr_price,
                            realized_timestamp=now.isoformat(),
                        )
                        if resolved:
                            resolved_list.append(resolved)
            except Exception:
                continue

        return resolved_list

    # ── Performance Telemetry ─────────────────────────────────────────────────

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Returns real-time performance metrics computed from realized outcomes:
        - Directional accuracy vs 50% baseline
        - Realized trades count, win rate, total PnL %
        - Concept drift events count
        - Model Brier score
        """
        df = self.feature_store.get_resolved_dataset()
        status = self.online_learner.get_status()

        if df.empty:
            return {
                "total_resolved": 0,
                "pending_count": len(self.feature_store.get_pending_predictions()),
                "online_accuracy": 0.5,
                "brier_score": 0.25,
                "trade_win_rate": 0.0,
                "total_pnl_pct": 0.0,
                "drift_events_count": len(self.feature_store.get_drift_events()),
                "status": status,
            }

        total_resolved = len(df)
        correct_online = df["correct_online"].sum()
        online_acc = round(float(correct_online / total_resolved), 4)

        # Trading performance (only BUY or SELL signals)
        traded = df[df["action_online"].isin(["BUY", "SELL"])]
        if not traded.empty:
            wins = (traded["pnl_pct"] > 0).sum()
            trade_wr = round(float(wins / len(traded)), 4)
            total_pnl = round(float(traded["pnl_pct"].sum()), 4)
        else:
            trade_wr = 0.0
            total_pnl = 0.0

        return {
            "total_resolved": total_resolved,
            "pending_count": len(self.feature_store.get_pending_predictions()),
            "online_accuracy": online_acc,
            "brier_score": status["brier_score"],
            "trades_executed": len(traded),
            "trade_win_rate": trade_wr,
            "total_pnl_pct": total_pnl,
            "drift_events_count": len(self.feature_store.get_drift_events()),
            "status": status,
        }
