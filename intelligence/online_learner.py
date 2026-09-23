"""
ARGUS V5 — Continuous Online Stream Learner
Wraps River's HoeffdingAdaptiveTreeClassifier + StandardScaler + ADWIN drift detection.
Continuously updates split statistics per-sample without requiring full batch retraining.
"""

import os
import math
import pickle
import threading
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple
import numpy as np

from river import tree, forest, preprocessing, drift, metrics
from config import BASE_DIR
from data.feature_store import FeatureStore

DEFAULT_MODEL_DIR = os.path.join(BASE_DIR, "data", "models")
DEFAULT_ONLINE_MODEL_PATH = os.path.join(DEFAULT_MODEL_DIR, "online_learner.pkl")


class OnlineLearner:
    """
    Incremental ML classifier for continuous 15-minute directional market prediction.
    Features:
    - Incremental Hoeffding Adaptive Tree (HAT) or Adaptive Random Forest (ARF)
    - Online standard scaling of incoming numerical features
    - Probability recalibration (Platt scaling / Isotonic regression) on held-out validation fold
    - ADWIN (Adaptive Windowing) concept drift detection on classification error
    - Real-time calibration tracking via Brier Score and Rolling Accuracy
    - Auditable model checkpointing
    """

    def __init__(
        self,
        model_path: str = DEFAULT_ONLINE_MODEL_PATH,
        feature_store: Optional[FeatureStore] = None,
        grace_period: int = 50,
        adwin_delta: float = 0.002,
        model_type: Optional[str] = None,
        calibration: Optional[str] = None,
        n_models: int = 10,
    ):
        self.model_path = model_path
        self.feature_store = feature_store or FeatureStore()
        self.grace_period = grace_period
        self.adwin_delta = adwin_delta
        self.n_models = n_models

        # Architecture & calibration configuration
        env_model = os.environ.get("ARGUS_MODEL_TYPE", "HAT").upper()
        self.model_type = (model_type or env_model).upper()

        env_calib = os.environ.get("ARGUS_CALIBRATION", "none").lower()
        self.calibration_mode = (calibration or env_calib).lower()

        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

        # Pipeline: Online Scaler -> Classifier
        if self.model_type == "ARF":
            classifier = forest.ARFClassifier(
                n_models=self.n_models,
                grace_period=self.grace_period,
                split_criterion="gini",
                delta=1e-5,
                tau=0.05,
                leaf_prediction="nba",
                seed=42,
            )
            self.model_name = "ARFClassifier"
        else:
            classifier = tree.HoeffdingAdaptiveTreeClassifier(
                grace_period=self.grace_period,
                split_criterion="gini",
                delta=1e-5,
                tau=0.05,
                leaf_prediction="nba",
            )
            self.model_name = "HoeffdingAdaptiveTreeClassifier"

        self.pipeline = preprocessing.StandardScaler() | classifier

        # Calibration state (held-out validation fold)
        self.calibrator = None
        self.is_calibrated = False
        self.val_fold = []
        self.calib_fold_size = int(os.environ.get("ARGUS_CALIB_FOLD_SIZE", "50"))

        # ADWIN drift monitor on absolute prediction error
        self.drift_detector = drift.ADWIN(delta=self.adwin_delta)

        # Streaming evaluation metrics (Brier score is MSE between prob and binary label)
        self.accuracy_metric = metrics.Accuracy()
        self.brier_metric = metrics.MSE()
        self.samples_seen = 0
        self.drift_count = 0
        self.last_drift_timestamp = None
        self.prob_history = []

        # Load existing weights if present
        self._load_checkpoint()

    # ── Inference ─────────────────────────────────────────────────────────────

    def fit_calibrator(self):
        """Fit Platt scaling or Isotonic regression on held-out validation fold."""
        if len(self.val_fold) < 10:
            return

        targets = [y for _, y in self.val_fold]
        if len(set(targets)) < 2:
            return

        probs = [p for p, _ in self.val_fold]

        if self.calibration_mode == "platt":
            from sklearn.linear_model import LogisticRegression

            def to_logit(p):
                p_c = max(1e-5, min(1.0 - 1e-5, p))
                return math.log(p_c / (1.0 - p_c))

            X = np.array([[to_logit(p)] for p in probs])
            y = np.array(targets)
            clf = LogisticRegression(C=1.0, solver="lbfgs")
            clf.fit(X, y)
            self.calibrator = clf
            self.is_calibrated = True
        elif self.calibration_mode == "isotonic":
            from sklearn.isotonic import IsotonicRegression
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.05, y_max=0.95)
            iso.fit(probs, targets)
            self.calibrator = iso
            self.is_calibrated = True

    def predict_proba(self, features: Dict[str, Any]) -> Dict[int, float]:
        """
        Returns calibrated class probabilities: {0: P(down/flat), 1: P(up)}.
        """
        with self._lock:
            # Filter out non-numeric entries
            clean_feats = {k: float(v) for k, v in features.items() if isinstance(v, (int, float))}
            proba = self.pipeline.predict_proba_one(clean_feats)

            # Ensure both classes exist with smoothing if unobserved
            p_up = proba.get(1, 0.5 if self.samples_seen == 0 else 0.5)
            p_down = proba.get(0, 1.0 - p_up)

            # Normalize to sum to 1.0
            total = p_up + p_down
            if total > 0:
                p_up = p_up / total
                p_down = p_down / total
            else:
                p_up = 0.5
                p_down = 0.5

            if self.is_calibrated and self.calibrator is not None:
                if self.calibration_mode == "platt":
                    p_c = max(1e-5, min(1.0 - 1e-5, p_up))
                    logit_p = math.log(p_c / (1.0 - p_c))
                    cal_p = float(self.calibrator.predict_proba([[logit_p]])[0, 1])
                    p_up = max(0.0001, min(0.9999, cal_p))
                elif self.calibration_mode == "isotonic":
                    cal_p = float(self.calibrator.predict([p_up])[0])
                    p_up = max(0.0001, min(0.9999, cal_p))
                p_down = 1.0 - p_up

            return {0: round(float(p_down), 4), 1: round(float(p_up), 4)}

    def predict_up_probability(self, features: Dict[str, Any]) -> float:
        """Convenience method returning calibrated probability of upward move."""
        proba = self.predict_proba(features)
        return proba[1]

    # ── Learning & Drift Monitoring ───────────────────────────────────────────

    def learn_one(
        self,
        features: Dict[str, Any],
        target: int,
        timestamp: Optional[str] = None,
        pred_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Per-sample incremental update:
        1. Evaluates pre-update prediction error for ADWIN drift detection
        2. Collects validation fold predictions for calibration if enabled
        3. Calls learn_one() on the River pipeline
        4. Updates running Accuracy and Brier metrics
        5. Logs drift events to the feature store if detected

        Returns summary of post-update metrics.
        """
        if target not in (0, 1):
            raise ValueError(f"Target must be binary 0 or 1, got {target}")

        clean_feats = {k: float(v) for k, v in features.items() if isinstance(v, (int, float))}
        now_ts = timestamp or datetime.now(timezone.utc).isoformat()

        with self._lock:
            # 1. Pre-update prediction for honest online error tracking
            pre_proba = self.pipeline.predict_proba_one(clean_feats)
            p_up = pre_proba.get(1, 0.5)

            # Collect held-out validation samples for calibration
            if self.calibration_mode in ("platt", "isotonic") and not self.is_calibrated:
                self.val_fold.append((p_up, target))
                if len(self.val_fold) >= self.calib_fold_size:
                    self.fit_calibrator()

            # If calibrated, compute calibrated probability for metrics
            eval_p_up = p_up
            if self.is_calibrated and self.calibrator is not None:
                if self.calibration_mode == "platt":
                    p_c = max(1e-5, min(1.0 - 1e-5, p_up))
                    logit_p = math.log(p_c / (1.0 - p_c))
                    eval_p_up = float(self.calibrator.predict_proba([[logit_p]])[0, 1])
                elif self.calibration_mode == "isotonic":
                    eval_p_up = float(self.calibrator.predict([p_up])[0])

            pred_class = 1 if eval_p_up >= 0.5 else 0
            error = abs(target - eval_p_up)

            # 2. Update model
            self.pipeline.learn_one(clean_feats, target)
            self.samples_seen += 1

            # 3. Update evaluation metrics
            self.accuracy_metric.update(target, pred_class)
            self.brier_metric.update(target, eval_p_up)

            # 4. Update ADWIN concept drift detector
            val_before = self.drift_detector.estimation
            self.drift_detector.update(error)
            val_after = self.drift_detector.estimation

            drift_flag = False
            if self.drift_detector.drift_detected:
                drift_flag = True
                self.drift_count += 1
                self.last_drift_timestamp = now_ts
                msg = (
                    f"ADWIN detected concept drift at sample #{self.samples_seen}. "
                    f"Rolling error rate shifted from {val_before:.4f} to {val_after:.4f}."
                )
                # Log to FeatureStore
                if self.feature_store:
                    self.feature_store.log_drift_event(
                        timestamp=now_ts,
                        metric_name="adwin_error_rate",
                        value_before=float(val_before) if val_before is not None else None,
                        value_after=float(val_after) if val_after is not None else None,
                        message=msg,
                    )

            # Auto-save checkpoint periodically (every 25 samples)
            if self.samples_seen % 25 == 0:
                self._save_checkpoint()

            return {
                "samples_seen": self.samples_seen,
                "accuracy": round(float(self.accuracy_metric.get()), 4),
                "brier_score": round(float(self.brier_metric.get()), 4),
                "drift_detected": drift_flag,
                "drift_count": self.drift_count,
                "last_drift": self.last_drift_timestamp,
            }

    # ── State Persistence ─────────────────────────────────────────────────────

    def _save_checkpoint(self):
        try:
            state = {
                "pipeline": self.pipeline,
                "drift_detector": self.drift_detector,
                "accuracy_metric": self.accuracy_metric,
                "brier_metric": self.brier_metric,
                "samples_seen": self.samples_seen,
                "drift_count": self.drift_count,
                "last_drift_timestamp": self.last_drift_timestamp,
                "calibrator": self.calibrator,
                "is_calibrated": self.is_calibrated,
            }
            with open(self.model_path, "wb") as f:
                pickle.dump(state, f)
        except Exception as e:
            # Don't break online inference if save fails
            pass

    def _load_checkpoint(self):
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, "rb") as f:
                    state = pickle.load(f)
                    self.pipeline = state.get("pipeline", self.pipeline)
                    self.drift_detector = state.get("drift_detector", self.drift_detector)
                    self.accuracy_metric = state.get("accuracy_metric", self.accuracy_metric)
                    self.brier_metric = state.get("brier_metric", self.brier_metric)
                    self.samples_seen = state.get("samples_seen", 0)
                    self.drift_count = state.get("drift_count", 0)
                    self.last_drift_timestamp = state.get("last_drift_timestamp", None)
                    self.calibrator = state.get("calibrator", self.calibrator)
                    self.is_calibrated = state.get("is_calibrated", self.is_calibrated)
            except Exception:
                pass

    def get_status(self) -> Dict[str, Any]:
        """Return diagnostic health and performance telemetry."""
        with self._lock:
            calib_suffix = f" + {self.calibration_mode.capitalize()} Calibration" if self.is_calibrated else ""
            return {
                "model_type": f"River {self.model_name} + StandardScaler{calib_suffix}",
                "samples_seen": self.samples_seen,
                "accuracy": round(float(self.accuracy_metric.get()), 4) if self.samples_seen > 0 else 0.5,
                "brier_score": round(float(self.brier_metric.get()), 4) if self.samples_seen > 0 else 0.25,
                "drift_count": self.drift_count,
                "last_drift_timestamp": self.last_drift_timestamp,
                "checkpoint_path": self.model_path,
                "is_calibrated": self.is_calibrated,
            }
