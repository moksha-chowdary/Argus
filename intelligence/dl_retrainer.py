"""
ARGUS V5 — Periodic Deep Learning Retrainer (LSTM Sequence Model)
Implements a rolling-window recurrent neural network in PyTorch for capturing
multi-timestep temporal dependencies across OHLCV and technical feature vectors.

===============================================================================
ARCHITECTURAL RATIONALE: Periodic Mini-Batch Retraining vs. Online Per-Sample Updates
===============================================================================
In streaming financial ML, there is a fundamental dichotomy between tree-based
online models (e.g. River's Hoeffding Adaptive Trees) and Deep Neural Networks (DNNs):

1. CATASTROPHIC FORGETTING:
   Stochastic gradient descent (SGD/Adam) applied with a batch size of 1 on continuous
   non-stationary financial streams induces severe catastrophic forgetting. The network's
   distributed weights overfit immediately to the latest single sample's noise, erasing
   previously learned representation geometry across different market regimes (e.g. high vs.
   low volatility states).

2. GRADIENT NOISE & WEIGHT DRIFT:
   Individual 5-minute intraday price changes have notoriously low signal-to-noise ratios.
   Per-sample gradient vectors fluctuate wildly, leading to unstable internal representations,
   exploding activations, and optimizer momentum degradation. Mini-batch training (B >= 32)
   averages out high-frequency microstructure noise to yield clean, directional gradients.

3. TEMPORAL SEQUENCE INTEGRITY:
   Recurrent architectures (LSTM / GRU) require coherent sequential sequences of length L
   (e.g., L = 15 bars = 75 minutes of market context) with standardized input distribution.
   Per-sample updates disrupt sequence continuity and recurrent hidden state initialization.

4. SEPARATION OF CONCERNS (ENSEMBLE DIVERSITY):
   - Fast Online Learner (River HAT): Responds in milliseconds to sudden intraday distribution
     shifts, updating local decision boundaries per-sample.
   - Periodic Deep Retrainer (PyTorch LSTM): Retrains every N samples or hourly over a rolling
     historical window (e.g. past 30 days) to identify persistent macro-temporal patterns.
===============================================================================
"""

import os
import threading
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from config import BASE_DIR
from data.feature_store import FeatureStore

DEFAULT_MODEL_DIR = os.path.join(BASE_DIR, "data", "models")
DEFAULT_DL_PATH = os.path.join(DEFAULT_MODEL_DIR, "dl_lstm.pt")


class MarketLSTM(nn.Module):
    """
    2-Layer LSTM with Input BatchNorm1d and GELU projection head for calibrated directional probability.
    Input shape:  (Batch, Seq_Len, Feature_Dim) [Default Feature_Dim = 40]
    Output shape: (Batch, 1) in [0.0, 1.0] representing P(Price_{t+15m} > Price_t)
    """

    def __init__(self, input_dim: int, hidden_dim: int = 48, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Input normalization across sequence dimension to stabilize multi-scale features
        self.bn = nn.BatchNorm1d(input_dim)

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 24),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(24, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, L, D)
        B, L, D = x.shape
        x_flat = x.reshape(B * L, D)
        # Handle batch norm for batch size >= 2 during training or eval
        if self.training and B * L > 1:
            x_norm = self.bn(x_flat).reshape(B, L, D)
        else:
            self.bn.eval()
            x_norm = self.bn(x_flat).reshape(B, L, D)

        lstm_out, _ = self.lstm(x_norm)
        last_hidden = lstm_out[:, -1, :]
        prob = self.head(last_hidden)
        return prob


class DeepLearningRetrainer:
    """
    Manages the periodic training, checkpointing, and inference of the PyTorch LSTM model.
    """

    def __init__(
        self,
        input_dim: int = 40,
        seq_len: int = 15,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        model_path: str = DEFAULT_DL_PATH,
        feature_store: Optional[FeatureStore] = None,
    ):
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.model_path = model_path
        self.feature_store = feature_store or FeatureStore()

        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MarketLSTM(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=2,
            dropout=0.2,
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self.criterion = nn.BCELoss()

        self.retrain_count = 0
        self.last_retrain_time = None
        self.last_train_loss = 0.0
        self.total_samples_trained = 0

        # Load existing weights if present
        self._load_checkpoint()

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict_proba(self, sequence_features: np.ndarray | List[List[float]]) -> float:
        """
        Calculates calibrated probability P(up) given an input sequence of shape (Seq_Len, Feature_Dim).
        """
        with self._lock:
            self.model.eval()
            arr = np.array(sequence_features, dtype=np.float32)

            # Pad or truncate to self.seq_len if needed
            if arr.ndim == 1:
                # Single feature vector: repeat to form minimal sequence
                arr = np.tile(arr, (self.seq_len, 1))
            elif arr.shape[0] < self.seq_len:
                pad_width = ((self.seq_len - arr.shape[0], 0), (0, 0))
                arr = np.pad(arr, pad_width, mode="edge")
            elif arr.shape[0] > self.seq_len:
                arr = arr[-self.seq_len:]

            tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(tensor)
                prob_up = float(output.item())

            return round(float(np.clip(prob_up, 0.001, 0.999)), 4)

    # ── Periodic Mini-Batch Retraining ─────────────────────────────────────────

    def retrain_batch(
        self,
        X_sequences: np.ndarray,
        y_labels: np.ndarray,
        epochs: int = 5,
        batch_size: int = 32,
    ) -> Dict[str, Any]:
        """
        Retrains the neural network on buffered sequences using mini-batch AdamW.
        X_sequences: shape (N, seq_len, input_dim)
        y_labels:    shape (N,) with binary 0 or 1
        """
        if len(X_sequences) < batch_size:
            return {
                "status": "skipped",
                "reason": f"Insufficient samples ({len(X_sequences)}) for batch_size ({batch_size})",
            }

        with self._lock:
            self.model.train()
            X_tensor = torch.tensor(X_sequences, dtype=torch.float32)
            y_tensor = torch.tensor(y_labels, dtype=torch.float32).unsqueeze(1)

            dataset = TensorDataset(X_tensor, y_tensor)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

            total_loss = 0.0
            num_batches = 0

            for epoch in range(epochs):
                epoch_loss = 0.0
                for batch_x, batch_y in loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)

                    self.optimizer.zero_grad()
                    preds = self.model(batch_x)
                    loss = self.criterion(preds, batch_y)
                    loss.backward()

                    # Gradient clipping to prevent exploding gradients in volatile market regimes
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                    epoch_loss += loss.item()
                    num_batches += 1
                total_loss = epoch_loss

            avg_loss = total_loss / max(1, len(loader))
            self.retrain_count += 1
            self.last_retrain_time = datetime.now(timezone.utc).isoformat()
            self.last_train_loss = round(float(avg_loss), 4)
            self.total_samples_trained += len(X_sequences)

            self._save_checkpoint()

            return {
                "status": "completed",
                "retrain_count": self.retrain_count,
                "epochs": epochs,
                "samples_trained": len(X_sequences),
                "avg_loss": self.last_train_loss,
                "timestamp": self.last_retrain_time,
            }

    def retrain_from_store(
        self,
        min_samples: int = 40,
        epochs: int = 5,
        batch_size: int = 32,
    ) -> Dict[str, Any]:
        """
        Fetches resolved dataset from SQLite feature store, builds temporal sequences,
        and executes a scheduled retraining cycle.
        """
        df = self.feature_store.get_resolved_dataset()
        if df.empty or len(df) < min_samples + self.seq_len:
            return {
                "status": "skipped",
                "reason": f"Need at least {min_samples + self.seq_len} resolved records in feature store",
            }

        # Extract numeric feature columns
        feat_cols = [c for c in df.columns if c.startswith("feat_")]
        if not feat_cols:
            return {"status": "error", "reason": "No feature columns found in dataset"}

        X_raw = df[feat_cols].values.astype(np.float32)
        # Impute any NaNs
        X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=1.0, neginf=-1.0)
        y_raw = df["realized_label"].values.astype(np.float32)

        # Construct sliding temporal sequences: (N - seq_len, seq_len, D)
        X_seqs = []
        y_seqs = []
        for i in range(len(X_raw) - self.seq_len):
            X_seqs.append(X_raw[i : i + self.seq_len])
            y_seqs.append(y_raw[i + self.seq_len])

        X_seqs = np.array(X_seqs)
        y_seqs = np.array(y_seqs)

        return self.retrain_batch(X_seqs, y_seqs, epochs=epochs, batch_size=batch_size)

    # ── Checkpointing ─────────────────────────────────────────────────────────

    def _save_checkpoint(self):
        try:
            state = {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "retrain_count": self.retrain_count,
                "last_retrain_time": self.last_retrain_time,
                "last_train_loss": self.last_train_loss,
                "total_samples_trained": self.total_samples_trained,
                "input_dim": self.input_dim,
                "seq_len": self.seq_len,
            }
            torch.save(state, self.model_path)
        except Exception:
            pass

    def _load_checkpoint(self):
        if os.path.exists(self.model_path):
            try:
                state = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(state["model_state_dict"])
                self.optimizer.load_state_dict(state["optimizer_state_dict"])
                self.retrain_count = state.get("retrain_count", 0)
                self.last_retrain_time = state.get("last_retrain_time", None)
                self.last_train_loss = state.get("last_train_loss", 0.0)
                self.total_samples_trained = state.get("total_samples_trained", 0)
            except Exception:
                pass

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "model_type": "PyTorch Rolling-Window LSTM (Periodic Mini-Batch Retrainer)",
                "input_dim": self.input_dim,
                "seq_len": self.seq_len,
                "device": str(self.device),
                "retrain_count": self.retrain_count,
                "last_train_loss": self.last_train_loss,
                "total_samples_trained": self.total_samples_trained,
                "last_retrain_time": self.last_retrain_time,
                "checkpoint_path": self.model_path,
            }
