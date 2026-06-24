"""
Offline Trainer — Batch Layer of Lambda Architecture.

Runs on a separate thread. Periodically retrains Autoformer and BiLSTM
using attention-weighted loss based on overload events from EventDatabase.

Key insight: When the scheduler encounters overloads that weren't predicted,
those time windows get higher attention weights during retraining, so the
model learns to focus on patterns that led to missed overloads.

Retrain schedule (from EDA):
  - BiLSTM:    every 7h simulation time (84 steps) — ACF peak at lag 7h
  - Autoformer: every 24h simulation time (288 steps) — daily cycle
"""

import os
import threading
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from collections import defaultdict

from config import Config
from autoformer_detector import AutoformerDetector
from event_database import EventDatabase
from model_registry import ModelRegistry


class OfflineTrainer:
    """
    Batch retraining module — runs asynchronously from the scheduler.

    Implements attention-weighted retraining:
      1. Fetch overload events from EventDatabase
      2. Compute Gaussian attention weights for training samples
      3. Retrain with weighted loss (samples near overloads get more focus)
      4. Save to Model Registry if improved
    """

    def __init__(self, event_db, model_registry, config=None):
        self.event_db = event_db
        self.registry = model_registry
        self.config = config or Config

        # Track last retrain timestamps
        self._last_lstm_retrain_step = 0
        self._last_af_retrain_step = 0

        # Lock for thread safety
        self._lock = threading.Lock()
        self._training_in_progress = {"bilstm": False, "autoformer": False}

    def should_retrain_lstm(self, current_step):
        """Check if BiLSTM should retrain from live process signals."""
        return self.config.should_retrain("bilstm", current_step)

    def should_retrain_autoformer(self, current_step):
        """Check if Autoformer should retrain from live process signals."""
        return self.config.should_retrain("autoformer", current_step)

    def trigger_lstm_retrain(self, cpu_history, current_step, sim_time, async_=True):
        """
        Trigger BiLSTM retraining.

        Args:
            cpu_history: dict {host_id: [cpu_util, ...]} — recent CPU data
            current_step: int
            sim_time: float
            async_: if True, run in background thread
        """
        if self._training_in_progress["bilstm"]:
            print("[Trainer] BiLSTM retrain already in progress, skipping")
            return

        self._last_lstm_retrain_step = current_step
        self.config.mark_retrain("bilstm", current_step)

        # Log retrain trigger
        if self.event_db:
            self.event_db.log_retrain_trigger(sim_time, current_step, "bilstm",
                                              reason="Process drift/SLA/failure trigger")

        if async_:
            t = threading.Thread(
                target=self._retrain_lstm_worker,
                args=(cpu_history, current_step),
                daemon=True
            )
            t.start()
        else:
            self._retrain_lstm_worker(cpu_history, current_step)

    def trigger_autoformer_retrain(self, cpu_history, current_step, sim_time, async_=True):
        """Trigger Autoformer retraining."""
        if self._training_in_progress["autoformer"]:
            print("[Trainer] Autoformer retrain already in progress, skipping")
            return

        self._last_af_retrain_step = current_step
        self.config.mark_retrain("autoformer", current_step)

        if self.event_db:
            self.event_db.log_retrain_trigger(sim_time, current_step, "autoformer",
                                              reason="Process drift/SLA/failure trigger")

        if async_:
            t = threading.Thread(
                target=self._retrain_autoformer_worker,
                args=(cpu_history, current_step),
                daemon=True
            )
            t.start()
        else:
            self._retrain_autoformer_worker(cpu_history, current_step)

    # ==================== BiLSTM Retraining ====================

    def _retrain_lstm_worker(self, cpu_history, current_step):
        """Background worker for BiLSTM retraining."""
        self._training_in_progress["bilstm"] = True
        print(f"[Trainer] BiLSTM retrain started at step {current_step}")

        try:
            # 1. Prepare training data from CPU history
            X, y, timestamps = self._prepare_lstm_data(cpu_history)

            if len(X) < 50:
                print(f"[Trainer] BiLSTM: Not enough data ({len(X)} samples), skipping")
                return

            # 2. Compute attention weights from overload events
            weights = self.event_db.compute_attention_weights(
                timestamps=timestamps,
                sigma=self.config.ATTENTION_SIGMA_STEPS,
                base_weight=self.config.ATTENTION_BASE_WEIGHT,
                event_multiplier=self.config.ATTENTION_EVENT_MULTIPLIER,
                since_step=max(0, current_step - self.config.LSTM_RETRAIN_INTERVAL_STEPS),
            )

            # 3. Train with weighted loss
            from lstm_underload_detector import UnderloadBiLSTM
            model = UnderloadBiLSTM()

            # Load current best as initialization
            current_best = self.registry.load_best_model("bilstm")
            if current_best is not None:
                model.load_state_dict(current_best)

            val_loss = self._train_lstm_weighted(model, X, y, weights)

            # 4. Save to registry
            self.registry.save_model(
                model, "bilstm",
                metrics={"val_loss": val_loss, "num_samples": len(X),
                         "step": current_step}
            )

            print(f"[Trainer] BiLSTM retrain complete: val_loss={val_loss:.6f}, "
                  f"samples={len(X)}")

        except Exception as e:
            print(f"[Trainer] BiLSTM retrain FAILED: {e}")
        finally:
            self._training_in_progress["bilstm"] = False

    def _prepare_lstm_data(self, cpu_history):
        """
        Prepare (X, y) pairs for BiLSTM from CPU history.

        X: [window] of CPU utilization
        y: 1 if ALL next [horizon] steps < underload_threshold, else 0
        """
        window = self.config.LSTM_WINDOW
        horizon = self.config.LSTM_HORIZON
        threshold = self.config.UNDERLOAD_THRESHOLD

        X_all, y_all, ts_all = [], [], []

        for host_id, series in cpu_history.items():
            if len(series) < window + horizon:
                continue
            for i in range(window, len(series) - horizon):
                x_seq = series[i - window:i]
                future = series[i:i + horizon]
                label = 1.0 if all(v < threshold for v in future) else 0.0
                X_all.append(x_seq)
                y_all.append(label)
                ts_all.append(i)  # step index as timestamp

        X = np.array(X_all, dtype=np.float32)
        y = np.array(y_all, dtype=np.float32)
        timestamps = np.array(ts_all, dtype=np.float64)

        return X, y, timestamps

    def _train_lstm_weighted(self, model, X, y, weights, epochs=30, lr=None):
        """Train BiLSTM with attention-weighted BCEWithLogitsLoss."""
        lr = lr or self.config.LSTM_LR

        # Split train/val (80/20)
        n = len(X)
        idx = np.random.permutation(n)
        split = int(0.8 * n)
        train_idx, val_idx = idx[:split], idx[split:]

        X_train = torch.FloatTensor(X[train_idx]).unsqueeze(-1)
        y_train = torch.FloatTensor(y[train_idx]).unsqueeze(-1)
        w_train = torch.FloatTensor(weights[train_idx])

        X_val = torch.FloatTensor(X[val_idx]).unsqueeze(-1)
        y_val = torch.FloatTensor(y[val_idx]).unsqueeze(-1)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss(reduction="none")

        model.train()
        best_val_loss = float("inf")

        for epoch in range(epochs):
            # Forward
            logits = model(X_train)
            raw_loss = criterion(logits, y_train).squeeze()
            # Apply attention weights
            weighted_loss = (raw_loss * w_train).mean()

            optimizer.zero_grad()
            weighted_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Validation
            if (epoch + 1) % 10 == 0:
                model.eval()
                with torch.no_grad():
                    val_logits = model(X_val)
                    val_loss = F.binary_cross_entropy_with_logits(
                        val_logits, y_val
                    ).item()
                model.train()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss

        return best_val_loss

    # ==================== Autoformer Retraining ====================

    def _retrain_autoformer_worker(self, cpu_history, current_step):
        """Background worker for Autoformer retraining."""
        self._training_in_progress["autoformer"] = True
        print(f"[Trainer] Autoformer retrain started at step {current_step}")

        try:
            # 1. Prepare training data
            X, y, timestamps = self._prepare_autoformer_data(cpu_history)

            if len(X) < 30:
                print(f"[Trainer] Autoformer: Not enough data ({len(X)} samples), skipping")
                return

            # 2. Compute attention weights
            weights = self.event_db.compute_attention_weights(
                timestamps=timestamps,
                sigma=self.config.ATTENTION_SIGMA_STEPS,
                base_weight=self.config.ATTENTION_BASE_WEIGHT,
                event_multiplier=self.config.ATTENTION_EVENT_MULTIPLIER,
                since_step=max(0, current_step - self.config.AUTOFORMER_RETRAIN_INTERVAL_STEPS),
            )

            # 3. Train with weighted loss
            model = AutoformerDetector(
                seq_len=self.config.AF_SEQ_LEN,
                pred_len=self.config.AF_PRED_LEN,
                d_model=self.config.AF_D_MODEL,
            )

            # Load current best
            current_best = self.registry.load_best_model("autoformer")
            if current_best is not None:
                model.load_state_dict(current_best)

            val_loss = self._train_autoformer_weighted(model, X, y, weights)

            # 4. Save to registry
            self.registry.save_model(
                model, "autoformer",
                metrics={"val_loss": val_loss, "num_samples": len(X),
                         "step": current_step}
            )

            print(f"[Trainer] Autoformer retrain complete: val_loss={val_loss:.6f}, "
                  f"samples={len(X)}")

        except Exception as e:
            print(f"[Trainer] Autoformer retrain FAILED: {e}")
        finally:
            self._training_in_progress["autoformer"] = False

    def _prepare_autoformer_data(self, cpu_history):
        """
        Prepare (X, y) for Autoformer.

        X: [seq_len] of past CPU utilization
        y: [pred_len] of future CPU utilization
        """
        seq_len = self.config.AF_SEQ_LEN
        pred_len = self.config.AF_PRED_LEN

        X_all, y_all, ts_all = [], [], []

        for host_id, series in cpu_history.items():
            if len(series) < seq_len + pred_len:
                continue
            for i in range(seq_len, len(series) - pred_len):
                x_seq = series[i - seq_len:i]
                y_seq = series[i:i + pred_len]
                X_all.append(x_seq)
                y_all.append(y_seq)
                ts_all.append(i)

        X = np.array(X_all, dtype=np.float32)
        y = np.array(y_all, dtype=np.float32)
        timestamps = np.array(ts_all, dtype=np.float64)

        return X, y, timestamps

    def _train_autoformer_weighted(self, model, X, y, weights, epochs=30, lr=None):
        """Train Autoformer with attention-weighted MSE loss."""
        lr = lr or self.config.AF_LR

        # Split
        n = len(X)
        idx = np.random.permutation(n)
        split = int(0.8 * n)
        train_idx, val_idx = idx[:split], idx[split:]

        X_train = torch.FloatTensor(X[train_idx])
        y_train = torch.FloatTensor(y[train_idx])
        w_train = torch.FloatTensor(weights[train_idx])

        X_val = torch.FloatTensor(X[val_idx])
        y_val = torch.FloatTensor(y[val_idx])

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        model.train()
        best_val_loss = float("inf")

        for epoch in range(epochs):
            preds = model(X_train)
            # Per-sample MSE
            raw_loss = ((preds - y_train) ** 2).mean(dim=1)  # [batch]
            # Weighted
            weighted_loss = (raw_loss * w_train).mean()

            optimizer.zero_grad()
            weighted_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                model.eval()
                with torch.no_grad():
                    val_preds = model(X_val)
                    val_loss = F.mse_loss(val_preds, y_val).item()
                model.train()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss

        return best_val_loss

    def is_training(self):
        """Check if any retraining is in progress."""
        return any(self._training_in_progress.values())
