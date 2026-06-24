"""
Event Database — Ghi nhận overload/anomaly events cho attention-weighted retraining.

Khi scheduler phát hiện host bị overload mà model không dự đoán được, event
được ghi vào DB. Khi retrain Autoformer/BiLSTM, các time windows gần overload
events sẽ có attention weight cao hơn trong loss function.

Schema:
  events:         (id, sim_time, step, event_type, host_id, cpu_util, predicted_state, details)
  model_versions: (id, model_type, version, path, val_loss, metrics_json, created_at)
"""

import os
import json
import sqlite3
import numpy as np
from datetime import datetime


class EventDatabase:
    """SQLite-based event store for overload logging and attention-weighted retraining."""

    EVENT_TYPES = ("overload_missed", "overload_detected", "underload_missed",
                   "anomaly", "model_swap", "retrain_trigger")

    def __init__(self, db_path="events.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        c = self.conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sim_time REAL NOT NULL,
                step INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                host_id INTEGER,
                cpu_util REAL,
                predicted_state TEXT,
                severity REAL DEFAULT 1.0,
                details TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS model_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_type TEXT NOT NULL,
                version INTEGER NOT NULL,
                path TEXT NOT NULL,
                val_loss REAL,
                metrics TEXT,
                is_active INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        c.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_sim_time
            ON events(sim_time)
        """)
        c.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_type
            ON events(event_type)
        """)
        c.execute("""
            CREATE INDEX IF NOT EXISTS idx_model_type_active
            ON model_versions(model_type, is_active)
        """)
        self.conn.commit()

    # ==================== Event Logging ====================

    def log_overload_missed(self, sim_time, step, host_id, cpu_util,
                            predicted_state="NORMAL", severity=1.0, details=None):
        """Log when an overload occurs but model predicted NORMAL."""
        self._insert_event(
            sim_time=sim_time, step=step,
            event_type="overload_missed",
            host_id=host_id, cpu_util=cpu_util,
            predicted_state=predicted_state,
            severity=severity,
            details=details
        )

    def log_overload_detected(self, sim_time, step, host_id, cpu_util,
                              details=None):
        """Log when an overload is correctly detected."""
        self._insert_event(
            sim_time=sim_time, step=step,
            event_type="overload_detected",
            host_id=host_id, cpu_util=cpu_util,
            predicted_state="OVERLOAD",
            severity=0.0,
            details=details
        )

    def log_anomaly(self, sim_time, step, metric_name, value, threshold,
                    details=None):
        """Log anomaly detected by monitoring system."""
        self._insert_event(
            sim_time=sim_time, step=step,
            event_type="anomaly",
            host_id=None,
            cpu_util=value,
            predicted_state=metric_name,
            severity=abs(value - threshold),
            details=details
        )

    def log_retrain_trigger(self, sim_time, step, model_type, reason=None):
        """Log when a retrain is triggered."""
        self._insert_event(
            sim_time=sim_time, step=step,
            event_type="retrain_trigger",
            host_id=None, cpu_util=None,
            predicted_state=model_type,
            severity=0.0,
            details=reason
        )

    def _insert_event(self, **kwargs):
        c = self.conn.cursor()
        c.execute("""
            INSERT INTO events (sim_time, step, event_type, host_id, cpu_util,
                                predicted_state, severity, details)
            VALUES (:sim_time, :step, :event_type, :host_id, :cpu_util,
                    :predicted_state, :severity, :details)
        """, kwargs)
        self.conn.commit()

    # ==================== Query Events ====================

    def get_events_since(self, since_step=0, event_type=None):
        """Get events since a given step."""
        c = self.conn.cursor()
        if event_type:
            c.execute(
                "SELECT * FROM events WHERE step >= ? AND event_type = ? ORDER BY step",
                (since_step, event_type)
            )
        else:
            c.execute(
                "SELECT * FROM events WHERE step >= ? ORDER BY step",
                (since_step,)
            )
        return [dict(row) for row in c.fetchall()]

    def get_overload_events(self, since_step=0):
        """Get all missed overload events for attention weighting."""
        return self.get_events_since(since_step, "overload_missed")

    def get_all_events_summary(self):
        """Get summary counts by event type."""
        c = self.conn.cursor()
        c.execute("""
            SELECT event_type, COUNT(*) as count,
                   AVG(severity) as avg_severity,
                   MAX(severity) as max_severity
            FROM events
            GROUP BY event_type
        """)
        return {row["event_type"]: dict(row) for row in c.fetchall()}

    # ==================== Attention Weights ====================

    def compute_attention_weights(self, timestamps, sigma=42, base_weight=1.0,
                                  event_multiplier=3.0, since_step=0):
        """
        Compute attention weights for training samples based on overload events.

        Samples near overload events get higher weight → model focuses on
        patterns that led to missed overloads.

        Args:
            timestamps: array of step numbers for training samples
            sigma: Gaussian kernel width (steps) — half of retrain interval
            base_weight: minimum weight for all samples
            event_multiplier: extra weight scale for event-adjacent samples

        Returns:
            weights: array of shape [len(timestamps)], normalized
        """
        events = self.get_overload_events(since_step=since_step)
        timestamps = np.asarray(timestamps, dtype=np.float64)
        weights = np.full(len(timestamps), base_weight, dtype=np.float64)

        if not events:
            return weights / weights.sum()

        for event in events:
            event_step = event["step"]
            severity = event.get("severity", 1.0) or 1.0

            # Gaussian attention: peak at event time, decay with distance
            distances = np.abs(timestamps - event_step)
            attention = np.exp(-distances ** 2 / (2 * sigma ** 2))
            weights += attention * severity * event_multiplier

        # Normalize so weights sum to len(timestamps) (like uniform = 1 each)
        weights = weights / weights.mean()
        return weights

    # ==================== Cleanup ====================

    def clear_events(self):
        """Clear all events (for testing)."""
        c = self.conn.cursor()
        c.execute("DELETE FROM events")
        self.conn.commit()

    def close(self):
        self.conn.close()

    def __del__(self):
        try:
            self.conn.close()
        except Exception:
            pass
