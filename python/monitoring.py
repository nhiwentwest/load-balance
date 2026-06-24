"""
Monitoring System — Prometheus/Ganglia-like metrics collection and anomaly detection.

Collects metrics from CloudSim simulation at each step:
  - CPU utilization (avg, std, min, max)
  - Active host ratio
  - Energy consumption
  - SLATAH, PDM
  - Migration count
  - Overload/underload counts

Anomaly detection via z-score on rolling statistics.
When anomaly detected → logs to EventDatabase.
"""

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np
from collections import defaultdict, deque
from config import Config


class MonitoringSystem:
    """Simulated monitoring system with real-time anomaly detection."""

    # Metrics to track
    METRICS = [
        "cpu_avg", "cpu_std", "active_ratio", "energy_kwh",
        "slatah", "migrations", "overloads", "critical_overloads",
        "capacity_overloads",
        "actionable_overloads", "selected_overload_sources",
        "guard_overload_sources", "unactionable_overloads",
        "gpu_blocked_overloads", "capacity_blocked_overloads",
        "single_vm_overloads", "a2_true_positive",
        "a2_false_positive", "a2_false_negative", "underloads",
    ]

    def __init__(self, event_db=None, config=None):
        self.event_db = event_db
        self.config = config or Config

        # Rolling history for each metric (for z-score anomaly detection)
        self._history = {m: deque(maxlen=200) for m in self.METRICS}
        # Full history for export
        self._full_history = {m: [] for m in self.METRICS}
        self._latest_metrics = {m: 0.0 for m in self.METRICS}
        self._timestamps = []
        self._step_count = 0

    def record_step(self, step, sim_time, info, global_state=None):
        """
        Record metrics from one simulation step.

        Args:
            step: int, current step number
            sim_time: float, simulation time in seconds
            info: dict from env.step() — contains slatah, pdm, energy_kwh, etc.
            global_state: np.array, global state vector [8 dims]
        """
        metrics = {}

        # From global state
        if global_state is not None and len(global_state) >= 8:
            metrics["cpu_avg"] = float(global_state[0])
            metrics["cpu_std"] = float(global_state[1])
            metrics["active_ratio"] = float(global_state[2])
        else:
            metrics["cpu_avg"] = 0.0
            metrics["cpu_std"] = 0.0
            metrics["active_ratio"] = 1.0

        # From info dict
        metrics["energy_kwh"] = info.get("energy_kwh", 0.0)
        metrics["slatah"] = info.get("slatah", 0.0)
        metrics["migrations"] = info.get("migrations", 0)
        metrics["overloads"] = info.get("overloads", 0)
        metrics["critical_overloads"] = info.get("critical_overloads", 0)
        metrics["capacity_overloads"] = info.get("capacity_overloads", 0)
        metrics["actionable_overloads"] = info.get("actionable_overloads", 0)
        metrics["selected_overload_sources"] = info.get("selected_overload_sources", 0)
        metrics["guard_overload_sources"] = info.get("guard_overload_sources", 0)
        metrics["unactionable_overloads"] = info.get("unactionable_overloads", 0)
        metrics["gpu_blocked_overloads"] = info.get("gpu_blocked_overloads", 0)
        metrics["capacity_blocked_overloads"] = info.get("capacity_blocked_overloads", 0)
        metrics["single_vm_overloads"] = info.get("single_vm_overloads", 0)
        metrics["a2_true_positive"] = info.get("a2_true_positive", 0)
        metrics["a2_false_positive"] = info.get("a2_false_positive", 0)
        metrics["a2_false_negative"] = info.get("a2_false_negative", 0)
        metrics["underloads"] = info.get("underloads", 0)

        # Store
        self._timestamps.append(sim_time)
        for m in self.METRICS:
            val = metrics.get(m, 0.0)
            self._history[m].append(val)
            self._full_history[m].append(val)
            self._latest_metrics[m] = float(val)

        self._step_count += 1

        # Periodic anomaly check
        if self._step_count % self.config.MONITORING_INTERVAL_STEPS == 0:
            self._check_anomalies(step, sim_time, metrics)

    def _check_anomalies(self, step, sim_time, current_metrics):
        """Check for anomalies using z-score on rolling window."""
        if len(self._history["cpu_avg"]) < 30:
            return  # Need enough history

        anomalies = []

        for metric_name in ["cpu_avg", "slatah", "overloads"]:
            history = np.array(self._history[metric_name])
            current = current_metrics.get(metric_name, 0.0)

            mean = np.mean(history)
            std = np.std(history)

            if std < 1e-8:
                continue

            z_score = abs(current - mean) / std

            if z_score > self.config.ANOMALY_ZSCORE_THRESHOLD:
                anomalies.append({
                    "metric": metric_name,
                    "value": current,
                    "mean": mean,
                    "std": std,
                    "z_score": z_score,
                })

                # Log to event database
                if self.event_db:
                    self.event_db.log_anomaly(
                        sim_time=sim_time,
                        step=step,
                        metric_name=metric_name,
                        value=current,
                        threshold=mean + self.config.ANOMALY_ZSCORE_THRESHOLD * std,
                        details=f"z={z_score:.2f}, μ={mean:.4f}, σ={std:.4f}"
                    )

        if anomalies:
            print(f"[Monitor] Step {step}: {len(anomalies)} anomaly(ies) detected:")
            for a in anomalies:
                print(f"  {a['metric']}: {a['value']:.4f} "
                      f"(z={a['z_score']:.2f}, μ={a['mean']:.4f})")

    def get_dashboard_data(self):
        """Export all metrics for visualization/reporting."""
        return {
            "timestamps": list(self._timestamps),
            "metrics": {m: list(self._full_history[m]) for m in self.METRICS},
            "total_steps": self._step_count,
        }

    def get_prometheus_metrics(self):
        """Render current monitor/config state in Prometheus text format."""
        def sample(name, value, labels=None):
            label_text = ""
            if labels:
                pairs = [f'{k}="{str(v).replace(chr(34), chr(92) + chr(34))}"'
                         for k, v in sorted(labels.items())]
                label_text = "{" + ",".join(pairs) + "}"
            return f"{name}{label_text} {float(value):.12g}"

        lines = [
            "# HELP dacn_monitor_step_count Number of CloudSim scheduler steps observed.",
            "# TYPE dacn_monitor_step_count gauge",
            sample("dacn_monitor_step_count", self._step_count),
            "# HELP dacn_config_source_info Hyperparameter source; value 1 means active.",
            "# TYPE dacn_config_source_info gauge",
            sample("dacn_config_source_info", 1.0, {"source": self.config.SOURCE}),
        ]

        for metric_name in self.METRICS:
            prom_name = f"dacn_monitor_{metric_name}"
            lines.extend([
                f"# HELP {prom_name} Latest {metric_name} value from MonitoringSystem.",
                f"# TYPE {prom_name} gauge",
                sample(prom_name, self._latest_metrics.get(metric_name, 0.0)),
            ])

        config_samples = {
            "underload_threshold": self.config.UNDERLOAD_THRESHOLD,
            "overload_threshold": self.config.OVERLOAD_THRESHOLD,
            "critical_overload_threshold": self.config.CRITICAL_OVERLOAD_THRESHOLD,
            "underload_prior": self.config.UNDERLOAD_PRIOR,
            "overload_prior": self.config.OVERLOAD_PRIOR,
            "critical_overload_prior": self.config.CRITICAL_OVERLOAD_PRIOR,
            "lstm_prob_threshold": self.config.LSTM_PROB_THRESHOLD,
            "anomaly_zscore_threshold": self.config.ANOMALY_ZSCORE_THRESHOLD,
            "anomaly_zscore_prior": self.config.ANOMALY_ZSCORE_PRIOR,
            "process_sample_count": self.config.PROCESS_SAMPLE_COUNT,
            "eda_prior_loaded": 1.0 if self.config.EDA_PRIOR_LOADED else 0.0,
            "eda_prior_sample_count": self.config.EDA_PRIOR_SAMPLE_COUNT,
            "eda_prior_effective_weight": self.config.EDA_PRIOR_WEIGHT,
            "process_to_prior_weight_ratio": (
                self.config.PROCESS_SAMPLE_COUNT
                / max(1.0, float(self.config.EDA_PRIOR_WEIGHT))
            ),
            "last_drift_score": self.config.LAST_DRIFT_SCORE,
            "lstm_retrain_interval_steps": self.config.LSTM_RETRAIN_INTERVAL_STEPS,
            "autoformer_retrain_interval_steps": self.config.AUTOFORMER_RETRAIN_INTERVAL_STEPS,
            "autoformer_seq_len": self.config.AF_SEQ_LEN,
            "ppo_lr": self.config.PPO_LR,
            "ppo_gamma": self.config.PPO_GAMMA,
            "ppo_gae_lambda": self.config.PPO_GAE_LAMBDA,
            "ppo_clip_epsilon": self.config.PPO_CLIP_EPSILON,
            "ppo_update_epochs": self.config.PPO_UPDATE_EPOCHS,
            "hpo_results_available": 1.0 if self.config.HPO_RESULTS_AVAILABLE else 0.0,
            "hpo_results_loaded": 1.0 if self.config.HPO_RESULTS_LOADED else 0.0,
            "hpo_promotion_ready": 1.0 if self.config.HPO_PROMOTION_READY else 0.0,
            "hpo_tuning_trials": self.config.HPO_TUNING_TRIALS,
            "hpo_last_tuning_score": self.config.HPO_LAST_TUNING_SCORE,
            "hpo_last_test_score": self.config.HPO_LAST_TEST_SCORE,
            "meta_tuning_enabled": 1.0 if self.config.META_TUNING_ENABLED else 0.0,
            "meta_last_objective": self.config.META_LAST_OBJECTIVE,
            "meta_last_safety_pressure": self.config.META_LAST_SAFETY_PRESSURE,
            "meta_last_migration_pressure": self.config.META_LAST_MIGRATION_PRESSURE,
            "reward_overload_sla_manageable_weight": self.config.REWARD_OVERLOAD_SLA_MANAGEABLE_WEIGHT,
            "reward_placer_sla_weight": self.config.REWARD_PLACER_SLA_WEIGHT,
            "reward_placer_failure_weight": self.config.REWARD_PLACER_FAILURE_WEIGHT,
            "reward_excess_migration_weight": self.config.REWARD_EXCESS_MIGRATION_WEIGHT,
            "reward_underload_active_ratio_weight": self.config.REWARD_UNDERLOAD_ACTIVE_RATIO_WEIGHT,
            "reward_placer_shutdown_weight": self.config.REWARD_PLACER_SHUTDOWN_WEIGHT,
            "robust_opt_enabled": 1.0 if self.config.ROBUST_OPT_ENABLED else 0.0,
            "robust_guard_evaluations": self.config.ROBUST_GUARD_EVALUATIONS,
            "robust_guard_filtered_candidates": self.config.ROBUST_GUARD_FILTERED_CANDIDATES,
            "robust_guard_fallbacks": self.config.ROBUST_GUARD_FALLBACKS,
            "robust_last_margin": self.config.ROBUST_LAST_MARGIN,
            "robust_last_selected_risk": self.config.ROBUST_LAST_SELECTED_RISK,
            "robust_last_selected_projected_util": self.config.ROBUST_LAST_SELECTED_PROJECTED_UTIL,
            "ora_lite_enabled": 1.0 if self.config.ORA_LITE_ENABLED else 0.0,
            "ora_lite_history_loaded": 1.0 if self.config.ORA_LITE_HISTORY_LOADED else 0.0,
            "ora_lite_history_sample_count": self.config.ORA_LITE_HISTORY_SAMPLE_COUNT,
            "ora_lite_runtime_mean_hours": self.config.ORA_LITE_LAST_RUNTIME_MEAN_HOURS,
            "ora_lite_runtime_std_hours": self.config.ORA_LITE_LAST_RUNTIME_STD_HOURS,
            "ora_lite_confidence": self.config.ORA_LITE_LAST_CONFIDENCE,
            "ora_lite_neighbors": self.config.ORA_LITE_LAST_NEIGHBORS,
            "ora_lite_resource_pressure": self.config.ORA_LITE_LAST_RESOURCE_PRESSURE,
        }
        for key, value in config_samples.items():
            prom_name = f"dacn_config_{key}"
            lines.extend([
                f"# HELP {prom_name} Process-adaptive Config.{key.upper()} value.",
                f"# TYPE {prom_name} gauge",
                sample(prom_name, value),
            ])

        return "\n".join(lines) + "\n"

    def get_summary(self):
        """Get aggregate summary of monitored metrics."""
        summary = {}
        for m in self.METRICS:
            vals = self._full_history[m]
            if vals:
                arr = np.array(vals)
                summary[m] = {
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                    "last": float(arr[-1]),
                }
        return summary

    def reset(self):
        """Reset monitoring state for a new episode."""
        self._history = {m: deque(maxlen=200) for m in self.METRICS}
        self._full_history = {m: [] for m in self.METRICS}
        self._latest_metrics = {m: 0.0 for m in self.METRICS}
        self._timestamps = []
        self._step_count = 0

    def start_http_endpoint(self, port=None):
        """Start a real Prometheus-scrapeable HTTP endpoint at /metrics.

        Serves get_prometheus_metrics() over HTTP in a daemon thread so a
        Prometheus server can scrape live config/runtime state. Returns the
        bound port. Raises OSError if the port is taken -- no silent fallback.
        """
        if port is None:
            port = getattr(self.config, "PROMETHEUS_PORT", 8000)
        monitor = self

        class _MetricsHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path.rstrip("/") in ("", "/metrics"):
                    body = monitor.get_prometheus_metrics().encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain; version=0.0.4")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.send_error(404)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("0.0.0.0", port), _MetricsHandler)
        self._http_server = server
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self._http_thread = thread
        print(f"[Monitor] Prometheus endpoint live at http://0.0.0.0:{port}/metrics")
        return port

    def stop_http_endpoint(self):
        """Shut down the HTTP endpoint if running."""
        server = getattr(self, "_http_server", None)
        if server is not None:
            server.shutdown()
            server.server_close()
            self._http_server = None
