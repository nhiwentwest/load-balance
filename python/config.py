"""
Process-adaptive hyperparameter configuration.

EDA is used only as a cold-start prior and guardrail. The running CloudSim
process feeds CPU, SLA, migration, failure, reward, and critic-loss signals into
Config.observe_* methods. Thresholds and retrain triggers then adapt with the
observed process instead of being copied from a report or a fixed grid.
"""

import csv
import json
import math
import os
from collections import deque
from pathlib import Path

import numpy as np


class Config:
    SOURCE = "PROCESS_ADAPTIVE"

    # CloudSim structural constants, mirrored from Py4jBridge.
    NUM_HOSTS = 20
    INTERVAL_SEC = 300
    MAX_TIME_SEC = 24 * 3600
    STEPS_PER_EPISODE = MAX_TIME_SEC // INTERVAL_SEC

    # Warm-up values are replaced by EDA priors at import when available, then
    # regularized by live process samples.
    UNDERLOAD_THRESHOLD = 0.0
    OVERLOAD_THRESHOLD = 1.0
    CRITICAL_OVERLOAD_THRESHOLD = 1.0
    LSTM_PROB_THRESHOLD = 0.5
    ANOMALY_ZSCORE_THRESHOLD = 3.0

    EDA_PRIOR_PATH = Path(__file__).resolve().parent.parent / "eda_data" / "eda_results.json"
    EDA_PRIOR_LOADED = False
    EDA_PRIOR_SAMPLE_COUNT = 0
    EDA_PRIOR_WEIGHT = 0.0
    UNDERLOAD_PRIOR = UNDERLOAD_THRESHOLD
    OVERLOAD_PRIOR = OVERLOAD_THRESHOLD
    CRITICAL_OVERLOAD_PRIOR = CRITICAL_OVERLOAD_THRESHOLD
    ANOMALY_ZSCORE_PRIOR = ANOMALY_ZSCORE_THRESHOLD
    CPU_MEAN_PRIOR = 0.0
    CPU_STD_PRIOR = 0.0

    LSTM_RETRAIN_INTERVAL_STEPS = 1
    AUTOFORMER_RETRAIN_INTERVAL_STEPS = 1
    LSTM_RETRAIN_INTERVAL_SIM_HOURS = 0.0
    AUTOFORMER_RETRAIN_INTERVAL_SIM_HOURS = 0.0
    ATTENTION_SIGMA_STEPS = 1

    AF_SEQ_LEN = 20
    AF_PRED_LEN = 5
    AF_D_MODEL = 32
    LSTM_HIDDEN = 32
    LSTM_LAYERS = 2
    LSTM_WINDOW = 10
    LSTM_HORIZON = 5
    LSTM_COOLDOWN_STEPS = LSTM_WINDOW + LSTM_HORIZON
    PROCESS_CALIBRATION_MIN_SAMPLES = NUM_HOSTS * LSTM_WINDOW

    PPO_LR = 3e-4
    PPO_LR_INIT = 3e-4  # initial LR for cosine schedule (FIX-2)
    PPO_GAMMA = 0.99
    PPO_GAE_LAMBDA = 0.95
    PPO_CLIP_EPSILON = 0.2
    PPO_UPDATE_EPOCHS = 4
    NUM_EPISODES = int(os.environ.get("NUM_EPISODES", "400"))

    HPO_RESULTS_PATH = Path(__file__).resolve().parent / "tuned_hyperparams.json"
    HPO_RESULTS_AVAILABLE = False
    HPO_RESULTS_LOADED = False
    HPO_PROMOTION_READY = False
    HPO_METHOD = "none"
    HPO_LAST_TUNING_SCORE = 0.0
    HPO_LAST_TEST_SCORE = 0.0
    HPO_TUNING_TRIALS = 0
    HPO_TUNING_SEEDS = ""
    HPO_TEST_SEEDS = ""

    SATURATION_ACTIVE_RATIO_THRESHOLD = 0.95
    SATURATION_OVERLOAD_RATIO_THRESHOLD = 0.50
    EXCESS_MIGRATION_START = max(1, int(round(math.sqrt(NUM_HOSTS))))

    REWARD_UNDERLOAD_SHUTDOWN_WEIGHT = 5.0
    REWARD_UNDERLOAD_ACTIVE_RATIO_WEIGHT = 3.0
    REWARD_UNDERLOAD_THRASH_WEIGHT = 0.05
    REWARD_OVERLOAD_SLA_SATURATED_WEIGHT = 5.0
    REWARD_OVERLOAD_SLA_MANAGEABLE_WEIGHT = 20.0
    REWARD_OVERLOAD_DETECTION_BONUS = 2.0
    REWARD_OVERLOAD_FALSE_POSITIVE_WEIGHT = 1.0
    REWARD_VM_SUCCESS_WEIGHT = 2.0
    REWARD_VM_MIGRATION_BASE_WEIGHT = 0.5
    REWARD_EXCESS_MIGRATION_WEIGHT = 0.2
    REWARD_PLACER_UTIL_WEIGHT = 2.0
    REWARD_PLACER_SHUTDOWN_WEIGHT = 3.0
    REWARD_PLACER_FAILURE_WEIGHT = 2.0
    REWARD_PLACER_SLA_WEIGHT = 25.0
    META_TUNING_ENABLED = os.environ.get("META_TUNING_ENABLED", "1") != "0"
    META_TUNING_GRANULARITY = os.environ.get("META_TUNING_GRANULARITY", "step").lower()
    META_LAST_OBJECTIVE = 0.0
    META_LAST_SAFETY_PRESSURE = 0.0
    META_LAST_MIGRATION_PRESSURE = 0.0
    META_LAST_CONSOLIDATION_PRESSURE = 0.0

    AF_LR = 1e-3
    LSTM_LR = 1e-3
    MONITORING_INTERVAL_STEPS = 1

    ROBUST_OPT_ENABLED = os.environ.get("ROBUST_OPT_ENABLED", "1") != "0"
    ROBUST_GUARD_EVALUATIONS = 0
    ROBUST_GUARD_FILTERED_CANDIDATES = 0
    ROBUST_GUARD_FALLBACKS = 0
    ROBUST_LAST_MARGIN = 0.0
    ROBUST_LAST_SELECTED_RISK = 0.0
    ROBUST_LAST_SELECTED_PROJECTED_UTIL = 0.0

    ORA_LITE_ENABLED = os.environ.get("ORA_LITE_ENABLED", "1") != "0"
    ORA_LITE_HISTORY_PATH = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "Gen-Parallel-Workloads"
        / "Philly"
        / "training_data"
        / "philly_data_training.csv"
    )
    ORA_LITE_HISTORY_LOADED = False
    ORA_LITE_HISTORY_SAMPLE_COUNT = 0
    ORA_LITE_LAST_RUNTIME_MEAN_HOURS = 0.0
    ORA_LITE_LAST_RUNTIME_STD_HOURS = 0.0
    ORA_LITE_LAST_CONFIDENCE = 0.0
    ORA_LITE_LAST_NEIGHBORS = 0
    ORA_LITE_LAST_RESOURCE_PRESSURE = 0.0

    RUNTIME_MEAN_PRIOR_HOURS = 0.0
    RUNTIME_MEDIAN_PRIOR_HOURS = 0.0
    RUNTIME_STD_PRIOR_HOURS = 0.0
    RUNTIME_Q90_PRIOR_HOURS = 0.0
    RUNTIME_Q95_PRIOR_HOURS = 0.0

    MODEL_DIR = "models"
    BEST_MODEL_SELECTION_METRIC = "val_loss"
    EVENT_DB_PATH = "events.db"
    ATTENTION_BASE_WEIGHT = 1.0
    ATTENTION_EVENT_MULTIPLIER = 3.0

    CPU_MEAN = 0.0
    CPU_STD = 0.0
    PROCESS_SAMPLE_COUNT = 0
    LAST_DRIFT_SCORE = 0.0
    LAST_CALIBRATION_STEP = -1

    _cpu_values = deque(maxlen=4096)
    _cpu_avg_history = deque(maxlen=512)
    _slatah_history = deque(maxlen=512)
    _migration_history = deque(maxlen=512)
    _failure_history = deque(maxlen=512)
    _drift_history = deque(maxlen=512)
    _reward_history = deque(maxlen=128)
    _critic_loss_history = deque(maxlen=128)
    _meta_objective_history = deque(maxlen=128)
    _last_retrain_step = {"bilstm": 0, "autoformer": 0}
    _retrain_events = {"bilstm": [], "autoformer": []}
    _runtime_resource_history = np.asarray([], dtype=float)
    _runtime_hours_history = np.asarray([], dtype=float)
    _reward_weight_priors = {}

    @classmethod
    def load_eda_prior(cls, path=None):
        prior_path = Path(path) if path is not None else cls.EDA_PRIOR_PATH
        try:
            with prior_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError):
            cls.EDA_PRIOR_LOADED = False
            cls.SOURCE = "PROCESS_ADAPTIVE"
            cls._sync_priors_from_current()
            return False

        derived = data.get("derived_hyperparameters", {})
        cpu_dist = data.get("cpu_utilization_analysis", {}).get("distribution", {})
        runtime_dist = data.get("runtime_analysis", {}).get("runtime_hours", {})

        cls.UNDERLOAD_PRIOR = cls._hp_value(derived, "underload_threshold", cls.UNDERLOAD_THRESHOLD)
        cls.OVERLOAD_PRIOR = cls._hp_value(derived, "overload_threshold", cls.OVERLOAD_THRESHOLD)
        cls.CRITICAL_OVERLOAD_PRIOR = cls._hp_value(
            derived, "critical_overload_threshold", cls.CRITICAL_OVERLOAD_THRESHOLD
        )
        cls.ANOMALY_ZSCORE_PRIOR = cls._hp_value(
            derived, "anomaly_zscore_threshold", cls.ANOMALY_ZSCORE_THRESHOLD
        )
        cls.AF_SEQ_LEN = int(round(cls._hp_value(derived, "autoformer_seq_len", cls.AF_SEQ_LEN)))

        lstm_hours = cls._hp_value(derived, "lstm_retrain_interval_hours", cls.LSTM_RETRAIN_INTERVAL_SIM_HOURS)
        af_hours = cls._hp_value(
            derived, "autoformer_retrain_interval_hours", cls.AUTOFORMER_RETRAIN_INTERVAL_SIM_HOURS
        )
        cls.LSTM_RETRAIN_INTERVAL_STEPS = cls._hours_to_steps(lstm_hours)
        cls.AUTOFORMER_RETRAIN_INTERVAL_STEPS = cls._hours_to_steps(af_hours)
        cls.LSTM_RETRAIN_INTERVAL_SIM_HOURS = cls.LSTM_RETRAIN_INTERVAL_STEPS * cls.INTERVAL_SEC / 3600.0
        cls.AUTOFORMER_RETRAIN_INTERVAL_SIM_HOURS = cls.AUTOFORMER_RETRAIN_INTERVAL_STEPS * cls.INTERVAL_SEC / 3600.0

        cls.CPU_MEAN_PRIOR = float(cpu_dist.get("mean", cls.CPU_MEAN_PRIOR))
        cls.CPU_STD_PRIOR = float(cpu_dist.get("std", cls.CPU_STD_PRIOR))
        cls.EDA_PRIOR_SAMPLE_COUNT = int(cpu_dist.get("total_observations", 0) or 0)
        cls.RUNTIME_MEAN_PRIOR_HOURS = float(runtime_dist.get("mean", cls.RUNTIME_MEAN_PRIOR_HOURS) or 0.0)
        cls.RUNTIME_MEDIAN_PRIOR_HOURS = float(runtime_dist.get("median", cls.RUNTIME_MEDIAN_PRIOR_HOURS) or 0.0)
        cls.RUNTIME_STD_PRIOR_HOURS = float(runtime_dist.get("std", cls.RUNTIME_STD_PRIOR_HOURS) or 0.0)
        cls.RUNTIME_Q90_PRIOR_HOURS = float(runtime_dist.get("q90", cls.RUNTIME_Q90_PRIOR_HOURS) or 0.0)
        cls.RUNTIME_Q95_PRIOR_HOURS = float(runtime_dist.get("q95", cls.RUNTIME_Q95_PRIOR_HOURS) or 0.0)
        # EDA is a cold-start prior, not a permanent majority vote. Give it
        # one model context window of effective evidence; live process samples
        # then become dominant within the same episode/deployment stream.
        cls.EDA_PRIOR_WEIGHT = float(min(
            cls.EDA_PRIOR_SAMPLE_COUNT,
            cls.PROCESS_CALIBRATION_MIN_SAMPLES,
        ))
        cls.EDA_PRIOR_LOADED = True
        cls.SOURCE = "PROCESS_ADAPTIVE_EDA_PRIOR"

        cls.UNDERLOAD_THRESHOLD = cls.UNDERLOAD_PRIOR
        cls.OVERLOAD_THRESHOLD = cls.OVERLOAD_PRIOR
        cls.CRITICAL_OVERLOAD_THRESHOLD = cls.CRITICAL_OVERLOAD_PRIOR
        cls.ANOMALY_ZSCORE_THRESHOLD = cls.ANOMALY_ZSCORE_PRIOR
        cls.ATTENTION_SIGMA_STEPS = max(1, cls.LSTM_RETRAIN_INTERVAL_STEPS // 2)
        cls._set_ora_lite_prediction(
            cls.RUNTIME_MEAN_PRIOR_HOURS,
            cls.RUNTIME_STD_PRIOR_HOURS,
            1.0 if runtime_dist else 0.0,
            0,
            0.0,
        )
        cls.load_runtime_history_prior()
        cls.load_hpo_results()
        return True

    @classmethod
    def observe_process_step(cls, cpu_values=None, info=None, global_state=None, step=None):
        vals = cls._as_cpu_array(cpu_values)
        if vals.size:
            for value in vals:
                cls._cpu_values.append(float(value))
            cls.PROCESS_SAMPLE_COUNT += int(vals.size)
            cls.CPU_MEAN = float(np.mean(cls._cpu_values))
            cls.CPU_STD = float(np.std(cls._cpu_values))
            cls._calibrate_thresholds(step)

        if global_state is not None:
            gs = np.asarray(global_state, dtype=float).reshape(-1)
            if gs.size:
                cls._bounded_append(cls._cpu_avg_history, float(np.clip(gs[0], 0.0, 1.0)))

        if info:
            cls._bounded_append(cls._slatah_history, float(info.get("slatah", 0.0)))
            cls._bounded_append(cls._migration_history, float(info.get("migrations", 0.0)))
            cls._bounded_append(cls._failure_history, float(info.get("failures", 0.0)))

        cls._update_drift_score()
        cls._update_process_intervals()

    @classmethod
    def observe_training_episode(cls, reward, critic_loss, actor_losses=None):
        cls._bounded_append(cls._reward_history, float(reward))
        cls._bounded_append(cls._critic_loss_history, float(critic_loss))
        if len(cls._reward_history) < 3:
            return

        rewards = np.asarray(cls._reward_history, dtype=float)
        losses = np.asarray(cls._critic_loss_history, dtype=float)
        recent_reward = float(np.mean(rewards[-3:]))
        previous_reward = float(np.mean(rewards[:-3])) if len(rewards) > 3 else recent_reward
        recent_loss = float(np.mean(losses[-3:]))
        stable_loss = recent_loss <= float(np.median(losses) + np.std(losses))

        # FIX-2: Heuristic LR schedule disabled.  PPO LR is now set by
        # cosine annealing in the training loop (Loshchilov & Hutter 2017).
        # The asymmetric 5%/30% rule caused LR collapse to floor after ~20 eps.
        # if recent_reward >= previous_reward and stable_loss:
        #     cls.PPO_LR = cls._bounded(cls.PPO_LR * 1.05, 1e-6, 1e-2)
        # else:
        #     cls.PPO_LR = cls._bounded(cls.PPO_LR * 0.7, 1e-6, 1e-2)
        pass  # LR managed by cosine schedule in marl_v4_train.py

        if actor_losses:
            loss_scale = float(np.mean(np.abs(actor_losses)))
            cls.PPO_CLIP_EPSILON = cls._bounded(math.sqrt(loss_scale + 1e-12), 0.05, 0.4)

    @classmethod
    def observe_step_outcome(cls, info=None, rewards=None):
        if cls.META_TUNING_GRANULARITY != "step":
            return
        cls._observe_outcome(info, rewards)

    @classmethod
    def observe_episode_outcome(cls, info=None, rewards=None):
        cls._observe_outcome(info, rewards)

    @classmethod
    def _observe_outcome(cls, info=None, rewards=None):
        if not cls.META_TUNING_ENABLED or not info:
            return
        cls._ensure_reward_weight_priors()

        reward_total = float(sum((rewards or {}).values())) if rewards else 0.0
        slatah = max(0.0, float(info.get("slatah", 0.0)))
        failures = max(0.0, float(info.get("interval_failures", info.get("failures", 0.0))))
        migrations = max(0.0, float(info.get("migrations", 0.0)))
        active_ratio = cls._bounded(float(info.get("active_ratio", 0.0)), 0.0, 1.0)

        safety_pressure = cls._constraint_pressure(
            slatah + failures / max(1.0, float(cls.NUM_HOSTS)),
            cls._slatah_history,
        )
        migration_pressure = cls._constraint_pressure(
            migrations / max(1.0, float(cls.EXCESS_MIGRATION_START)),
            cls._migration_history,
        )
        consolidation_pressure = active_ratio - float(np.median(cls._cpu_avg_history)) if cls._cpu_avg_history else 0.0
        consolidation_pressure = cls._bounded(consolidation_pressure, -1.0, 1.0)

        cls.META_LAST_OBJECTIVE = reward_total - safety_pressure - migration_pressure
        cls.META_LAST_SAFETY_PRESSURE = safety_pressure
        cls.META_LAST_MIGRATION_PRESSURE = migration_pressure
        cls.META_LAST_CONSOLIDATION_PRESSURE = consolidation_pressure
        cls._bounded_append(cls._meta_objective_history, cls.META_LAST_OBJECTIVE)

        cls._meta_update_weight("REWARD_OVERLOAD_SLA_MANAGEABLE_WEIGHT", safety_pressure)
        cls._meta_update_weight("REWARD_PLACER_SLA_WEIGHT", safety_pressure)
        cls._meta_update_weight("REWARD_PLACER_FAILURE_WEIGHT", failures / max(1.0, float(cls.NUM_HOSTS)))
        cls._meta_update_weight("REWARD_EXCESS_MIGRATION_WEIGHT", migration_pressure)
        cls._meta_update_weight("REWARD_UNDERLOAD_ACTIVE_RATIO_WEIGHT", consolidation_pressure)
        cls._meta_update_weight("REWARD_PLACER_SHUTDOWN_WEIGHT", consolidation_pressure)

    @classmethod
    def load_hpo_results(cls, path=None):
        result_path = Path(path) if path is not None else cls.HPO_RESULTS_PATH
        try:
            with result_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError):
            cls.HPO_RESULTS_AVAILABLE = False
            cls.HPO_RESULTS_LOADED = False
            cls.HPO_PROMOTION_READY = False
            return False

        cls.HPO_RESULTS_AVAILABLE = True
        cls.HPO_PROMOTION_READY = bool(data.get("promotion_ready", False))
        cls.HPO_METHOD = str(data.get("tuning_method", "autorl_hpo"))
        cls.HPO_TUNING_TRIALS = int(data.get("total_trials", 0) or 0)
        cls.HPO_TUNING_SEEDS = ",".join(str(x) for x in data.get("tuning_seeds", []))
        cls.HPO_TEST_SEEDS = ",".join(str(x) for x in data.get("test_seeds", []))
        tuning_metrics = data.get("best_tuning_metrics") or data.get("best_metrics") or {}
        test_metrics = data.get("best_test_metrics") or {}
        cls.HPO_LAST_TUNING_SCORE = float(tuning_metrics.get("score", 0.0) or 0.0)
        cls.HPO_LAST_TEST_SCORE = float(test_metrics.get("score", 0.0) or 0.0)

        if not cls.HPO_PROMOTION_READY and os.environ.get("ALLOW_UNPROMOTED_HPO", "0") != "1":
            cls.HPO_RESULTS_LOADED = False
            return False

        params = data.get("best_params") or {}
        if not isinstance(params, dict):
            cls.HPO_RESULTS_LOADED = False
            return False

        cls.apply_hpo_params(params)
        cls.HPO_RESULTS_LOADED = True
        cls.SOURCE = f"{cls.SOURCE}_HPO"
        return True

    @classmethod
    def apply_hpo_params(cls, params):
        key_map = {
            "underload_threshold": "UNDERLOAD_PRIOR",
            "overload_threshold": "OVERLOAD_PRIOR",
            "critical_threshold": "CRITICAL_OVERLOAD_PRIOR",
            "critical_overload_threshold": "CRITICAL_OVERLOAD_PRIOR",
            "anomaly_zscore": "ANOMALY_ZSCORE_PRIOR",
            "anomaly_zscore_threshold": "ANOMALY_ZSCORE_PRIOR",
            "ppo_lr": "PPO_LR",
            "ppo_learning_rate": "PPO_LR",
            "ppo_gamma": "PPO_GAMMA",
            "ppo_gae_lambda": "PPO_GAE_LAMBDA",
            "ppo_clip": "PPO_CLIP_EPSILON",
            "ppo_clip_epsilon": "PPO_CLIP_EPSILON",
            "ppo_update_epochs": "PPO_UPDATE_EPOCHS",
            "saturation_active_ratio": "SATURATION_ACTIVE_RATIO_THRESHOLD",
            "saturation_overload_ratio": "SATURATION_OVERLOAD_RATIO_THRESHOLD",
            "excess_migration_start": "EXCESS_MIGRATION_START",
            "reward_overload_sla_manageable_weight": "REWARD_OVERLOAD_SLA_MANAGEABLE_WEIGHT",
            "reward_placer_sla_weight": "REWARD_PLACER_SLA_WEIGHT",
            "reward_placer_failure_weight": "REWARD_PLACER_FAILURE_WEIGHT",
            "reward_excess_migration_weight": "REWARD_EXCESS_MIGRATION_WEIGHT",
            "reward_underload_active_ratio_weight": "REWARD_UNDERLOAD_ACTIVE_RATIO_WEIGHT",
            "reward_placer_shutdown_weight": "REWARD_PLACER_SHUTDOWN_WEIGHT",
        }
        for key, attr in key_map.items():
            if key not in params:
                continue
            value = params[key]
            if isinstance(value, dict):
                value = value.get("value")
            if value is None:
                continue
            if attr in {"PPO_UPDATE_EPOCHS", "EXCESS_MIGRATION_START"}:
                setattr(cls, attr, int(max(1, round(float(value)))))
            else:
                setattr(cls, attr, float(value))

        cls.UNDERLOAD_THRESHOLD = cls._bounded(cls.UNDERLOAD_PRIOR, 0.0, 1.0)
        cls.OVERLOAD_THRESHOLD = cls._bounded(cls.OVERLOAD_PRIOR, cls.UNDERLOAD_THRESHOLD + 1e-6, 1.0)
        cls.CRITICAL_OVERLOAD_THRESHOLD = cls._bounded(
            cls.CRITICAL_OVERLOAD_PRIOR,
            cls.OVERLOAD_THRESHOLD + 1e-6,
            1.0,
        )
        cls.ANOMALY_ZSCORE_THRESHOLD = cls._bounded(cls.ANOMALY_ZSCORE_PRIOR, 1.0, 5.0)
        cls.PPO_LR_INIT = cls.PPO_LR
        cls._ensure_reward_weight_priors(reset=True)

    @classmethod
    def reset_runtime_state(cls, reset_thresholds=True):
        cls.CPU_MEAN = 0.0
        cls.CPU_STD = 0.0
        cls.PROCESS_SAMPLE_COUNT = 0
        cls.LAST_DRIFT_SCORE = 0.0
        cls.LAST_CALIBRATION_STEP = -1
        cls.ROBUST_GUARD_EVALUATIONS = 0
        cls.ROBUST_GUARD_FILTERED_CANDIDATES = 0
        cls.ROBUST_GUARD_FALLBACKS = 0
        cls.ROBUST_LAST_MARGIN = 0.0
        cls.ROBUST_LAST_SELECTED_RISK = 0.0
        cls.ROBUST_LAST_SELECTED_PROJECTED_UTIL = 0.0
        cls.META_LAST_OBJECTIVE = 0.0
        cls.META_LAST_SAFETY_PRESSURE = 0.0
        cls.META_LAST_MIGRATION_PRESSURE = 0.0
        cls.META_LAST_CONSOLIDATION_PRESSURE = 0.0

        for history in (
            cls._cpu_values,
            cls._cpu_avg_history,
            cls._slatah_history,
            cls._migration_history,
            cls._failure_history,
            cls._drift_history,
            cls._reward_history,
            cls._critic_loss_history,
            cls._meta_objective_history,
        ):
            history.clear()

        cls._last_retrain_step = {"bilstm": 0, "autoformer": 0}
        cls._retrain_events = {"bilstm": [], "autoformer": []}
        if reset_thresholds:
            cls.UNDERLOAD_THRESHOLD = cls.UNDERLOAD_PRIOR
            cls.OVERLOAD_THRESHOLD = cls.OVERLOAD_PRIOR
            cls.CRITICAL_OVERLOAD_THRESHOLD = cls.CRITICAL_OVERLOAD_PRIOR
            cls.ANOMALY_ZSCORE_THRESHOLD = cls.ANOMALY_ZSCORE_PRIOR

    @classmethod
    def robust_margin(cls, vm_cpu_ratio=0.0, runtime_uncertainty_hours=None):
        host_windows = max(1.0, cls.PROCESS_SAMPLE_COUNT / max(1, cls.NUM_HOSTS))
        process_std = cls.CPU_STD if cls.PROCESS_SAMPLE_COUNT >= cls.NUM_HOSTS else cls.CPU_STD_PRIOR
        if cls.EDA_PRIOR_LOADED:
            std = cls._blend_with_prior(process_std, cls.CPU_STD_PRIOR)
        else:
            std = process_std
        drift_pressure = cls.LAST_DRIFT_SCORE / max(cls.ANOMALY_ZSCORE_THRESHOLD, 1e-9)
        margin = cls.ANOMALY_ZSCORE_THRESHOLD * std / math.sqrt(host_windows)
        margin *= 1.0 + min(1.0, max(0.0, drift_pressure))
        margin += 0.10 * max(0.0, float(vm_cpu_ratio))
        margin += cls._runtime_uncertainty_margin(runtime_uncertainty_hours)
        limit = max(0.01, cls.CRITICAL_OVERLOAD_THRESHOLD - cls.UNDERLOAD_THRESHOLD)
        return cls._bounded(margin, 0.0, min(0.25, limit))

    @classmethod
    def robust_candidate_risk(cls, candidate_util, vm_cpu_ratio=0.0, runtime_uncertainty_hours=None):
        projected = cls._bounded(float(candidate_util) + max(0.0, float(vm_cpu_ratio)), 0.0, 1.5)
        margin = cls.robust_margin(vm_cpu_ratio, runtime_uncertainty_hours)
        return projected + margin - cls.OVERLOAD_THRESHOLD, projected, margin

    @classmethod
    def load_runtime_history_prior(cls, path=None):
        history_path = Path(path) if path is not None else cls.ORA_LITE_HISTORY_PATH
        if not cls.ORA_LITE_ENABLED:
            cls.ORA_LITE_HISTORY_LOADED = False
            return False
        try:
            with history_path.open("r", encoding="utf-8", newline="") as fh:
                rows = list(csv.DictReader(fh))
        except OSError:
            cls.ORA_LITE_HISTORY_LOADED = False
            return False

        parsed = []
        for row in rows:
            try:
                gpu = float(row.get("gpu_num", 0.0) or 0.0)
                node = float(row.get("node_num", 0.0) or 0.0)
                runtime_hours = float(row.get("run_time", 0.0) or 0.0) / 3600.0
            except ValueError:
                continue
            if runtime_hours <= 0.0:
                continue
            parsed.append((gpu, node, runtime_hours))

        if not parsed:
            cls.ORA_LITE_HISTORY_LOADED = False
            return False

        max_gpu = max(max(row[0] for row in parsed), 1.0)
        max_node = max(max(row[1] for row in parsed), 1.0)
        resource = []
        runtime = []
        for gpu, node, runtime_hours in parsed:
            gpu_pressure = gpu / max_gpu
            node_pressure = node / max_node
            resource.append(cls._bounded(max(gpu_pressure, node_pressure), 0.0, 1.0))
            runtime.append(runtime_hours)

        cls._runtime_resource_history = np.asarray(resource, dtype=float)
        cls._runtime_hours_history = np.asarray(runtime, dtype=float)
        cls.ORA_LITE_HISTORY_SAMPLE_COUNT = int(cls._runtime_hours_history.size)
        cls.ORA_LITE_HISTORY_LOADED = cls.ORA_LITE_HISTORY_SAMPLE_COUNT > 0

        if cls.ORA_LITE_HISTORY_LOADED and cls.RUNTIME_MEAN_PRIOR_HOURS <= 0.0:
            cls.RUNTIME_MEAN_PRIOR_HOURS = float(np.mean(cls._runtime_hours_history))
            cls.RUNTIME_MEDIAN_PRIOR_HOURS = float(np.median(cls._runtime_hours_history))
            cls.RUNTIME_STD_PRIOR_HOURS = float(np.std(cls._runtime_hours_history))
            cls.RUNTIME_Q90_PRIOR_HOURS = float(np.quantile(cls._runtime_hours_history, 0.90))
            cls.RUNTIME_Q95_PRIOR_HOURS = float(np.quantile(cls._runtime_hours_history, 0.95))
        return cls.ORA_LITE_HISTORY_LOADED

    @classmethod
    def predict_runtime_uncertainty(cls, features=None):
        if not cls.ORA_LITE_ENABLED:
            cls._set_ora_lite_prediction(0.0, 0.0, 0.0, 0, 0.0)
            return 0.0, 0.0, 0.0

        resource_pressure = cls._runtime_resource_pressure(features)
        if cls.ORA_LITE_HISTORY_LOADED and cls._runtime_hours_history.size:
            distances = np.abs(cls._runtime_resource_history - resource_pressure)
            neighbor_count = max(1, int(round(math.sqrt(float(distances.size)))))
            neighbor_count = min(neighbor_count, int(distances.size))
            if neighbor_count < distances.size:
                idx = np.argpartition(distances, neighbor_count - 1)[:neighbor_count]
            else:
                idx = np.arange(distances.size)
            samples = cls._runtime_hours_history[idx]
            mean_hours = float(np.mean(samples))
            std_hours = float(np.std(samples))
            scale = max(float(np.std(cls._runtime_resource_history)), 1e-9)
            distance_penalty = float(np.mean(distances[idx])) / scale
            confidence = 1.0 / (1.0 + max(0.0, distance_penalty))
            cls._set_ora_lite_prediction(mean_hours, std_hours, confidence, neighbor_count, resource_pressure)
            return mean_hours, std_hours, confidence

        confidence = 1.0 if cls.RUNTIME_MEAN_PRIOR_HOURS > 0.0 else 0.0
        cls._set_ora_lite_prediction(
            cls.RUNTIME_MEAN_PRIOR_HOURS,
            cls.RUNTIME_STD_PRIOR_HOURS,
            confidence,
            0,
            resource_pressure,
        )
        return cls.RUNTIME_MEAN_PRIOR_HOURS, cls.RUNTIME_STD_PRIOR_HOURS, confidence

    @classmethod
    def observe_robust_guard(cls, base_mask, robust_mask, selected_risk, selected_projected_util, margin):
        cls.ROBUST_GUARD_EVALUATIONS += 1
        base = np.asarray(base_mask, dtype=bool)
        robust = np.asarray(robust_mask, dtype=bool)
        if base.size == robust.size:
            cls.ROBUST_GUARD_FILTERED_CANDIDATES += int(np.sum(base & ~robust))
            if bool(base.any()) and not bool(robust.any()):
                cls.ROBUST_GUARD_FALLBACKS += 1
        cls.ROBUST_LAST_SELECTED_RISK = float(selected_risk)
        cls.ROBUST_LAST_SELECTED_PROJECTED_UTIL = float(selected_projected_util)
        cls.ROBUST_LAST_MARGIN = float(margin)

    @classmethod
    def should_retrain(cls, model_type, current_step):
        if cls.PROCESS_SAMPLE_COUNT < cls.NUM_HOSTS:
            return False
        last = cls._last_retrain_step.get(model_type, 0)
        elapsed = current_step - last
        min_gap = cls._min_gap_steps(model_type)
        if elapsed < min_gap:
            return False

        recent_calibration = (
            cls.LAST_CALIBRATION_STEP >= 0
            and current_step - cls.LAST_CALIBRATION_STEP <= min_gap
        )
        drift_trigger = cls._robust_high(cls._drift_history, cls.LAST_DRIFT_SCORE)
        sla_trigger = bool(cls._slatah_history) and cls._robust_high(cls._slatah_history, cls._slatah_history[-1])
        failure_trigger = bool(cls._failure_history) and cls._robust_high(cls._failure_history, cls._failure_history[-1])
        migration_trigger = (
            model_type == "autoformer"
            and bool(cls._migration_history)
            and cls._robust_high(cls._migration_history, cls._migration_history[-1])
        )
        return recent_calibration or drift_trigger or sla_trigger or failure_trigger or migration_trigger

    @classmethod
    def mark_retrain(cls, model_type, current_step):
        last = cls._last_retrain_step.get(model_type, 0)
        elapsed = max(1, current_step - last)
        cls._last_retrain_step[model_type] = current_step
        cls._retrain_events.setdefault(model_type, []).append(current_step)
        if model_type == "bilstm":
            cls.LSTM_RETRAIN_INTERVAL_STEPS = elapsed
        else:
            cls.AUTOFORMER_RETRAIN_INTERVAL_STEPS = elapsed
        cls._update_process_intervals()

    @classmethod
    def _calibrate_thresholds(cls, step=None):
        values = np.asarray(cls._cpu_values, dtype=float)
        if values.size < cls.PROCESS_CALIBRATION_MIN_SAMPLES:
            return
        raw_ul, raw_ol, raw_critical = cls._otsu_three_regimes(values)
        ul, ol, critical = cls._regularize_thresholds(raw_ul, raw_ol, raw_critical)
        changed = (
            abs(ul - cls.UNDERLOAD_THRESHOLD) > 1e-6
            or abs(ol - cls.OVERLOAD_THRESHOLD) > 1e-6
            or abs(critical - cls.CRITICAL_OVERLOAD_THRESHOLD) > 1e-6
        )
        cls.UNDERLOAD_THRESHOLD = ul
        cls.OVERLOAD_THRESHOLD = ol
        cls.CRITICAL_OVERLOAD_THRESHOLD = critical
        if changed and step is not None:
            cls.LAST_CALIBRATION_STEP = int(step)

    @classmethod
    def _otsu_three_regimes(cls, values):
        vals = np.sort(cls._as_cpu_array(values))
        if vals.size < 3 or float(vals[-1] - vals[0]) < 1e-12:
            mean = float(np.mean(vals)) if vals.size else 0.0
            std = float(np.std(vals)) if vals.size else 0.0
            low = cls._bounded(mean - std, 0.0, 1.0)
            high = cls._bounded(mean + std, low, 1.0)
            return low, high, cls._bounded(high + std, high, 1.0)

        if vals.size > 1200:
            idx = np.linspace(0, vals.size - 1, 1200).round().astype(int)
            vals = vals[idx]

        n = vals.size
        prefix = np.concatenate(([0.0], np.cumsum(vals)))
        total_mean = float(prefix[-1] / n)
        best_score = -float("inf")
        best_i = max(1, n // 3)
        best_j = max(best_i + 1, (2 * n) // 3)

        for i in range(1, n - 1):
            n0 = i
            mean0 = prefix[i] / n0
            js = np.arange(i + 1, n)
            n1 = js - i
            n2 = n - js
            mean1 = (prefix[js] - prefix[i]) / n1
            mean2 = (prefix[n] - prefix[js]) / n2
            scores = (
                n0 * (mean0 - total_mean) ** 2
                + n1 * (mean1 - total_mean) ** 2
                + n2 * (mean2 - total_mean) ** 2
            )
            local_idx = int(np.argmax(scores))
            local_score = float(scores[local_idx])
            if local_score > best_score:
                best_score = local_score
                best_i = i
                best_j = int(js[local_idx])

        underload = float((vals[best_i - 1] + vals[best_i]) / 2.0)
        overload = float((vals[best_j - 1] + vals[best_j]) / 2.0)
        # The robust guard represents the critical upper tail, not the center
        # of Otsu's high-utilization regime. Keeping the same empirical Q95
        # definition used by the EDA prior prevents normal regime shifts from
        # turning the safety guard into a migration trigger.
        critical = cls._bounded(float(np.percentile(vals, 95)), overload, 1.0)
        if underload >= overload:
            mean = float(np.mean(vals))
            std = float(np.std(vals))
            underload = cls._bounded(mean - std, 0.0, 1.0)
            overload = cls._bounded(mean + std, underload, 1.0)
            critical = cls._bounded(overload + std, overload, 1.0)
        return underload, overload, critical

    @classmethod
    def _update_drift_score(cls):
        values = np.asarray(cls._cpu_values, dtype=float)
        if values.size < cls.NUM_HOSTS * 2:
            return
        window = max(cls.NUM_HOSTS, int(math.sqrt(values.size)))
        if values.size < window * 2:
            return
        prev = values[-2 * window:-window]
        recent = values[-window:]
        scale = max(float(np.std(values)), 1e-9)
        drift = abs(float(np.mean(recent) - np.mean(prev))) / scale
        drift += abs(float(np.std(recent) - np.std(prev))) / scale
        cls.LAST_DRIFT_SCORE = float(drift)
        cls._bounded_append(cls._drift_history, cls.LAST_DRIFT_SCORE)
        process_limit = max(1.0, cls._robust_limit(cls._drift_history))
        if cls.EDA_PRIOR_LOADED:
            cls.ANOMALY_ZSCORE_THRESHOLD = cls._bounded(
                cls._blend_with_prior(process_limit, cls.ANOMALY_ZSCORE_PRIOR),
                1.0,
                5.0,
            )
        else:
            cls.ANOMALY_ZSCORE_THRESHOLD = process_limit

    @classmethod
    def _update_process_intervals(cls):
        for model_type, attr in (("bilstm", "LSTM_RETRAIN_INTERVAL_STEPS"), ("autoformer", "AUTOFORMER_RETRAIN_INTERVAL_STEPS")):
            min_gap = cls._min_gap_steps(model_type)
            events = cls._retrain_events.get(model_type, [])
            if len(events) >= 2:
                interval = int(max(min_gap, round(float(np.median(np.diff(events))))))
            else:
                interval = max(getattr(cls, attr), min_gap)
            setattr(cls, attr, int(interval))
        cls.LSTM_RETRAIN_INTERVAL_SIM_HOURS = cls.LSTM_RETRAIN_INTERVAL_STEPS * cls.INTERVAL_SEC / 3600.0
        cls.AUTOFORMER_RETRAIN_INTERVAL_SIM_HOURS = cls.AUTOFORMER_RETRAIN_INTERVAL_STEPS * cls.INTERVAL_SEC / 3600.0
        cls.ATTENTION_SIGMA_STEPS = max(1, cls.LSTM_RETRAIN_INTERVAL_STEPS // 2)
        cls.MONITORING_INTERVAL_STEPS = max(1, int(math.sqrt(max(1, cls.PROCESS_SAMPLE_COUNT))))

    @classmethod
    def _min_gap_steps(cls, model_type="bilstm"):
        host_windows = max(1, cls.PROCESS_SAMPLE_COUNT // max(1, cls.NUM_HOSTS))
        process_floor = max(1, int(math.sqrt(host_windows)))
        if not cls.EDA_PRIOR_LOADED:
            return process_floor
        if model_type == "autoformer":
            prior_gap = max(1, int(cls.AUTOFORMER_RETRAIN_INTERVAL_STEPS))
        else:
            prior_gap = max(1, int(cls.LSTM_RETRAIN_INTERVAL_STEPS))
        drift_pressure = cls.LAST_DRIFT_SCORE / max(cls.ANOMALY_ZSCORE_THRESHOLD, 1e-9)
        adaptive_gap = int(round(prior_gap / (1.0 + max(0.0, drift_pressure))))
        return max(process_floor, adaptive_gap)

    @classmethod
    def _regularize_thresholds(cls, underload, overload, critical):
        if not cls.EDA_PRIOR_LOADED:
            return underload, overload, critical
        ul = cls._blend_with_prior(underload, cls.UNDERLOAD_PRIOR)
        ol = cls._blend_with_prior(overload, cls.OVERLOAD_PRIOR)
        crit = cls._blend_with_prior(critical, cls.CRITICAL_OVERLOAD_PRIOR)
        ol = max(ol, ul + 1e-6)
        crit = max(crit, ol + 1e-6)
        return (
            cls._bounded(ul, 0.0, 1.0),
            cls._bounded(ol, 0.0, 1.0),
            cls._bounded(crit, 0.0, 1.0),
        )

    @classmethod
    def _blend_with_prior(cls, process_value, prior_value):
        if not cls.EDA_PRIOR_LOADED or cls.EDA_PRIOR_WEIGHT <= 0:
            return float(process_value)
        drift_pressure = cls.LAST_DRIFT_SCORE / max(cls.ANOMALY_ZSCORE_THRESHOLD, 1e-9)
        prior_weight = cls.EDA_PRIOR_WEIGHT / (1.0 + max(0.0, drift_pressure))
        process_weight = max(1.0, float(cls.PROCESS_SAMPLE_COUNT))
        return float((process_weight * process_value + prior_weight * prior_value) / (process_weight + prior_weight))

    @classmethod
    def _sync_priors_from_current(cls):
        cls.UNDERLOAD_PRIOR = cls.UNDERLOAD_THRESHOLD
        cls.OVERLOAD_PRIOR = cls.OVERLOAD_THRESHOLD
        cls.CRITICAL_OVERLOAD_PRIOR = cls.CRITICAL_OVERLOAD_THRESHOLD
        cls.ANOMALY_ZSCORE_PRIOR = cls.ANOMALY_ZSCORE_THRESHOLD

    @classmethod
    def _hp_value(cls, derived, key, default):
        value = derived.get(key, default)
        if isinstance(value, dict):
            value = value.get("value", default)
        return float(value)

    @classmethod
    def _hours_to_steps(cls, hours):
        return max(1, int(round(float(hours) * 3600.0 / cls.INTERVAL_SEC)))

    @classmethod
    def _robust_limit(cls, history):
        arr = np.asarray(history, dtype=float)
        if arr.size == 0:
            return 1.0
        center = float(np.median(arr))
        mad = float(np.median(np.abs(arr - center)))
        if mad <= 1e-12:
            return float(np.mean(arr) + np.std(arr))
        return center + 3.0 * 1.4826 * mad

    @classmethod
    def _runtime_uncertainty_margin(cls, runtime_uncertainty_hours=None):
        if not cls.ORA_LITE_ENABLED:
            return 0.0
        if runtime_uncertainty_hours is None:
            runtime_uncertainty_hours = cls.ORA_LITE_LAST_RUNTIME_STD_HOURS
        std_hours = max(0.0, float(runtime_uncertainty_hours or 0.0))
        if std_hours <= 0.0:
            return 0.0
        runtime_scale = max(
            cls.RUNTIME_Q95_PRIOR_HOURS,
            cls.RUNTIME_MEAN_PRIOR_HOURS,
            cls.RUNTIME_MEDIAN_PRIOR_HOURS,
            1e-9,
        )
        uncertainty_pressure = cls._bounded(std_hours / runtime_scale, 0.0, 1.0)
        safety_band = max(0.0, cls.CRITICAL_OVERLOAD_THRESHOLD - cls.OVERLOAD_THRESHOLD)
        return safety_band * uncertainty_pressure

    @classmethod
    def _runtime_resource_pressure(cls, features=None):
        if features is None:
            return 0.0
        if isinstance(features, dict):
            values = [
                features.get("vm_cpu_ratio", 0.0),
                features.get("vm_ram_ratio", 0.0),
                features.get("vm_gpu_ratio", 0.0),
                features.get("source_util", 0.0),
                features.get("source_gpu_util", 0.0),
                features.get("active_ratio", 0.0),
            ]
        else:
            values = np.asarray(features, dtype=float).reshape(-1)[:4]
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return 0.0
        return cls._bounded(float(np.max(arr)), 0.0, 1.0)

    @classmethod
    def _set_ora_lite_prediction(cls, mean_hours, std_hours, confidence, neighbors, resource_pressure):
        cls.ORA_LITE_LAST_RUNTIME_MEAN_HOURS = max(0.0, float(mean_hours or 0.0))
        cls.ORA_LITE_LAST_RUNTIME_STD_HOURS = max(0.0, float(std_hours or 0.0))
        cls.ORA_LITE_LAST_CONFIDENCE = cls._bounded(float(confidence or 0.0), 0.0, 1.0)
        cls.ORA_LITE_LAST_NEIGHBORS = int(max(0, neighbors or 0))
        cls.ORA_LITE_LAST_RESOURCE_PRESSURE = cls._bounded(float(resource_pressure or 0.0), 0.0, 1.0)

    @classmethod
    def _reward_weight_names(cls):
        return (
            "REWARD_UNDERLOAD_SHUTDOWN_WEIGHT",
            "REWARD_UNDERLOAD_ACTIVE_RATIO_WEIGHT",
            "REWARD_UNDERLOAD_THRASH_WEIGHT",
            "REWARD_OVERLOAD_SLA_SATURATED_WEIGHT",
            "REWARD_OVERLOAD_SLA_MANAGEABLE_WEIGHT",
            "REWARD_OVERLOAD_DETECTION_BONUS",
            "REWARD_OVERLOAD_FALSE_POSITIVE_WEIGHT",
            "REWARD_VM_SUCCESS_WEIGHT",
            "REWARD_VM_MIGRATION_BASE_WEIGHT",
            "REWARD_EXCESS_MIGRATION_WEIGHT",
            "REWARD_PLACER_UTIL_WEIGHT",
            "REWARD_PLACER_SHUTDOWN_WEIGHT",
            "REWARD_PLACER_FAILURE_WEIGHT",
            "REWARD_PLACER_SLA_WEIGHT",
        )

    @classmethod
    def _ensure_reward_weight_priors(cls, reset=False):
        if reset or not cls._reward_weight_priors:
            cls._reward_weight_priors = {
                name: max(abs(float(getattr(cls, name))), 1e-9)
                for name in cls._reward_weight_names()
            }

    @classmethod
    def _meta_update_weight(cls, attr, pressure):
        """FIX-3: EMA-based self-tuning with tighter bounds.

        Changes from original:
        - EMA smoothing (alpha=0.01) instead of exp(step_size * pressure)
          to prevent bang-bang oscillation between bounds.
        - Tighter bounds: [prior/2, prior*2] instead of [prior/sqrt(N), prior*sqrt(N)]
          to keep reward scale stable during training.
        - step_size floor prevents wild swings when sample count is low.
        Ref: Meta SAC-Lag (2024) for principled self-tuning of constraint weights.
        """
        if attr not in cls._reward_weight_names():
            return
        cls._ensure_reward_weight_priors()
        current = max(abs(float(getattr(cls, attr))), 1e-9)
        prior = cls._reward_weight_priors.get(attr, current)
        bounded_pressure = cls._bounded(float(pressure), -1.0, 1.0)
        # EMA toward pressure-adjusted target (max ±50% deviation from prior)
        ema_alpha = 0.01
        target = prior * (1.0 + 0.5 * bounded_pressure)
        candidate = (1.0 - ema_alpha) * current + ema_alpha * target
        # Tighter bounds: [prior/2, prior*2]
        setattr(cls, attr, cls._bounded(candidate, prior / 2.0, prior * 2.0))

    @classmethod
    def _constraint_pressure(cls, value, history):
        value = max(0.0, float(value))
        if len(history) < 5:
            return cls._bounded(value, 0.0, 1.0)
        limit = max(1e-9, cls._robust_limit(history))
        return cls._bounded((value - limit) / limit, -1.0, 1.0)

    @classmethod
    def _robust_high(cls, history, current):
        if len(history) < 5:
            return False
        return float(current) > cls._robust_limit(history)

    @classmethod
    def _as_cpu_array(cls, values):
        if values is None:
            return np.asarray([], dtype=float)
        arr = np.asarray(values, dtype=float).reshape(-1)
        arr = arr[np.isfinite(arr)]
        return np.clip(arr, 0.0, 1.0)

    @classmethod
    def _bounded(cls, value, low, high):
        return float(min(max(value, low), high))

    @classmethod
    def _bounded_append(cls, target, value):
        target.append(float(value))

    @classmethod
    def summary(cls):
        print(f"\n{'=' * 70}")
        print(f"CONFIG [{cls.SOURCE}]")
        print(f"{'=' * 70}")
        if cls.EDA_PRIOR_LOADED:
            print(
                "  Source: live CloudSim process regularized by EDA cold-start "
                f"prior (raw samples={cls.EDA_PRIOR_SAMPLE_COUNT}, "
                f"effective weight={cls.EDA_PRIOR_WEIGHT:.0f})"
            )
        else:
            print("  Source: live CloudSim process, no EDA prior loaded")
        print(f"  Samples={cls.PROCESS_SAMPLE_COUNT}, drift={cls.LAST_DRIFT_SCORE:.4f}")
        print(f"  Thresholds: UL={cls.UNDERLOAD_THRESHOLD:.4f}, OL={cls.OVERLOAD_THRESHOLD:.4f}, CRIT={cls.CRITICAL_OVERLOAD_THRESHOLD:.4f}, LSTM_p={cls.LSTM_PROB_THRESHOLD:.4f}")
        print(f"  EDA priors: UL={cls.UNDERLOAD_PRIOR:.4f}, OL={cls.OVERLOAD_PRIOR:.4f}, CRIT={cls.CRITICAL_OVERLOAD_PRIOR:.4f}, z={cls.ANOMALY_ZSCORE_PRIOR:.4f}")
        print(f"  Retrain trigger: process drift/SLA/failure; current gaps BiLSTM={cls.LSTM_RETRAIN_INTERVAL_STEPS} steps, Autoformer={cls.AUTOFORMER_RETRAIN_INTERVAL_STEPS} steps")
        print(
            f"  PPO algorithmic schedule: lr={cls.PPO_LR:.6g}, "
            f"self-tuned clip={cls.PPO_CLIP_EPSILON:.4f}, episodes={cls.NUM_EPISODES}"
        )
        print(f"  HPO: available={cls.HPO_RESULTS_AVAILABLE}, loaded={cls.HPO_RESULTS_LOADED}, promoted={cls.HPO_PROMOTION_READY}, method={cls.HPO_METHOD}, trials={cls.HPO_TUNING_TRIALS}, tuning_score={cls.HPO_LAST_TUNING_SCORE:.4f}, test_score={cls.HPO_LAST_TEST_SCORE:.4f}")
        print(f"  Meta tuning: enabled={cls.META_TUNING_ENABLED}, objective={cls.META_LAST_OBJECTIVE:.4f}, safety={cls.META_LAST_SAFETY_PRESSURE:.4f}, migration={cls.META_LAST_MIGRATION_PRESSURE:.4f}")
        print(f"  Monitor: adaptive z={cls.ANOMALY_ZSCORE_THRESHOLD:.4f}")
        print(f"  Robust guard: enabled={cls.ROBUST_OPT_ENABLED}, evals={cls.ROBUST_GUARD_EVALUATIONS}, filtered={cls.ROBUST_GUARD_FILTERED_CANDIDATES}, fallbacks={cls.ROBUST_GUARD_FALLBACKS}, margin={cls.ROBUST_LAST_MARGIN:.4f}")
        print(f"  ORA-lite: enabled={cls.ORA_LITE_ENABLED}, history={cls.ORA_LITE_HISTORY_LOADED}, samples={cls.ORA_LITE_HISTORY_SAMPLE_COUNT}, runtime_std={cls.ORA_LITE_LAST_RUNTIME_STD_HOURS:.4f}h, confidence={cls.ORA_LITE_LAST_CONFIDENCE:.4f}")
        print(f"  Reward weights: OL_SLA={cls.REWARD_OVERLOAD_SLA_MANAGEABLE_WEIGHT:.4f}, A4_SLA={cls.REWARD_PLACER_SLA_WEIGHT:.4f}, A4_FAIL={cls.REWARD_PLACER_FAILURE_WEIGHT:.4f}, MIG_EXCESS={cls.REWARD_EXCESS_MIGRATION_WEIGHT:.4f}")
        print(f"{'=' * 70}\n")


Config.load_eda_prior()
