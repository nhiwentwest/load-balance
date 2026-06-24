"""
Online Scheduler — Speed Layer of Lambda Architecture.

Inference-only: uses pre-trained MARL agents to make real-time decisions.
Does NOT perform gradient updates — that's the Offline Trainer's job.

Responsibilities:
  1. Load best models from Model Registry
  2. Run 4-agent inference pipeline each step
  3. Log overload events (missed detections) to EventDatabase
  4. Hot-swap to new models when Offline Trainer produces better ones
  5. Feed metrics to Monitoring System
"""

import numpy as np
import torch
from config import Config
from autoformer_detector import AutoformerDetector, detect_pm_state
from lstm_underload_detector import LSTMUnderloadDetector
from models import select_action


class OnlineScheduler:
    """
    Real-time scheduler — inference path of the Lambda architecture.

    Replaces the monolithic training loop's inference section
    with a clean, separated decision-making module.
    """

    def __init__(self, env, agents, central_critic, autoformer,
                 event_db=None, model_registry=None, monitoring=None):
        """
        Args:
            env: CloudSimEnv instance
            agents: dict with keys 'agent1', 'agent2', 'agent3', 'agent4'
            central_critic: CentralizedCritic network (for value estimation)
            autoformer: AutoformerDetector for proactive detection
            event_db: EventDatabase for logging
            model_registry: ModelRegistry for hot-swap
            monitoring: MonitoringSystem for metrics
        """
        self.env = env
        self.agents = agents
        self.central_critic = central_critic
        self.autoformer = autoformer
        self.event_db = event_db
        self.registry = model_registry
        self.monitoring = monitoring

        # Set all agents to eval mode
        for agent in self.agents.values():
            agent.eval()
        if self.central_critic:
            self.central_critic.eval()
        if self.autoformer:
            self.autoformer.eval()

        # BiLSTM underload detector
        bilstm_path = None
        if self.registry and self.registry.has_model("bilstm"):
            bilstm_path = self.registry.get_best_model_path("bilstm")
        if bilstm_path is None:
            bilstm_path = "lstm_underload.pt"
        self.lstm_detector = LSTMUnderloadDetector(
            model_path=bilstm_path,
            num_hosts=env.num_hosts,
            window=Config.LSTM_WINDOW,
            threshold=Config.LSTM_PROB_THRESHOLD,
            cooldown_steps=Config.LSTM_COOLDOWN_STEPS,
        )

        # Track active model versions for hot-swap detection
        self._active_versions = {}
        if self.registry:
            for mt in ["autoformer", "bilstm", "agent1", "agent2", "agent3", "agent4"]:
                self._active_versions[mt] = self.registry.get_active_version(mt)

        # CPU history buffer for overload detection
        self._host_cpu_history = {}
        self._last_robust_candidate_mask = None

    def step(self, global_state, step_num, sim_time):
        """
        Execute one scheduling decision step.

        Args:
            global_state: np.array [global_dim]
            step_num: int — current step
            sim_time: float — current simulation time

        Returns:
            underload_indices, overload_indices, selection_action, placement_actions
        """
        agent1 = self.agents["agent1"]
        agent2 = self.agents["agent2"]
        agent3 = self.agents["agent3"]
        agent4 = self.agents["agent4"]

        env = self.env

        # ===== Build Autoformer predictions =====
        autoformer_preds = np.zeros(env.num_hosts, dtype=np.float32)
        if self.autoformer is not None:
            history = env.get_host_history()
            for h_idx in range(env.num_hosts):
                host_hist = list(history[h_idx])
                self._update_host_history(h_idx, history[h_idx, -1])
                if len(host_hist) >= Config.AF_SEQ_LEN:
                    seq = np.array(host_hist[-Config.AF_SEQ_LEN:], dtype=np.float32)
                    # AutoformerDetector expects [Batch, seq_len] and adds the
                    # channel dim internally; it returns [Batch, pred_len].
                    inp = torch.FloatTensor(seq).unsqueeze(0)
                    with torch.no_grad():
                        pred = self.autoformer(inp)
                    autoformer_preds[h_idx] = pred[0, -1].item()
                else:
                    # Not enough history yet -- naive persistence is the defined
                    # behaviour here, not an error fallback.
                    autoformer_preds[h_idx] = history[h_idx, -1]
        else:
            history = env.get_host_history()
            autoformer_preds = history[:, -1].copy()
            for h_idx in range(env.num_hosts):
                self._update_host_history(h_idx, history[h_idx, -1])

        # Build detector observations
        det_obs = env.get_detector_obs(autoformer_preds)
        current_utils = np.array([det_obs[h * 2] for h in range(env.num_hosts)])
        Config.observe_process_step(cpu_values=current_utils, step=step_num)

        # ===== LEVEL 1: Detection (Agent 1 + Agent 2) =====
        ul_mask = env.get_detector_masks('underload')
        ol_mask = env.get_detector_masks('overload')

        # Agent 1: Underload Detection
        with torch.no_grad():
            a1_action, _ = select_action(agent1, det_obs, ul_mask, mode="eval")

        # Use LSTM for proactive underload detection
        predicted_underloads = None
        if self.lstm_detector is not None:
            self.lstm_detector.threshold = Config.LSTM_PROB_THRESHOLD
            self.lstm_detector.update(current_utils)
            predicted_underloads, _ = self.lstm_detector.detect_with_probs()

        # Agent 2: Overload Detection
        with torch.no_grad():
            a2_action, _ = select_action(agent2, det_obs, ol_mask, mode="eval")

        underload_indices, overload_indices, detector_context = (
            env.resolve_detector_actions(
                det_obs,
                a1_action,
                a2_action,
                predicted_underloads=predicted_underloads,
            )
        )
        if self.lstm_detector is not None and underload_indices:
            self.lstm_detector.cooldown.mark_shutdown(underload_indices[0])

        if self.event_db and detector_context["a2_true_positive"]:
            self.event_db.log_overload_detected(
                sim_time=sim_time,
                step=step_num,
                host_id=a2_action,
                cpu_util=float(current_utils[a2_action]),
            )
        if self.event_db:
            policy_sources = {a2_action} if detector_context["a2_true_positive"] else set()
            for h_idx in overload_indices:
                if h_idx not in policy_sources:
                    self.event_db.log_overload_missed(
                        sim_time=sim_time,
                        step=step_num,
                        host_id=h_idx,
                        cpu_util=float(current_utils[h_idx]),
                        predicted_state="ROBUST_GUARD",
                        severity=float(current_utils[h_idx] - Config.OVERLOAD_THRESHOLD),
                    )

        # ===== LEVEL 2: Tactical Decisions (Agent 3 + Agent 4) =====
        sel_obs = env.build_selector_obs(
            global_state, overload_indices, det_obs,
            underload_indices=underload_indices,
        )
        sel_mask = np.ones(env.num_sel_actions, dtype=bool)
        with torch.no_grad():
            selection_action, _ = select_action(agent3, sel_obs, sel_mask, mode="eval")

        # Agent 4: VM Placement
        placement_actions = []
        migration_sources = list(overload_indices) + list(underload_indices)
        if migration_sources:
            vm_states = env.get_migration_placer_obs(
                overload_indices, underload_indices, selection_action
            )
            for vs in vm_states:
                place_mask = env.get_placer_mask(vs)
                action_mask = self._robust_placement_mask(vs, place_mask)
                with torch.no_grad():
                    a4_act, _ = select_action(agent4, vs, action_mask, mode="eval")
                self._record_robust_placement(vs, place_mask, action_mask, a4_act)
                placement_actions.append(a4_act)

        return (
            underload_indices,
            overload_indices,
            selection_action,
            placement_actions,
            det_obs,
            detector_context,
        )

    def check_model_update(self):
        """Check if Model Registry has new best models → hot-swap."""
        if not self.registry:
            return False

        swapped = False
        for mt in ["agent1", "agent2", "agent3", "agent4"]:
            current_ver = self._active_versions.get(mt)
            latest_ver = self.registry.get_active_version(mt)
            if latest_ver and latest_ver != current_ver:
                # Hot-swap
                state_dict = self.registry.load_best_model(mt)
                if state_dict is not None:
                    try:
                        self.agents[mt].load_state_dict(state_dict)
                    except RuntimeError as exc:
                        print(f"[Scheduler] Skip incompatible {mt} model; retrain required. {exc}")
                        continue
                    self.agents[mt].eval()
                    self._active_versions[mt] = latest_ver
                    print(f"[Scheduler] Hot-swapped {mt}: v{current_ver} → v{latest_ver}")
                    swapped = True

        # Check autoformer
        af_ver = self.registry.get_active_version("autoformer")
        if af_ver and af_ver != self._active_versions.get("autoformer"):
            state_dict = self.registry.load_best_model("autoformer")
            if state_dict is not None and self.autoformer is not None:
                self.autoformer.load_state_dict(state_dict)
                self.autoformer.eval()
                self._active_versions["autoformer"] = af_ver
                print(f"[Scheduler] Hot-swapped autoformer → v{af_ver}")
                swapped = True

        return swapped

    def _resolve_conflicts(self, underload, overload):
        """Ensure no host is in both underload and overload lists."""
        conflict = set(underload) & set(overload)
        if conflict:
            underload = [h for h in underload if h not in conflict]
        return underload, overload

    def _get_host_history(self, host_id):
        """Get CPU utilization history for a host."""
        return self._host_cpu_history.get(host_id, [])

    def _update_host_history(self, host_id, cpu_util):
        """Update CPU history buffer (keep last 40 values for Autoformer)."""
        if host_id not in self._host_cpu_history:
            self._host_cpu_history[host_id] = []
        self._host_cpu_history[host_id].append(float(cpu_util))
        # Keep only last 40
        if len(self._host_cpu_history[host_id]) > 40:
            self._host_cpu_history[host_id] = self._host_cpu_history[host_id][-40:]

    def _robust_placement_mask(self, vm_state, base_mask):
        if not Config.ROBUST_OPT_ENABLED:
            return base_mask
        vm_state = np.asarray(vm_state, dtype=np.float32)
        base_mask = np.asarray(base_mask, dtype=bool)
        top_k = self.env.top_k
        vm_cpu_ratio = self._vm_cpu_ratio(vm_state)
        _, runtime_std_hours, _ = Config.predict_runtime_uncertainty(
            self._ora_lite_features(vm_state)
        )

        robust_mask = base_mask.copy()
        for idx in range(min(top_k, len(vm_state), len(base_mask))):
            if not base_mask[idx] or vm_state[idx] < 0:
                robust_mask[idx] = False
                continue
            risk, _, _ = Config.robust_candidate_risk(
                vm_state[idx],
                vm_cpu_ratio,
                runtime_std_hours,
            )
            robust_mask[idx] = risk <= 0.0

        self._last_robust_candidate_mask = robust_mask.copy()
        return robust_mask if robust_mask.any() else base_mask

    def _record_robust_placement(self, vm_state, base_mask, action_mask, action_idx):
        if not Config.ROBUST_OPT_ENABLED:
            return
        top_k = self.env.top_k
        vm_cpu_ratio = self._vm_cpu_ratio(vm_state)
        if action_idx < 0 or action_idx >= min(top_k, len(vm_state)):
            return
        _, runtime_std_hours, _ = Config.predict_runtime_uncertainty(
            self._ora_lite_features(vm_state)
        )
        risk, projected, margin = Config.robust_candidate_risk(
            vm_state[action_idx],
            vm_cpu_ratio,
            runtime_std_hours,
        )
        robust_mask = self._last_robust_candidate_mask
        if robust_mask is None or len(robust_mask) != len(action_mask):
            robust_mask = action_mask
        Config.observe_robust_guard(base_mask, robust_mask, risk, projected, margin)

    def _vm_cpu_ratio(self, vm_state):
        idx = self._a4_extra_base(vm_state)
        if idx < len(vm_state) and np.isfinite(vm_state[idx]):
            return float(max(0.0, vm_state[idx]))
        return 0.0

    def _ora_lite_features(self, vm_state):
        top_k = self.env.top_k
        vm_state = np.asarray(vm_state, dtype=np.float32)
        base = self._a4_extra_base(vm_state)
        values = [self._vm_state_value(vm_state, base + offset) for offset in range(8)]
        if self._has_gpu_aware_placer_state(vm_state):
            return {
                "vm_cpu_ratio": values[0],
                "vm_ram_ratio": values[1],
                "vm_gpu_ratio": values[2],
                "source_util": values[3],
                "source_gpu_util": values[4],
                "active_ratio": values[5],
            }
        return {
            "vm_cpu_ratio": values[0],
            "vm_ram_ratio": values[1],
            "source_util": values[2],
            "active_ratio": values[3],
        }

    def _has_gpu_aware_placer_state(self, vm_state):
        return len(vm_state) >= self.env.top_k * 3 + 8

    def _a4_extra_base(self, vm_state):
        if self._has_gpu_aware_placer_state(vm_state):
            return self.env.top_k * 3
        return self.env.top_k * 2

    def _vm_state_value(self, vm_state, idx):
        if idx < len(vm_state) and np.isfinite(vm_state[idx]):
            return float(max(0.0, vm_state[idx]))
        return 0.0

    def reset(self):
        """Reset scheduler state for a new episode."""
        self._host_cpu_history = {}
        if self.lstm_detector is not None:
            self.lstm_detector.reset()
