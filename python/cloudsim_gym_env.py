"""
CloudSim Plus Gymnasium Environment Wrapper for 4-Agent Hierarchical MARL V4.
Wraps Py4jBridge as a multi-agent Gymnasium environment.

Architecture:
  Level 1 (Strategic): Agent 1 (Underload Detector) + Agent 2 (Overload Detector)
  Level 2 (Tactical):  Agent 3 (VM Selector) + Agent 4 (VM Placer)
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from py4j.java_gateway import JavaGateway
from config import Config


# SLA Tier weights based on VM RAM
def get_vm_tier_weight(ram_mb):
    if ram_mb >= 4096:
        return 3.0  # Gold
    elif ram_mb >= 2048:
        return 1.5  # Silver
    else:
        return 0.5  # Bronze


class CloudSimEnv:
    """
    Multi-agent environment wrapping Py4jBridge.
    Not a standard Gym env — instead exposes per-agent observation/action spaces
    and a hierarchical step() interface.
    """
    def __init__(self, gateway=None):
        if gateway is None:
            import os
            from py4j.java_gateway import GatewayParameters
            port = int(os.environ.get("BRIDGE_PORT", "25333"))
            self.gateway = JavaGateway(gateway_parameters=GatewayParameters(port=port))
        else:
            self.gateway = gateway
        self.bridge = self.gateway.entry_point

        # Dimensions from Java bridge
        try:
            self.bridge.setHistoryLen(int(Config.AF_SEQ_LEN))
        except Exception:
            pass
        self.global_dim = self.bridge.getGlobalStateDim()
        self.vm_dim = self.bridge.getVmStateDim()
        self.top_k = self.bridge.getTopK()
        self.num_hosts = self.bridge.getNumHosts()
        self.num_sel_actions = self.bridge.getNumSelectionActions()
        try:
            self.history_len = int(self.bridge.getHistoryLen())
        except Exception:
            self.history_len = 20

        # ===== Agent Observation/Action Spaces =====

        # Detector actions: one host action plus an explicit NO_OP. Physical
        # feasibility belongs in the mask; adaptive thresholds remain labels.
        self.detector_noop_action = self.num_hosts

        # Agent 1 (Underload Detector): Observe per-host util + Autoformer pred
        self.a1_obs_dim = self.num_hosts * 2  # [current_util, predicted_util] per host
        self.a1_action_n = self.num_hosts + 1

        # Agent 2 (Overload Detector): Same obs structure
        self.a2_obs_dim = self.num_hosts * 2
        self.a2_action_n = self.num_hosts + 1

        # Agent 3 (VM Selector): For each triggered host, pick which VM
        # Obs: host util + VM features. Action: VM index within host
        self.a3_obs_dim = self.global_dim + 8  # global + host-specific features
        self.a3_max_vms = 10  # max VMs per host to consider

        # Agent 4 (VM Placer): Given a VM, pick target host
        self.a4_obs_dim = self.vm_dim
        self.a4_action_n = self.top_k

        # Runtime state
        self.global_state = None
        self.host_history = None
        self.autoformer_preds = None  # Populated by Autoformer
        self.done = False
        self.step_count = 0
        self.total_reward = 0

        # Per-episode metrics
        self.ep_energy = 0
        self.ep_migrations = 0
        self.ep_sla_cost = 0
        self.ep_overloads = 0
        self.ep_actionable_overloads = 0
        self.ep_selected_overload_sources = 0
        self.ep_guard_overload_sources = 0
        self.ep_unactionable_overloads = 0
        self.ep_gpu_blocked_overloads = 0
        self.ep_capacity_blocked_overloads = 0
        self.ep_single_vm_overloads = 0
        self.ep_capacity_overloads = 0
        self.ep_critical_overloads = 0
        self.ep_underloads = 0
        self.ep_a2_true_positives = 0
        self.ep_a2_false_positives = 0
        self.ep_a2_false_negatives = 0
        self.ep_failures = 0
        self.ep_no_candidates = 0
        self.ep_same_host_skips = 0

    def reset(self):
        """Reset the CloudSim simulation."""
        self._sync_adaptive_thresholds()
        raw = np.array(self.bridge.reset())
        self.global_state = raw[:self.global_dim]
        self.done = False
        self.step_count = 0
        self.total_reward = 0
        self.ep_energy = 0
        self.ep_migrations = 0
        self.ep_sla_cost = 0
        self.ep_overloads = 0
        self.ep_actionable_overloads = 0
        self.ep_selected_overload_sources = 0
        self.ep_guard_overload_sources = 0
        self.ep_unactionable_overloads = 0
        self.ep_gpu_blocked_overloads = 0
        self.ep_capacity_blocked_overloads = 0
        self.ep_single_vm_overloads = 0
        self.ep_capacity_overloads = 0
        self.ep_critical_overloads = 0
        self.ep_underloads = 0
        self.ep_a2_true_positives = 0
        self.ep_a2_false_positives = 0
        self.ep_a2_false_negatives = 0
        self.ep_failures = 0
        self.ep_no_candidates = 0
        self.ep_same_host_skips = 0
        self._prev_slatah = 0.0
        self._sync_adaptive_thresholds()
        return self.global_state

    def _sync_adaptive_thresholds(self):
        process_mean = (
            Config.CPU_MEAN
            if Config.PROCESS_SAMPLE_COUNT >= Config.NUM_HOSTS
            else Config.CPU_MEAN_PRIOR
        )
        try:
            self.bridge.setProcessProfile(
                float(Config.OVERLOAD_THRESHOLD),
                float(Config.CRITICAL_OVERLOAD_THRESHOLD),
                float(process_mean),
            )
        except Exception:
            try:
                self.bridge.setAdaptiveThresholds(
                    float(Config.OVERLOAD_THRESHOLD),
                    float(Config.CRITICAL_OVERLOAD_THRESHOLD),
                )
            except Exception:
                pass

    def get_host_history(self):
        """Get host CPU history matrix [num_hosts, history_len]."""
        return np.array(self.bridge.getHostHistory())

    def get_host_vm_counts(self):
        """Get current VM count per host from the Java bridge."""
        try:
            return np.array(self.bridge.getHostVmCounts(), dtype=np.int32)
        except Exception:
            return np.ones(self.num_hosts, dtype=np.int32)

    def get_movable_host_mask(self):
        """Hosts that have at least one VM with a feasible migration target."""
        try:
            return np.array(self.bridge.getMovableHostMask(), dtype=np.int32) > 0
        except Exception:
            return self.get_host_vm_counts() > 1

    def get_host_mobility_reason_codes(self):
        """0=movable, 1=single/empty, 2=CPU/RAM/BW blocked, 3=GPU blocked."""
        try:
            return np.array(self.bridge.getHostMobilityReasonCodes(), dtype=np.int32)
        except Exception:
            counts = self.get_host_vm_counts()
            return np.where(counts > 1, 0, 1).astype(np.int32)

    def get_detector_obs(self, autoformer_preds):
        """
        Build observation for Agent 1 & 2 (Detectors).
        Shape: [num_hosts * 2] = [current_util_0, pred_util_0, ..., current_util_N, pred_util_N]
        """
        history = self.get_host_history()
        current_utils = history[:, -1]  # Last column = current utilization
        obs = np.zeros(self.num_hosts * 2, dtype=np.float32)
        for i in range(self.num_hosts):
            obs[i * 2] = current_utils[i]
            obs[i * 2 + 1] = autoformer_preds[i] if i < len(autoformer_preds) else 0.0
        return obs

    def get_detector_masks(self, mode='underload', ensure_action=True):
        """
        Physical action mask for detector policies.

        Thresholds are deliberately excluded: they are adaptive labels the
        detector must learn, not rules that pre-select the detector's action.
        NO_OP is always valid, so a numerical fallback is never needed.
        """
        history = self.get_host_history()
        current_utils = history[:, -1]
        vm_counts = self.get_host_vm_counts()
        movable_hosts = self.get_movable_host_mask()
        mask = np.zeros(self.num_hosts + 1, dtype=bool)
        for i in range(self.num_hosts):
            if mode == 'underload':
                mask[i] = vm_counts[i] > 0
            else:  # overload
                mask[i] = movable_hosts[i]
        mask[self.detector_noop_action] = True
        return mask

    def resolve_detector_actions(
        self,
        det_obs,
        underload_action,
        overload_action,
        predicted_underloads=None,
    ):
        """Resolve detector decisions once for train, serving, eval, and HPO."""
        current_utils = np.array(
            [det_obs[h * 2] for h in range(self.num_hosts)],
            dtype=np.float32,
        )
        vm_counts = self.get_host_vm_counts()
        movable_hosts = self.get_movable_host_mask()

        underload_labels = (
            (vm_counts > 0)
            & (current_utils > 0.001)
            & (current_utils < Config.UNDERLOAD_THRESHOLD)
        )
        if predicted_underloads is not None:
            for host_idx in predicted_underloads:
                if 0 <= int(host_idx) < self.num_hosts and vm_counts[int(host_idx)] > 0:
                    underload_labels[int(host_idx)] = True

        overload_labels = movable_hosts & (current_utils > Config.OVERLOAD_THRESHOLD)
        critical_labels = movable_hosts & (current_utils > Config.CRITICAL_OVERLOAD_THRESHOLD)

        a1_host_action = 0 <= int(underload_action) < self.num_hosts
        a2_host_action = 0 <= int(overload_action) < self.num_hosts
        a1_true_positive = bool(a1_host_action and underload_labels[int(underload_action)])
        a2_true_positive = bool(a2_host_action and overload_labels[int(overload_action)])

        underload_indices = [int(underload_action)] if a1_true_positive else []
        policy_overload_indices = [int(overload_action)] if a2_true_positive else []
        guard_overload_indices = [
            int(host_idx)
            for host_idx in np.flatnonzero(critical_labels)
            if int(host_idx) not in policy_overload_indices
        ]
        overload_indices = policy_overload_indices + guard_overload_indices

        conflict = set(underload_indices) & set(overload_indices)
        if conflict:
            underload_indices = [h for h in underload_indices if h not in conflict]

        context = {
            "underload_action": int(underload_action),
            "overload_action": int(overload_action),
            "a1_true_positive": a1_true_positive,
            "a1_false_positive": bool(a1_host_action and not a1_true_positive),
            "a1_false_negative": bool(underload_labels.any() and not a1_true_positive),
            "a2_true_positive": a2_true_positive,
            "a2_false_positive": bool(a2_host_action and not a2_true_positive),
            "a2_false_negative": bool(overload_labels.any() and not a2_true_positive),
            "eligible_underloads": int(np.sum(underload_labels)),
            "eligible_overloads": int(np.sum(overload_labels)),
            "eligible_critical_overloads": int(np.sum(critical_labels)),
            "policy_overload_sources": len(policy_overload_indices),
            "guard_overload_sources": len(guard_overload_indices),
        }
        return underload_indices, overload_indices, context

    def build_selector_obs(self, global_state, overload_indices, det_obs,
                           underload_indices=None):
        """Build Agent 3 (VM Selector) observation, filling ALL 8 feature slots.

        Single source of truth used by training, serving and HPO so the policy
        sees an identical observation in every mode. Previously the training
        loop populated only 3 of the 8 reserved slots, leaving 5 dims always
        zero -- wasted capacity and a train/serve mismatch risk.

        Layout (after the global_dim block):
          +0 flagged-overload count ratio
          +1 mean current util of flagged hosts
          +2 max  current util of flagged hosts
          +3 std  current util of flagged hosts
          +4 mean 5-step rolling average over flagged hosts
          +5 mean 5-step rolling std over flagged hosts
          +6 underload count ratio (tactical context)
          +7 active-host ratio (global pressure)
        """
        obs = np.zeros(self.a3_obs_dim, dtype=np.float32)
        obs[:self.global_dim] = global_state
        g = self.global_dim
        if overload_indices:
            history = self.get_host_history()
            cur = np.array(
                [det_obs[h * 2] for h in overload_indices], dtype=np.float32
            )
            roll_avgs = np.array(
                [np.mean(history[h, -5:]) for h in overload_indices], dtype=np.float32
            )
            roll_stds = np.array(
                [np.std(history[h, -5:]) for h in overload_indices], dtype=np.float32
            )
            obs[g + 0] = len(overload_indices) / self.num_hosts
            obs[g + 1] = float(np.mean(cur))
            obs[g + 2] = float(np.max(cur))
            obs[g + 3] = float(np.std(cur)) if len(cur) > 1 else 0.0
            obs[g + 4] = float(np.mean(roll_avgs))
            obs[g + 5] = float(np.mean(roll_stds))
        if underload_indices:
            obs[g + 6] = len(underload_indices) / self.num_hosts
        try:
            obs[g + 7] = float(self.get_movable_host_mask().mean())
        except Exception:
            obs[g + 7] = 0.0
        return obs

    def get_placer_obs(self, overloaded_indices, selection_action):
        """Get VM state observations from Java bridge for Agent 4."""
        java_overloaded = self._to_java_int_array(overloaded_indices)
        preview = np.array(self.bridge.previewVmStates(java_overloaded, selection_action))
        return self._decode_vm_state_preview(preview)

    def get_migration_placer_obs(self, overloaded_indices, underloaded_indices, selection_action):
        """Get Agent 4 VM states for every VM that Java step() will migrate."""
        java_overloaded = self._to_java_int_array(overloaded_indices)
        java_underloaded = self._to_java_int_array(underloaded_indices)
        try:
            preview = np.array(self.bridge.previewMigrationVmStates(
                java_overloaded, java_underloaded, selection_action
            ))
        except Exception:
            # Backward-compatible fallback for older bridges.
            combined = list(overloaded_indices) + list(underloaded_indices)
            java_sources = self._to_java_int_array(combined)
            preview = np.array(self.bridge.previewVmStates(java_sources, selection_action))
        return self._decode_vm_state_preview(preview)

    def _decode_vm_state_preview(self, preview):
        num_vms = int(preview[0])
        vm_states = []
        for i in range(num_vms):
            start = 1 + i * self.vm_dim
            end = start + self.vm_dim
            if end <= len(preview):
                vm_states.append(preview[start:end].astype(np.float32))
        return vm_states

    def get_placer_mask(self, vm_state):
        """
        Action mask for Agent 4 (VM Placer).
        Masks out:
        - Candidates that don't exist (util = -1)
        - Candidates already near adaptive overload threshold to prevent cascade
        - GPU-infeasible candidates when the Java bridge exposes GPU features
        """
        mask = np.zeros(self.top_k, dtype=bool)
        vm_gpu_idx = self.top_k * 3 + 2
        vm_uses_gpu = (
            self._has_gpu_aware_placer_state(vm_state)
            and vm_gpu_idx < len(vm_state)
            and vm_state[vm_gpu_idx] > 0.0
        )
        gpu_headroom_candidates = []
        if vm_uses_gpu:
            for i in range(self.top_k):
                gpu_idx = self._candidate_gpu_free_idx(vm_state, i)
                if (
                    vm_state[i] >= 0
                    and gpu_idx is not None
                    and vm_state[gpu_idx] > 0.0
                ):
                    gpu_headroom_candidates.append(i)
        for i in range(self.top_k):
            if vm_state[i] < -0.5:  # Invalid candidate (no host)
                continue
            gpu_idx = self._candidate_gpu_free_idx(vm_state, i)
            if gpu_idx is not None and vm_state[gpu_idx] < -0.5:
                continue
            if gpu_headroom_candidates and i not in gpu_headroom_candidates:
                continue
            if vm_state[i] > Config.OVERLOAD_THRESHOLD:  # Host already near overload — skip
                continue
            mask[i] = True
        # Fallback: if all masked out, allow the least loaded valid host
        if not mask.any():
            best = -1
            best_util = 2.0
            for i in range(self.top_k):
                if vm_state[i] >= 0 and vm_state[i] < best_util:
                    best_util = vm_state[i]
                    best = i
            if best >= 0:
                mask[best] = True
            else:
                mask[0] = True
        return mask

    def _has_gpu_aware_placer_state(self, vm_state):
        return len(vm_state) >= self.top_k * 3 + 8

    def _candidate_gpu_free_idx(self, vm_state, candidate_idx):
        if not self._has_gpu_aware_placer_state(vm_state):
            return None
        idx = self.top_k * 2 + candidate_idx
        return idx if idx < len(vm_state) else None

    def step(
        self,
        underload_indices,
        overload_indices,
        selection_action,
        placement_actions,
        detector_context=None,
    ):
        """
        Execute one step in CloudSim Plus.
        
        Args:
            underload_indices: list of host indices marked as underloaded (from Agent 1)
            overload_indices: list of host indices marked as overloaded (from Agent 2)
            selection_action: int, VM selection heuristic (0=MMT, 1=MaxCPU, 2=Random)
            placement_actions: list of int, target host index per migrated VM
            detector_context: output from resolve_detector_actions()
        
        Returns:
            next_global_state, rewards_dict, done, info
        """
        java_overloaded = self._to_java_int_array(overload_indices)
        java_underloaded = self._to_java_int_array(underload_indices)
        java_placements = self._to_java_int_array(placement_actions)

        self._sync_adaptive_thresholds()
        result = np.array(self.bridge.step(java_overloaded, java_underloaded,
                                            selection_action, java_placements))

        next_global = result[:self.global_dim]
        raw_reward = result[self.global_dim]
        done_flag = result[self.global_dim + 1]
        interval_mig = int(result[self.global_dim + 2])
        step_num = int(result[self.global_dim + 3])

        self.done = done_flag > 0.5
        self.global_state = next_global
        self.step_count += 1
        self.ep_migrations += interval_mig

        # Get SLAV metrics
        slav_metrics = np.array(self.bridge.getSlavMetrics())
        slatah, pdm, slav, energy_kwh, tot_mig, tot_steps = slav_metrics
        delta_energy = next_global[3] if self.global_dim > 3 else 0

        # Extract from Java: SLA cost, attempted, failures, and consolidation metrics
        interval_sla_cost = result[self.global_dim + 4] if len(result) > self.global_dim + 4 else 0.0
        normalized_interval_sla_cost = interval_sla_cost / max(1.0, float(Config.INTERVAL_SEC))
        java_attempted = int(result[self.global_dim + 5]) if len(result) > self.global_dim + 5 else 0
        java_failed = int(result[self.global_dim + 6]) if len(result) > self.global_dim + 6 else 0
        avg_target_util = result[self.global_dim + 7] if len(result) > self.global_dim + 7 else 0.0
        active_host_delta = result[self.global_dim + 8] if len(result) > self.global_dim + 8 else 0.0
        active_ratio = result[self.global_dim + 9] if len(result) > self.global_dim + 9 else 1.0
        java_no_candidates = int(result[self.global_dim + 10]) if len(result) > self.global_dim + 10 else 0
        java_same_host_skips = int(result[self.global_dim + 11]) if len(result) > self.global_dim + 11 else 0
        java_blocked = java_no_candidates + java_same_host_skips
        try:
            raw_demands = np.array(self.bridge.getHostRawDemandRatios(), dtype=np.float32)
        except Exception:
            raw_demands = self.get_host_history()[:, -1]

        detector_context = detector_context or {
            "a1_true_positive": bool(underload_indices),
            "a1_false_positive": False,
            "a1_false_negative": False,
            "a2_true_positive": bool(overload_indices),
            "a2_false_positive": False,
            "a2_false_negative": False,
            "eligible_overloads": len(overload_indices),
            "policy_overload_sources": len(overload_indices),
            "guard_overload_sources": 0,
        }

        # ===== Per-Agent Rewards — CAUSAL CHAIN DECOMPOSITION (V6) =====
        # Each agent only receives reward for signals its action directly causes.
        # No cross-agent credit leakage. Ref: PRD-MAPPO (RLC 2024)
        n_overloads = int(np.sum(raw_demands > Config.OVERLOAD_THRESHOLD))
        n_critical_overloads = int(np.sum(raw_demands > Config.CRITICAL_OVERLOAD_THRESHOLD))
        n_capacity_overloads = int(np.sum(raw_demands > 1.0))
        movable_after_step = self.get_movable_host_mask()
        mobility_reasons = self.get_host_mobility_reason_codes()
        overload_mask = raw_demands > Config.OVERLOAD_THRESHOLD
        actionable_mask = overload_mask & movable_after_step
        unactionable_mask = overload_mask & ~movable_after_step
        n_actionable_overloads = int(np.sum(actionable_mask))
        n_unactionable_overloads = int(np.sum(unactionable_mask))
        n_gpu_blocked_overloads = int(np.sum(unactionable_mask & (mobility_reasons == 3)))
        n_capacity_blocked_overloads = int(np.sum(unactionable_mask & (mobility_reasons == 2)))
        n_single_vm_overloads = int(np.sum(unactionable_mask & (mobility_reasons == 1)))
        n_selected_overload_sources = int(detector_context.get("policy_overload_sources", 0))
        n_guard_overload_sources = int(detector_context.get("guard_overload_sources", 0))
        n_underloads = len(underload_indices)
        self.ep_overloads += n_overloads
        self.ep_actionable_overloads += n_actionable_overloads
        self.ep_selected_overload_sources += n_selected_overload_sources
        self.ep_guard_overload_sources += n_guard_overload_sources
        self.ep_unactionable_overloads += n_unactionable_overloads
        self.ep_gpu_blocked_overloads += n_gpu_blocked_overloads
        self.ep_capacity_blocked_overloads += n_capacity_blocked_overloads
        self.ep_single_vm_overloads += n_single_vm_overloads
        self.ep_capacity_overloads += n_capacity_overloads
        self.ep_critical_overloads += n_critical_overloads
        self.ep_underloads += n_underloads
        self.ep_a2_true_positives += int(detector_context.get("a2_true_positive", False))
        self.ep_a2_false_positives += int(detector_context.get("a2_false_positive", False))
        self.ep_a2_false_negatives += int(detector_context.get("a2_false_negative", False))

        # Track SLATAH delta for Agent 2
        slatah_delta = slatah - self._prev_slatah if hasattr(self, '_prev_slatah') else 0.0
        self._prev_slatah = slatah

        # Agent 1: decision quality plus downstream consolidation it caused.
        r1 = 0.0
        if detector_context.get("a1_true_positive", False):
            r1 += Config.REWARD_UNDERLOAD_THRASH_WEIGHT
        if detector_context.get("a1_false_positive", False):
            r1 -= Config.REWARD_UNDERLOAD_THRASH_WEIGHT
        if detector_context.get("a1_false_negative", False):
            r1 -= Config.REWARD_UNDERLOAD_THRASH_WEIGHT
        if n_underloads > 0:
            if active_host_delta < 0:
                r1 += Config.REWARD_UNDERLOAD_SHUTDOWN_WEIGHT * abs(active_host_delta)
            r1 += (1.0 - active_ratio) * Config.REWARD_UNDERLOAD_ACTIVE_RATIO_WEIGHT

        # Agent 2: explicit detector decision quality plus SLA consequence when
        # an actionable overload existed at decision time.
        # Ref: MAPPO-LCE (NeurIPS 2025), EcoFair-CH-MARL (arXiv:2603.14625)
        # Key insight: When datacenter is capacity-saturated (all hosts active +
        # massive overloads), SLA violations are PHYSICALLY INEVITABLE -- not
        # Agent 2 fault. Penalizing detection in this regime produces
        # incorrect gradients that suppress future detection sensitivity.
        is_saturated = (
            active_ratio > Config.SATURATION_ACTIVE_RATIO_THRESHOLD
            and n_critical_overloads > self.num_hosts * Config.SATURATION_OVERLOAD_RATIO_THRESHOLD
        )

        # Detection-quality signal is the PRIMARY learning signal for Agent 2.
        r2 = 0.0
        if detector_context.get("a2_true_positive", False):
            r2 += Config.REWARD_OVERLOAD_DETECTION_BONUS
        if detector_context.get("a2_false_positive", False):
            r2 -= Config.REWARD_OVERLOAD_FALSE_POSITIVE_WEIGHT
        if detector_context.get("a2_false_negative", False):
            r2 -= Config.REWARD_OVERLOAD_DETECTION_BONUS
        # SLA consequence is charged to Agent 2 ONLY when it actually missed an
        # actionable overload (false negative). If A2 correctly detected and
        # acted (true positive), a subsequent SLATAH rise is a placement issue
        # (A3/A4), not a detection failure -- charging A2 here produced the
        # incorrect gradients that previously froze its FP/FN rates. Saturated
        # regimes (physically inevitable violations) use a reduced weight.
        if detector_context.get("a2_false_negative", False) and slatah_delta > 0:
            if is_saturated:
                r2 -= Config.REWARD_OVERLOAD_SLA_SATURATED_WEIGHT * slatah_delta
            else:
                r2 -= Config.REWARD_OVERLOAD_SLA_MANAGEABLE_WEIGHT * slatah_delta

        # Agent 3 (VM Selector): ONLY migration success signals
        # Causal: A3 picks which VM → affects migration success/failure rate
        r3 = 0.0
        if overload_indices:
            if java_attempted > 0:
                success_rate = 1.0 - (java_failed / max(1, java_attempted))
                r3 += Config.REWARD_VM_SUCCESS_WEIGHT * success_rate
            if interval_mig > 0:
                r3 += Config.REWARD_VM_MIGRATION_BASE_WEIGHT
            if java_blocked > 0:
                r3 -= Config.REWARD_PLACER_FAILURE_WEIGHT * java_blocked
            r3 -= Config.REWARD_EXCESS_MIGRATION_WEIGHT * max(
                0,
                interval_mig - Config.EXCESS_MIGRATION_START,
            )
        # REMOVED: interval_sla_cost — that's placement-related (A4), not selection (A3)

        # Agent 4 (VM Placer): ONLY placement quality signals
        # Causal: A4 picks target host → affects consolidation + SLA at target
        r4_consol = 0.0
        if interval_mig > 0 and avg_target_util > 0:
            r4_consol = Config.REWARD_PLACER_UTIL_WEIGHT * avg_target_util * interval_mig
        r4_shutdown = 0.0
        if active_host_delta < 0:
            r4_shutdown = Config.REWARD_PLACER_SHUTDOWN_WEIGHT * abs(active_host_delta)
        r4_fail = -Config.REWARD_PLACER_FAILURE_WEIGHT * (java_failed + java_blocked)
        # Convert migration SLA cost from weighted downtime seconds into a
        # dimensionless fraction of the scheduling interval before applying
        # the adaptive weight. Raw seconds otherwise dominate every reward.
        r4_sla = -normalized_interval_sla_cost * Config.REWARD_PLACER_SLA_WEIGHT
        # REMOVED: delta_energy — global metric, not per-placement
        
        r4 = r4_consol + r4_shutdown + r4_fail + r4_sla
        self.ep_failures += java_failed
        self.ep_no_candidates += java_no_candidates
        self.ep_same_host_skips += java_same_host_skips
        
        # Store R4 breakdown for logging
        self._r4_breakdown = {
            'sla': r4_sla, 'consol': r4_consol,
            'shutdown': r4_shutdown, 'fail': r4_fail,
            'avg_tgt_util': avg_target_util, 'active_delta': active_host_delta,
            'active_ratio': active_ratio
        }

        rewards = {'underload_det': r1, 'overload_det': r2, 
                   'vm_selector': r3, 'vm_placer': r4}

        info = {
            'step': step_num,
            'slatah': slatah,
            'pdm': pdm,
            'slav': slav,
            'energy_kwh': energy_kwh,
            'migrations': interval_mig,
            'total_migrations': int(tot_mig),
            'overloads': n_overloads,
            'critical_overloads': n_critical_overloads,
            'capacity_overloads': n_capacity_overloads,
            'actionable_overloads': n_actionable_overloads,
            'selected_overload_sources': n_selected_overload_sources,
            'guard_overload_sources': n_guard_overload_sources,
            'unactionable_overloads': n_unactionable_overloads,
            'gpu_blocked_overloads': n_gpu_blocked_overloads,
            'capacity_blocked_overloads': n_capacity_blocked_overloads,
            'single_vm_overloads': n_single_vm_overloads,
            'episode_overloads': self.ep_overloads,
            'episode_critical_overloads': self.ep_critical_overloads,
            'episode_capacity_overloads': self.ep_capacity_overloads,
            'episode_actionable_overloads': self.ep_actionable_overloads,
            'episode_selected_overload_sources': self.ep_selected_overload_sources,
            'episode_guard_overload_sources': self.ep_guard_overload_sources,
            'episode_unactionable_overloads': self.ep_unactionable_overloads,
            'episode_gpu_blocked_overloads': self.ep_gpu_blocked_overloads,
            'episode_capacity_blocked_overloads': self.ep_capacity_blocked_overloads,
            'episode_single_vm_overloads': self.ep_single_vm_overloads,
            'a2_true_positive': int(detector_context.get("a2_true_positive", False)),
            'a2_false_positive': int(detector_context.get("a2_false_positive", False)),
            'a2_false_negative': int(detector_context.get("a2_false_negative", False)),
            'episode_a2_true_positives': self.ep_a2_true_positives,
            'episode_a2_false_positives': self.ep_a2_false_positives,
            'episode_a2_false_negatives': self.ep_a2_false_negatives,
            'underloads': n_underloads,
            'failures': self.ep_failures,
            'interval_failures': java_failed,
            'no_candidates': self.ep_no_candidates,
            'interval_no_candidates': java_no_candidates,
            'same_host_skips': self.ep_same_host_skips,
            'interval_same_host_skips': java_same_host_skips,
            'placement_blocked': self.ep_no_candidates + self.ep_same_host_skips,
            'interval_placement_blocked': java_blocked,
            'active_ratio': active_ratio,
            'active_host_delta': active_host_delta,
            'avg_target_util': avg_target_util,
            'interval_sla_cost': interval_sla_cost,
            'normalized_interval_sla_cost': normalized_interval_sla_cost,
        }

        self.total_reward += sum(rewards.values())
        Config.observe_step_outcome(info, rewards)

        return next_global, rewards, self.done, info

    def _to_java_int_array(self, py_list):
        arr = self.gateway.new_array(self.gateway.jvm.int, len(py_list))
        for i, val in enumerate(py_list):
            arr[i] = int(val)
        return arr

    def close(self):
        try:
            self.gateway.shutdown()
        except Exception:
            pass
