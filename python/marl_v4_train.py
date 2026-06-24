"""
MARL V8: 4-Agent CTDE PPO + Centralized Critic + Parallel Envs + Batched GPU.

V8 changes over V6/V7 (single-env):
  - PARALLEL: N CloudSim bridges run concurrently (one Java process per port).
    Each env steps in its own thread; py4j releases the GIL on the socket wait,
    so N simulations advance truly in parallel on the available CPU cores.
  - BATCHED GPU: every agent forward pass and the Autoformer prediction are
    batched across all N envs (and across all hosts for the Autoformer) into a
    single CUDA call, giving the GPU enough work to be worth using at scale.
  - SCALE: NUM_HOSTS is read from the env var on both the Java and Python side,
    enabling the 100-host experiment where the larger observation/action space
    makes the networks big enough for GPU acceleration to pay off.

Inherited fixes (still active):
  - FIX 0: Agent 4 real log_prob   FIX 1: causal reward decomposition
  - FIX 2: centralized critic       P1-P4: A2 reward, invariant, A3 obs, Huber critic

Single-env behaviour is recovered with NUM_ENVS=1.
"""
import os
import csv
import math
import random
import threading
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor
from py4j.java_gateway import JavaGateway, GatewayParameters

from autoformer_detector import AutoformerDetector, detect_pm_state
from cloudsim_gym_env import CloudSimEnv
from config import Config
from lstm_underload_detector import LSTMUnderloadDetector
from models import Actor, CentralizedCritic, select_action

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ========== CTDE PPO Agent (batched, GPU-resident) ==========
class CTDEPPOAgent:
    """Decentralized actor + centralized value. Buffers collect transitions
    from ALL parallel envs; updates run as one big batch on DEVICE."""

    def __init__(self, obs_dim, action_dim, lr=None, gamma=None,
                 gae_lambda=None, clip_ratio=None, epochs=None,
                 ent_coef=0.01, name="agent"):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        lr = Config.PPO_LR if lr is None else lr
        self.gamma = Config.PPO_GAMMA if gamma is None else gamma
        self.gae_lambda = Config.PPO_GAE_LAMBDA if gae_lambda is None else gae_lambda
        self.clip_ratio = Config.PPO_CLIP_EPSILON if clip_ratio is None else clip_ratio
        self.epochs = Config.PPO_UPDATE_EPOCHS if epochs is None else epochs
        self.ent_coef = ent_coef
        self.name = name

        self.actor = Actor(obs_dim, action_dim).to(DEVICE)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        # Per-env rollout buffers (list-of-lists, one inner list per env) so
        # GAE is computed along each env's own trajectory, never across envs.
        self.n_envs = 1
        self._reset_buffers()
        self._last_returns = None
        self._last_global_obs = None
        self._updated_last_call = False

    def set_num_envs(self, n):
        self.n_envs = n
        self._reset_buffers()

    def _reset_buffers(self):
        self.obs_buf = [[] for _ in range(self.n_envs)]
        self.act_buf = [[] for _ in range(self.n_envs)]
        self.rew_buf = [[] for _ in range(self.n_envs)]
        self.val_buf = [[] for _ in range(self.n_envs)]
        self.logp_buf = [[] for _ in range(self.n_envs)]
        self.mask_buf = [[] for _ in range(self.n_envs)]
        self.done_buf = [[] for _ in range(self.n_envs)]
        self.global_obs_buf = [[] for _ in range(self.n_envs)]

    def act_batch(self, obs_batch, mask_batch=None, deterministic=False):
        """Batched action selection across envs. obs_batch: [B, obs_dim]."""
        obs_t = torch.as_tensor(np.asarray(obs_batch), dtype=torch.float32, device=DEVICE)
        mask_t = None
        if mask_batch is not None:
            mask_t = torch.as_tensor(np.asarray(mask_batch), dtype=torch.bool, device=DEVICE)
        with torch.no_grad():
            logits = self.actor(obs_t, mask_t)
            probs = F.softmax(logits, dim=-1)
            if deterministic:
                actions = probs.argmax(dim=-1)
            else:
                actions = torch.distributions.Categorical(probs).sample()
            logps = torch.log(probs.gather(1, actions.unsqueeze(1)).squeeze(1) + 1e-8)
        return actions.cpu().numpy(), logps.cpu().numpy()

    def store(self, env_i, obs, action, reward, central_value, logp, mask, done, global_obs):
        self.obs_buf[env_i].append(obs)
        self.act_buf[env_i].append(action)
        self.rew_buf[env_i].append(reward)
        self.val_buf[env_i].append(central_value)
        self.logp_buf[env_i].append(logp)
        self.mask_buf[env_i].append(mask)
        self.done_buf[env_i].append(done)
        self.global_obs_buf[env_i].append(global_obs)

    def mark_episode_boundary(self):
        for e in range(self.n_envs):
            if self.done_buf[e]:
                self.done_buf[e][-1] = 1.0

    def _gae_one(self, rew, val, done, last_value=0.0):
        T = len(rew)
        adv = np.zeros(T, dtype=np.float32)
        ret = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            if t == T - 1:
                next_val, next_done = last_value, 1.0
            else:
                next_val, next_done = val[t + 1], done[t + 1]
            delta = rew[t] + self.gamma * next_val * (1 - next_done) - val[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - next_done) * gae
            adv[t] = gae
            ret[t] = gae + val[t]
        return adv, ret

    def update_actor(self):
        self._updated_last_call = False
        self._last_returns = None
        self._last_global_obs = None

        all_obs, all_act, all_adv, all_logp, all_mask, all_ret, all_gobs = \
            [], [], [], [], [], [], []
        for e in range(self.n_envs):
            if not self.obs_buf[e]:
                continue
            adv, ret = self._gae_one(self.rew_buf[e], self.val_buf[e], self.done_buf[e])
            all_obs.extend(self.obs_buf[e])
            all_act.extend(self.act_buf[e])
            all_adv.append(adv)
            all_ret.append(ret)
            all_logp.extend(self.logp_buf[e])
            all_mask.extend(self.mask_buf[e])
            all_gobs.extend(self.global_obs_buf[e])

        if len(all_obs) < 8:
            return 0.0

        obs_t = torch.as_tensor(np.array(all_obs), dtype=torch.float32, device=DEVICE)
        act_t = torch.as_tensor(np.array(all_act), dtype=torch.long, device=DEVICE)
        adv_t = torch.as_tensor(np.concatenate(all_adv), dtype=torch.float32, device=DEVICE)
        old_logp_t = torch.as_tensor(np.array(all_logp), dtype=torch.float32, device=DEVICE)
        mask_t = torch.as_tensor(np.array(all_mask), dtype=torch.bool, device=DEVICE)

        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        total_loss = 0.0
        for _ in range(self.epochs):
            logits = self.actor(obs_t, mask_t)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            new_logp = dist.log_prob(act_t)
            entropy = dist.entropy().mean()
            ratio = torch.exp(new_logp - old_logp_t)
            surr1 = ratio * adv_t
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv_t
            policy_loss = -torch.min(surr1, surr2).mean()
            loss = policy_loss - self.ent_coef * entropy
            self.actor_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optimizer.step()
            total_loss += loss.item()

        self._last_returns = torch.as_tensor(np.concatenate(all_ret), dtype=torch.float32, device=DEVICE)
        self._last_global_obs = torch.as_tensor(np.array(all_gobs), dtype=torch.float32, device=DEVICE)
        self._updated_last_call = True
        return total_loss / self.epochs

    def clear_buffer(self):
        self._reset_buffers()
        self._last_returns = None
        self._last_global_obs = None

    def save(self, path):
        torch.save(self.actor.state_dict(), path)

    def load(self, path):
        if os.path.exists(path):
            try:
                self.actor.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
                return True
            except RuntimeError as exc:
                print(f"[RESUME] Skipping incompatible {self.name} checkpoint {path}; {exc}")
                return False
        return False


# ========== Conflict Resolver (kept for eval_v7_demo import) ==========
def resolve_conflicts(underload_indices, overload_indices):
    """OL takes priority over UL: SLA violations > energy savings."""
    ul_set = set(underload_indices)
    ol_set = set(overload_indices)
    conflicts = ul_set & ol_set
    clean_ul = [i for i in underload_indices if i not in conflicts]
    clean_ol = list(overload_indices)
    return clean_ul, clean_ol


# ========== Parallel env helpers ==========
def _make_env(port):
    gw = JavaGateway(gateway_parameters=GatewayParameters(port=port, auto_convert=True))
    return CloudSimEnv(gw)


def _parallel(pool, fn, items):
    """Run fn(item) across the thread pool, preserving order."""
    return list(pool.map(fn, items))


# ========== Training Loop ==========
def train():
    num_envs = max(1, int(os.environ.get("NUM_ENVS", "4")))
    base_port = int(os.environ.get("BRIDGE_BASE_PORT", "25333"))
    ports = [base_port + i for i in range(num_envs)]

    print("=" * 70)
    print("MARL V8: 4-Agent CTDE PPO + Parallel Envs + Batched GPU")
    print(f"  Device   : {DEVICE}")
    print(f"  Parallel : {num_envs} envs on ports {ports}")
    print(f"  Hosts    : {Config.NUM_HOSTS} (NUM_HOSTS env)")
    print("=" * 70)

    pool = ThreadPoolExecutor(max_workers=num_envs)
    envs = _parallel(pool, _make_env, ports)
    env0 = envs[0]

    fresh_train = os.environ.get("FRESH_TRAIN", "0").strip().lower() in {"1", "true", "yes", "y"}
    fresh_detectors = os.environ.get("FRESH_DETECTORS", "0").strip().lower() in {"1", "true", "yes", "y"}
    save_models = os.environ.get("SAVE_MODELS", "1").strip().lower() not in {"0", "false", "no", "n"}
    summary_path = os.environ.get("RUN_SUMMARY_PATH", "run_summary_v6.csv")
    smoke_max_steps = max(0, int(os.environ.get("SMOKE_MAX_STEPS", "0")))
    if smoke_max_steps:
        print(f"[SMOKE] Limiting each episode to {smoke_max_steps} process steps.")

    # ---- Detectors (shared, GPU-resident, batched) ----
    af_seq_len = int(getattr(env0, "history_len", Config.AF_SEQ_LEN))
    autoformer = AutoformerDetector(seq_len=af_seq_len, pred_len=Config.AF_PRED_LEN,
                                    d_model=Config.AF_D_MODEL).to(DEVICE)
    if fresh_detectors:
        print("[FRESH] Autoformer from random weights.")
    elif os.path.exists("autoformer_pretrained.pt"):
        try:
            autoformer.load_state_dict(torch.load("autoformer_pretrained.pt",
                                                  map_location=DEVICE, weights_only=True))
            print("[V8] Loaded pre-trained Autoformer.")
        except RuntimeError as exc:
            print(f"[V8] Skipping incompatible Autoformer checkpoint; {exc}")

    # One LSTM detector per env (each tracks its own host CPU buffers).
    lstm_detectors = []
    lstm_model_path = "lstm_underload.pt"
    for _ in range(num_envs):
        if (not fresh_detectors) and os.path.exists(lstm_model_path):
            lstm_detectors.append(LSTMUnderloadDetector(
                model_path=lstm_model_path, num_hosts=env0.num_hosts,
                window=Config.LSTM_WINDOW, threshold=Config.LSTM_PROB_THRESHOLD,
                cooldown_steps=Config.LSTM_COOLDOWN_STEPS))
        else:
            lstm_detectors.append(None)
    if lstm_detectors[0] is not None:
        print("[V8] Loaded LSTM Underload Predictor (per-env).")

    print(f"Config: hosts={env0.num_hosts}, top_k={env0.top_k}, "
          f"global_dim={env0.global_dim}, vm_dim={env0.vm_dim}, history_len={af_seq_len}")
    Config.META_TUNING_GRANULARITY = "episode"

    # ---- Centralized critic ----
    central_obs_dim = env0.global_dim + env0.num_hosts * 2 + 3
    central_critic = CentralizedCritic(central_obs_dim).to(DEVICE)
    critic_optimizer = optim.Adam(central_critic.parameters(), lr=Config.PPO_LR)
    print(f"[V8] Centralized Critic input_dim={central_obs_dim}")

    # ---- Agents ----
    agent1 = CTDEPPOAgent(env0.a1_obs_dim, env0.a1_action_n, ent_coef=0.05, name="underload_detector")
    agent2 = CTDEPPOAgent(env0.a2_obs_dim, env0.a2_action_n, ent_coef=0.05, name="overload_detector")
    agent3 = CTDEPPOAgent(env0.a3_obs_dim, env0.num_sel_actions, ent_coef=0.01, name="vm_selector")
    agent4 = CTDEPPOAgent(env0.vm_dim, env0.top_k, ent_coef=0.01, name="vm_placer")
    agents = [agent1, agent2, agent3, agent4]
    for ag in agents:
        ag.set_num_envs(num_envs)

    NUM_EPISODES = Config.NUM_EPISODES
    episode_rewards = []
    start_ep = 0
    if fresh_train:
        print("[FRESH] Training agents + critic from scratch.")

    # CSV logger (identical schema to single-env so analysis scripts still work)
    csv_file = open(summary_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "episode", "steps", "total_reward", "avg20_reward",
        "r_underload", "r_overload", "r_selector", "r_placer",
        "slatah", "pdm", "slav", "energy_kwh",
        "overloads", "critical_overloads", "capacity_overloads",
        "actionable_overloads", "selected_overload_sources", "guard_overload_sources",
        "unactionable_overloads", "gpu_blocked_overloads", "capacity_blocked_overloads",
        "single_vm_overloads", "a2_true_positives", "a2_false_positives", "a2_false_negatives",
        "underloads", "migrations", "failures", "no_candidates", "same_host_skips", "critic_loss",
        "num_envs",
    ])

    def autoformer_preds_batch(histories):
        """Batch Autoformer over all (env,host) rows in one CUDA call.
        histories: list of [num_hosts, seq_len] arrays. Returns list of
        [num_hosts] max-pred vectors."""
        rows, idx = [], []
        for ei, h in enumerate(histories):
            for hi in range(h.shape[0]):
                seq = h[hi]
                if seq.max() > 0.01:
                    rows.append(seq)
                    idx.append((ei, hi))
        preds = [np.zeros(h.shape[0], dtype=np.float32) for h in histories]
        if rows:
            x = torch.as_tensor(np.array(rows), dtype=torch.float32, device=DEVICE)
            autoformer.eval()
            with torch.no_grad():
                out = autoformer(x).cpu().numpy()  # [R, pred_len]
            for k, (ei, hi) in enumerate(idx):
                preds[ei][hi] = out[k].max()
        return preds

    for ep in range(start_ep, NUM_EPISODES):
        Config.reset_runtime_state(reset_thresholds=True)
        cosine_lr = max(1e-5, Config.PPO_LR_INIT * 0.5 * (1 + math.cos(math.pi * ep / NUM_EPISODES)))
        Config.PPO_LR = cosine_lr
        for opt in [critic_optimizer] + [a.actor_optimizer for a in agents]:
            for g in opt.param_groups:
                g["lr"] = cosine_lr

        _parallel(pool, lambda e: e.reset(), envs)
        for d in lstm_detectors:
            if d:
                d.reset()
        global_states = [e.global_state for e in envs]
        active = [True] * num_envs
        ep_rewards = [0.0] * num_envs
        ep_r = [[0.0, 0.0, 0.0, 0.0] for _ in range(num_envs)]
        ep_infos = [{} for _ in range(num_envs)]
        steps = 0

        while any(active) and (smoke_max_steps == 0 or steps < smoke_max_steps):
            live = [i for i in range(num_envs) if active[i]]

            # ---- Level 1: detectors (batched) ----
            histories = _parallel(pool, lambda i: envs[i].get_host_history(), live)
            af_preds = autoformer_preds_batch(histories)
            det_obs_list = [envs[i].get_detector_obs(af_preds[k]) for k, i in enumerate(live)]

            for k, i in enumerate(live):
                cu = np.array([det_obs_list[k][h * 2] for h in range(envs[i].num_hosts)])
                Config.observe_process_step(cpu_values=cu, step=steps)
                if lstm_detectors[i]:
                    lstm_detectors[i].threshold = Config.LSTM_PROB_THRESHOLD
                    lstm_detectors[i].update(cu)

            ul_masks = [envs[i].get_detector_masks(mode='underload') for i in live]
            ol_masks = [envs[i].get_detector_masks(mode='overload') for i in live]
            a1_acts, a1_lps = agent1.act_batch(det_obs_list, ul_masks)
            a2_acts, a2_lps = agent2.act_batch(det_obs_list, ol_masks)

            # ---- Resolve detector decisions per env ----
            ul_idx_list, ol_idx_list, det_ctx_list = [], [], []
            for k, i in enumerate(live):
                pred_ul = None
                if lstm_detectors[i]:
                    pred_ul, _ = lstm_detectors[i].detect_with_probs()
                ul_idx, ol_idx, ctx = envs[i].resolve_detector_actions(
                    det_obs_list[k], int(a1_acts[k]), int(a2_acts[k]),
                    predicted_underloads=pred_ul)
                if lstm_detectors[i] and ul_idx:
                    lstm_detectors[i].cooldown.mark_shutdown(ul_idx[0])
                ul_idx_list.append(ul_idx)
                ol_idx_list.append(ol_idx)
                det_ctx_list.append(ctx)

            # ---- Level 2: selector (batched) ----
            sel_obs_list = [envs[i].build_selector_obs(global_states[i], ol_idx_list[k],
                                                       det_obs_list[k], underload_indices=ul_idx_list[k])
                            for k, i in enumerate(live)]
            sel_masks = [np.ones(envs[i].num_sel_actions, dtype=bool) for i in live]
            a3_acts, a3_lps = agent3.act_batch(sel_obs_list, sel_masks)

            # ---- Level 2: placer (collect VM states across envs, batch) ----
            vm_states_per = []
            for k, i in enumerate(live):
                srcs = list(ol_idx_list[k]) + list(ul_idx_list[k])
                if srcs:
                    vs = envs[i].get_migration_placer_obs(ol_idx_list[k], ul_idx_list[k], int(a3_acts[k]))
                else:
                    vs = []
                vm_states_per.append(vs)
            masks_per = [[] for _ in live]  # per env-slot placer masks, aligned with vm_states_per
            flat_vs, flat_masks, flat_loc = [], [], []
            for k, vs in enumerate(vm_states_per):
                for vsi in vs:
                    m = envs[live[k]].get_placer_mask(vsi)
                    masks_per[k].append(m)
                    flat_vs.append(vsi)
                    flat_masks.append(m)
                    flat_loc.append(k)
            placements_per = [[] for _ in live]
            a4lps_per = [[] for _ in live]
            if flat_vs:
                a4_acts, a4_lps = agent4.act_batch(flat_vs, flat_masks)
                for j, k in enumerate(flat_loc):
                    placements_per[k].append(int(a4_acts[j]))
                    a4lps_per[k].append(float(a4_lps[j]))

            # ---- Centralized value (batched) ----
            central_obs_list = []
            for k, i in enumerate(live):
                sel_ctx = sel_obs_list[k][envs[i].global_dim:envs[i].global_dim + 3]
                central_obs_list.append(np.concatenate([global_states[i], det_obs_list[k], sel_ctx]))
            with torch.no_grad():
                cobs_t = torch.as_tensor(np.array(central_obs_list), dtype=torch.float32, device=DEVICE)
                cvals = central_critic(cobs_t).squeeze(-1).cpu().numpy()

            # ---- Step all envs in parallel ----
            def _do_step(k):
                i = live[k]
                return envs[i].step(ul_idx_list[k], ol_idx_list[k], int(a3_acts[k]),
                                    placements_per[k], detector_context=det_ctx_list[k])
            results = _parallel(pool, _do_step, list(range(len(live))))

            # ---- Store transitions + bookkeeping ----
            for k, i in enumerate(live):
                next_global, rewards, done, info = results[k]
                Config.observe_process_step(info=info, global_state=next_global, step=steps)
                ep_infos[i] = info
                cval = float(cvals[k])
                cobs = central_obs_list[k]
                agent1.store(i, det_obs_list[k], int(a1_acts[k]), rewards['underload_det'],
                             cval, float(a1_lps[k]), ul_masks[k], float(done), cobs)
                agent2.store(i, det_obs_list[k], int(a2_acts[k]), rewards['overload_det'],
                             cval, float(a2_lps[k]), ol_masks[k], float(done), cobs)
                if ol_idx_list[k]:
                    agent3.store(i, sel_obs_list[k], int(a3_acts[k]), rewards['vm_selector'],
                                 cval, float(a3_lps[k]), sel_masks[k], float(done), cobs)
                for vi, vsi in enumerate(vm_states_per[k]):
                    if vi < len(placements_per[k]):
                        agent4.store(i, vsi, placements_per[k][vi], rewards['vm_placer'],
                                     cval, a4lps_per[k][vi], masks_per[k][vi],
                                     float(done), cobs)
                ep_rewards[i] += sum(rewards.values())
                ep_r[i][0] += rewards['underload_det']
                ep_r[i][1] += rewards['overload_det']
                ep_r[i][2] += rewards['vm_selector']
                ep_r[i][3] += rewards['vm_placer']
                global_states[i] = next_global
                if done:
                    active[i] = False
            steps += 1

        # ---- PPO update (one big batch across all envs) ----
        for ag in agents:
            ag.mark_episode_boundary()
        loss1 = agent1.update_actor()
        loss2 = agent2.update_actor()
        loss3 = agent3.update_actor()
        loss4 = agent4.update_actor()

        critic_loss_total, critic_updates = 0.0, 0
        for ag in agents:
            if ag._last_returns is not None and len(ag._last_returns) > 0:
                pred_v = central_critic(ag._last_global_obs).squeeze(-1)
                c_loss = F.smooth_l1_loss(pred_v, ag._last_returns)
                critic_optimizer.zero_grad()
                c_loss.backward()
                torch.nn.utils.clip_grad_norm_(central_critic.parameters(), 0.5)
                critic_optimizer.step()
                critic_loss_total += c_loss.item()
                critic_updates += 1
        avg_critic_loss = critic_loss_total / max(1, critic_updates)

        # ---- Aggregate metrics across envs (sum counts, mean of rates) ----
        agg = _aggregate_envs(envs, ep_infos, ep_rewards, ep_r)
        for ag in agents:
            if ag._updated_last_call:
                ag.clear_buffer()

        # Feed self-tuning with the aggregate (use env0's tuning hook contract)
        Config.observe_episode_outcome(agg["tuning_info"], agg["tuning_rewards"])
        Config.observe_training_episode(agg["total_reward"], avg_critic_loss,
                                        [loss1, loss2, loss3, loss4])

        episode_rewards.append(agg["total_reward"])
        avg_reward = float(np.mean(episode_rewards[-20:]))

        print(f"Ep {ep+1:3d}/{NUM_EPISODES} | Envs:{num_envs} Steps:{steps:3d} | "
              f"R(mean/env): {agg['total_reward']:8.2f} | Avg20: {avg_reward:8.2f} | "
              f"OL:{agg['overloads']} CritOL:{agg['critical_overloads']} "
              f"UL:{agg['underloads']} Mig:{agg['migrations']} "
              f"Fail:{agg['failures']} | SLATAH:{agg['slatah']:.4f} E:{agg['energy_kwh']:.2f}kWh")
        print(f"       A2 TP/FP/FN:{agg['a2_tp']}/{agg['a2_fp']}/{agg['a2_fn']} | "
              f"R1:{agg['r1']:.1f} R2:{agg['r2']:.1f} R3:{agg['r3']:.1f} R4:{agg['r4']:.1f} | "
              f"L1:{loss1:.4f} L2:{loss2:.4f} L3:{loss3:.4f} L4:{loss4:.4f} CriticL:{avg_critic_loss:.4f}")

        csv_writer.writerow([
            ep + 1, steps, f"{agg['total_reward']:.4f}", f"{avg_reward:.4f}",
            f"{agg['r1']:.4f}", f"{agg['r2']:.4f}", f"{agg['r3']:.4f}", f"{agg['r4']:.4f}",
            f"{agg['slatah']:.6f}", f"{agg['pdm']:.8f}", f"{agg['slav']:.10f}", f"{agg['energy_kwh']:.4f}",
            agg['overloads'], agg['critical_overloads'], agg['capacity_overloads'],
            agg['actionable_overloads'], agg['selected_overload_sources'], agg['guard_overload_sources'],
            agg['unactionable_overloads'], agg['gpu_blocked_overloads'], agg['capacity_blocked_overloads'],
            agg['single_vm_overloads'], agg['a2_tp'], agg['a2_fp'], agg['a2_fn'],
            agg['underloads'], agg['migrations'], agg['failures'],
            agg['no_candidates'], agg['same_host_skips'], f"{avg_critic_loss:.6f}", num_envs,
        ])
        csv_file.flush()

        if save_models and (ep + 1) % 10 == 0:
            _save_all(agents, central_critic, autoformer)
            print(f"[CHECKPOINT] saved at episode {ep+1}")

    csv_file.close()
    if save_models:
        _save_all(agents, central_critic, autoformer)
    print("=" * 70)
    print("V8 Training complete!" + (" Models saved." if save_models else " Smoke: not saved."))
    print("=" * 70)
    for e in envs:
        try:
            e.close()
        except Exception:
            pass
    pool.shutdown(wait=False)


def _aggregate_envs(envs, ep_infos, ep_rewards, ep_r):
    n = len(envs)
    s = lambda attr: int(sum(getattr(e, attr) for e in envs))
    out = {
        "overloads": s("ep_overloads"), "critical_overloads": s("ep_critical_overloads"),
        "capacity_overloads": s("ep_capacity_overloads"),
        "actionable_overloads": s("ep_actionable_overloads"),
        "selected_overload_sources": s("ep_selected_overload_sources"),
        "guard_overload_sources": s("ep_guard_overload_sources"),
        "unactionable_overloads": s("ep_unactionable_overloads"),
        "gpu_blocked_overloads": s("ep_gpu_blocked_overloads"),
        "capacity_blocked_overloads": s("ep_capacity_blocked_overloads"),
        "single_vm_overloads": s("ep_single_vm_overloads"),
        "a2_tp": s("ep_a2_true_positives"), "a2_fp": s("ep_a2_false_positives"),
        "a2_fn": s("ep_a2_false_negatives"), "underloads": s("ep_underloads"),
        "migrations": s("ep_migrations"), "failures": s("ep_failures"),
        "no_candidates": s("ep_no_candidates"), "same_host_skips": s("ep_same_host_skips"),
    }
    out["slatah"] = float(np.mean([ep_infos[i].get('slatah', 0) for i in range(n)]))
    out["pdm"] = float(np.mean([ep_infos[i].get('pdm', 0) for i in range(n)]))
    out["slav"] = float(np.mean([ep_infos[i].get('slav', 0) for i in range(n)]))
    out["energy_kwh"] = float(np.mean([ep_infos[i].get('energy_kwh', 0) for i in range(n)]))
    out["total_reward"] = float(np.mean(ep_rewards))
    out["r1"] = float(np.mean([r[0] for r in ep_r]))
    out["r2"] = float(np.mean([r[1] for r in ep_r]))
    out["r3"] = float(np.mean([r[2] for r in ep_r]))
    out["r4"] = float(np.mean([r[3] for r in ep_r]))
    # tuning hook expects single-env-style dicts; use representative env0 info
    ti = dict(ep_infos[0])
    ti.update({
        "migrations": envs[0].ep_migrations,
        "failures": envs[0].ep_failures + envs[0].ep_no_candidates + envs[0].ep_same_host_skips,
        "overloads": envs[0].ep_overloads,
        "critical_overloads": envs[0].ep_critical_overloads,
        "actionable_overloads": envs[0].ep_actionable_overloads,
        "a2_true_positives": envs[0].ep_a2_true_positives,
        "a2_false_positives": envs[0].ep_a2_false_positives,
        "a2_false_negatives": envs[0].ep_a2_false_negatives,
        "underloads": envs[0].ep_underloads,
    })
    out["tuning_info"] = ti
    out["tuning_rewards"] = {"underload_det": ep_r[0][0], "overload_det": ep_r[0][1],
                             "vm_selector": ep_r[0][2], "vm_placer": ep_r[0][3]}
    return out


def _save_all(agents, central_critic, autoformer):
    agents[0].save("agent1_underload_det_v6.pt")
    agents[1].save("agent2_overload_det_v6.pt")
    agents[2].save("agent3_vm_selector_v6.pt")
    agents[3].save("agent4_vm_placer_v6.pt")
    torch.save(central_critic.state_dict(), "central_critic_v6.pt")
    torch.save(autoformer.state_dict(), "autoformer_detector.pt")


if __name__ == "__main__":
    train()
