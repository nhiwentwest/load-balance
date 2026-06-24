"""
Lambda Architecture Runner — Main entry point.

Orchestrates:
  1. Speed Layer: Online Scheduler (inference, real-time decisions)
  2. Batch Layer: Offline Trainer (periodic retraining, separate thread)
  3. Monitoring: Metrics collection + anomaly detection

Usage:
  # Phase 1: Train baseline models (uses existing marl_v4_train.py)
  python marl_v4_train.py

  # Phase 2: Run Lambda architecture evaluation
  python lambda_runner.py
"""

import os
import sys
import csv
import time
import json
import numpy as np
import torch
import torch.nn as nn

from config import Config
from event_database import EventDatabase
from model_registry import ModelRegistry
from monitoring import MonitoringSystem
from scheduler import OnlineScheduler
from offline_trainer import OfflineTrainer
from cloudsim_gym_env import CloudSimEnv
from autoformer_detector import AutoformerDetector
from lstm_underload_detector import LSTMUnderloadDetector


# ==================== Resolve conflicts (shared util) ====================
def resolve_conflicts(underload, overload):
    conflict = set(underload) & set(overload)
    if conflict:
        underload = [h for h in underload if h not in conflict]
    return underload, overload


from models import Actor, CentralizedCritic, select_action


# ==================== Initialize from baseline models ====================
def initialize_from_baseline(registry, baseline_dir="."):
    """Register baseline models from marl_v4_train.py output."""

    model_files = {
        "agent1": "agent1_underload_det_v6.pt",
        "agent2": "agent2_overload_det_v6.pt",
        "agent3": "agent3_vm_selector_v6.pt",
        "agent4": "agent4_vm_placer_v6.pt",
        "central_critic": "central_critic_v6.pt",
        "autoformer": "autoformer_detector.pt",
        "bilstm": "lstm_underload.pt",
    }

    for model_type, filename in model_files.items():
        filepath = os.path.join(baseline_dir, filename)
        if os.path.exists(filepath):
            registry.register_initial_model(
                model_type, filepath,
                metrics={"val_loss": 0.0, "source": "baseline_v7"}
            )
        else:
            print(f"[Init] WARNING: {filepath} not found, skipping {model_type}")


def load_compatible_state_dict(module, state_dict, name):
    if not state_dict:
        return False
    try:
        module.load_state_dict(state_dict)
        module.eval()
        return True
    except RuntimeError as exc:
        print(f"[Init] Skipping incompatible {name} checkpoint; retrain required. {exc}")
        return False


# ==================== Main Lambda Runner ====================
def run_lambda(num_episodes=5, baseline_dir="."):
    """
    Run the Lambda architecture evaluation.

    Flow:
      1. Initialize all components
      2. Load baseline models into registry
      3. For each episode:
         a. Scheduler makes decisions (Speed Layer)
         b. Check retrain triggers (Batch Layer)
         c. Hot-swap models if trainer produced better ones
         d. Monitoring records metrics + checks anomalies
      4. Report results
    """

    print("=" * 70)
    print("LAMBDA ARCHITECTURE — MARL VM Placement")
    print("=" * 70)
    print(f"  Speed Layer: Online Scheduler (inference)")
    print("  Batch Layer: Offline Trainer (process drift/SLA/failure-triggered)")
    print(f"  Monitoring:  z-score anomaly detection (threshold={Config.ANOMALY_ZSCORE_THRESHOLD})")
    print(f"  Thresholds:  underload={Config.UNDERLOAD_THRESHOLD}, "
          f"overload={Config.OVERLOAD_THRESHOLD}, "
          f"critical={Config.CRITICAL_OVERLOAD_THRESHOLD}")
    print("=" * 70)

    # 1. Initialize infrastructure
    event_db = EventDatabase(Config.EVENT_DB_PATH)
    registry = ModelRegistry(Config.MODEL_DIR)
    monitoring = MonitoringSystem(event_db)
    # Open a real Prometheus-scrapeable HTTP endpoint for live config/runtime state.
    prom_port = int(os.environ.get("PROMETHEUS_PORT", "8000"))
    monitoring.start_http_endpoint(prom_port)

    # 2. Load baseline models
    print("\n[Init] Loading baseline models...")
    initialize_from_baseline(registry, baseline_dir)
    registry.summary()

    # 3. Initialize environment
    print("\n[Init] Starting CloudSim Plus...")
    env = CloudSimEnv()

    # 4. Create agent networks and load weights
    a1_dim = env.a1_obs_dim
    a2_dim = env.a2_obs_dim
    agent1 = Actor(obs_dim=a1_dim, action_dim=env.a1_action_n)
    agent2 = Actor(obs_dim=a2_dim, action_dim=env.a2_action_n)
    agent3 = Actor(obs_dim=env.a3_obs_dim, action_dim=env.num_sel_actions)
    agent4 = Actor(obs_dim=env.a4_obs_dim, action_dim=env.a4_action_n)
    central_obs_dim = env.global_dim + env.num_hosts * 2 + 3
    central_critic = CentralizedCritic(central_obs_dim)

    # Load from registry
    loaded_agents = {}
    for mt, agent in [("agent1", agent1), ("agent2", agent2),
                       ("agent3", agent3), ("agent4", agent4)]:
        sd = registry.load_best_model(mt)
        loaded_agents[mt] = load_compatible_state_dict(agent, sd, mt)
    if not loaded_agents.get("agent1") or not loaded_agents.get("agent2"):
        raise RuntimeError(
            "Serving blocked: detector checkpoints predate the N+NO_OP action "
            "contract. Complete a fresh training run and promote compatible models."
        )

    sd = registry.load_best_model("central_critic")
    load_compatible_state_dict(central_critic, sd, "central_critic")

    autoformer = AutoformerDetector(
        seq_len=int(getattr(env, "history_len", Config.AF_SEQ_LEN)),
        pred_len=Config.AF_PRED_LEN,
        d_model=Config.AF_D_MODEL,
    )
    sd = registry.load_best_model("autoformer")
    if not load_compatible_state_dict(autoformer, sd, "autoformer"):
        raise RuntimeError(
            "Serving blocked: Autoformer checkpoint is incompatible with the "
            f"process-derived history_len={env.history_len}. Retrain and promote "
            "a compatible offline detector first."
        )

    # 5. Create Lambda components
    scheduler = OnlineScheduler(
        env=env,
        agents={"agent1": agent1, "agent2": agent2,
                "agent3": agent3, "agent4": agent4},
        central_critic=central_critic,
        autoformer=autoformer,
        event_db=event_db,
        model_registry=registry,
        monitoring=monitoring,
    )

    trainer = OfflineTrainer(event_db, registry)

    # 6. Run evaluation episodes
    print(f"\n{'='*70}")
    print(f"Running {num_episodes} episodes with Lambda architecture...")
    print(f"{'='*70}\n")

    all_results = []
    csv_path = "lambda_results.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "episode", "steps", "total_reward",
        "r_underload", "r_overload", "r_selector", "r_placer",
        "slatah", "pdm", "energy_kwh",
        "overloads", "underloads", "migrations", "failures",
        "retrains_lstm", "retrains_autoformer", "model_swaps",
    ])

    for ep in range(num_episodes):
        global_state = env.reset()
        scheduler.reset()
        monitoring.reset()

        total_reward = 0
        ep_r1, ep_r2, ep_r3, ep_r4 = 0, 0, 0, 0
        steps = 0
        ep_info = {}
        retrains_lstm = 0
        retrains_af = 0
        model_swaps = 0

        # Collect CPU history for retraining
        cpu_history_buffer = {}

        while not env.done:
            # Get detection observations handled inside scheduler
            sim_time = steps * Config.INTERVAL_SEC

            # ===== SPEED LAYER: Online Scheduler =====
            ul_indices, ol_indices, sel_action, place_actions, det_obs, detector_context = \
                scheduler.step(global_state, steps, sim_time)

            # ===== EXECUTE STEP =====
            next_global, rewards, done, info = env.step(
                ul_indices, ol_indices, sel_action, place_actions,
                detector_context=detector_context,
            )
            Config.observe_process_step(info=info, global_state=next_global, step=steps)

            # ===== MONITORING =====
            monitoring.record_step(steps, sim_time, info, next_global)

            # ===== Accumulate CPU history for retraining =====
            for h_idx in range(env.num_hosts):
                if h_idx * 2 < len(det_obs):
                    if h_idx not in cpu_history_buffer:
                        cpu_history_buffer[h_idx] = []
                    cpu_history_buffer[h_idx].append(float(det_obs[h_idx * 2]))

            # ===== BATCH LAYER: Check retrain triggers =====
            if trainer.should_retrain_lstm(steps):
                trainer.trigger_lstm_retrain(
                    cpu_history_buffer, steps, sim_time, async_=True
                )
                retrains_lstm += 1

            if trainer.should_retrain_autoformer(steps):
                trainer.trigger_autoformer_retrain(
                    cpu_history_buffer, steps, sim_time, async_=True
                )
                retrains_af += 1

            # ===== HOT-SWAP CHECK =====
            if steps % 20 == 0:  # Check every 20 steps
                if scheduler.check_model_update():
                    model_swaps += 1

            # Accumulate rewards
            total_reward += sum(rewards.values())
            ep_r1 += rewards["underload_det"]
            ep_r2 += rewards["overload_det"]
            ep_r3 += rewards["vm_selector"]
            ep_r4 += rewards["vm_placer"]
            ep_info = info
            global_state = next_global
            steps += 1

        # Episode summary
        slatah = ep_info.get("slatah", 0)
        pdm = ep_info.get("pdm", 0)
        energy = ep_info.get("energy_kwh", 0)

        result = {
            "episode": ep + 1,
            "steps": steps,
            "total_reward": total_reward,
            "r1": ep_r1, "r2": ep_r2, "r3": ep_r3, "r4": ep_r4,
            "slatah": slatah, "pdm": pdm, "energy": energy,
            "overloads": env.ep_overloads,
            "underloads": env.ep_underloads,
            "migrations": env.ep_migrations,
            "failures": env.ep_failures,
            "retrains_lstm": retrains_lstm,
            "retrains_af": retrains_af,
            "model_swaps": model_swaps,
        }
        all_results.append(result)

        csv_writer.writerow([
            ep + 1, steps, f"{total_reward:.4f}",
            f"{ep_r1:.4f}", f"{ep_r2:.4f}", f"{ep_r3:.4f}", f"{ep_r4:.4f}",
            f"{slatah:.6f}", f"{pdm:.8f}", f"{energy:.4f}",
            env.ep_overloads, env.ep_underloads, env.ep_migrations, env.ep_failures,
            retrains_lstm, retrains_af, model_swaps,
        ])
        csv_file.flush()

        print(f"\nEp {ep+1}/{num_episodes} | Steps: {steps} | "
              f"R: {total_reward:.2f} | "
              f"SLATAH: {slatah:.4f} E: {energy:.2f}kWh | "
              f"OL: {env.ep_overloads} UL: {env.ep_underloads} "
              f"Mig: {env.ep_migrations} Fail: {env.ep_failures}")
        print(f"  R1={ep_r1:.2f} R2={ep_r2:.2f} R3={ep_r3:.2f} R4={ep_r4:.2f} | "
              f"Retrains: LSTM={retrains_lstm} AF={retrains_af} | "
              f"Swaps: {model_swaps}")

    csv_file.close()

    # ===== FINAL SUMMARY =====
    print("\n" + "=" * 70)
    print("LAMBDA ARCHITECTURE — EVALUATION SUMMARY")
    print("=" * 70)

    rewards = [r["total_reward"] for r in all_results]
    energies = [r["energy"] for r in all_results]
    slatahs = [r["slatah"] for r in all_results]
    migs = [r["migrations"] for r in all_results]

    print(f"  Episodes:        {num_episodes}")
    print(f"  Avg Reward:      {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"  Avg Energy:      {np.mean(energies):.2f} kWh")
    print(f"  Avg SLATAH:      {np.mean(slatahs):.6f}")
    print(f"  Avg Migrations:  {np.mean(migs):.1f}")
    print(f"  Total Failures:  {sum(r['failures'] for r in all_results)}")
    print(f"  Total Retrains:  LSTM={sum(r['retrains_lstm'] for r in all_results)}, "
          f"AF={sum(r['retrains_af'] for r in all_results)}")
    print(f"  Total Swaps:     {sum(r['model_swaps'] for r in all_results)}")

    # Event database summary
    event_summary = event_db.get_all_events_summary()
    print(f"\n  Event DB Summary:")
    for etype, info in event_summary.items():
        print(f"    {etype}: {info['count']} events "
              f"(avg_severity={info['avg_severity']:.4f})")

    # Monitoring summary
    mon_summary = monitoring.get_summary()
    print(f"\n  Monitoring Summary:")
    for metric, stats in mon_summary.items():
        print(f"    {metric}: mean={stats['mean']:.4f}, "
              f"std={stats['std']:.4f}, max={stats['max']:.4f}")

    # Registry summary
    registry.summary()

    print(f"\n  Results saved to: {csv_path}")
    print("=" * 70)

    # Cleanup
    env.close()
    event_db.close()


if __name__ == "__main__":
    n = int(os.environ.get("LAMBDA_EPISODES", "5"))
    baseline = os.environ.get("BASELINE_DIR", ".")
    run_lambda(num_episodes=n, baseline_dir=baseline)
