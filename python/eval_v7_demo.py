import csv
import os
import sys

import numpy as np
import torch


ROOT = os.environ.get("DACN_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PY_DIR = os.path.join(ROOT, "python")
if PY_DIR not in sys.path:
    sys.path.insert(0, PY_DIR)

from autoformer_detector import AutoformerDetector
from cloudsim_gym_env import CloudSimEnv
from config import Config
from lstm_underload_detector import LSTMUnderloadDetector
from models import Actor, select_action
from marl_v4_train import resolve_conflicts


def load_actor(obs_dim, action_dim, checkpoint):
    actor = Actor(obs_dim, action_dim)
    state = torch.load(os.path.join(ROOT, checkpoint), map_location="cpu", weights_only=True)
    try:
        actor.load_state_dict(state)
    except RuntimeError as exc:
        raise RuntimeError(
            f"{checkpoint} is incompatible with obs_dim={obs_dim}, action_dim={action_dim}. "
            "Retrain/fine-tune the agent after the GPU-aware Agent 4 state change."
        ) from exc
    actor.eval()
    return actor


def act(actor, obs, mask):
    action, _ = select_action(actor, obs, mask, mode="eval")
    return action


def run_eval(num_episodes=3, out_csv="eval_v7_azure_demo.csv"):
    os.chdir(ROOT)
    np.random.seed(42)
    torch.manual_seed(42)
    dataset_label = os.environ.get("DATASET_LABEL", "Azure unseen traces")
    out_csv = os.environ.get("OUT_CSV", out_csv)

    print("=" * 80)
    print(f"V7 DEMO EVAL: deterministic inference from checkpoint on {dataset_label}")
    print("=" * 80)

    env = CloudSimEnv()
    print(
        f"Config: hosts={env.num_hosts}, top_k={env.top_k}, "
        f"global_dim={env.global_dim}, vm_dim={env.vm_dim}"
    )

    print("Loading V7 actor checkpoints (files still use _v6 names)...")
    agent1 = load_actor(env.a1_obs_dim, env.a1_action_n, "agent1_underload_det_v6.pt")
    agent2 = load_actor(env.a2_obs_dim, env.a2_action_n, "agent2_overload_det_v6.pt")
    agent3 = load_actor(env.a3_obs_dim, env.num_sel_actions, "agent3_vm_selector_v6.pt")
    agent4 = load_actor(env.vm_dim, env.top_k, "agent4_vm_placer_v6.pt")

    autoformer = AutoformerDetector(
        seq_len=int(getattr(env, "history_len", Config.AF_SEQ_LEN)),
        pred_len=Config.AF_PRED_LEN,
        d_model=Config.AF_D_MODEL,
    )
    autoformer_path = "autoformer_detector.pt"
    if not os.path.exists(os.path.join(ROOT, autoformer_path)):
        autoformer_path = "autoformer_pretrained.pt"
    autoformer.load_state_dict(
        torch.load(os.path.join(ROOT, autoformer_path), map_location="cpu", weights_only=True)
    )
    autoformer.eval()

    lstm_detector = LSTMUnderloadDetector(
        model_path=os.path.join(ROOT, "lstm_underload.pt"),
        num_hosts=env.num_hosts,
        window=10,
        threshold=0.75,
        cooldown_steps=20,
    )

    rows = []
    for ep in range(1, num_episodes + 1):
        global_state = env.reset()
        lstm_detector.reset()

        total_reward = 0.0
        ep_r1 = ep_r2 = ep_r3 = ep_r4 = 0.0
        steps = 0
        ep_info = {}

        while not env.done:
            history = env.get_host_history()
            autoformer_preds = np.zeros(env.num_hosts, dtype=np.float32)
            with torch.no_grad():
                for h_idx in range(env.num_hosts):
                    hist = history[h_idx].tolist()
                    if max(hist) > 0.01:
                        x = torch.tensor([hist], dtype=torch.float32)
                        pred = autoformer(x).squeeze(0).numpy()
                        autoformer_preds[h_idx] = pred.max()

            det_obs = env.get_detector_obs(autoformer_preds)
            ul_mask = env.get_detector_masks(mode="underload")
            ol_mask = env.get_detector_masks(mode="overload")

            current_utils = np.array([det_obs[h * 2] for h in range(env.num_hosts)])
            lstm_detector.update(current_utils)

            a1_action = act(agent1, det_obs, ul_mask)

            a2_action = act(agent2, det_obs, ol_mask)
            predicted_underloads, _ = lstm_detector.detect_with_probs()
            underload_indices, overload_indices, detector_context = (
                env.resolve_detector_actions(
                    det_obs,
                    a1_action,
                    a2_action,
                    predicted_underloads=predicted_underloads,
                )
            )
            if underload_indices:
                lstm_detector.cooldown.mark_shutdown(underload_indices[0])

            sel_obs = env.build_selector_obs(
                global_state, overload_indices, det_obs,
                underload_indices=underload_indices,
            )
            sel_mask = np.ones(env.num_sel_actions, dtype=bool)
            selection_action = act(agent3, sel_obs, sel_mask)

            placement_actions = []
            migration_sources = list(overload_indices) + list(underload_indices)
            if migration_sources:
                vm_states = env.get_migration_placer_obs(
                    overload_indices, underload_indices, selection_action
                )
                for vm_state in vm_states:
                    place_mask = env.get_placer_mask(vm_state)
                    placement_actions.append(act(agent4, vm_state, place_mask))

            next_global, rewards, done, info = env.step(
                underload_indices, overload_indices, selection_action, placement_actions,
                detector_context=detector_context,
            )

            step_reward = sum(rewards.values())
            total_reward += step_reward
            ep_r1 += rewards["underload_det"]
            ep_r2 += rewards["overload_det"]
            ep_r3 += rewards["vm_selector"]
            ep_r4 += rewards["vm_placer"]
            ep_info = info
            global_state = next_global
            steps += 1

        row = {
            "episode": ep,
            "steps": steps,
            "total_reward": total_reward,
            "r_underload": ep_r1,
            "r_overload": ep_r2,
            "r_selector": ep_r3,
            "r_placer": ep_r4,
            "slatah": ep_info.get("slatah", 0.0),
            "pdm": ep_info.get("pdm", 0.0),
            "slav": ep_info.get("slav", 0.0),
            "energy_kwh": ep_info.get("energy_kwh", 0.0),
            "overloads": env.ep_overloads,
            "underloads": env.ep_underloads,
            "migrations": env.ep_migrations,
            "failures": env.ep_failures,
        }
        rows.append(row)
        print(
            f"Eval Ep {ep:02d}/{num_episodes} | R={total_reward:8.2f} | "
            f"OL={env.ep_overloads:4d} UL={env.ep_underloads:3d} "
            f"Mig={env.ep_migrations:3d} Fail={env.ep_failures:2d} | "
            f"SLATAH={row['slatah']:.6f} PDM={row['pdm']:.8f} "
            f"E={row['energy_kwh']:.2f}kWh"
        )
        print(
            f"  R1={ep_r1:.2f} R2={ep_r2:.2f} R3={ep_r3:.2f} R4={ep_r4:.2f}"
        )

    out_path = os.path.join(ROOT, out_csv)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    rewards = np.array([r["total_reward"] for r in rows], dtype=np.float32)
    energies = np.array([r["energy_kwh"] for r in rows], dtype=np.float32)
    slatahs = np.array([r["slatah"] for r in rows], dtype=np.float32)
    migrations = np.array([r["migrations"] for r in rows], dtype=np.float32)
    failures = np.array([r["failures"] for r in rows], dtype=np.float32)

    print("=" * 80)
    print("SUMMARY")
    print(f"episodes={num_episodes}")
    print(f"avg_reward={rewards.mean():.2f}, std_reward={rewards.std():.2f}")
    print(f"avg_energy_kwh={energies.mean():.2f}")
    print(f"avg_slatah={slatahs.mean():.6f}")
    print(f"avg_migrations={migrations.mean():.2f}")
    print(f"total_failures={int(failures.sum())}")
    print(f"csv={out_path}")
    print("=" * 80)
    env.close()


if __name__ == "__main__":
    n = int(os.environ.get("EVAL_EPISODES", "3"))
    run_eval(num_episodes=n)
