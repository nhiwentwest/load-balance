import os
import sys
import json
import time
import socket
import subprocess
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY_DIR = os.path.join(ROOT, "python")
if PY_DIR not in sys.path:
    sys.path.insert(0, PY_DIR)

from autoformer_detector import AutoformerDetector
from cloudsim_gym_env import CloudSimEnv
from config import Config
from lstm_underload_detector import LSTMUnderloadDetector
from models import Actor, select_action


def is_port_open(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0


def kill_port_owner(port):
    try:
        cmd = f"lsof -tiTCP:{port} -sTCP:LISTEN | xargs kill -9 2>/dev/null || true"
        subprocess.run(cmd, shell=True)
        time.sleep(1)
    except Exception:
        pass


def load_actor(obs_dim, action_dim, checkpoint):
    actor = Actor(obs_dim, action_dim)
    state = torch.load(os.path.join(ROOT, checkpoint), map_location="cpu", weights_only=True)
    actor.load_state_dict(state)
    actor.eval()
    return actor


def enable_caching(env):
    cache = {}
    
    orig_reset = env.reset
    orig_step = env.step
    orig_history = env.get_host_history
    orig_vm_counts = env.get_host_vm_counts
    orig_movable = env.get_movable_host_mask
    orig_mobility = env.get_host_mobility_reason_codes
    orig_sync = env._sync_adaptive_thresholds

    synced = [False]

    def cached_reset(*args, **kwargs):
        cache.clear()
        synced[0] = False
        return orig_reset(*args, **kwargs)

    def cached_step(*args, **kwargs):
        cache.clear()
        res = orig_step(*args, **kwargs)
        cache.clear()
        return res

    def cached_history():
        if 'history' not in cache:
            cache['history'] = orig_history()
        return cache['history']

    def cached_vm_counts():
        if 'vm_counts' not in cache:
            cache['vm_counts'] = orig_vm_counts()
        return cache['vm_counts']

    def cached_movable():
        if 'movable' not in cache:
            cache['movable'] = orig_movable()
        return cache['movable']

    def cached_mobility():
        if 'mobility' not in cache:
            cache['mobility'] = orig_mobility()
        return cache['mobility']

    def cached_sync():
        if not synced[0]:
            orig_sync()
            synced[0] = True

    env.reset = cached_reset
    env.step = cached_step
    env.get_host_history = cached_history
    env.get_host_vm_counts = cached_vm_counts
    env.get_movable_host_mask = cached_movable
    env.get_host_mobility_reason_codes = cached_mobility
    env._sync_adaptive_thresholds = cached_sync


def main():
    os.chdir(ROOT)
    torch.set_grad_enabled(False)
    port = 25333
    kill_port_owner(port)

    num_days = int(os.environ.get("NUM_DAYS", "30"))

    print(f"[Export] Starting Java CloudSim bridge for {num_days} days simulation...")
    bridge_cmd = f'java -cp "target/classes:target/dependency/*" com.dacn.advanced.Py4jBridge {port}'
    bridge_proc = subprocess.Popen(bridge_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Wait for bridge to start
    connected = False
    for i in range(15):
        if is_port_open(port):
            connected = True
            break
        time.sleep(1)

    if not connected:
        print("[Error] Could not connect to Java bridge.")
        sys.exit(1)

    print("[Export] Bridge connected! Loading actors...")

    try:
        env = CloudSimEnv()
        enable_caching(env)
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

        days_data = {}

        for day in range(1, num_days + 1):
            print(f"[Export] Simulating Day {day}/{num_days}...")
            global_state = env.reset()
            lstm_detector.reset()

            steps_data = []
            step_idx = 0
            
            while not env.done:
                if step_idx % 20 == 0:
                    print(f"  [Day {day}] Step {step_idx}/288... (time={time.time():.1f})")
                history = env.get_host_history()
                with torch.no_grad():
                    x = torch.tensor(history, dtype=torch.float32)
                    preds = autoformer(x).numpy() # shape [20, pred_len]
                    autoformer_preds = preds.max(axis=1)
                    for h_idx in range(env.num_hosts):
                        if max(history[h_idx]) <= 0.01:
                            autoformer_preds[h_idx] = 0.0

                det_obs = env.get_detector_obs(autoformer_preds)
                ul_mask = env.get_detector_masks(mode="underload")
                ol_mask = env.get_detector_masks(mode="overload")

                current_utils = np.array([det_obs[h * 2] for h in range(env.num_hosts)])
                lstm_detector.update(current_utils)

                # A1 action
                logits_1 = agent1(torch.FloatTensor(det_obs).unsqueeze(0), torch.BoolTensor(ul_mask).unsqueeze(0))
                a1_action = logits_1.squeeze(0).argmax().item()

                # A2 action
                logits_2 = agent2(torch.FloatTensor(det_obs).unsqueeze(0), torch.BoolTensor(ol_mask).unsqueeze(0))
                a2_action = logits_2.squeeze(0).argmax().item()

                predicted_underloads, _ = lstm_detector.detect_with_probs()
                underload_indices, overload_indices, detector_context = env.resolve_detector_actions(
                    det_obs,
                    a1_action,
                    a2_action,
                    predicted_underloads=predicted_underloads,
                )
                if underload_indices:
                    lstm_detector.cooldown.mark_shutdown(underload_indices[0])

                sel_obs = env.build_selector_obs(
                    global_state, overload_indices, det_obs,
                    underload_indices=underload_indices,
                )
                sel_mask = np.ones(env.num_sel_actions, dtype=bool)
                logits_3 = agent3(torch.FloatTensor(sel_obs).unsqueeze(0), torch.BoolTensor(sel_mask).unsqueeze(0))
                selection_action = logits_3.squeeze(0).argmax().item()

                placement_actions = []
                migration_sources = list(overload_indices) + list(underload_indices)
                if migration_sources:
                    vm_states = env.get_migration_placer_obs(
                        overload_indices, underload_indices, selection_action
                    )
                    for vm_state in vm_states:
                        place_mask = env.get_placer_mask(vm_state)
                        logits_4 = agent4(torch.FloatTensor(vm_state).unsqueeze(0), torch.BoolTensor(place_mask).unsqueeze(0))
                        placement_actions.append(logits_4.squeeze(0).argmax().item())

                # Compact Host states: [cpu, predicted_cpu, vm_count, gpu_util, active]
                pre_hosts = []
                gpu_utils = list(env.bridge.getHostGpuUtils())
                vm_counts = env.get_host_vm_counts()
                for h in range(env.num_hosts):
                    cpu_val = float(current_utils[h])
                    pred_val = float(autoformer_preds[h])
                    vm_count = int(vm_counts[h])
                    gpu_util = float(gpu_utils[h])
                    active = 1 if vm_count > 0 else 0
                    pre_hosts.append([cpu_val, pred_val, vm_count, gpu_util, active])

                next_global, rewards, done, info = env.step(
                    underload_indices, overload_indices, selection_action, placement_actions,
                    detector_context=detector_context,
                )

                # Get Java migrations list: [vm_id, source, target]
                if underload_indices or overload_indices:
                    java_migs = list(env.bridge.getLastStepMigrations())
                else:
                    java_migs = []
                migs_list = []
                for jm in java_migs:
                    parts = jm.split(",")
                    if len(parts) == 3:
                        migs_list.append([int(parts[0]), int(parts[1]), int(parts[2])])

                # Compact Agent actions
                a_data = [
                    int(a1_action),
                    int(a2_action),
                    int(selection_action),
                    [int(x) for x in placement_actions],
                    [int(x) for x in underload_indices],
                    [int(x) for x in overload_indices],
                    1 if detector_context.get("a1_true_positive", False) else 0,
                    1 if detector_context.get("a1_false_positive", False) else 0,
                    1 if detector_context.get("a1_false_negative", False) else 0,
                    1 if detector_context.get("a2_true_positive", False) else 0,
                    1 if detector_context.get("a2_false_positive", False) else 0,
                    1 if detector_context.get("a2_false_negative", False) else 0
                ]

                # Compact Metrics
                met_data = [
                    float(info.get("energy_kwh", 0.0)),
                    float(info.get("slatah", 0.0)),
                    float(info.get("pdm", 0.0)),
                    float(info.get("slav", 0.0)),
                    float(sum(rewards.values())),
                    float(rewards["underload_det"]),
                    float(rewards["overload_det"]),
                    float(rewards["vm_selector"]),
                    float(rewards["vm_placer"])
                ]

                steps_data.append({
                    "h": pre_hosts,
                    "a": a_data,
                    "m": migs_list,
                    "met": met_data
                })

                global_state = next_global
                step_idx += 1

            # Summary for this day
            rewards_list = [s["met"][4] for s in steps_data]
            energies_list = [s["met"][0] for s in steps_data]
            slatah_list = [s["met"][1] for s in steps_data]
            migrations_count = sum(len(s["m"]) for s in steps_data)

            days_data[str(day)] = {
                "summary": {
                    "total_reward": float(sum(rewards_list)),
                    "final_energy_kwh": float(energies_list[-1]),
                    "final_slatah": float(slatah_list[-1]),
                    "total_migrations": migrations_count,
                },
                "steps": steps_data
            }

        # Write to js file
        os.makedirs(os.path.join(ROOT, "visualization"), exist_ok=True)
        js_path = os.path.join(ROOT, "visualization", "episode_data.js")
        
        vis_data = {
            "num_days": num_days,
            "days": days_data
        }
        
        with open(js_path, "w", encoding="utf-8") as f:
            f.write(f"window.VIS_DATA = {json.dumps(vis_data)};\n")

        print(f"[Export] Successfully generated 30 days visualization data at: {js_path}")

    finally:
        env.close()
        kill_port_owner(port)


if __name__ == "__main__":
    main()
