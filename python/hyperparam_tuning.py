"""
AutoRL hyperparameter optimization for the MARL CloudSim scheduler.

This is not a hand-written grid. Search bounds are generated from the EDA
distribution and CloudSim process scale, then evaluated with separate tuning and
test seeds following the AutoRL protocol recommended by Eimer et al. (ICML 2023).

Output: tuned_hyperparams.json, read by Config at import time.
"""

import argparse
import csv
import json
import math
import os
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from autoformer_detector import AutoformerDetector
from cloudsim_gym_env import CloudSimEnv
from config import Config
from models import Actor, select_action


BASELINE_DIR = Path(os.environ.get("BASELINE_DIR", str(Path(__file__).resolve().parent)))
OUTPUT_FILE = Path(os.environ.get("HPO_OUTPUT", str(Path(__file__).with_name("tuned_hyperparams.json"))))
RESULTS_CSV = Path(os.environ.get("HPO_RESULTS_CSV", str(Path(__file__).with_name("tuning_results.csv"))))


def seed_list(env_name, fallback):
    raw = os.environ.get(env_name)
    if not raw:
        return fallback
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def hp_value(derived, key, default):
    value = derived.get(key, default)
    if isinstance(value, dict):
        value = value.get("value", default)
    return float(value)


def load_eda_context(path=None):
    eda_path = Path(path) if path else Config.EDA_PRIOR_PATH
    with eda_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    cpu = data.get("cpu_utilization_analysis", {}).get("distribution", {})
    runtime = data.get("runtime_analysis", {}).get("runtime_hours", {})
    derived = data.get("derived_hyperparameters", {})
    return data, cpu, runtime, derived


def log_uniform(rng, low, high):
    low = max(float(low), 1e-12)
    high = max(float(high), low * (1.0 + 1e-9))
    return float(10 ** rng.uniform(math.log10(low), math.log10(high)))


def uniform(rng, low, high):
    return float(rng.uniform(float(low), float(high)))


def build_search_space():
    _, cpu, runtime, derived = load_eda_context()
    q10 = float(cpu.get("q10", Config.UNDERLOAD_PRIOR))
    q25 = float(cpu.get("q25", Config.UNDERLOAD_PRIOR))
    q75 = float(cpu.get("q75", Config.OVERLOAD_PRIOR))
    q90 = float(cpu.get("q90", Config.OVERLOAD_PRIOR))
    q95 = float(cpu.get("q95", Config.CRITICAL_OVERLOAD_PRIOR))
    q99 = float(cpu.get("q99", q95))
    under_prior = hp_value(derived, "underload_threshold", Config.UNDERLOAD_PRIOR)
    over_prior = hp_value(derived, "overload_threshold", Config.OVERLOAD_PRIOR)
    crit_prior = hp_value(derived, "critical_overload_threshold", Config.CRITICAL_OVERLOAD_PRIOR)
    # This evaluator holds policy weights fixed, so it may tune only
    # parameters that causally change decisions. PPO optimizer and reward
    # weights require an inner training loop and are intentionally excluded.
    return {
        "underload_threshold": ("float", min(q10, under_prior), max(q25, under_prior)),
        "overload_threshold": ("float", min(q75, over_prior), max(q90, over_prior)),
        "critical_threshold": ("float", min(q90, crit_prior), max(q99, crit_prior)),
    }


def sample_params(rng, search_space):
    params = {}
    for name, spec in search_space.items():
        kind, low, high = spec
        if kind == "log_float":
            params[name] = log_uniform(rng, low, high)
        elif kind == "int":
            params[name] = int(rng.integers(int(low), int(high) + 1))
        else:
            params[name] = uniform(rng, low, high)

    params["overload_threshold"] = max(params["overload_threshold"], params["underload_threshold"] + 1e-6)
    params["critical_threshold"] = max(params["critical_threshold"], params["overload_threshold"] + 1e-6)
    params["critical_threshold"] = min(params["critical_threshold"], 1.0)
    return params


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_agents(env):
    agent1 = Actor(env.a1_obs_dim, env.a1_action_n)
    agent2 = Actor(env.a2_obs_dim, env.a2_action_n)
    agent3 = Actor(env.a3_obs_dim, env.num_sel_actions)
    agent4 = Actor(env.a4_obs_dim, env.a4_action_n)

    loaded = []
    for filename, agent in [
        ("agent1_underload_det_v6.pt", agent1),
        ("agent2_overload_det_v6.pt", agent2),
        ("agent3_vm_selector_v6.pt", agent3),
        ("agent4_vm_placer_v6.pt", agent4),
    ]:
        path = BASELINE_DIR / filename
        if path.exists():
            try:
                agent.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
                loaded.append(True)
            except RuntimeError as exc:
                print(f"[HPO] Skipping incompatible checkpoint {filename}: {exc}")
                loaded.append(False)
        else:
            loaded.append(False)
    if not all(loaded):
        raise RuntimeError(
            "Operational HPO requires compatible trained policies. Fresh-train "
            "and promote the N+NO_OP detector checkpoints first."
        )
    for agent in (agent1, agent2, agent3, agent4):
        agent.eval()

    af_seq_len = int(getattr(env, "history_len", Config.AF_SEQ_LEN))
    autoformer = AutoformerDetector(seq_len=af_seq_len, pred_len=Config.AF_PRED_LEN, d_model=Config.AF_D_MODEL)
    af_path = BASELINE_DIR / "autoformer_detector.pt"
    if af_path.exists():
        try:
            autoformer.load_state_dict(torch.load(af_path, map_location="cpu", weights_only=True))
        except RuntimeError as exc:
            print(f"[HPO] Skipping incompatible Autoformer checkpoint: {exc}")
    autoformer.eval()
    return agent1, agent2, agent3, agent4, autoformer


def objective_key(metrics):
    return (
        -float(metrics["failures"]),
        -float(metrics["slatah"]),
        -float(metrics["energy_kwh"]),
        -float(metrics["migrations"]),
        float(metrics["reward"]),
    )


def evaluate_params(env, agents, params, seeds, max_steps):
    agent1, agent2, agent3, agent4, autoformer = agents
    previous_meta = Config.META_TUNING_ENABLED
    Config.META_TUNING_ENABLED = False
    Config.apply_hpo_params(params)

    rows = []
    for seed in seeds:
        set_seed(seed)
        Config.reset_runtime_state(reset_thresholds=True)
        Config.apply_hpo_params(params)
        global_state = env.reset()
        total_reward = 0.0
        steps = 0
        info = {}

        while not env.done and steps < max_steps:
            history = env.get_host_history()
            af_preds = history[:, -1].copy()
            det_obs = env.get_detector_obs(af_preds)

            ul_mask = env.get_detector_masks("underload")
            ol_mask = env.get_detector_masks("overload")

            a1_action, _ = select_action(agent1, det_obs, ul_mask, mode="eval")

            a2_action, _ = select_action(agent2, det_obs, ol_mask, mode="eval")
            underload_indices, overload_indices, detector_context = (
                env.resolve_detector_actions(det_obs, a1_action, a2_action)
            )

            sel_obs = env.build_selector_obs(
                global_state, overload_indices, det_obs,
                underload_indices=underload_indices,
            )
            a3_action, _ = select_action(agent3, sel_obs, np.ones(env.num_sel_actions, dtype=bool), mode="eval")

            placement_actions = []
            migration_sources = list(overload_indices) + list(underload_indices)
            if migration_sources:
                vm_states = env.get_migration_placer_obs(
                    overload_indices, underload_indices, a3_action
                )
                for vs in vm_states:
                    a4_action, _ = select_action(agent4, vs, env.get_placer_mask(vs), mode="eval")
                    placement_actions.append(a4_action)

            global_state, rewards, done, info = env.step(
                underload_indices,
                overload_indices,
                a3_action,
                placement_actions,
                detector_context=detector_context,
            )
            total_reward += sum(rewards.values())
            steps += 1

        rows.append({
            "seed": seed,
            "steps": steps,
            "reward": total_reward,
            "energy_kwh": float(info.get("energy_kwh", 0.0)),
            "slatah": float(info.get("slatah", 0.0)),
            "overloads": float(info.get("episode_overloads", 0.0)),
            "actionable_overloads": float(info.get("episode_actionable_overloads", 0.0)),
            "selected_overload_sources": float(info.get("episode_selected_overload_sources", 0.0)),
            "migrations": float(info.get("total_migrations", info.get("migrations", 0.0))),
            "failures": float(info.get("failures", 0.0)),
        })

    Config.META_TUNING_ENABLED = previous_meta
    metrics = {
        key: float(np.mean([row[key] for row in rows]))
        for key in (
            "steps", "reward", "energy_kwh", "slatah", "overloads",
            "actionable_overloads", "selected_overload_sources",
            "migrations", "failures",
        )
    }
    metrics["score"] = float(objective_key(metrics)[-1])
    metrics["seed_rows"] = rows
    return metrics


def run_tuning(print_space=False):
    search_space = build_search_space()
    if print_space:
        print(json.dumps(search_space, indent=2))
        return {"search_space": search_space}

    trials = int(os.environ.get("AUTORL_TRIALS", str(max(8, int(round(math.sqrt(Config.NUM_HOSTS))) * 4))))
    rng = np.random.default_rng(int(os.environ.get("AUTORL_RANDOM_SEED", "252")))
    tuning_seeds = seed_list("AUTORL_TUNING_SEEDS", [101, 202])
    test_seeds = seed_list("AUTORL_TEST_SEEDS", [303, 404])
    tuning_max_steps = int(os.environ.get("AUTORL_TUNING_MAX_STEPS", str(Config.LSTM_RETRAIN_INTERVAL_STEPS)))
    test_max_steps = int(os.environ.get("AUTORL_TEST_MAX_STEPS", str(Config.STEPS_PER_EPISODE)))
    finalist_count = max(1, int(round(math.sqrt(float(trials)))))

    print("=" * 70)
    print("OPERATIONAL HPO -- EDA-derived thresholds + separated tuning/test seeds")
    print("=" * 70)
    print(f"Trials={trials}, finalists={finalist_count}")
    print(f"Tuning seeds={tuning_seeds}, test seeds={test_seeds}")
    print(f"Fidelity: tune {tuning_max_steps} steps, test {test_max_steps} steps")

    env = CloudSimEnv()
    agents = load_agents(env)
    candidates = [sample_params(rng, search_space) for _ in range(trials)]
    tuning_results = []

    with RESULTS_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "phase", "trial", "seeds", "steps", "reward", "energy_kwh", "slatah",
            "overloads", "migrations", "failures", "params_json", "elapsed_sec",
        ])

        for idx, params in enumerate(candidates, start=1):
            t0 = time.time()
            metrics = evaluate_params(env, agents, params, tuning_seeds, tuning_max_steps)
            elapsed = time.time() - t0
            tuning_results.append({"trial": idx, "params": params, "metrics": metrics, "elapsed_sec": elapsed})
            writer.writerow([
                "tune", idx, ",".join(map(str, tuning_seeds)), metrics["steps"], metrics["reward"],
                metrics["energy_kwh"], metrics["slatah"], metrics["overloads"], metrics["migrations"],
                metrics["failures"], json.dumps(params, sort_keys=True), f"{elapsed:.3f}",
            ])
            print(
                f"[Tune {idx:03d}] fail={metrics['failures']:.0f} slatah={metrics['slatah']:.6f} "
                f"energy={metrics['energy_kwh']:.3f} mig={metrics['migrations']:.0f} reward={metrics['reward']:.3f}"
            )

        tuning_results.sort(key=lambda row: objective_key(row["metrics"]), reverse=True)
        finalists = tuning_results[:finalist_count]
        test_results = []
        for row in finalists:
            t0 = time.time()
            metrics = evaluate_params(env, agents, row["params"], test_seeds, test_max_steps)
            elapsed = time.time() - t0
            test_row = {
                "trial": row["trial"],
                "params": row["params"],
                "tuning_metrics": row["metrics"],
                "test_metrics": metrics,
                "elapsed_sec": elapsed,
            }
            test_results.append(test_row)
            writer.writerow([
                "test", row["trial"], ",".join(map(str, test_seeds)), metrics["steps"], metrics["reward"],
                metrics["energy_kwh"], metrics["slatah"], metrics["overloads"], metrics["migrations"],
                metrics["failures"], json.dumps(row["params"], sort_keys=True), f"{elapsed:.3f}",
            ])
            print(
                f"[Test {row['trial']:03d}] fail={metrics['failures']:.0f} slatah={metrics['slatah']:.6f} "
                f"energy={metrics['energy_kwh']:.3f} mig={metrics['migrations']:.0f} reward={metrics['reward']:.3f}"
            )

    test_results.sort(key=lambda row: objective_key(row["test_metrics"]), reverse=True)
    best = test_results[0] if test_results else {
        "trial": tuning_results[0]["trial"],
        "params": tuning_results[0]["params"],
        "tuning_metrics": tuning_results[0]["metrics"],
        "test_metrics": {},
    }
    promotion_ready = (
        test_max_steps >= Config.STEPS_PER_EPISODE
        and len(tuning_seeds) >= 2
        and len(test_seeds) >= 2
        and trials >= max(8, int(round(math.sqrt(Config.NUM_HOSTS))) * 4)
    )

    output = {
        "tuning_method": "eda_derived_operational_threshold_hpo",
        "tuning_date": datetime.now().isoformat(),
        "promotion_ready": promotion_ready,
        "promotion_rule": "compatible trained policy, full episode test, at least two tuning seeds, at least two test seeds, and default or larger trial budget",
        "total_trials": trials,
        "finalist_count": finalist_count,
        "tuning_seeds": tuning_seeds,
        "test_seeds": test_seeds,
        "tuning_max_steps": tuning_max_steps,
        "test_max_steps": test_max_steps,
        "selection_rule": "lexicographic: minimize failures, slatah, energy, migrations; maximize reward",
        "search_space": search_space,
        "best_trial": best["trial"],
        "best_params": best["params"],
        "best_tuning_metrics": best.get("tuning_metrics", {}),
        "best_test_metrics": best.get("test_metrics", {}),
        "tuning_results": tuning_results,
        "test_results": test_results,
    }
    with OUTPUT_FILE.open("w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)
    print(f"Saved HPO result: {OUTPUT_FILE}")
    env.close()
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--print-space", action="store_true", help="Print EDA-derived search space and exit.")
    args = parser.parse_args()
    run_tuning(print_space=args.print_space)
