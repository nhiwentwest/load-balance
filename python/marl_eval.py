"""
Compatibility wrapper for deterministic MARL evaluation.

The previous version hardcoded legacy Agent 4 dimensions.  Evaluation now uses
the current CloudSimEnv schema so GPU-aware placer states are handled by the
same path as eval_v7_demo.py.
"""
from eval_v7_demo import run_eval


def evaluate_model(num_episodes=10):
    return run_eval(num_episodes=num_episodes, out_csv="marl_eval_results.csv")


if __name__ == "__main__":
    evaluate_model(num_episodes=10)
