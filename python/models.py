"""
models.py — Single source of truth for MARL network definitions.

Before this module existed, ``Actor`` and ``CentralizedCritic`` were copy-pasted
into marl_v4_train.py, lambda_runner.py and hyperparam_tuning.py, each with a
*different* action-selection method (sampling vs. argmax vs. deterministic flag).
Training with one variant and serving with another silently changed the policy.

Every layer now imports these definitions, and every action selection goes through
``select_action(actor, obs, mask, mode)``:
  - mode="train": stochastic sample, returns (action, real log_prob)  -- PPO needs this
  - mode="eval" : deterministic argmax, returns (action, log_prob)    -- serving/HPO

References:
  - MAPPO (Yu et al., NeurIPS 2022)
  - CTDE survey (arXiv 2409.03052, 2024)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    """Actor network — sees only local observation (decentralized execution)."""

    def __init__(self, obs_dim, action_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.head = nn.Linear(hidden, action_dim)
        self.obs_dim = obs_dim
        self.action_dim = action_dim

    def forward(self, x, mask=None):
        h = self.net(x)
        logits = self.head(h)
        if mask is not None:
            logits = logits.masked_fill(~mask, float("-inf"))
        return logits

    def get_action(self, obs, mask=None, deterministic=False):
        """Tensor-in action selection. obs/mask are already batched tensors.

        Kept for the training loop, which prepares tensors itself. New code should
        prefer the module-level select_action() helper.
        """
        with torch.no_grad():
            logits = self.forward(obs, mask).squeeze(0)
            probs = F.softmax(logits, dim=-1)
            if deterministic:
                action = probs.argmax()
            else:
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
            log_prob = torch.log(probs[action] + 1e-8)
        return action.item(), log_prob.item()


class CentralizedCritic(nn.Module):
    """Centralized critic sees GLOBAL state during training (CTDE paradigm)."""

    def __init__(self, global_obs_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)


def select_action(actor, obs, mask=None, mode="train"):
    """Unified action selection for every layer.

    Args:
        actor: an Actor instance.
        obs:   np.ndarray or tensor of local observation (unbatched).
        mask:  optional np.ndarray/tensor action mask (unbatched).
        mode:  "train" -> stochastic sample (returns real log_prob for PPO);
               "eval"  -> deterministic argmax (serving / HPO).

    Returns:
        (action: int, log_prob: float)
    """
    if mode not in ("train", "eval"):
        raise ValueError(f"select_action mode must be 'train' or 'eval', got {mode!r}")
    if isinstance(obs, np.ndarray):
        obs = torch.FloatTensor(obs).unsqueeze(0)
    if mask is not None and isinstance(mask, np.ndarray):
        mask = torch.BoolTensor(mask).unsqueeze(0)
    return actor.get_action(obs, mask, deterministic=(mode == "eval"))
